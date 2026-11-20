"""Lossy bounded observer: actual training frames, no RNG, no reevaluation.

The training thread never waits for the GUI or disk. A single background writer
replaces a small mailbox, without fsync/retry sleeps. Locked files drop telemetry,
never checkpoints. Sampling is wall-clock throttled and does not enter fitness.
"""
from collections import deque
from .settings import Limits
import json
import os
from pathlib import Path
import queue
import threading
import time
import uuid


def write_mailbox(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.tmp')
    try:
        temp.write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')
        os.replace(temp, path)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


class LiveObserver:
    def __init__(self, experiment, interval=.1, max_frames=128):
        self.folder = Path(experiment) / 'live'
        self.interval = max(.02, interval)
        self.run_id = uuid.uuid4().hex
        self.frames = deque(maxlen=max_frames)
        self.path = deque(maxlen=512)
        self.mailbox = queue.Queue(maxsize=1)
        self.closed = threading.Event()
        self.last_publish = 0.
        self.last_episode_at = 0.
        self.last_control = 0.
        self.selector = {'individual': -1, 'scenario': 'all', 'stride': 1}
        self.context = {}
        self.packet = {}
        self.episode_number = 0
        self.sequence = 0
        self.dropped = 0
        self.write_errors = 0
        self.thread = threading.Thread(target=self._writer, name='live-observer', daemon=True)
        self.thread.start()

    def set_context(self, **context):
        self.context = dict(context)
        now = time.monotonic()
        if now - self.last_control >= .5:
            self.last_control = now
            try:
                data = json.loads((self.folder / 'control.json').read_text(encoding='utf-8'))
                individual = int(data.get('individual', -1))
                self.selector = {'individual': individual if -1 <= individual <= Limits.MAX_POPULATION else -1,
                                 'scenario': str(data.get('scenario', 'all')),
                                 'stride': max(1, min(100, int(data.get('stride', 1))))}
            except (OSError, ValueError, TypeError, AttributeError):
                pass

    def begin_episode(self, net, scenario, seed):
        slot = self.selector['individual']
        selected = ((slot < 0 or slot == self.context.get('individual')) and
                    self.selector['scenario'] in ('all', scenario['name']))
        now=time.monotonic()
        if not selected or now-self.last_episode_at<self.interval:
            return False
        self.last_episode_at=now
        self.episode_number += 1
        self.frames.clear()
        self.path.clear()
        self.packet = {'run_id': self.run_id, 'episode_id': self.episode_number,
                       'experiment': self.folder.parent.name, **self.context,
                       'scenario': scenario['name'], 'seed': seed,
                       'map': scenario['map'], 'brain': net.export(),
                       'source': 'Live Training', 'status': 'running'}
        return True

    def add_frame(self, frame):
        self.path.append({'t':frame['t'],'position':list(frame['predator'])})
        if frame['t'] % self.selector['stride'] and not frame.get('captured'):
            return
        self.frames.append(frame)
        self.publish()

    def end_episode(self):
        self.publish(force=True)

    def publish(self, force=False, status=None):
        now = time.monotonic()
        if not force and now - self.last_publish < self.interval:
            return
        self.last_publish = now
        self.sequence += 1
        packet = {**self.packet, 'run_id': self.run_id, 'sequence': self.sequence,
                  'timestamp': time.time(), 'frames': list(self.frames), 'trajectory':list(self.path),
                  'dropped_packets': self.dropped, 'write_errors': self.write_errors}
        if status:
            packet['status'] = status
        try:
            self.mailbox.put_nowait(packet)
        except queue.Full:
            try:
                self.mailbox.get_nowait()
            except queue.Empty:
                pass
            self.dropped += 1
            try:
                self.mailbox.put_nowait(packet)
            except queue.Full:
                pass

    def _writer(self):
        while not self.closed.is_set() or not self.mailbox.empty():
            try:
                packet = self.mailbox.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                write_mailbox(self.folder / 'latest.json', packet)
            except (OSError, ValueError, TypeError):
                self.write_errors += 1

    def close(self, status='finished'):
        self.publish(force=True, status=status)
        self.closed.set()
        self.thread.join(timeout=1.)


class LiveBuffer:
    """GUI-side deduplication, bounded playback and explicit stale status."""
    def __init__(self, limit=512, individual=-1, scenario="all"):
        self.individual=individual
        self.scenario=scenario
        self.pending = deque(maxlen=limit)
        self.current = None
        self.run_id = None
        self.sequence = -1
        self.last_key = (-1, -1)
        self.timestamp = 0.
        self.status = 'unavailable'
        self.error = ''
        self.file_stamp = None
        self.last_read = 0.

    def poll(self, experiment):
        try:
            path=Path(experiment)/'live/latest.json'
            stat=path.stat()
            stamp=(path,stat.st_ino,stat.st_size,stat.st_mtime_ns)
            now=time.monotonic()
            # Atomic replacement changes file identity. Recheck periodically on
            # filesystems with coarse timestamps or unavailable file IDs.
            if stamp==self.file_stamp and now-self.last_read<1. and not self.error:
                return False
            packet=json.loads(path.read_text(encoding='utf-8'))
            self.file_stamp=stamp
            self.last_read=now
            if ((self.individual>=0 and packet.get('individual')!=self.individual) or
                    self.scenario not in ('all',packet.get('scenario'))):
                self.status='in attesa della selezione richiesta'
                self.timestamp=packet.get('timestamp',0.)
                return False
            if packet.get('run_id') != self.run_id:
                self.pending.clear()
                self.current = None
                self.last_key = (-1, -1)
                self.sequence = -1
                self.run_id = packet.get('run_id')
            if packet['sequence'] <= self.sequence:
                self.error=''
                return False
            self.sequence = packet['sequence']
            self.timestamp = packet['timestamp']
            self.status = packet.get('status', 'running')
            metadata = {k: v for k, v in packet.items() if k != 'frames'}
            for frame in packet['frames']:
                key = (packet.get('episode_id', 0), frame['t'])
                if key > self.last_key:
                    self.pending.append((metadata, frame))
                    self.last_key = key
            self.error = ''
            return True
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            self.error = type(exc).__name__
            return False

    def advance(self):
        if self.pending:
            self.current = self.pending.popleft()
        return self.current

    def latest(self):
        if self.pending:
            self.current = self.pending[-1]
            self.pending.clear()
        return self.current

    def label(self):
        age = time.time() - self.timestamp
        if not self.timestamp:
            return 'Live non disponibile: in attesa di episodi realmente eseguiti.'
        status = self.status
        if self.error or (age > 3 and status == 'running'):
            status = 'nessun dato recente / cache / processo interrotto'
        return f'{status} · ultimo pacchetto {max(0, age):.1f}s fa · buffer {len(self.pending)}'
