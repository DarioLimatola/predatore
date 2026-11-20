"""Versioned, self-contained experiment vault with atomic writes and checksums."""
import hashlib
import json
import os
import tempfile
import time
import uuid
import warnings
import re
import shutil
from pathlib import Path

LEGACY_SCHEMA = 'predatore.evolution-vault/3.0'
SCHEMA = 'predatore.evolution-vault/3.1'
CHECKPOINT_HASH = '_checkpoint_sha256'

# UI-generated logs and stop signals are volatile: do not include them in an immutable experiment seal.
VOLATILE_FILES = {'gui_worker.log', 'stop_requested'}

def _replace_with_retry(src, dst, attempts=5):
    """Handle short Windows/OneDrive/AV sharing violations without losing the old file.

    We *never* delete the destination first: the old checkpoint must survive.
    """
    for trial in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError as exc:
            # Only retry known transient access/sharing failures. Permanent access
            # errors are still reported to the caller once attempts are exhausted.
            if trial == attempts - 1:
                raise
            time.sleep(min(0.12 * (2 ** trial), 1.0))


def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(obj, f, separators=(',',':'), sort_keys=True)
            f.flush(); os.fsync(f.fileno())
        _replace_with_retry(tmp, path)
    finally:
        if os.path.exists(tmp):
            try: os.unlink(tmp)
            except PermissionError:
                # An external scanner may still briefly hold this temporary file.
                # The complete target, if any, remains available for recovery.
                pass


def checkpoint_paths(directory):
    """All recoverable snapshots, including alternative saves when latest is locked."""
    directory = Path(directory)
    paths = []
    primary = directory / 'latest.json'
    if primary.is_file(): paths.append(primary)
    paths.extend(directory.glob('latest.g*.fallback-*.json'))
    previous = directory / 'previous.json'
    if previous.is_file():
        paths.append(previous)
    return paths


def _checkpoint_generation(path):
    try:
        state = read_json(path)
        if (isinstance(state, dict) and type(state.get('generation')) is int
                and state['generation'] >= 0 and isinstance(state.get('history'), list)
                and isinstance(state.get('config'), dict)
                and (isinstance(state.get('population'), list) or
                     state.get('schema') == 'smith.supervisor/2' and isinstance(state.get('eva'), dict))):
            return state['generation']
    except (OSError, ValueError, TypeError):
        pass
    return -1


def latest_checkpoint_path(directory):
    """Prefer the newest verified commit, including a retained ordinary predecessor."""
    directory = Path(directory)
    primary = directory / 'latest.json'
    generation = _checkpoint_generation(primary)
    best = primary if generation >= 0 else None
    if best is None:
        previous = directory / 'previous.json'
        generation = _checkpoint_generation(previous)
        if generation >= 0:
            best = previous
    fallbacks = []
    for path in directory.glob('latest.g*.fallback-*.json'):
        match = re.fullmatch(r'latest\.g(\d+)\.fallback-[a-f0-9]+\.json', path.name)
        if match and int(match[1]) > generation:
            fallbacks.append((int(match[1]), path))
    for expected, path in sorted(fallbacks, reverse=True):
        if _checkpoint_generation(path) == expected:
            return path
    return best


def _retain_previous(path, generation):
    if not path.is_file():
        return
    current = _checkpoint_generation(path)
    if current < 0 or current >= generation:
        return
    temp = path.parent / ('.previous-' + uuid.uuid4().hex + '.tmp')
    try:
        try:
            os.link(path, temp)
        except OSError:
            shutil.copyfile(path, temp)
        _replace_with_retry(temp, path.parent / 'previous.json')
    finally:
        temp.unlink(missing_ok=True)


def save_checkpoint(path, obj):
    """Commit JSON and its digest together; rotate the previous successful generation."""
    path = Path(path)
    payload = json.dumps(obj, separators=(',', ':'), sort_keys=True, allow_nan=False)
    digest = hashlib.sha256(payload.encode('utf-8')).hexdigest()
    signed = '{"' + CHECKPOINT_HASH + '":"' + digest + '",' + payload[1:]
    try:
        _retain_previous(path, obj['generation'])
        _atomic_text(path, signed)
        return path
    except PermissionError as original_error:
        generation = int(obj['generation'])
        fallback = path.with_name(f'latest.g{generation:09d}.fallback-{uuid.uuid4().hex[:12]}.json')
        try:
            _atomic_text(fallback, signed)
        except OSError as secondary_error:
            raise OSError(
                f'Checkpoint {generation} was NOT saved. Windows denied both '
                f'{path} and {fallback}. Close file viewers and move Predatore '
                f'out of OneDrive; earlier valid checkpoints were not removed.'
            ) from secondary_error
        warnings.warn(
            f'Windows temporarily denied replacement of {path.name}; '
            f'generation {generation} safely saved as {fallback.name}. '
            f'Resume and GUI will detect this snapshot automatically. '
            f'Consider moving Predatore out of OneDrive. '
            f'Original error: {original_error}', RuntimeWarning, stacklevel=2)
        return fallback

def _atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.' + path.name + '-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temp, path)
    finally:
        if os.path.exists(temp):
            try: os.unlink(temp)
            except PermissionError: pass


def reconcile_evolution_logs(experiment_dir, committed_generation):
    """On resume, archive and remove event lines newer than the saved checkpoint.

    Each generation is committed only when its checkpoint is durable. The old
    training loop appended events *before* committing its checkpoint. On an
    unclean exit, those extra rows must not be counted a second time.
    Mutation/compression rows describe offspring for generation gen+1, so
    they are committed through saved_generation+1.
    """
    experiment_dir = Path(experiment_dir)
    cutoffs = {'generations.jsonl': committed_generation,
               'consolidation.jsonl': committed_generation,
               'mutations.jsonl': committed_generation + 1,
               }
    for filename, cutoff in cutoffs.items():
        path = experiment_dir / 'evolution' / filename
        if not path.is_file():
            continue
        raw = path.read_text(encoding='utf-8')
        kept, discarded = [], []
        for line in raw.splitlines(keepends=True):
            try:
                event = json.loads(line)
                if isinstance(event, dict) and type(event.get('generation')) is int and event['generation'] > cutoff:
                    discarded.append(line)
                else:
                    kept.append(line)
            except (ValueError, TypeError):
                # Preserve but isolate incomplete trailing lines after a crash.
                discarded.append(line)
        if discarded:
            # Keep the original uncommitted evidence, so no past research data
            # is silently destroyed by log reconciliation.
            recovery = experiment_dir / 'recovery' / f'pre_resume_g{committed_generation:09d}_{filename[:-6]}_{uuid.uuid4().hex[:8]}.jsonl'
            _atomic_text(recovery, ''.join(discarded))
            _atomic_text(path, ''.join(kept))
            warnings.warn(f'Archived {len(discarded)} uncommitted event lines from {filename} to {recovery.name}',
                          RuntimeWarning, stacklevel=2)

def read_json(path):

    path = Path(path)
    if path.name not in {'latest.json', 'previous.json'} and not path.name.startswith('latest.g'):
        return json.loads(path.read_bytes())
    raw = path.read_bytes()
    prefix = b'{"_checkpoint_sha256":"'
    if raw.startswith(prefix):
        start = len(prefix)
        if raw[start + 64:start + 66] != b'",':
            raise ValueError(f'Invalid checkpoint header in {path.name}')
        payload = b'{' + raw[start + 66:]
        if hashlib.sha256(payload).hexdigest().encode('ascii') != raw[start:start + 64]:
            raise ValueError(f'Checkpoint checksum mismatch in {path.name}')
        return json.loads(payload)
    state = json.loads(raw)
    if isinstance(state, dict) and (CHECKPOINT_HASH in state or state.get('schema') == SCHEMA
                                   or ('transaction' in state and 'engine_version' in state)):
        raise ValueError(f'Missing or displaced checkpoint checksum in {path.name}')
    return state

def append_jsonl(path, item):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(item, sort_keys=True) + '\n')

def fingerprint(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def experiment_input_hashes(folder):
    """Fingerprint the fixed inputs used to compare generations of this run."""
    folder = Path(folder)
    paths = [folder / name for name in ('config.json', 'manifest.json', 'brains/initial_eva.json',
             'brains/initial_explorers.json', 'environment/seed_protocol.json')]
    paths.extend(sorted((folder / 'environment/suites').glob('*.json')))
    hashes = {}
    for path in paths:
        if not path.is_file():
            continue
        name = path.relative_to(folder).as_posix()
        if name in ('manifest.json', 'config.json'):
            # Run creation time and the initial requested stop generation do not define its science.
            fields = read_json(path)
            fields.pop('created_unix' if name == 'manifest.json' else 'generations', None)
            encoded = json.dumps(fields, sort_keys=True, separators=(',', ':')).encode()
            hashes[name] = hashlib.sha256(encoded).hexdigest()
        else:
            hashes[name] = fingerprint(path)
    return hashes


def verify_experiment_inputs(folder, snapshot):
    expected = snapshot.get('input_sha256')
    actual = experiment_input_hashes(folder)
    changed = set()
    if expected is not None:
        changed.update(k for k in actual.keys() | expected.keys() if actual.get(k) != expected.get(k))
    seal_path = folder / 'integrity/checksums.json'
    if seal_path.exists():
        historical = read_json(seal_path)['files']
        for name, digest in historical.items():
            if name in {'config.json', 'manifest.json', 'brains/initial_eva.json',
                        'brains/initial_explorers.json', 'environment/seed_protocol.json'} or name.startswith('environment/suites/'):
                if name not in actual or fingerprint(folder/name) != digest:
                    changed.add(name)
    if changed:
        raise ValueError('Input sperimentali modificati: ' + ', '.join(sorted(changed)))
    return actual


def seal(folder):
    folder=Path(folder); result={}
    for path in sorted(folder.rglob('*')):
        if path.is_file() and path.relative_to(folder).as_posix() != 'integrity/checksums.json' and path.name not in VOLATILE_FILES and 'live' not in path.relative_to(folder).parts:
            result[path.relative_to(folder).as_posix()]=fingerprint(path)
    atomic_json(folder/'integrity'/'checksums.json', {'schema':SCHEMA,'files':result})
    return len(result)

def verify(folder):
    folder=Path(folder); manifest=read_json(folder/'integrity'/'checksums.json')
    return {name:(folder/name).exists() and fingerprint(folder/name)==digest for name,digest in manifest['files'].items()}
