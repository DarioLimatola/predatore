"""Versioned, self-contained experiment vault with atomic writes and checksums."""
import hashlib
import json
import os
import tempfile
import time
import uuid
import warnings
import re
from pathlib import Path

SCHEMA = 'predatore.evolution-vault/2.0'

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
            json.dump(obj, f, indent=2, sort_keys=True)
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
    return paths


def latest_checkpoint_path(directory):
    """Newest valid checkpoint; efficient with many immutable backup snapshots."""
    directory = Path(directory)
    primary = directory / 'latest.json'
    best_generation = -1
    best_path = None
    if primary.is_file():
        try:
            state = read_json(primary)
            if (type(state.get('generation')) is int and state['generation'] >= 0
                    and (isinstance(state.get('population'), list) or
                         state.get('schema') == 'smith.supervisor/1' and isinstance(state.get('eva'), dict))
                    and isinstance(state.get('history'), list)
                    and isinstance(state.get('config'), dict)):
                best_generation, best_path = state['generation'], primary
        except (OSError, ValueError, TypeError):
            pass

    # Read at most a few candidate JSON files rather than parsing the entire
    # checkpoint history for every 1.5-second GUI refresh.
    fallbacks = []
    for path in directory.glob('latest.g*.fallback-*.json'):
        match = re.fullmatch(r'latest\.g(\d+)\.fallback-[a-f0-9]+\.json', path.name)
        if match and int(match.group(1)) > best_generation:
            fallbacks.append((int(match.group(1)), path))
    for generation, path in sorted(fallbacks, reverse=True):
        if generation <= best_generation:
            break
        try:
            state = read_json(path)
            if (type(state.get('generation')) is int and state['generation'] == generation
                    and (isinstance(state.get('population'), list) or
                         state.get('schema') == 'smith.supervisor/1' and isinstance(state.get('eva'), dict))
                    and isinstance(state.get('history'), list)
                    and isinstance(state.get('config'), dict)):
                return path
        except (OSError, ValueError, TypeError):
            pass
    return best_path

def save_checkpoint(path, obj):
    """Commit a recoverable checkpoint even when Windows locks latest.json.

    The normal on-disk representation is unchanged. If the standard rename is
    denied repeatedly, commit the same JSON to a unique, immutable sibling and
    teach both the GUI and resume logic to select its higher generation.
    """
    path = Path(path)
    try:
        atomic_json(path, obj)
        return path
    except PermissionError as original_error:
        generation = int(obj['generation'])
        fallback = path.with_name(f'latest.g{generation:09d}.fallback-{uuid.uuid4().hex[:12]}.json')
        try:
            atomic_json(fallback, obj)
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
    return json.loads(Path(path).read_text(encoding='utf-8'))

def append_jsonl(path, item):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(item, sort_keys=True) + '\n')

def fingerprint(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

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
