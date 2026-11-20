"""Parallel execution of independent, isolated work (assessments, explorer training).

Every job works on its own disposable brain; nothing flows back into persistent
agents except through the caller, in a fixed order. Results are therefore the same
with 1 or N workers (tests/test_v7_parallel.py).

- Native kernel available: a thread pool. Episodes run in Rust without Python's
  global lock, so threads share brains without copying them between processes.
- Pure Python: a process pool (spawn), which receives exported brains.

PREDATORE_WORKERS sets the number of workers (default 16, at most the CPU threads).
PREDATORE_WORKERS=1 runs everything in the calling thread.
"""
import atexit
import os
import sys
import threading

DEFAULT_WORKERS = 16

_threads = None
_processes = None
_lock = threading.Lock()
_fallbacks = 0


def workers():
    value = os.environ.get('PREDATORE_WORKERS')
    if value is not None:
        return max(1, int(value))
    return max(1, min(DEFAULT_WORKERS, os.cpu_count() or 1))


def native():
    from boolean_engine import native_backend
    return native_backend.AVAILABLE


def status():
    return {'workers': workers(), 'mode': ('threads' if native() else 'processes') if workers() > 1 else 'serial',
            'fallbacks': _fallbacks}


def _thread_pool():
    global _threads
    with _lock:
        if _threads is None:
            from concurrent.futures import ThreadPoolExecutor
            _threads = ThreadPoolExecutor(workers(), thread_name_prefix='predatore')
            atexit.register(shutdown)
        return _threads


def _process_pool():
    global _processes
    with _lock:
        if _processes is None:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            # 'spawn' behaves the same on Windows, Linux and macOS.
            _processes = ProcessPoolExecutor(workers(), mp_context=mp.get_context('spawn'))
            atexit.register(shutdown)
        return _processes


def shutdown():
    global _threads, _processes
    for pool in (_threads, _processes):
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
    _threads = _processes = None


def _report_fallback(exc):
    global _fallbacks
    _fallbacks += 1
    print(f'[parallel] pool failed ({type(exc).__name__}: {exc}); continuing in one thread',
          file=sys.stderr, flush=True)


def map_ordered(fn, items):
    """fn over items, results in input order. Threads when native, else in-process."""
    items = list(items)
    if workers() <= 1 or len(items) < 2 or not native():
        return [fn(item) for item in items]
    try:
        return list(_thread_pool().map(fn, items))
    except RuntimeError as exc:  # pool shut down during interpreter exit
        _report_fallback(exc)
        return [fn(item) for item in items]


def _assess(net, suites, seed, count, steps, split):
    from .continual import assessment
    return assessment(net, suites, seed, count, steps, split)


def _assess_exported(exported, suites, seed, count, steps, split):
    from .boolean_net import BooleanNet
    return _assess(BooleanNet.restore(exported), suites, seed, count, steps, split)


def assess_many(jobs):
    """jobs: list of (brain, suites, seed, count, steps, split). Ordered results.

    `brain` is a network or an exported brain dict. Every assessment works on its own
    evaluation copy, so jobs never share state.
    """
    from .boolean_net import BooleanNet
    jobs = [((BooleanNet.restore(b) if isinstance(b, dict) else b), *rest) for b, *rest in jobs]
    if workers() <= 1 or len(jobs) < 2:
        return [_assess(*job) for job in jobs]
    if native():
        return map_ordered(lambda job: _assess(*job), jobs)
    try:
        exported = [(net.export(), *rest) for net, *rest in jobs]
        return list(_process_pool().map(_assess_exported, *zip(*exported)))
    except Exception as exc:
        # A broken pool (e.g. a killed worker) must not stop training: visible fallback.
        _report_fallback(exc)
        shutdown()
        return [_assess(*job) for job in jobs]
