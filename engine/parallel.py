"""Process pool for independent, isolated assessments.

Each job restores its own disposable copy from an exported brain, exactly like
``evaluation_copy``; nothing flows back into persistent agents. Results are the
same as the sequential path, only computed on several CPU cores.

PREDATORE_WORKERS=1 disables the pool; any other integer fixes the worker count.
"""
import atexit
import os

_pool = None
_size = None


def workers():
    value = os.environ.get('PREDATORE_WORKERS')
    if value is not None:
        return max(1, int(value))
    return max(1, min(8, (os.cpu_count() or 1) - 1))


def _get_pool():
    global _pool, _size
    if _pool is None:
        _size = workers()
        if _size <= 1:
            return None
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        # 'spawn' behaves the same on Windows, Linux and macOS.
        _pool = ProcessPoolExecutor(_size, mp_context=mp.get_context('spawn'))
        atexit.register(shutdown)
    return _pool


def shutdown():
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None


def _assess(exported, suites, seed, count, steps, split):
    from .boolean_net import BooleanNet
    from .continual import assessment
    return assessment(BooleanNet.restore(exported), suites, seed, count, steps, split)


def assess_many(jobs):
    """jobs: list of (exported_brain, suites, seed, count, steps, split). Ordered results."""
    pool = _get_pool() if len(jobs) > 1 else None
    if pool is None:
        return [_assess(*job) for job in jobs]
    try:
        return list(pool.map(_assess, *zip(*jobs)))
    except Exception:
        # A broken pool (e.g. a killed worker) must not stop training: fall back.
        shutdown()
        return [_assess(*job) for job in jobs]
