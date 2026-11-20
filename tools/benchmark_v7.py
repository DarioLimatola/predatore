"""V7-A: measure Predatore on this machine (Windows, Linux, macOS).

For each worker count, the same short training is repeated and timed: per
generation, final evaluation, total. The final EVA identity must be the same for
every worker count (parallelism never changes results). The report is printed and
saved under validation/v7/.

    python tools/benchmark_v7.py                 # 1, 4, 8, 16 workers, 3 repetitions
    python tools/benchmark_v7.py --workers 1 16 --generations 10 --repeat 5
    python tools/benchmark_v7.py --python        # the pure-Python reference, for comparison
"""
import argparse
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def one_run(generations, seed, runs):
    """`runs` consecutive runs on one SMITH profile (later runs carry more retained tasks)."""
    from engine import continual
    from engine.evolution import Config, run
    from engine.continual import load_profile
    from boolean_engine import BinaryVectorNetwork
    root = tempfile.mkdtemp(prefix='predatore-bench-')
    timings = []
    original_finish = continual._finish
    for r in range(runs):
        gens, finish, last = [], [0.0], [time.perf_counter()]

        def progress(*_a, **_k):
            now = time.perf_counter()
            gens.append(now - last[0])
            last[0] = now

        def timed_finish(*a, **k):
            t = time.perf_counter()
            try:
                return original_finish(*a, **k)
            finally:
                finish[0] = time.perf_counter() - t
        continual._finish = timed_finish
        start = time.perf_counter()
        try:
            run(root, Config(generations=generations, seed=seed + r), progress=progress)
        finally:
            continual._finish = original_finish
        timings.append({'run': r + 1, 'total_s': time.perf_counter() - start,
                        'generation_s': gens, 'final_evaluation_s': finish[0]})
    eva = BinaryVectorNetwork.restore(load_profile(root)['eva']).snapshot_id()
    return timings, eva


def child(args):
    """Runs inside a fresh interpreter so the worker count and kernel choice apply cleanly."""
    from boolean_engine import native_backend
    from engine import parallel
    timings, eva = one_run(args.generations, args.seed, args.runs)
    print(json.dumps({'timings': timings, 'eva': eva, 'native': native_backend.status(),
                      'parallel': parallel.status()}))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workers', type=int, nargs='+', default=[1, 4, 8, 16])
    p.add_argument('--generations', type=int, default=5)
    p.add_argument('--runs', type=int, default=2, help='consecutive runs on one profile')
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--python', action='store_true', help='disable the native kernel (reference)')
    p.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.child:
        return child(args)
    import subprocess
    report = {'machine': {'system': platform.platform(), 'python': sys.version.split()[0],
                          'cpu_threads': os.cpu_count(), 'processor': platform.processor()},
              'config': {'generations': args.generations, 'runs': args.runs, 'repeat': args.repeat,
                         'python_reference': args.python}, 'results': []}
    print(f"Predatore V7 benchmark · {report['machine']['system']} · {os.cpu_count()} thread CPU")
    identities = set()
    for workers in args.workers:
        env = dict(os.environ, PREDATORE_WORKERS=str(workers))
        if args.python:
            env['PREDATORE_NATIVE'] = '0'
        totals, rows = [], []
        for _ in range(args.repeat):
            out = subprocess.run([sys.executable, __file__, '--child', '--generations', str(args.generations),
                                  '--runs', str(args.runs), '--seed', str(args.seed)],
                                 env=env, capture_output=True, text=True, cwd=str(ROOT))
            if out.returncode:
                print(out.stderr[-2000:])
                raise SystemExit(f'benchmark failed with {workers} workers')
            row = json.loads(out.stdout.strip().splitlines()[-1])
            rows.append(row)
            totals.append(sum(t['total_s'] for t in row['timings']))
            identities.add(row['eva'])
        gens = [g for row in rows for t in row['timings'] for g in t['generation_s']]
        entry = {'workers': workers, 'mode': rows[0]['parallel']['mode'], 'native': rows[0]['native'],
                 'total_median_s': statistics.median(totals), 'total_min_s': min(totals), 'total_max_s': max(totals),
                 'generation_median_s': statistics.median(gens) if gens else None, 'runs': rows}
        report['results'].append(entry)
        print(f"  {workers:>2} lavoratori ({entry['mode']:<9}) · totale mediano {entry['total_median_s']:7.2f} s "
              f"[{entry['total_min_s']:.2f}–{entry['total_max_s']:.2f}] · generazione mediana {entry['generation_median_s']:.2f} s")
    best = min(report['results'], key=lambda e: e['total_median_s'])
    report['same_result_for_every_worker_count'] = len(identities) == 1
    report['fastest_workers'] = best['workers']
    print(f"Stesso EVA con ogni numero di lavoratori: {'sì' if len(identities) == 1 else 'NO'}")
    print(f"Più veloce: {best['workers']} lavoratori. Per usarlo sempre: set PREDATORE_WORKERS={best['workers']}")
    out_dir = ROOT / 'validation' / 'v7'
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"benchmark_{platform.node() or 'machine'}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    (out_dir / name).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Report: {out_dir / name}')


if __name__ == '__main__':
    main()
