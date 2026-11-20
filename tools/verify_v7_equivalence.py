"""V7 gate: the same training gives the same saved results in every execution mode.

Compares, byte for byte after canonical JSON, the complete SMITH profile (EVA,
history, SMITH memory, events) and the benchmark files of the run, for:
pure Python (PREDATORE_NATIVE=0), native with 1 worker, native with N workers.

    python tools/verify_v7_equivalence.py --generations 6 --runs 2
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def child(generations, runs, seed):
    from engine.evolution import Config, run
    from engine.continual import load_profile
    from engine.vault import read_json
    root = tempfile.mkdtemp(prefix='predatore-eq-')
    digests = {}
    for r in range(runs):
        folder, _ = run(root, Config(generations=generations, seed=seed + r))
        for name in ('benchmarks/results.json', 'benchmarks/population_comparison.json', 'benchmarks/learning_comparison.json'):
            digests[f'run{r+1}/{name}'] = read_json(folder / name)
        for rec in sorted((folder / 'recordings').glob('*.json')):
            digests[f'run{r+1}/recordings/{rec.name}'] = read_json(rec)['frames']
    profile = load_profile(root)
    digests['profile'] = profile
    out = {k: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest() for k, v in digests.items()}
    print(json.dumps(out))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--generations', type=int, default=4)
    p.add_argument('--runs', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--child', action='store_true')
    a = p.parse_args()
    if a.child:
        return child(a.generations, a.runs, a.seed)
    modes = {'python': {'PREDATORE_NATIVE': '0', 'PREDATORE_WORKERS': '1'},
             'native-1': {'PREDATORE_WORKERS': '1'},
             f'native-{a.workers}': {'PREDATORE_WORKERS': str(a.workers)}}
    results = {}
    for name, extra in modes.items():
        env = dict(os.environ, **extra)
        out = subprocess.run([sys.executable, __file__, '--child', '--generations', str(a.generations),
                              '--runs', str(a.runs), '--seed', str(a.seed)], env=env, capture_output=True, text=True, cwd=str(ROOT))
        if out.returncode:
            print(out.stderr[-3000:]); raise SystemExit(f'{name} failed')
        results[name] = json.loads(out.stdout.strip().splitlines()[-1])
        print(f'{name}: {len(results[name])} files hashed')
    reference = results['python']
    ok = True
    for name, digests in results.items():
        diff = [k for k in reference if reference[k] != digests.get(k)] + [k for k in digests if k not in reference]
        print(f'{name} vs python: ' + ('IDENTICAL' if not diff else f'DIFFERENT in {diff[:6]}'))
        ok &= not diff
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
