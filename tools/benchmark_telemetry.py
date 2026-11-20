"""Paired total runtime with/without the LiveObserver. Run from the source root."""
import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.evolution import Config, run
from engine.telemetry import LiveObserver


def measure(seed, live):
    c=Config(population=4,generations=2,episodes=8,steps=24,size=8,seed=seed,
             initial_neurons=32,max_neurons=36,validation_episodes=8,
             holdout_episodes=8,memory_capacity=32,experience_capacity=32,vector_width=16)
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp)/'EvolutionVault/experiments'/f'EXP_seed{seed}_size8'
        start=time.perf_counter()
        cpu=time.process_time()
        observer=LiveObserver(folder) if live else None
        try:
            run(tmp,c,observer=observer)
        finally:
            if observer:observer.close()
        elapsed=time.perf_counter()-start
        cpu=time.process_time()-cpu
        result=json.loads((folder/'results.json').read_text())
        return {'seed':seed,'live':live,'wall_s':elapsed,'cpu_s':cpu,
                'packets':observer.sequence if live else 0,
                'sampled_episodes':observer.episode_number if live else 0,
                'champion_id':result['champion_id'],
                'holdout_captures':result['benchmarks']['holdout']['capture_count']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    rows=[]
    pairs=[]
    for i,seed in enumerate((611,612,613,614,615,616)):
        pair={enabled:measure(seed,enabled) for enabled in ((True,False) if i%2 else (False,True))}
        rows.extend(pair.values())
        a,b=pair[False],pair[True]
        if (a['champion_id'],a['holdout_captures'])!=(b['champion_id'],b['holdout_captures']):
            raise RuntimeError(f'Paired outcomes differ at seed {seed}')
        pairs.append(100*(b['wall_s']/a['wall_s']-1))
    report={'python':platform.python_version(),'platform':platform.platform(),
            'configuration':'classic; 4 explorers, 2 generations, 8 episodes/split, 24 steps, 8x8; fresh profiles',
            'main_process':'whole training including final diagnostics, saves and observer shutdown; no Tk window',
            'predeclared_threshold_pct':10,'rows':rows,'paired_wall_delta_pct':pairs,
            'median_paired_wall_delta_pct':statistics.median(pairs)}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
