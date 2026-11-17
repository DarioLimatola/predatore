"""Sequential, reproducible comparison on the four existing 2D scenarios."""
import argparse
import cProfile
from dataclasses import asdict
import json
from pathlib import Path
import platform
import pstats
import random
import statistics
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from boolean_engine import BinaryVectorNetwork
from engine.continual import load_profile
from engine.evolution import Config,run
from engine.vault import read_json


def case(root,seed,modules,generations):
    c=Config(seed=seed,population=6,generations=generations,episodes=8,steps=60,size=8,
             minimum_active_neurons=modules,initial_neurons=modules,max_neurons=modules,
             vector_shape='8,8',max_vector_width=64,memory_capacity=256,experience_capacity=256,
             validation_episodes=8,holdout_episodes=32,retention_tasks=2)
    wall=time.perf_counter();cpu=time.process_time()
    folder,_=run(root,c)
    seconds=time.perf_counter()-wall;cpu_seconds=time.process_time()-cpu
    learning=read_json(folder/'benchmarks/learning_comparison.json')
    compare=read_json(folder/'benchmarks/population_comparison.json')
    state=load_profile(root)
    initial=BinaryVectorNetwork.restore(read_json(folder/'brains/initial_eva.json'))
    final=BinaryVectorNetwork.restore(state['eva'])
    history=read_json(folder/'checkpoints/latest.json')['history']
    row={'config':asdict(c),'seconds':seconds,'cpu_seconds':cpu_seconds,
         'initial':learning['initial']['capture_rate'],'final':learning['final']['capture_rate'],
         'population':compare['population_mean_capture'],'initial_id':initial.snapshot_id(),
         'final_id':final.snapshot_id(),'executed':len(final._output_ancestors()),
         'personal_regressions':sum(h['post_learning_regression'] for h in history),
         'transfers':sum(h['eva_promoted'] for h in history),
         'decoder_counts':{k:sum(h['decoder_counts'][k] for h in history) for k in final.decoder_counts},
         'replay_episodes':sum(h['replayed_training_episodes'] for h in history),
         'brain_bytes':(folder/'brains/eva.json').stat().st_size,
         'checkpoint_bytes':(folder/'checkpoints/latest.json').stat().st_size}
    try:
        import resource
        row['peak_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)
    except ImportError:row['peak_mib']=None
    return row


def micro():
    rng=random.Random(8301)
    observations=[[rng.randrange(2) for _ in range(15)] for _ in range(1500)]
    rows=[]
    for modules in (32,64,96):
        base=BinaryVectorNetwork(42,neurons=modules)
        durations={}; outputs={}
        for sparse in (False,True):
            trials=[]
            for _ in range(3):
                net=BinaryVectorNetwork.restore(base.export())
                net.sparse_enabled=sparse;net.associative_enabled=False
                start=time.perf_counter();actions=[net.decide(obs) for obs in observations]
                trials.append(time.perf_counter()-start)
            outputs[sparse]=actions;durations['sparse' if sparse else 'full']=statistics.median(trials)
        assert outputs[False]==outputs[True]
        rows.append({'modules':modules,'executed':len(base._output_ancestors()),'decisions':len(observations),
                     **durations,'speedup':durations['full']/durations['sparse']})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'validation/v605')
    p.add_argument('--generations',type=int,default=10)
    p.add_argument('--worker',type=int);p.add_argument('--modules',type=int,default=64)
    p.add_argument('--profile',action='store_true')
    a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    def save(name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    if a.worker is not None:
        with tempfile.TemporaryDirectory() as root:
            if a.profile:
                profile=cProfile.Profile()
                result=profile.runcall(case,root,a.worker,a.modules,a.generations)
                with (out/'profile.txt').open('w',encoding='utf-8') as fp:
                    pstats.Stats(profile,stream=fp).strip_dirs().sort_stats('cumulative').print_stats(30)
            else:result=case(root,a.worker,a.modules,a.generations)
        print(json.dumps(result));return
    def worker(seed,modules,generations,*extra):
        result=subprocess.run([sys.executable,__file__,'--worker',str(seed),'--modules',str(modules),
               '--generations',str(generations),'--output',str(out),*extra],check=True,capture_output=True,text=True)
        return json.loads(result.stdout)
    rows=[]
    for modules in (32,64,96):
        for seed in (42,117,809):
            row=worker(seed,modules,a.generations)
            rows.append(row);save('controlled.json',rows)
            print(modules,seed,row['initial'],row['final'],row['population'],round(row['seconds'],2),flush=True)
    continuity=[]
    for seed in (801,811):
        with tempfile.TemporaryDirectory() as root:
            first=case(root,seed,64,a.generations)
            second=case(root,seed+1,64,a.generations)
            assert first['final_id']==second['initial_id']
            continuity.append([first,second]);save('continuity.json',continuity)
    save('inference.json',micro())
    save('profile_run.json',worker(42,64,min(3,a.generations),'--profile'))
    save('environment.json',{'platform':platform.platform(),'python':sys.version,'generations':a.generations})


if __name__=='__main__':main()
