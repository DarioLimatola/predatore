"""Small reproducible multi-run check; no promise of monotonic learning."""
from dataclasses import asdict
from pathlib import Path
import json
import statistics
import sys
import tempfile
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine.evolution import Config,run
from engine.continual import assessment,load_profile
from engine.boolean_net import BooleanNet
from engine.challenges import load_suites
from engine.vault import read_json


def main():
    records=[]
    start=time.perf_counter()
    for lineage in (801,811,821):
        with tempfile.TemporaryDirectory() as root:
            previous=None
            for seed in (lineage,lineage+1):
                c=Config(seed=seed,population=8,generations=6,episodes=16,steps=80,size=8,
                         initial_neurons=64,minimum_active_neurons=32,max_neurons=128,
                         vector_shape='8,8',memory_capacity=256,experience_capacity=256,
                         validation_episodes=16,holdout_episodes=80,retention_tasks=4)
                folder,_=run(root,c)
                state=load_profile(root)
                initial=BooleanNet.restore(read_json(folder/'brains/initial_eva.json'))
                if previous is not None:assert initial.snapshot_id()==previous
                previous=BooleanNet.restore(state['eva']).snapshot_id()
                suites=load_suites(folder,'holdout')
                baseline=assessment(initial,suites,c.seed,c.holdout_episodes//len(suites),c.steps,'holdout')
                cp=read_json(folder/'checkpoints/latest.json')
                history=cp['history']
                test=read_json(folder/'benchmarks/population_comparison.json')
                item={'seed':seed,'config':asdict(c),'initial_holdout':baseline['capture_rate'],
                      'final_holdout':test['eva']['capture_rate'],'population_holdout':test['population_mean_capture'],
                      'final_training_eva':history[-1]['capture_rate'],
                      'final_training_population':history[-1]['population_mean_capture'],
                      'training_generations_at_least_population_mean':sum(h['capture_rate']>=h['population_mean_capture'] for h in history),
                      'eva_lifetime_changes':sum(h['eva_lifetime_changed'] for h in history),
                      'transfers':sum(h['eva_promoted'] for h in history),
                      'context_writes_consolidated':sum(h['distillation']['contexts_consolidated'] for h in history),
                      'eva_neurons':len(state['eva']['nodes']),
                      'explorer_neuron_range':[min(h['population_min_neurons'] for h in history),max(h['population_max_neurons'] for h in history)],
                      'smith_remembered_runs':len(state['meta']['run_memory']),
                      'elapsed_seconds':read_json(folder/'results.json')['elapsed_this_session_seconds']}
                records.append(item)
                print(json.dumps({k:v for k,v in item.items() if k!='config'}),flush=True)
    report={'protocol':'Three independent lineages, two sequential runs each; default neuron bounds. Same adaptive held-out evaluation for initial EVA, final EVA and explorers. Descriptive small sample; no causal ablation or significance claim.',
            'runs':records,'elapsed_seconds':time.perf_counter()-start}
    target=Path(__file__).resolve().parents[1]/'validation/v604/benchmark.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
