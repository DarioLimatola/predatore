"""Independent probes of saved brains; never resumes or changes training."""
from pathlib import Path
import random
import time
from .boolean_net import BooleanNet
from .world import World
from .vault import read_json, atomic_json
from .progress_monitor import completion_metrics


def probe_world(record,seed,visibility,barrier=False,pounce=True):
    world=World(record,seed=seed,visibility=visibility,prey_moves=False)
    world.pounce_enabled=pounce
    if barrier:
        mid=record['size']//2
        world.predator=(mid-1,seed%record['size'])
        world.prey=(mid+1,(seed//record['size'])%record['size'])
        world.visited={world.predator}
    return world


def run_probes(experiment,output='Analysis',episodes=100,seed=901):
    exp=Path(experiment);cfg=read_json(exp/'config.json');size=cfg['size']
    brain=read_json(exp/'brains/eva.json')
    native=exp/'environment/suites/holdout_static.json'
    normal=read_json(native if native.exists() else exp/'environment/holdout.json')
    barrier={'name':'barriera_salto','size':size,'obstacles':[[size//2,y] for y in range(size)]}
    scenes=[('Preda ferma, vista 6',normal,6,False,True),
            ('Visibilità ridotta, vista 3',normal,3,False,True),
            ('Preda invisibile, vista 0',normal,0,False,True),
            ('Barriera: salto necessario',barrier,6,True,True),
            ('Barriera: salto disabilitato',barrier,6,True,False)]
    report={'protocol':{'episodes_per_condition':episodes,'seed':seed,'steps':cfg['steps'],
             'note':'Prove appaiate su cervelli salvati; nessun training. Stessa mappa nativa per visibilità 6/3/0. Barriera separata artificiale. Baseline 4 e 8 azioni.'},'scenarios':{}}
    for name,record,visibility,is_barrier,pounce in scenes:
        results={}
        for actor in ('EVA','Casuale 4 azioni','Casuale 8 azioni'):
            rows=[];attempts=successes=0
            for i in range(episodes):
                world=probe_world(record,seed+i,visibility,is_barrier,pounce)
                net=BooleanNet.restore(brain);net.reset();rng=random.Random(seed+i+5000);trail=[]
                for _ in range(cfg['steps']):
                    action=net.decide(world.observe()) if actor=='EVA' else rng.randrange(4 if actor=='Casuale 4 azioni' else 8)
                    previous=world.predator;reward,captured=world.step(action)
                    if action>=4:attempts+=1;successes+=world.predator!=previous
                    if actor=='EVA' and hasattr(net,'feedback'):net.feedback(reward)
                    trail.append(world.predator)
                    if captured or len(trail)>=20 and len(set(trail[-20:]))<=2:break
                rows.append({'capture':int(world.captured),'steps':world.steps})
            results[actor]={**completion_metrics(rows),'capture_rate':sum(r['capture'] for r in rows)/episodes,
                            'pounce_attempts':attempts,'pounce_legal_moves':successes}
        report['scenarios'][name]=results
    out=Path(output)/f'{exp.name}_probes_{time.time_ns()}'
    atomic_json(out/'supplementary.json',report)
    return out,report
