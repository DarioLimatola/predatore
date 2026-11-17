"""Deterministic multi-scenario curriculum for V6 EVA training."""
from pathlib import Path
from collections import deque
import json
import random
from .world import DIRS
from .vault import read_json

# Keep all tasks solvable with the historical 15 sensor contract.
SCENARIOS = {
    'static': {'density': .06, 'visibility': 6, 'prey_moves': False},
    'moving': {'density': .06, 'visibility': 6, 'prey_moves': True},
    'maze': {'density': .13, 'visibility': 6, 'prey_moves': False},
    'hidden': {'density': .08, 'visibility': 3, 'prey_moves': False},
}

def create_connected_map(path, size, density, seed):
    """Procedurally generated connected map, including large 40x40 arenas.

    Unlike repeatedly rejecting random layouts, this generator starts with a
    sampled obstacle field and carves corridors joining every free component.
    Only world geometry changes; no EVA action is imposed or privileged.
    """
    rng=random.Random(seed)
    obstacles={(x,y) for y in range(size) for x in range(size) if rng.random()<density}
    free={(x,y) for y in range(size) for x in range(size)}-obstacles
    if len(free)<2:
        free={(0,0),(1,0)}
    unseen=set(free)
    components=[]
    while unseen:
        start=min(unseen);unseen.remove(start)
        component={start};front=deque([start])
        while front:
            x,y=front.popleft()
            for dx,dy in DIRS:
                q=(x+dx,y+dy)
                if q in unseen:
                    unseen.remove(q);front.append(q);component.add(q)
        components.append(component)
    components.sort(key=lambda c:-len(c))
    main=min(components[0])
    # Every originally free component is connected to the main component.
    for component in components[1:]:
        dest=min(component)
        x,y=main
        while x!=dest[0]:
            x+=1 if dest[0]>x else -1
            obstacles.discard((x,y))
        while y!=dest[1]:
            y+=1 if dest[1]>y else -1
            obstacles.discard((x,y))
    record={'version':1,'name':Path(path).stem if path is not None else 'training','size':size,'density':len(obstacles)/(size*size),
            'requested_density':density,'seed':seed,
            'obstacles':[list(p) for p in sorted(obstacles)]}
    if path is not None:
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        Path(path).write_text(json.dumps(record,indent=2),encoding='utf-8')
    return record


def prepare_suites(folder, size, seed):
    """Stable, disjoint maps per split and task. Reuses saved maps on resume."""
    result={}
    for split, offset in (('train',1000), ('validation',2000), ('holdout',3000)):
        result[split]=[]
        for i,(name,spec) in enumerate(SCENARIOS.items()):
            relative=f'suites/{split}_{name}'
            path=Path(folder)/'environment'/f'{relative}.json'
            if not path.is_file():
                create_connected_map(path,size,spec['density'],seed+offset+i*197)
            result[split].append({'name':name,'map':read_json(path),
                                  'map_key':relative,'visibility':spec['visibility'],
                                  'prey_moves':spec['prey_moves']})
    return result


def seeds_for(seed, split, scenario, n):
    split_index={'train':1,'validation':2,'holdout':3}[split]
    scenario_index=list(SCENARIOS).index(scenario)
    return [seed*100000+split_index*10000+scenario_index*1000+i for i in range(n)]


def load_suites(folder, split):
    """Read existing scenarios without generating or modifying experiment data."""
    return [{'name':name, 'map':read_json(Path(folder)/'environment'/'suites'/f'{split}_{name}.json'),
             'map_key':f'suites/{split}_{name}', 'visibility':spec['visibility'],
             'prey_moves':spec['prey_moves']} for name,spec in SCENARIOS.items()]


def training_generation(size,seed,generation,count,archive,run):
    """One shared training schedule; replay entries come only from completed training."""
    import hashlib
    from .settings import Limits
    def derived(*parts):
        return int.from_bytes(hashlib.sha256(':'.join(map(str,parts)).encode()).digest()[:8],'little')
    suites=[]; fresh=[]; replayed=0
    for i,(name,spec) in enumerate(SCENARIOS.items()):
        scene={'name':name,'map':create_connected_map(None,size,spec['density'],derived(seed,generation,name,'map')),
               'map_key':f'train/g{generation}/{name}','visibility':spec['visibility'],'prey_moves':spec['prey_moves']}
        old=[record for record in archive if record['scenario']['name']==name]
        episodes=[]
        for index in range(count):
            episode_seed=derived(seed,generation,name,index,'episode')
            if old and (i*count+index+generation)%Limits.TRAIN_REPLAY_DENOMINATOR==0:
                chosen=random.Random(episode_seed).choice(old)
                episodes.append({**chosen,'replay':True});replayed+=1
            else:
                record={'id':f'{run}/g{generation}/{name}/{index}','seed':episode_seed,'scenario':scene}
                fresh.append(record);episodes.append({**record,'replay':False})
        suites.append({**scene,'episode_specs':episodes})
    retained=sorted(archive+fresh,key=lambda r:derived(r['id'],'sample'))[:Limits.TRAIN_REPLAY_CAPACITY]
    return suites,retained,replayed


def episode_specs(scenario,base_seed,split,count):
    if 'episode_specs' in scenario:
        if split!='train':raise ValueError('Training replay outside training')
        return [(item['scenario'],item['seed']) for item in scenario['episode_specs'][:count]]
    return [(scenario,seed) for seed in seeds_for(base_seed,split,scenario['name'],count)]
