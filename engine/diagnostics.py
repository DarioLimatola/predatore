"""Descriptive behavior and paired causal interventions, never used for selection."""
from collections import Counter
import math
import statistics
from .boolean_net import BooleanNet
from .challenges import seeds_for
from .vault import atomic_json


def output_ancestors(net):
    pending = list(net.outputs)
    found = set()
    while pending:
        i = pending.pop()
        if i in found or not net.nodes[i].active:
            continue
        found.add(i)
        n = net.nodes[i]
        sources = [n.a, n.b, net.rules[i]['hold']]
        pending.extend(s - net.sensors for s in sources if s >= net.sensors)
    return sorted(found)


def describe(frames):
    steps = [f for f in frames if 'action' in f]
    counts = Counter(f['action'] for f in steps)
    n = max(1, len(steps))
    positions = [tuple(f['predator']) for f in frames]
    cells = Counter(positions)
    length = max(1, len(positions))
    width = len(steps[0]['neural_state']) if steps else 0
    return {
        'action_counts': [counts[i] for i in range(8)],
        'action_entropy_bits': -sum((v/n)*math.log2(v/n) for v in counts.values()),
        'unique_cells': len(cells),
        'position_entropy_bits': -sum((v/length)*math.log2(v/length) for v in cells.values()),
        'revisited_steps': length - len(cells),
        'two_step_returns': sum(positions[i] == positions[i-2] for i in range(2,len(positions))),
        'neuron_on_fraction': [sum(f['neural_state'][i] for f in steps)/n for i in range(width)],
        'neuron_switch_fraction': [sum(f['neural_state'][i] != f['previous_state'][i] for f in steps)/n
                                   for i in range(width)],
        'plastic_updates': sum(len(f.get('plastic_updates', []))+len(f.get('vector_updates', [])) for f in steps),
        'start_manhattan': sum(abs(a-b) for a,b in zip(frames[0]['predator'],frames[0]['prey'])),
        'end_manhattan': sum(abs(a-b) for a,b in zip(frames[-1]['predator'],frames[-1]['prey'])),
    }


def assess_champion(net, suites, seed, episodes, steps, folder, neuron_ablations=False):
    from .evaluation import rollout, genome_id
    variants = ['normal', 'memory_off']
    variants.insert(1, 'plasticity_off')
    variants += ['associative_off', 'learned_memory_cleared', 'contextual_off']
    if neuron_ablations:
        variants += [f'neuron_off_{i}' for i,n in enumerate(net.nodes) if n.active]
    results = {}
    for variant in variants:
        clone = BooleanNet.restore(net.export())
        clone.plasticity_enabled = net.plasticity_enabled
        if variant == 'associative_off':clone.associative_enabled=False
        if variant == 'learned_memory_cleared':
            clone.knowledge.clear();clone.contextual_memory.rows.clear()
        if variant == 'contextual_off':clone.contextual_enabled=False
        if variant == 'plasticity_off':
            clone.plasticity_enabled = False
        if variant == 'memory_off':
            clone.memory_enabled = False
        if variant.startswith('neuron_off_'):
            clone.nodes[int(variant.rsplit('_',1)[1])].active = False
            clone._compiled = None
        rows = []
        for scenario in suites:
            for episode_seed in seeds_for(seed, 'holdout', scenario['name'], episodes):
                metric, frames = rollout(clone, scenario, episode_seed, steps, record=True)
                rows.append({'scenario':scenario['name'], 'seed':episode_seed, **metric,
                             **describe(frames)})
        results[variant] = {'capture_rate':statistics.mean(r['capture'] for r in rows),
                            'mean_reward':statistics.mean(r['reward'] for r in rows),
                            'mean_unique_cells':statistics.mean(r['unique_cells'] for r in rows),
                            'plastic_updates':sum(r['plastic_updates'] for r in rows),
                            'episodes':rows}
    normal = results['normal']
    for name, value in results.items():
        value['capture_delta_vs_normal'] = value['capture_rate'] - normal['capture_rate']
        value['paired_capture_deltas'] = [a['capture']-b['capture']
                                          for a,b in zip(value['episodes'],normal['episodes'])]
    report = {'genome_id':genome_id(net), 'episodes_per_scenario':episodes,
              'protocol':'paired held-out seeds; isolated adaptive copies; temporary recurrent state reset; no selection feedback',
              'note':'Ablations measure this champion on this sample; no general learning claim.',
              'output_ancestors':output_ancestors(net), 'variants':results}
    atomic_json(folder/'neuroscience'/'behavioral_diagnostics.json', report)
    return report
