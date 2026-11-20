"""Paired, frozen transfer probe for the experimental structural readout."""
import argparse
import json
import platform
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boolean_engine import BinaryVectorNetwork
from engine.challenges import build_scenario
from engine.evaluation import rollout

SEEDS = (42, 117, 809)
TASKS = ('static', 'hidden', 'maze_novel', 'delayed_reward')
GATE = ('On delayed_reward and maze_novel, horizon 8 must exceed both reference '
        'and horizon 1 in pooled captures and win on at least two of three '
        'initialization seeds per task. No lower pooled capture on other tasks.')


def schedule(seed, scenario_index, episodes, held_out):
    name = TASKS[scenario_index]
    offset = seed * 100_000 + scenario_index * 10_000 + (5000 if held_out else 0)
    return [(build_scenario(name, 10 if held_out else 8, offset + 2*i),
             offset + 2*i + 1) for i in range(episodes)]


def probe(net, episodes, steps):
    # A fresh frozen copy prevents state exchange between train and test.
    clone = BinaryVectorNetwork.restore(net.export())
    clone.plasticity_enabled = False
    rows = [rollout(clone, scene, seed, steps)[0] for scene, seed in episodes]
    captures = [row['capture'] for row in rows]
    times = [row['steps'] for row in rows if row['capture']]
    decisions = Counter()
    for row in rows:
        decisions.update(row['decoder_counts'])
    return {'outcomes': captures, 'captures': sum(captures),
            'completion_mean': statistics.mean(times) if times else None,
            'steps': sum(row['steps'] for row in rows), 'decoders': dict(decisions)}


def single(seed, task, n_train, n_test, steps):
    train = schedule(seed, task, n_train, False)
    test = schedule(seed, task, n_test, True)
    results = {}
    for mode, horizon in [('reference', 0), ('local_lut', 0), ('immediate', 1), ('temporal', 8)]:
        net = BinaryVectorNetwork(seed, neurons=32, shape=(8, 8))
        if horizon:
            net.enable_relation_credit(horizon)
        net.collect_experience = mode == 'local_lut'
        start = time.perf_counter()
        training_captures = 0
        for scene, episode_seed in train:
            metric, _ = rollout(net, scene, episode_seed, steps)
            training_captures += metric['capture']
        net.collect_experience = False
        bit = None
        if mode == 'local_lut':
            proposals = net.local_circuit_candidates(limit=1)
            if proposals:
                bit = proposals[0]
                net.apply_local_circuit_candidate(bit)
        training_seconds = time.perf_counter() - start
        start = time.perf_counter()
        observed = probe(net, test, steps)
        observed.update(training_seconds=training_seconds,
                        probe_seconds=time.perf_counter()-start,
                        train_captures=training_captures, lut_mutation=bit,
                        relation_count=len(net.relation_credit.weights) if horizon else 0)
        results[mode] = observed
        if mode == 'temporal':
            # Counterfactual on the same learned state. The contextual and
            # associative decoders retain everything learned during training.
            net.relation_credit = None
            results['temporal_readout_removed'] = probe(net, test, steps)
    return {'network_seed': seed, 'scenario': TASKS[task], 'test_map_seeds':
            [scene['map']['seed'] for scene, _ in test],
            'test_episode_seeds': [ep_seed for _, ep_seed in test], 'variants': results}


def summarize(rows):
    summaries = {}
    for task in TASKS:
        group = [row['variants'] for row in rows if row['scenario'] == task]
        summary = {}
        for name in group[0]:
            pairs = [(a, b) for row in group for a, b in
                     zip(row[name]['outcomes'], row['reference']['outcomes'])]
            summary[name] = {'captures': sum(a for a, _ in pairs), 'episodes': len(pairs),
                             'wins_vs_reference': sum(a > b for a, b in pairs),
                             'losses_vs_reference': sum(a < b for a, b in pairs),
                             'seeds_won_vs_reference':
                             sum(row[name]['captures'] > row['reference']['captures'] for row in group),
                             'seeds_won_vs_immediate':
                             sum(row[name]['captures'] > row['immediate']['captures'] for row in group)}
        summaries[task] = summary
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'validation/p3/relations.json')
    options = parser.parse_args()
    n_train, n_test, steps = 8, 16, 48
    rows = [single(seed, task, n_train, n_test, steps)
            for seed in SEEDS for task in range(len(TASKS))]
    summary = summarize(rows)
    passes = True
    for name, item in summary.items():
        ref = item['reference']['captures']
        temp = item['temporal']
        if name in ('maze_novel', 'delayed_reward'):
            passes &= (temp['captures'] > max(ref, item['immediate']['captures']) and
                       temp['seeds_won_vs_reference'] >= 2 and temp['seeds_won_vs_immediate'] >= 2)
        else:
            passes &= temp['captures'] >= ref
    result = {'protocol': 'P3/1', 'python': platform.python_version(), 'seeds': SEEDS,
              'modules': 32, 'width': 64, 'train_per_task_seed': n_train,
              'test_per_task_seed': n_test, 'max_steps': steps, 'train_size': 8,
              'test_size': 10, 'preregistered_gate': GATE, 'passes': passes,
              'summary': summary, 'runs': rows}
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passes': passes, 'summary': summary}))


if __name__ == '__main__':
    main()
