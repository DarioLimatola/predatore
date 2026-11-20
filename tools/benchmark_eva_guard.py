"""Paired 2x2 comparison of EVA rollback and explorer transfer-gain selection."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.boolean_net import BooleanNet
from engine.challenges import load_suites
from engine.continual import assessment, load_profile
from engine.evolution import Config, run
from engine.vault import read_json


def measurements(metrics):
    fields = ('capture_count', 'episode_count', 'successful_mean_steps', 'stall_count', 'mean_reward')
    return {name: {field: row[field] for field in fields} for name, row in metrics['by_scenario'].items()}


def experiment(seed, guarded, gain):
    with tempfile.TemporaryDirectory() as tmp:
        first_folder = previous_id = None
        records = []
        for run_seed in (seed, seed + 1):
            config = Config(seed=run_seed, size=8, generations=2, population=4, episodes=8, steps=32,
                            minimum_active_neurons=32, initial_neurons=32, max_neurons=32, max_vector_width=64,
                            memory_capacity=64, experience_capacity=96, validation_episodes=12, holdout_episodes=32,
                            retention_tasks=2, max_scenario_loss=0 if guarded else None, select_for_learning=gain)
            wall = time.perf_counter()
            cpu = time.process_time()
            folder, _ = run(tmp, config)
            elapsed, cpu_elapsed = time.perf_counter() - wall, time.process_time() - cpu
            initial = BooleanNet.restore(read_json(folder/'brains/initial_eva.json'))
            state = load_profile(tmp)
            final = BooleanNet.restore(state['eva'])
            if previous_id is not None and initial.snapshot_id() != previous_id:
                raise RuntimeError('Persistent brain differs from preceding final snapshot')
            previous_id = final.snapshot_id()
            learning = read_json(folder/'benchmarks/learning_comparison.json')
            baseline = read_json(folder/'benchmarks/baselines.json')
            history = read_json(folder/'checkpoints/latest.json')['history']
            details = [json.loads(line) for line in (folder/'evolution/consolidation.jsonl').read_text().splitlines()]
            item = {
                'config': asdict(config), 'initial': measurements(learning['initial']), 'final': measurements(learning['final']),
                'initial_id': initial.snapshot_id(), 'final_id': previous_id,
                'baselines': {
                    name: {scenario: {'capture_count': m['capture_count'], 'episode_count': m['episode_count']}
                           for scenario, m in row['by_scenario'].items()}
                    for name, row in baseline.items()
                },
                'rollbacks': sum(r['personal_rollback'] for r in history),
                'regressions_after_personal_training': sum(r['post_learning_regression'] for r in history),
                'proposals_rejected_for_scenario_loss': sum(d['reason'] == 'scenario_loss'
                                                           for event in details for d in event['decisions']),
                'seconds': elapsed, 'cpu_seconds': cpu_elapsed,
                'smith_run_memory': len(state['meta']['run_memory']),
            }
            if first_folder is not None:
                old = assessment(final, load_suites(first_folder, 'holdout'), seed, 8, config.steps, 'holdout')
                item['previous_run_holdout_after_second_run'] = measurements(old)
            first_folder = folder
            records.append(item)
    return {'lineage': seed, 'guard': guarded, 'learning_selection': gain, 'runs': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'validation/p4/paired.json')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 117, 809])
    args = parser.parse_args()
    results = []
    for seed in args.seeds:
        for guard, gain in ((False, False), (True, False), (False, True), (True, True)):
            results.append(experiment(seed, guard, gain))
            print(seed, guard, gain, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
