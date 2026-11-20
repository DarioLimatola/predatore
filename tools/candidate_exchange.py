"""Export a training candidate or evaluate one against a saved brain in isolation.

Run from the extracted project directory: python -m tools.candidate_exchange --help
"""
import argparse
import json
from pathlib import Path

from boolean_engine import VectorBooleanNet
from engine.candidate_exchange import BASE_SCENARIOS, circuit_candidate, experience_candidate, evaluate_candidate
from engine.vault import atomic_json


def read(path):
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError('Input exceeds the 64 MiB exchange budget')
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def save_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the name exclusively before atomic replacement. Never replace a checkpoint.
    with path.open('x', encoding='utf-8'):
        pass
    try:
        atomic_json(path, data)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    model = commands.add_parser('export-circuit', help='Export only the donor circuit')
    events = commands.add_parser('export-experience', help='Run a fresh teacher episode and export named observations')
    check = commands.add_parser('evaluate', help='Compare to baseline on disjoint validation data')
    for cmd in (model, events, check):
        cmd.add_argument('brain', type=Path, help='brains/eva.json or another exported arena brain')
        cmd.add_argument('output', type=Path, help='New output file')
    for cmd in (model, events):
        cmd.add_argument('--source', required=True, help='Persistent teacher or project ID')
        cmd.add_argument('--license', required=True, help='License or stated permission for the material')
        cmd.add_argument('--run', required=True)
        cmd.add_argument('--generation', required=True, type=int)
    model.add_argument('--training-seed', type=int, required=True)
    events.add_argument('--map-seed', type=int, required=True)
    events.add_argument('--episode-seed', type=int, required=True)
    events.add_argument('--scenario', choices=BASE_SCENARIOS, default='hidden')
    for cmd in (events, check):
        cmd.add_argument('--steps', type=int, default=64)
        cmd.add_argument('--size', type=int, default=8)
    check.add_argument('--candidate', type=Path, required=True)
    check.add_argument('--seed', type=int, required=True, help='Fresh validation seed')
    check.add_argument('--episodes', type=int, default=16, help='Per scenario')
    args = parser.parse_args(argv)
    try:
        net = VectorBooleanNet.restore(read(args.brain))
        if args.command == 'evaluate':
            result, winner = evaluate_candidate(net, read(args.candidate), seed=args.seed,
                                                 episodes=args.episodes, steps=args.steps, size=args.size)
            if winner is not None:
                result['accepted_brain'] = winner.export()
        else:
            common = dict(source=args.source, license_name=args.license, run=args.run, generation=args.generation)
            if args.command == 'export-circuit':
                result = circuit_candidate(net, **common, training_seed=args.training_seed)
            else:
                result = experience_candidate(net, **common, scenario=args.scenario, map_seed=args.map_seed,
                                              episode_seed=args.episode_seed, size=args.size, steps=args.steps)
        save_new(args.output, result)
    except (ValueError, OSError, TypeError, KeyError, IndexError) as exc:
        parser.exit(2, f'Exchange failed: {exc}\n')
    print(f"{result['status']} {result['candidate_id']} {args.output}")


if __name__ == '__main__':
    main()
