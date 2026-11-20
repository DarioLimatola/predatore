"""Repeat the nine V6.05 cases with isolated counterfactual probes of personal learning."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine import continual
from engine.challenges import load_suites
from engine.evolution import Config, run
from engine.evaluation import evaluate_suite
from engine.vault import atomic_json, read_json


def diagnose(reference):
    cfg = Config(**reference['config'])
    records = []
    original = continual.train_actor
    with tempfile.TemporaryDirectory() as root:
        folder = Path(root)/'EvolutionVault'/'experiments'/f'EXP_seed{cfg.seed}_size{cfg.size}'

        def audit_actor(net, source, suites, config, count, sample_seed, observer=None):
            if not source.endswith('/EVA'):
                return original(net, source, suites, config, count, sample_seed, observer)
            before = continual.evaluation_copy(net)
            training = original(net, source, suites, config, count, sample_seed, observer)

            def probe(model, decoder=True):
                copy = continual.evaluation_copy(model)
                copy.associative_enabled = decoder
                maps = load_suites(folder, 'validation')
                return evaluate_suite(copy, maps, cfg.seed, 'validation', 2, cfg.steps)

            thinned = continual.evaluation_copy(before)
            feedback = thinned.feedback
            transitions = 0
            def alternate_feedback(reward):
                nonlocal transitions
                transitions += 1
                thinned.plasticity_enabled = transitions % 2 == 1
                try:
                    feedback(reward)
                finally:
                    thinned.plasticity_enabled = True
            thinned.feedback = alternate_feedback
            original(thinned, source, suites, config, count, sample_seed)
            controls = {'frozen_before': probe(before), 'actual_after': probe(net),
                        'alternating_feedback': probe(thinned), 'decoder_off_before': probe(before, False),
                        'decoder_off_after': probe(net, False)}
            records.append({'generation': len(records)+1,
                            'arms': {name: {'captures': metrics['capture_count'],
                                            'episodes': metrics['episode_count'],
                                            'selection_key': [str(v) for v in continual.selection_key([metrics], len(net.nodes))],
                                            'by_scenario': {s: m['capture_count'] for s,m in metrics['by_scenario'].items()}}
                                     for name, metrics in controls.items()},
                            'regression': continual.selection_key([controls['actual_after']], len(net.nodes)) <
                                          continual.selection_key([controls['frozen_before']], len(net.nodes)),
                            'thinned_beats_after': continual.selection_key([controls['alternating_feedback']], len(net.nodes)) >
                                                  continual.selection_key([controls['actual_after']], len(net.nodes))})
            return training

        with patch.object(continual, 'train_actor', side_effect=audit_actor):
            run(root, cfg)
        history = read_json(folder/'checkpoints/latest.json')['history']
        for row, h in zip(records, history, strict=True):
            if row['regression'] != h['post_learning_regression']:
                raise RuntimeError('Audit clone diverges from validation gate')
            row['selected_teacher_or_proposal'] = h['promotion_reason']
        final = read_json(folder/'benchmarks/learning_comparison.json')['final']['capture_rate']
        return {'seed': cfg.seed, 'modules': cfg.initial_neurons, 'reference_final_holdout': reference['final'],
                'final_holdout': final, 'reference_personal_regressions': reference['personal_regressions'],
                'actual_personal_regressions': sum(r['regression'] for r in records), 'records': records}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'validation/p4/diagnosis.json')
    p.add_argument('--modules', nargs='+', type=int, default=[32, 64, 96])
    args = p.parse_args()
    records = []
    for r in read_json(ROOT/'validation/v605/controlled.json'):
        if r['config']['initial_neurons'] in args.modules:
            records.append(diagnose(r))
            atomic_json(args.output, records)
            print(records[-1]['seed'],records[-1]['modules'],records[-1]['actual_personal_regressions'],flush=True)


if __name__ == '__main__':
    main()
