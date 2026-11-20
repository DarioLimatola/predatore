"""Scenario veto, EVA rollback provenance, frozen train transfer and historic resume defaults."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from engine.boolean_net import BooleanNet
from engine import continual
from engine.evolution import Config, run
from engine.vault import read_json
from engine.vector_boolean import VectorBooleanNet


def row(captures, total=4):
    return dict(capture_count=captures, episode_count=total, stall_count=0, mean_reward=0,
                paired_outcomes=dict(episodes=total, captures=(1<<captures)-1, stalls=0))


def result(static, hidden):
    return {**row(static+hidden, 8), 'by_scenario': {'static': row(static), 'hidden': row(hidden)}}


class EVAGuardTests(unittest.TestCase):
    def test_scenario_veto_rejects_aggregate_gain_over_tolerance(self):
        eva, teacher = VectorBooleanNet(1), VectorBooleanNet(2)
        before, after = [result(0, 3)], [result(3, 2)]
        for limit, accepted in ((0, False), (1, True), (None, True)):
            with self.subTest(limit=limit), patch.object(continual, 'probe_scores', side_effect=[before, after]):
                winner, _, decisions = continual.select_transfer(eva, [('teacher_1', teacher, 0)], [{'run':'r'}],
                                                                  max_scenario_loss=limit)
                self.assertEqual(winner is not None, accepted)
                self.assertEqual(decisions[0]['channel'], 'adoption')
                self.assertEqual(decisions[0]['diagnostics']['scenarios'][0]['capture_delta'], -1)

    def test_guard_restores_full_brain_and_separates_raw_and_published_performance(self):
        # Drive only validation probes; exercise real personal training, checkpoints and holdout.
        old, new = [result(0, 2)], [result(3, 1)]
        with tempfile.TemporaryDirectory() as root, patch.object(continual, 'probe_scores', side_effect=[old, old, new]), \
                patch.object(continual, 'select_transfer', return_value=(None, False, [])):
            folder, _ = run(root, Config(seed=79, generations=1, population=4, episodes=4, steps=10,
                                         validation_episodes=4, holdout_episodes=4, max_scenario_loss=0))
            state = continual.load_profile(root)
            saved = state['eva_guard']
            self.assertTrue(saved['rolled_back'])
            retained = BooleanNet.restore(state['eva']).snapshot_id()
            self.assertEqual(retained, saved['rollback_id'])
            self.assertEqual(BooleanNet.restore(saved['stable_before_training']).snapshot_id(), retained)
            self.assertNotEqual(saved['candidate_id'], retained)
            self.assertEqual(BooleanNet.restore(saved['personal_candidate']).snapshot_id(), saved['candidate_id'])
            summary = state['history'][-1]
            self.assertEqual(summary['learning_channels']['personal'], {'attempted': 1, 'accepted': 0, 'rejected': 1})
            self.assertTrue(summary['personal_rollback'])
            self.assertFalse(summary['eva_lifetime_changed'])
            self.assertEqual(summary['episode_count'], 4)
            self.assertTrue(read_json(folder/'results.json')['complete'])

    def test_learning_probe_uses_other_training_maps_and_frozen_copies(self):
        observed = []
        real = continual.evaluate_suite
        def observe(net, suites, seed, split, count, steps, observer=None, **kwargs):
            if split == 'train':
                observed.append((seed, [s['map']['seed'] for s in suites], net.plasticity_enabled))
            return real(net, suites, seed, split, count, steps, observer=observer, **kwargs)
        with tempfile.TemporaryDirectory() as root, patch.object(continual, 'evaluate_suite', side_effect=observe):
            cfg = Config(seed=8, generations=1, population=4, episodes=4, steps=8, validation_episodes=4,
                         holdout_episodes=4, select_for_learning=True, max_scenario_loss=0)
            folder, _ = run(root, cfg)
            probes = [maps for seed, maps, adaptive in observed if seed == -9 and not adaptive]
            live = [maps for seed, maps, adaptive in observed if seed == 8 and adaptive]
            self.assertEqual(len(probes), cfg.population * 2)
            self.assertEqual(len(live), cfg.population + 1)
            self.assertFalse(set(probes[0]) & set(live[0]))
            cp = read_json(folder/'checkpoints/latest.json')
            for metrics, _ in cp['evaluated_population']:
                self.assertEqual(metrics['transfer_probe']['split'], 'train')
            self.assertTrue(all(e['source'].startswith('EXP_seed8_size8/g1/') for e in cp['supervisor_state']['events']))

    def test_invalid_guard_does_not_create_experiment(self):
        with tempfile.TemporaryDirectory() as root, self.assertRaises(ValueError):
            run(root, Config(max_scenario_loss=-1))


if __name__ == '__main__':
    unittest.main()
