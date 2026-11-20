"""Recovery, historical forks and integrity regressions in the V6.06 vault."""
import json
from pathlib import Path
import tempfile
import unittest

from engine.evolution import Config, run
from engine.lineage import fork_v605
from engine.settings import Limits
from engine.vault import (LEGACY_SCHEMA, SCHEMA, atomic_json, fingerprint,
                          latest_checkpoint_path, read_json, save_checkpoint, verify)


class LineageTests(unittest.TestCase):
    def test_corrupt_parseable_checkpoint_prefers_newest_verified_predecessor(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            latest = folder / 'latest.json'
            def data(generation):
                return dict(schema=SCHEMA, generation=generation, config={}, history=[], population=[])
            save_checkpoint(folder / 'latest.g000000001.fallback-abcd.json', data(1))
            save_checkpoint(latest, data(2))
            save_checkpoint(latest, data(3))
            corrupt = latest.read_bytes().replace(b'"generation":3', b'"generation":9')
            latest.write_bytes(corrupt)
            with self.assertRaises(ValueError):
                read_json(latest)
            self.assertEqual(latest_checkpoint_path(folder), folder / 'previous.json')
            # Stripping the digest from a release checkpoint cannot make it valid.
            latest.write_text(json.dumps(data(9)), encoding='utf-8')
            self.assertEqual(read_json(latest_checkpoint_path(folder))['generation'], 2)

    def test_corrupt_local_latest_recovers_from_previous_even_after_smith_commit(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = Config(population=4, generations=2, episodes=4, steps=12,
                         size=6, initial_neurons=32, max_neurons=32, validation_episodes=4, holdout_episodes=4)
            folder, _ = run(temp, cfg)
            champion = read_json(folder/'brains/eva.json')
            (folder/'checkpoints/latest.json').write_text('{"generation":')
            with self.assertWarns(RuntimeWarning):
                run(temp, cfg, resume=True)
            self.assertEqual(read_json(folder/'brains/eva.json'), champion)
            self.assertTrue(all(verify(folder).values()))
            lines = (folder / 'evolution/generations.jsonl').read_text().splitlines()
            self.assertEqual([json.loads(line)['generation'] for line in lines], [1, 2])

    def test_historical_fork_preserves_transaction_and_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'old' / 'EvolutionVault' / 'experiments' / 'EXP_seed42_size8'
            config = {'smith_profile': 'SMITH', 'seed': 42, 'size': 8}
            state = dict(generation=5, history=[], config={}, eva={}, schema=Limits.PROFILE_SCHEMA,
                         transaction='EXP_seed42_size8:2')
            local = dict(schema=LEGACY_SCHEMA, generation=2, config=config, population=[],
                         history=[], supervisor_state=state)
            original = source / 'checkpoints/latest.json'
            atomic_json(original, local)
            atomic_json(source/'manifest.json', {'schema': LEGACY_SCHEMA})
            digest = fingerprint(original)
            new_root = root / 'new'
            folder = fork_v605(source, new_root)
            self.assertEqual(fingerprint(original), digest)
            copied = read_json(folder/'checkpoints/latest.json')
            self.assertEqual(copied['origin']['checkpoint_sha256'], digest)
            self.assertEqual(copied['schema'], SCHEMA)
            self.assertTrue(all(verify(folder).values()))
            self.assertEqual(read_json(new_root/'EvolutionVault/SMITH/SMITH/latest.json'),
                             copied['supervisor_state'])
            with self.assertRaises(FileExistsError):
                fork_v605(source, new_root)

    def test_stop_before_generation_one_can_resume_with_identical_brains(self):
        with tempfile.TemporaryDirectory() as stopped, tempfile.TemporaryDirectory() as uninterrupted:
            cfg = Config(population=4, generations=1, episodes=4, steps=8, size=6,
                         initial_neurons=32, max_neurons=32, validation_episodes=4, holdout_episodes=4)
            folder, generation = run(stopped, cfg, should_stop=lambda: True)
            self.assertEqual(generation, 0)
            self.assertFalse((Path(stopped)/'EvolutionVault/SMITH/SMITH/latest.json').exists())
            run(stopped, cfg, resume=True)
            expected, _ = run(uninterrupted, cfg)
            self.assertEqual(read_json(folder/'brains/eva.json'), read_json(expected/'brains/eva.json'))
            self.assertEqual(read_json(folder/'checkpoints/latest.json')['population'],
                             read_json(expected/'checkpoints/latest.json')['population'])

    def test_resume_refuses_modified_holdout_without_resealing(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = Config(population=4, generations=1, episodes=4, steps=8, size=6,
                         initial_neurons=32, max_neurons=32, validation_episodes=4, holdout_episodes=4)
            folder, _ = run(root, cfg)
            seal = folder/'integrity/checksums.json'
            original = fingerprint(seal)
            map_path = folder/'environment/suites/holdout_static.json'
            changed = read_json(map_path)
            changed['seed'] += 1
            atomic_json(map_path, changed)
            for requested in (1, 2):
                cfg.generations = requested
                with self.assertRaisesRegex(ValueError, 'holdout_static'):
                    run(root, cfg, resume=True)
            self.assertEqual(fingerprint(seal), original)
