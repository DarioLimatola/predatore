"""Regression tests for OneDrive/Windows WinError 5 checkpoint rename locks."""
import contextlib
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from engine.evolution import Config, run
from engine import vault


class CheckpointLocks(unittest.TestCase):
    @staticmethod
    def deny_latest(original):
        def guarded(src, dst):
            if Path(dst).name == 'latest.json':
                raise PermissionError(5, 'Access is denied', str(dst))
            return original(src, dst)
        return guarded

    def test_locked_latest_creates_recoverable_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = root/'checkpoints'/'latest.json'
            original = vault.os.replace
            payload = lambda generation: {'generation':generation,'config':{},'history':[], 'population':[]}
            vault.atomic_json(primary, payload(5))
            with patch.object(vault.os, 'replace', side_effect=self.deny_latest(original)), \
                 patch.object(vault.time, 'sleep', return_value=None), \
                 warnings.catch_warnings(record=True) as events:
                candidate = vault.save_checkpoint(primary, payload(10))
            self.assertNotEqual(candidate, primary)
            self.assertEqual(vault.read_json(primary)['generation'],5)
            self.assertEqual(vault.read_json(candidate)['generation'],10)
            self.assertEqual(vault.latest_checkpoint_path(primary.parent),candidate)
            self.assertTrue(events)
            # Later successful standard checkpoint supersedes the fallback.
            vault.save_checkpoint(primary,payload(15))
            self.assertEqual(vault.latest_checkpoint_path(primary.parent),primary)

    def test_corrupt_latest_finds_previous_good_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            ck=root/'latest.json'
            vault.atomic_json(ck, {'generation':1,'population':[],'history':[],'config':{}})
            fallback=root/'latest.g000000002.fallback-aaaa.json'
            vault.atomic_json(fallback, {'generation':2,'population':[],'history':[],'config':{}})
            ck.write_text('{broken json')
            self.assertEqual(vault.latest_checkpoint_path(root),fallback)

    def test_training_and_resume_when_latest_is_locked(self):
        with tempfile.TemporaryDirectory() as baseline, tempfile.TemporaryDirectory() as locked:
            cfg=Config(population=4,generations=4,episodes=2,steps=12,size=6,
                       validation_episodes=3,holdout_episodes=4)
            expected,_=run(baseline,cfg)
            original=vault.os.replace
            with patch.object(vault.os, 'replace', side_effect=self.deny_latest(original)), \
                 patch.object(vault.time, 'sleep', return_value=None), \
                 warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                cfg.generations=2
                folder,g=run(locked,cfg)
                self.assertEqual(g,2)
                self.assertFalse((folder/'checkpoints/latest.json').exists())
                self.assertEqual(vault.read_json(vault.latest_checkpoint_path(folder/'checkpoints'))['generation'],2)
                cfg.generations=4
                folder,g=run(locked,cfg,resume=True)
            self.assertEqual(g,4)
            self.assertEqual(vault.read_json(expected/'brains/eva.json'),vault.read_json(folder/'brains/eva.json'))
            self.assertEqual(vault.read_json(expected/'benchmarks/results.json'),vault.read_json(folder/'benchmarks/results.json'))
            self.assertTrue(all(vault.verify(folder).values()))


if __name__=='__main__': unittest.main()

class CrashLogReconciliation(unittest.TestCase):
    def test_uncommitted_generation_rows_archived_before_resume(self):
        from engine.vault import reconcile_evolution_logs
        with tempfile.TemporaryDirectory() as directory:
            exp=Path(directory)
            evolution=exp/'evolution'
            evolution.mkdir(parents=True)
            (evolution/'generations.jsonl').write_text(''.join(
                __import__('json').dumps({'generation':i})+'\n' for i in (1,2,3,4)), encoding='utf-8')
            (evolution/'mutations.jsonl').write_text(''.join(
                __import__('json').dumps({'generation':i})+'\n' for i in (2,3,4,5)), encoding='utf-8')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                reconcile_evolution_logs(exp, 2)
            assert [__import__('json').loads(s)['generation'] for s in (evolution/'generations.jsonl').read_text().splitlines()] == [1,2]
            assert [__import__('json').loads(s)['generation'] for s in (evolution/'mutations.jsonl').read_text().splitlines()] == [2,3]
            backups=list((exp/'recovery').glob('*.jsonl'))
            self.assertEqual(len(backups),2)
            self.assertEqual(sum(len(f.read_text().splitlines()) for f in backups),4)
