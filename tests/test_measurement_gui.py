"""Matched measurements, exact paired outcomes, and experiment-switch regressions."""
import json
import os
import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from engine.paired_evidence import paired_capture_evidence
from engine.continual import assessment, evaluation_copy
from engine.boolean_net import BooleanNet
from engine.challenges import load_suites
from engine.evolution import run
from engine.vault import read_json
from test_vector_engine import config
from test_gui_callbacks import CallbackContracts, Widget


class EvidenceTests(unittest.TestCase):
    def test_masks_match_independent_episode_pairs(self):
        rng=random.Random(19)
        for count in (1,6,25,64,129):
            old=[rng.randrange(2) for _ in range(count)]
            new=[rng.randrange(2) for _ in range(count)]
            def row(bits):return {'paired_outcomes':{'episodes':count,'captures':sum(b<<i for i,b in enumerate(bits))}}
            result=paired_capture_evidence(row(old),row(new))
            self.assertEqual(result['gained_captures'],sum(a==0 and b==1 for a,b in zip(old,new)))
            self.assertEqual(result['lost_captures'],sum(a==1 and b==0 for a,b in zip(old,new)))
            self.assertEqual(result['capture_delta_numerator'],sum(new)-sum(old))

    def test_exact_tail_and_equal_outcomes(self):
        old={'paired_outcomes':{'episodes':6,'captures':0}}
        new={'paired_outcomes':{'episodes':6,'captures':63}}
        result=paired_capture_evidence(old,new)
        self.assertEqual((result['one_sided_sign_p_numerator'],result['one_sided_sign_p_denominator']),(1,64))
        tied=paired_capture_evidence(old,old)
        self.assertEqual((tied['one_sided_sign_p_numerator'],tied['one_sided_sign_p_denominator']),(1,1))

    def test_adaptive_population_comparison_uses_exact_same_test_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder,_=run(tmp,config(generations=1))
            report=read_json(folder/'benchmarks/population_comparison.json')
            self.assertEqual(report['eva'],read_json(folder/'benchmarks/results.json')['holdout'])
            self.assertEqual(report['population_mean_capture'],
                             sum(m['capture_rate'] for m in report['individuals'])/4)
            for individual in report['individuals']:
                self.assertEqual(individual['episode_count'],report['eva']['episode_count'])
            # Bit masks encode the observed captures, without a probability approximation.
            for row in report['eva']['by_scenario'].values():
                self.assertEqual(row['paired_outcomes']['captures'].bit_count(),row['capture_count'])


class RefreshTests(CallbackContracts):
    def test_programmatic_switch_clears_previous_run_even_with_equal_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            first,_=run(tmp,config(seed=731,generations=1))
            second,_=run(tmp,config(seed=732,generations=1))
            for folder in (first,second):
                for file in folder.rglob('*'):
                    if file.is_file():os.utime(file,ns=(1700000000000000000,1700000000000000000))
            def hook(app,ui):
                ui['exp_name'].set(first.name)
                self.assertIn(first.name,ui['diagnostics_text'].text)
                ui['exp_name'].set(second.name)
                self.assertEqual(ui['selected_source'][0],second)
                self.assertIn(second.name,ui['diagnostics_text'].text)
                self.assertEqual(ui['last_refresh'][0][0],str(second.resolve()))
                pending=Path(tmp)/'EvolutionVault/experiments/EXP_seed999_size6'
                pending.mkdir()
                ui['exp_name'].set(pending.name)
                self.assertEqual(ui['table'].get_children(),[])
                self.assertEqual(ui['plot'].items,{})
                self.assertEqual(ui['replay_group'],[])
                self.assertEqual(ui['replay_frames'][0],[])
                self.assertIsNone(ui['inspector'].brain)
                self.assertIn(pending.name,ui['diagnostics_text'].text)
                ui['close_window']()
            self.harness(Path(tmp),hook)

    def test_worker_without_config_keeps_new_run_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            first,_=run(tmp,config(seed=733,generations=1))
            def hook(app,ui):
                ui['fields']['seed'].set('734');ui['fields']['size'].set('6')
                with patch('gui.laboratory.subprocess.Popen') as start:
                    start.return_value.poll.return_value=None
                    ui['start_training']()
                    ui['periodic']()
                    self.assertEqual(ui['exp_name'].get(),'EXP_seed734_size6')
                    self.assertEqual(ui['table'].get_children(),[])
                def labels(widget):
                    return [widget.options.get('text','')]+[label for child in widget.children for label in labels(child)]
                self.assertFalse(any('Esegui prove' in str(label) for label in labels(ui['panels']['diagnostics'])))
                ui['close_window']()
            self.harness(Path(tmp),hook)


if __name__=='__main__':unittest.main()
