import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from engine.evolution import run
from engine.progress_monitor import ProgressMonitor, completion_metrics, population_completion
from engine.vault import read_json,latest_checkpoint_path
from engine.supplementary import probe_world,run_probes
from test_vector_engine import config
import test_gui_callbacks as callbacks
from test_gui_callbacks import Widget
from gui.progress_chart import draw_progress

class ProgressTests(unittest.TestCase):
    def test_failed_episodes_do_not_enter_completion_mean(self):
        result=completion_metrics([{'capture':1,'steps':4},{'capture':0,'steps':200},{'capture':1,'steps':10}])
        self.assertEqual(result['successful_mean_steps'],7)
        self.assertEqual(result['best_capture_steps'],4)
        self.assertIsNone(completion_metrics([{'capture':0,'steps':20}])['successful_mean_steps'])
        other=completion_metrics([{'capture':1,'steps':1}])
        self.assertEqual(population_completion([(result,None),(other,None)])['population_successful_mean_steps'],5)

    def test_monitor_does_not_change_population_rng_or_champion(self):
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            cfg=config(generations=3)
            original,_=run(a,cfg)
            folder=Path(b)/'EvolutionVault/experiments'/f'EXP_seed{cfg.seed}_size{cfg.size}'
            monitor=ProgressMonitor(folder,cfg,every=2)
            observed,_=run(b,cfg,monitor=monitor)
            self.assertIsNone(monitor.error)
            for relative in ('brains/eva.json','benchmarks/results.json'):
                self.assertEqual(read_json(original/relative),read_json(observed/relative))
            cp1=read_json(latest_checkpoint_path(original/'checkpoints'))
            cp2=read_json(latest_checkpoint_path(observed/'checkpoints'))
            self.assertEqual(cp1,cp2)
            snap=read_json(observed/'monitor/latest.json')
            self.assertEqual(set(snap['baselines']),{'train','validation'})
            self.assertEqual([r['generation'] for r in monitor.history],[1,2,3])

    def test_monitor_resume_discards_future_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg=config(seed=29,generations=2);folder=Path(temp)/'EvolutionVault/experiments/EXP_seed29_size6'
            run(temp,cfg,monitor=ProgressMonitor(folder,cfg,1))
            cfg.generations=3
            monitor=ProgressMonitor(folder,cfg,1)
            run(temp,cfg,resume=True,monitor=monitor)
            self.assertEqual([r['generation'] for r in monitor.history],[1,2,3])

    def test_invisible_and_required_pounce_geometry(self):
        record={'size':6,'obstacles':[[3,y] for y in range(6)]}
        world=probe_world(record,0,0,True,True)
        self.assertFalse(any(world.observe()[:5]));self.assertEqual(world.predator,(2,0))
        world.step(7);self.assertEqual(world.predator,(4,0));self.assertTrue(world.captured)
        world=probe_world(record,0,6,True,False)
        world.step(7);self.assertEqual(world.predator,(2,0))

    def test_probes_preserve_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder,_=run(tmp,config(generations=1))
            def fingerprint():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob('*') if p.is_file()}
            before=fingerprint()
            _,report=run_probes(folder,Path(tmp)/'Analysis',episodes=4)
            self.assertEqual(before,fingerprint())
            for item in report['scenarios']['Barriera: salto disabilitato'].values():self.assertEqual(item['capture_rate'],0)
            self.assertEqual(report['scenarios']['Barriera: salto necessario']['Casuale 4 azioni']['capture_rate'],0)

    def test_chart_labels_real_generations_and_missing_data(self):
        c=Widget()
        draw_progress(c,[{'generation':15,'capture_rate':.2},{'generation':100,'capture_rate':.4}],'Catture')
        labels=[r.get('text') for r in c.items.values()]
        self.assertIn('15',labels);self.assertIn('100',labels);self.assertIn('Generazione',labels)
        draw_progress(c,[{'generation':15},{'generation':100}],'Passi per cattura')

class NewGuiContracts(unittest.TestCase):
    def test_new_controls_and_monitor_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=config(seed=29,generations=1)
            folder=Path(tmp)/'EvolutionVault/experiments/EXP_seed29_size6'
            run(tmp,cfg,monitor=ProgressMonitor(folder,cfg,1))
            def hook(app,ui):
                ui['mode'].set('replay')
                ui['replay_all'].set(True)
                # EVA and the four explorers, all on the same recorded training episode.
                self.assertEqual(len(ui['replay_group']),5)
                self.assertEqual(len({str(item['frames'][0]['predator']) for item in ui['replay_group']}),1)
                for metric in ('Catture','Passi per cattura','Neuroni attivi'):ui['graph_metric'].set(metric)
                ui['graph_scenario'].set('Preda mobile')
                self.assertGreater(len(ui['table'].get_children()),4)
                ui['live_mosaic'].set(True);ui['mode'].set('live')
                ui['draw_arena']();ui['close_window']()
            callbacks.CallbackContracts().harness(Path(tmp),hook)
