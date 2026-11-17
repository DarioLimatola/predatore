"""Learning roles, symmetric structural evolution and reviewable teacher selection."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from engine.boolean_net import BooleanNet
from engine.continual import (build_transfer_proposals, evaluation_copy, fit_topology,
                              load_profile, ProbeCache, select_transfer, selection_key)
from engine.evolution import Config, run
from engine.settings import Limits
from engine.vector_boolean import VectorBooleanNet
from engine.vault import read_json
from gui.progress_chart import draw_progress, EVA_COLOR, POPULATION_COLOR
from test_gui_callbacks import CallbackContracts, Widget
from test_vector_engine import config


def metrics(captures, episodes=4, reward=0, stalls=0):
    row={'capture_count':captures,'episode_count':episodes,'stall_count':stalls,
         'capture_rate':captures/episodes,'stall_rate':stalls/episodes,'mean_reward':reward,
         'paired_outcomes':{'captures':(1<<captures)-1,'episodes':episodes,'stalls':(1<<stalls)-1}}
    return {**row,'by_scenario':{'static':dict(row)}}


class DevelopmentTests(unittest.TestCase):
    def test_default_limits_and_removed_schedules(self):
        c=Config()
        self.assertEqual((c.minimum_active_neurons,c.initial_neurons,c.max_neurons),(32,64,128))
        self.assertEqual(len(VectorBooleanNet().nodes),64)
        self.assertFalse({'prune_every','growth_every','evolution_mode','eva_profile'} & asdict(c).keys())
        d=VectorBooleanNet().export();d['engine']='boolean-vector/3'
        with self.assertRaises(ValueError):BooleanNet.restore(d)

    def test_neurons_can_increase_or_decrease_without_waiting(self):
        deltas=set()
        for seed in range(100):
            net=VectorBooleanNet(seed)
            report=net.vary_topology(seed,32,128)
            deltas.add(report['topology_delta'])
            self.assertEqual(len(net.nodes),64+report['topology_delta'])
        self.assertEqual(deltas,{-1,0})
        growth=[]
        for seed in range(12):
            net=VectorBooleanNet(seed,neurons=32)
            for i,node in enumerate(net.nodes):node.a=net.sensors+max(0,i-1)
            net.outputs=[31]*3;net.invalidate()
            growth.append(net.vary_topology(seed,32,128)['topology_delta'])
        self.assertIn(1,growth)

    def test_removal_remaps_cycles_hold_gates_outputs_and_reload(self):
        rng=random.Random(54)
        net=VectorBooleanNet(54)
        for _ in range(300):
            net.recovery_mutate(rng.randrange(99999),70,allow_growth=False,allow_reactivate=False)
            net.vary_topology(rng.randrange(99999),32,70)
            self.assertTrue(32<=len(net.nodes)<=70)
            net=BooleanNet.restore(net.export())
            self.assertIn(net.decide([rng.randrange(2) for _ in range(15)]),range(8))
        fit_topology(net,32,32,1)
        self.assertEqual(len(net.nodes),32)
        for seed in range(100):net.vary_topology(seed,32,32)
        self.assertEqual(len(net.nodes),32)
        BooleanNet.restore(net.export()).decide([0]*15)

    def test_structural_resume_preserves_every_learned_state(self):
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            cfg=config(initial_neurons=40,max_neurons=48,generations=5)
            fa,_=run(a,cfg)
            run(b,config(initial_neurons=40,max_neurons=48,generations=2))
            fb,_=run(b,cfg,resume=True)
            self.assertEqual(load_profile(a),load_profile(b))
            self.assertEqual(read_json(fa/'checkpoints/latest.json'),read_json(fb/'checkpoints/latest.json'))

    def test_teacher_keeps_intact_circuit_and_memory_as_candidate(self):
        eva=VectorBooleanNet(5);teacher=VectorBooleanNet(8,neurons=65)
        for _ in range(3):teacher.learn(0,3,True)
        proposals=build_transfer_proposals(eva,[(metrics(4),teacher)],[])
        direct=next(n for label,n,_ in proposals if label.startswith('teacher_') and label!='teacher_distilled') if any(label=='teacher_1' for label,_,_ in proposals) else next(n for label,n,_ in proposals if label=='teacher_distilled')
        self.assertEqual(direct.snapshot_id(),teacher.snapshot_id())

    def test_independent_probe_cache_uses_initial_memory_and_preserves_actor(self):
        cache=ProbeCache();net=VectorBooleanNet(4)
        task={'suites':[],'seed':1,'episodes':1,'steps':1}
        with patch('engine.continual.assessment',return_value=metrics(1)) as score:
            cache.assess(net,task);cache.assess(net,task)
            net.learn(123,2,True);cache.assess(net,task)
        self.assertEqual(score.call_count,2)
        before=copy.deepcopy(net.export());clone=evaluation_copy(net)
        clone.decide([0]*15);clone.feedback(-1)
        self.assertEqual(net.export(),before)
        self.assertNotEqual(clone.knowledge,net.knowledge)

    def test_retention_loss_can_be_offset_by_larger_current_gain(self):
        eva=VectorBooleanNet(1);teacher=VectorBooleanNet(2,neurons=65)
        tasks=[{'run':'past'},{'run':'current'}]
        before=[metrics(3),metrics(0)]
        after=[metrics(2,reward=-10),metrics(3,reward=-10)]
        with patch('engine.continual.probe_scores',side_effect=[before,after]):
            selected,improved,decisions=select_transfer(eva,[('teacher',teacher,0)],tasks)
        self.assertIs(selected[1],teacher)
        self.assertTrue(improved)
        self.assertEqual(decisions[0]['diagnostics']['scenarios'][0]['capture_delta'],-1)
        self.assertEqual(len(selected[1].nodes),65)
        with patch('engine.continual.probe_scores',side_effect=[before,[metrics(1),metrics(0)]]):
            self.assertIsNone(select_transfer(eva,[('teacher',teacher,0)],tasks)[0])

    def test_equal_performance_selects_smaller_teacher_without_timer(self):
        eva=VectorBooleanNet(1);teacher=VectorBooleanNet(2,neurons=63)
        rows=[metrics(2)]
        with patch('engine.continual.probe_scores',side_effect=[rows,rows]):
            selected,improved,_=select_transfer(eva,[('teacher',teacher,0)],[{'run':'current'}])
        self.assertIs(selected[1],teacher)
        self.assertTrue(improved)

    def test_all_past_tasks_share_weight_without_outvoting_current(self):
        old=metrics(4);now=metrics(0)
        self.assertEqual(selection_key([old,now],64),selection_key([old]*7+[now],64))

    def test_eva_keeps_learning_even_when_teacher_transfer_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('engine.continual.select_transfer',return_value=(None,False,[])):
                folder,_=run(tmp,config(generations=2))
            state=load_profile(tmp)
            for row in state['history']:
                self.assertFalse(row['eva_promoted'])
                self.assertTrue(row['eva_lifetime_changed'])
                self.assertTrue(row['eva_updated'])
            self.assertTrue(state['eva']['knowledge'])
            self.assertEqual(state['roles']['eva'],Limits.EVA_LABEL)
            self.assertEqual(state['roles']['smith'],Limits.SMITH_LABEL)
            self.assertEqual(BooleanNet.restore(state['eva']).snapshot_id(),read_json(folder/'checkpoints/latest.json')['history'][-1]['champion_id'])

    def test_live_eva_captures_equal_the_displayed_training_result(self):
        class Observer:
            def __init__(self):self.context={};self.episodes=[]
            def set_context(self,**kw):self.context=kw
            def begin_episode(self,net,scenario,seed):
                self.episodes.append({'phase':self.context['phase'],'individual':self.context['individual'],
                                      'seed':seed,'scenario':scenario['name'],'frames':[]})
                return True
            def add_frame(self,frame):self.episodes[-1]['frames'].append(frame)
            def end_episode(self):pass
        with tempfile.TemporaryDirectory() as tmp:
            observer=Observer();folder,_=run(tmp,config(generations=1),observer=observer)
            row=read_json(folder/'checkpoints/latest.json')['history'][0]
            eva=[e for e in observer.episodes if e['phase']=='eva']
            self.assertEqual(len(eva),row['episode_count'])
            self.assertEqual(sum(any(f.get('captured') for f in e['frames']) for e in eva),row['capture_count'])
            pairs={(e['scenario'],e['seed']) for e in eva}
            for slot in range(4):
                group=[e for e in observer.episodes if e['phase']=='explorer' and e['individual']==slot]
                self.assertEqual({(e['scenario'],e['seed']) for e in group},pairs)
            self.assertEqual(row['measurement'],'adaptive_training')

    def test_finalization_can_recover_after_last_generation_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('engine.continual._finish',side_effect=OSError('interrupted finalization')):
                with self.assertRaises(OSError):run(tmp,config(generations=1))
            folder,_=run(tmp,config(generations=1),resume=True)
            self.assertTrue(read_json(folder/'results.json')['complete'])

    def test_neuron_axis_is_integer_and_population_stays_green(self):
        canvas=Widget()
        rows=[{'generation':i,'active_neurons':64,'population_mean_neurons':63.5,
               'capture_rate':.2,'population_mean_capture':.3} for i in range(1,4)]
        caption=draw_progress(canvas,rows,'Neuroni attivi')
        self.assertIn('arrotondata',caption)
        self.assertFalse(any('.0' in str(v.get('text','')) for v in canvas.items.values()))
        caption=draw_progress(canvas,rows,'Catture')
        self.assertIn(Limits.EVA_LABEL,caption)
        self.assertIn(POPULATION_COLOR,{v.get('fill') for v in canvas.items.values()})
        self.assertNotIn('#94a3b8',{v.get('fill') for v in canvas.items.values()})


class DevelopmentGUI(CallbackContracts):
    def test_three_integer_slider_toggles_control_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            def hook(app,ui):
                for key,number in [('minimum',40),('base',72),('maximum',160)]:
                    control=ui['neuron_controls'][key]
                    self.assertEqual(control['scale'].options['state'],'disabled')
                    control['enabled'].set(True);control['toggle']()
                    self.assertEqual(control['spin'].options['state'],'normal')
                    control['value'].set(number)
                with patch('gui.laboratory.subprocess.Popen') as launch:
                    launch.return_value.poll.return_value=None
                    ui['start_training']()
                    cmd=launch.call_args.args[0]
                    for flag,value in [('--min-neurons','40'),('--base-neurons','72'),('--max-neurons','160'),('--smith-profile','SMITH')]:
                        self.assertEqual(cmd[cmd.index(flag)+1],value)
                    self.assertNotIn('--growth-every',cmd)
                    self.assertNotIn('--prune-every',cmd)
                ui['close_window']()
            self.harness(Path(tmp),hook)

    def test_smith_panel_follows_selected_run_and_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            first,_=run(tmp,config(generations=1,smith_profile='ALPHA'))
            second,_=run(tmp,config(seed=605,generations=1,smith_profile='BETA'))
            def hook(app,ui):
                for folder,name in [(first,'ALPHA'),(second,'BETA')]:
                    ui['exp_name'].set(folder.name)
                    self.assertIn('Profilo SMITH: '+name,ui['supervisor_text'].text)
                    self.assertIn(folder.name,ui['supervisor_text'].text)
                    self.assertNotIn('Profilo EVA',ui['supervisor_text'].text)
                    self.assertIn(Limits.EVA_LABEL,ui['train_status'].get())
                ui['close_window']()
            self.harness(Path(tmp),hook)

    def test_invalid_slider_order_does_not_launch_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            def hook(app,ui):
                control=ui['neuron_controls']['minimum']
                control['enabled'].set(True);control['value'].set(80)
                with patch('gui.laboratory.subprocess.Popen') as start,patch('gui.laboratory.messagebox.showerror') as error:
                    ui['start_training']()
                    start.assert_not_called();error.assert_called_once()
                ui['close_window']()
            self.harness(Path(tmp),hook)


if __name__=='__main__':unittest.main()
