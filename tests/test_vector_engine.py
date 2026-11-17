"""End-to-end contracts for the vector engine and persistent EVA lifecycle."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from boolean_engine.memory import counter_value
from engine.vector_boolean import VectorBooleanNet, BinarySpace, pack, unpack, rotate, masked_write
from engine.continual import (load_profile, profile_lock, profile_path,
                              fit_topology, evaluation_copy)

from engine.evolution import Config, run
from engine.boolean_net import BooleanNet
from engine.vault import read_json, verify
from engine.telemetry import LiveObserver
from engine.progress_monitor import ProgressMonitor


def config(**values):
    defaults=dict(population=4,generations=2,episodes=4,steps=15,
                  size=6,seed=610,initial_neurons=32,max_neurons=36,validation_episodes=4,
                  holdout_episodes=4,memory_capacity=32,experience_capacity=32,vector_width=16,
                  retention_tasks=2,minimum_active_neurons=32)
    defaults.update(values)
    return Config(**defaults)


class VectorCircuitTests(unittest.TestCase):
    def test_boolean_vector_matches_independent_lane_reference(self):
        net=VectorBooleanNet(4,neurons=5,shape=(2,8));rng=random.Random(123)
        net.associative_enabled=False;net.sparse_enabled=False
        for rule in net.rules:
            rule['lut']=[rng.getrandbits(16) for _ in range(4)]
            rule['write']=rng.getrandbits(16)
        for _ in range(20):
            obs=[rng.randrange(2) for _ in range(15)]
            old=list(net.vectors);packed=pack(obs)
            inputs=[rotate(packed,i,16) ^ (net.mask if b else 0) for i,b in enumerate(obs)]+old
            expected=[]
            for i,(n,r) in enumerate(zip(net.nodes,net.rules)):
                bits=[]
                for lane in range(16):
                    a=(inputs[n.a]>>((lane-r['shift_a'])%16))&1
                    b=(inputs[n.b]>>((lane-r['shift_b'])%16))&1
                    hold=(inputs[r['hold']]>>lane)&1 if r['hold']>=0 else 0
                    value=(r['lut'][2*a+b]>>lane)&1
                    bits.append(((old[i]>>lane)&1) if hold or not ((r['write']>>lane)&1) else value)
                expected.append(pack(bits))
            net.decide(obs)
            self.assertEqual(net.vectors,expected)

    def test_confirmed_evidence_does_not_cancel_and_can_reverse(self):
        n=VectorBooleanNet(neurons=3)
        for _ in range(10):n.learn(11,2,True)
        self.assertEqual(counter_value(n.knowledge['11'][0],2),10)
        for _ in range(3):n.learn(11,2,False)
        self.assertEqual(n.short_memory.choose('11',2),2)
        for _ in range(8):n.learn(11,2,False)
        self.assertNotEqual(n.short_memory.choose('11',2),2)

    def test_lifetime_learning_and_reset_retains_acquired_memory(self):
        n=VectorBooleanNet(neurons=3)
        obs=[0]*15
        wrong=n.decide(obs);n.feedback(-1)
        n.reset();self.assertEqual(n.decide(obs),wrong);n.feedback(-1)
        n.reset();self.assertNotEqual(n.decide(obs),wrong)
        n.plasticity_enabled=False;before=copy.deepcopy(n.knowledge)
        n.feedback(-1);self.assertEqual(n.knowledge,before)

    def test_local_boolean_candidates_are_trace_guided_and_bit_surgical(self):
        n=VectorBooleanNet(seed=4,neurons=5,shape=(2,8))
        n.associative_enabled=False
        n.collect_experience=True
        for _ in range(3):
            n.decide([0]*15);n.feedback(-1)
        candidates=n.local_circuit_candidates()
        self.assertTrue(candidates)
        node,pattern,lane=candidates[0]
        before=list(n.rules[node]['lut'])
        n.apply_local_circuit_candidate(candidates[0])
        self.assertEqual(n.rules[node]['lut'][pattern],before[pattern]^(1<<lane))
        self.assertEqual([value for i,value in enumerate(n.rules[node]['lut'])
                          if i!=pattern],
                         [value for i,value in enumerate(before) if i!=pattern])

    def test_positive_local_evidence_suppresses_candidate(self):
        n=VectorBooleanNet(seed=4,neurons=5)
        n.associative_enabled=False
        n.collect_experience=True
        for _ in range(3):
            n.decide([0]*15);n.feedback(-1)
        self.assertTrue(n.local_circuit_candidates())
        for _ in range(3):
            n.decide([0]*15);n.feedback(1)
        self.assertEqual(n.local_circuit_candidates(),[])

    def test_delayed_feedback_reaches_hidden_ancestor(self):
        from engine.boolean_net import Neuron
        n=VectorBooleanNet(seed=8,sensors=1,neurons=2,shape=(2,8))
        n.nodes=[Neuron(0,0,'COPY'),Neuron(1,1,'COPY')]
        n.outputs=[1,1,1];n.output_lanes=[0,0,0]
        n.sensor_encoding='broadcast'
        n.rules=[
            {'shift_a':0,'shift_b':0,'hold':-1,'write':n.mask,
             'lut':[0,0,0,n.mask]},
            {'shift_a':0,'shift_b':0,'hold':-1,'write':n.mask,
             'lut':[0,0,0,n.mask]}]
        n.reset();n.collect_experience=True;n.associative_enabled=False
        n.decide([1]);n.feedback(0)
        n.decide([1]);n.feedback(-1)
        low_neg=n.circuit_evidence[(0,3)][2]
        self.assertTrue(low_neg & 1)

    def test_decoder_override_is_not_credited_to_circuit(self):
        n=VectorBooleanNet(seed=4,neurons=5)
        observation=[0]*15
        proposed=n.decide(observation);n.reset()
        address=n._address(observation);alternative=(proposed+1)%8
        for _ in range(2):n.learn(address,alternative,True)
        n.collect_experience=True
        action=n.decide(observation)
        self.assertNotEqual(action,proposed)
        n.feedback(-1)
        self.assertFalse(n.circuit_evidence)

    def test_circuit_plasticity_ablation_disables_trace_and_candidates(self):
        n=VectorBooleanNet(seed=4,neurons=5)
        n.collect_experience=True;n.circuit_plasticity_enabled=False
        n.decide([0]*15);n.feedback(-1)
        self.assertFalse(n.circuit_trace)
        self.assertFalse(n.circuit_evidence)

    def test_cross_agent_experience_transfer(self):
        a=VectorBooleanNet(seed=2,neurons=3);events=[]
        a.collect_experience=True;a.experience_sink=events.append
        action=a.decide([0]*15);a.transition([1]*15,True);a.feedback(20)
        b=VectorBooleanNet(seed=3,neurons=3);b.replay_experience(events)
        self.assertFalse(b.knowledge)
        self.assertTrue(b.contextual_memory.teachers)
        self.assertEqual(b.decide([0]*15),action)

    def test_round_trip_continues_exact_state_and_evidence(self):
        a=VectorBooleanNet(9,neurons=4)
        for i in range(8):
            a.decide([i%2]*15);a.transition([(i+1)%2]*15);a.feedback(1)
        b=BooleanNet.restore(a.export());self.assertEqual(a.export(),b.export())
        for i in range(10):
            obs=[(j+i)%2 for j in range(15)]
            self.assertEqual(a.decide(obs),b.decide(obs))
            for n in (a,b):n.transition(obs);n.feedback(-1)
        self.assertEqual(a.export(),b.export())

    def test_growth_initially_preserves_decisions(self):
        a=VectorBooleanNet(8,neurons=5);b=VectorBooleanNet.restore(a.export())
        b.grow(55,10)
        for i in range(12):self.assertEqual(a.decide([i%2]*15),b.decide([i%2]*15))
        self.assertEqual(len(b.nodes),6)

    def test_growth_mutation_is_favored_and_can_connect_new_neuron(self):
        events=[VectorBooleanNet(seed=i,neurons=4).recovery_mutate(
                i+1000,max_neurons=40) for i in range(300)]
        growth=sum(event['kind'] in ('vector_grow','vector_grow_output') for event in events)
        self.assertGreater(growth/len(events),.3)
        net=VectorBooleanNet(seed=3,neurons=4)
        for seed in range(1000):
            event=net.recovery_mutate(seed,max_neurons=40)
            if event['kind']=='vector_grow_output':
                self.assertIn(len(net.nodes)-1,net.outputs)
                break
        else:self.fail('Expected a growth mutation connected to an output')

    def test_minimum_active_neurons_grows_only_to_requested_floor(self):
        net=VectorBooleanNet(seed=3,neurons=6)
        fit_topology(net,32,48,91)
        self.assertEqual(sum(node.active for node in net.nodes),32)
        node_count=len(net.nodes)
        fit_topology(net,32,48,91)
        self.assertEqual(len(net.nodes),node_count)

    def test_budgets_and_shape(self):
        n=VectorBooleanNet(neurons=3,memory_capacity=8)
        for i in range(40):n.learn(i,0,True)
        self.assertEqual(len(n.knowledge),8)
        n.reshape((2,4,8));self.assertEqual(n.width,64)
        with self.assertRaises(ValueError):n.reshape((3,4))
        with self.assertRaises(ValueError):VectorBooleanNet(internal_ticks=100)
        n.extend_width((2,64));self.assertEqual(n.width,128)

    def test_vocabulary_and_scalar_geometry(self):
        a=BinarySpace(64,1);x=a.bind('agent','EVA');a.register('temperature')
        self.assertEqual(x,a.bind('agent','EVA'))
        self.assertEqual((a.scalar('x',10,32)^a.scalar('x',11,32)).bit_count(),1)
        self.assertEqual((a.scalar('x',10,32)^a.scalar('x',20,32)).bit_count(),10)
        self.assertEqual(pack(unpack(177,8)),177)
        self.assertEqual(masked_write(15,0,3,4),12)

    def test_prediction_is_actual_observation_and_never_reward_or_coordinates(self):
        n=VectorBooleanNet(neurons=3)
        n.decide([0]*15);n.transition([1]*15);n.feedback(0)
        self.assertEqual(next(iter(n.predictions.values())),(1<<15)-1)
        n.reset();n.decide([0]*15);n.transition([0]*15);n.feedback(0)
        self.assertEqual(n.last_prediction_error,(1<<15)-1)

    def test_invalid_serialization_rejected(self):
        d=VectorBooleanNet(neurons=3).export();d['rules'][0]['lut'][0]=-1
        with self.assertRaises(ValueError):VectorBooleanNet.restore(d)


class ContinualLifecycleTests(unittest.TestCase):






    def test_circuit_plasticity_ablation_runs_paired_independent_lineages(self):
        with tempfile.TemporaryDirectory() as enabled_root,tempfile.TemporaryDirectory() as disabled_root:
            enabled,_=run(enabled_root,config(generations=1,circuit_plasticity_enabled=True))
            disabled,_=run(disabled_root,config(generations=1,circuit_plasticity_enabled=False))
            initial_enabled=read_json(enabled/'brains'/'initial_eva.json')
            initial_disabled=read_json(disabled/'brains'/'initial_eva.json')
            self.assertEqual(initial_enabled,initial_disabled)
            for folder,expected in ((enabled,True),(disabled,False)):
                saved=read_json(folder/'checkpoints'/'latest.json')
                self.assertEqual(saved['config']['circuit_plasticity_enabled'],expected)
                self.assertEqual(saved['history'][0]['circuit_plasticity_enabled'],expected)
                self.assertIn('holdout',read_json(folder/'benchmarks'/'results.json'))

    def test_training_starts_and_keeps_minimum_active_neuron_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder,_=run(tmp,config(generations=1,initial_neurons=32,
                                    minimum_active_neurons=32,max_neurons=40))
            initial=read_json(folder/'brains'/'initial_eva.json')
            checkpoint=read_json(folder/'checkpoints'/'latest.json')
            self.assertEqual(len(initial['nodes']),32)
            self.assertTrue(all(node['active'] for node in initial['nodes']))
            for net in checkpoint['population']:
                self.assertGreaterEqual(sum(node['active'] for node in net['nodes']),32)
            self.assertGreaterEqual(checkpoint['history'][0]['eva_active_neurons'],32)
            log=folder/'evolution'/'consolidation.jsonl'
            records=[json.loads(line) for line in log.read_text(encoding='utf-8').splitlines()]
            self.assertTrue(records)
            for decision in records[-1]['decisions']:
                detail=decision['diagnostics']
                self.assertTrue(detail['complete'])
                self.assertEqual(len(detail['scenarios']),4)

    def test_two_runs_inherit_and_all_agents_contribute(self):
        with tempfile.TemporaryDirectory() as tmp:
            first,_=run(tmp,config(generations=1));s1=load_profile(tmp)
            run(tmp,config(seed=611,generations=1));s2=load_profile(tmp)
            initial=read_json(Path(tmp)/'EvolutionVault/experiments/EXP_seed611_size6/brains/initial_eva.json')
            self.assertEqual(BooleanNet.restore(initial).snapshot_id(),BooleanNet.restore(s1['eva']).snapshot_id())
            self.assertEqual(s2['generation'],2)
            self.assertEqual(len(s2['contributors']),10)
            self.assertTrue(all(v>0 for v in s2['contributors'].values()))
            self.assertTrue(s2['events'])
            self.assertTrue(all(verify(first).values()))
            self.assertEqual(len(s2['tasks']),2)

    def test_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            fa,_=run(a,config(generations=3))
            run(b,config(generations=1))
            fb,_=run(b,config(generations=3),resume=True)
            self.assertEqual(load_profile(a),load_profile(b))
            self.assertEqual(read_json(fa/'checkpoints/latest.json')['population'],
                             read_json(fb/'checkpoints/latest.json')['population'])

    def test_stop_then_resume_and_no_historical_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls=[0]
            def stop():calls[0]+=1;return calls[0]>1
            _,last=run(tmp,config(generations=3),should_stop=stop);self.assertEqual(last,1)
            run(tmp,config(generations=3),resume=True)
            run(tmp,config(seed=612,generations=1))
            with self.assertRaises(ValueError):run(tmp,config(generations=4),resume=True)

    def test_live_and_monitor_do_not_change_learning(self):
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            c=config();run(a,c)
            folder=Path(b)/'EvolutionVault/experiments/EXP_seed610_size6'
            live=LiveObserver(folder,interval=.02)
            try:run(b,c,observer=live,monitor=ProgressMonitor(folder,c,1))
            finally:live.close()
            self.assertEqual(load_profile(a),load_profile(b))
            packet=read_json(folder/'live/latest.json')
            frames=[f for f in packet['frames'] if 'action' in f]
            self.assertTrue(frames);self.assertIn('vector_state',frames[-1])

    def test_no_holdout_contamination_and_probes_are_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            f,_=run(tmp,config(generations=1));state=load_profile(tmp)
            self.assertEqual(state,read_json(f/'checkpoints/latest.json')['supervisor_state'])
            self.assertTrue(all(('/explorer' in e['source'] or '/EVA' in e['source']) for e in state['events']))
            self.assertTrue(all('validation' in s['map_key'] for t in state['tasks'] for s in t['suites']))
            n=VectorBooleanNet.restore(state['eva']);before=n.export()
            clone=evaluation_copy(n);clone.decide([0]*15);clone.feedback(-1)
            self.assertEqual(n.export(),before)

    def test_profile_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            with profile_lock(Path(tmp)/'EVA'):
                with self.assertRaises(RuntimeError):
                    with profile_lock(Path(tmp)/'EVA'):pass
            with profile_lock(Path(tmp)/'EVA'):pass

    def test_regression_guard_and_fresh_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            run(tmp,config(generations=1));run(tmp,config(seed=613,generations=1,smith_profile='fresh'))
            self.assertEqual(load_profile(tmp,'fresh')['generation'],1)
            with self.assertRaises(ValueError):profile_path(tmp,'../bad')

    def test_repair_missing_profile_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            run(tmp,config(generations=1))
            (profile_path(tmp)/'latest.json').unlink()
            run(tmp,config(generations=2),resume=True)
            self.assertEqual(load_profile(tmp)['generation'],2)

if __name__=='__main__':unittest.main()
