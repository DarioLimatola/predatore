"""Observable-context transfer, bounded Boolean votes and persistent role separation."""
from collections import deque
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

from engine.contextual import ContextCodec,ContextMemory,distill_into,counter_value
from engine.vector_boolean import VectorBooleanNet
from engine.boolean_net import BooleanNet
from engine.continual import load_profile,evaluation_copy
from engine.evolution import run
from engine.settings import Limits
from engine import smith
from engine.vault import read_json
from gui.laboratory import available_agent_specs
from test_vector_engine import config
from test_gui_callbacks import CallbackContracts


def teaching_events(codec):
    histories=[[(1,0),(2,1),(3,2)],[(7,0),(2,1),(3,2)]]
    events=[]
    for source in ('explorer-A','explorer-B','explorer-C'):
        for history,target in zip(histories,(2,5)):
            key=codec.encode(4,history)
            events.append(dict(source=source,context_key=key,observation=4,previous=3,
                               valid=1,action=target,feedback=1,next=4,terminal_success=1,
                               eligible_contexts=[(key,target)]))
    return histories,events


class ContextDistillationTests(unittest.TestCase):
    def test_boolean_evidence_counters_match_scalar_reference_and_saturate(self):
        memory=ContextMemory(8);rng=random.Random(77)
        reference=[[0]*8,[0]*8]
        for _ in range(400):
            action=rng.randrange(8);positive=bool(rng.randrange(2))
            memory.update('0',action,positive)
            index=0 if positive else 1
            reference[index][action]=min((1<<Limits.CONTEXT_COUNTER_BITS)-1,reference[index][action]+1)
            for plane_index in (0,1):
                self.assertEqual([counter_value(memory.rows['0'][plane_index],a) for a in range(8)],reference[plane_index])
        expected=[a for a in range(8) if reference[0][a]>=2 and reference[0][a]>reference[1][a]]
        actual=memory.choose('0',3)
        if expected:
            best=max(reference[0][a]-reference[1][a] for a in expected)
            self.assertIn(actual,[a for a in expected if reference[0][a]-reference[1][a]==best])
        else:self.assertIsNone(actual)

    def test_longer_observable_context_separates_old_alias_and_transfers_to_eva(self):
        eva=VectorBooleanNet(3);histories,events=teaching_events(eva.context_codec)
        self.assertEqual(histories[0][-1],histories[1][-1])
        self.assertNotEqual(eva.context_codec.encode(4,histories[0]),eva.context_codec.encode(4,histories[1]))
        # Same current and previous observation would collide in the old decoder.
        self.assertEqual(events[0]['observation']|(events[0]['previous']<<15),
                         events[1]['observation']|(events[1]['previous']<<15))
        report=distill_into(eva,events,focus_terminal=True)
        self.assertEqual(report['contributors_taught'],3)
        self.assertEqual(report['contexts_taught'],2)
        reloaded=BooleanNet.restore(json.loads(json.dumps(eva.export())))
        for net in (eva,reloaded):
            for history,target in zip(histories,(2,5)):
                net.reset();net.observable_history=deque(history,maxlen=Limits.CONTEXT_HISTORY)
                self.assertEqual(net.decide([(4>>i)&1 for i in range(15)]),target)

    def test_archive_repetition_does_not_manufacture_extra_teacher_votes(self):
        eva=VectorBooleanNet();_,events=teaching_events(eva.context_codec)
        distill_into(eva,events);once=eva.contextual_memory.export()
        report=distill_into(eva,events)
        self.assertEqual(report['contexts_taught'],0)
        self.assertEqual(eva.contextual_memory.export(),once)
        other=VectorBooleanNet();distill_into(other,events*20)
        self.assertEqual(other.contextual_memory.export(),once)

    def test_contradictory_teachers_are_not_blindly_or_combined(self):
        eva=VectorBooleanNet();key=eva.context_codec.encode(0,[])
        # One positive and one negative opinion are equally strong, hence uncertain.
        events=[]
        for source,feedback in [('A',1),('B',2)]:
            for _ in range(4):
                events.append(dict(source=source,context_key=key,observation=0,previous=0,
                                   valid=0,action=2,feedback=feedback,next=0,terminal_success=0))
        report=distill_into(eva,events)
        self.assertEqual(report['contexts_taught'],0)
        self.assertIsNone(eva.contextual_memory.choose(key,0))

    def test_terminal_trace_teaches_only_after_real_feedback(self):
        n=VectorBooleanNet(8);stream=[];n.collect_experience=True;n.experience_sink=lambda e:stream.append({**e,'source':'teacher'})
        n.decide([0]*15)
        self.assertEqual(len(n.contextual_memory.rows),0)
        n.transition([1]*15,True);n.feedback(20)
        self.assertTrue(stream[-1]['terminal_success'])
        self.assertTrue(stream[-1]['eligible_contexts'])
        self.assertEqual(stream[-1]['context_key'],n.context_codec.encode(0,[]))
        student=VectorBooleanNet(9);distill_into(student,stream,focus_terminal=True)
        self.assertEqual(student.decide([0]*15),stream[-1]['action'])

    def test_context_memory_budget_and_frozen_probe_isolation(self):
        net=VectorBooleanNet(memory_capacity=8)
        for i in range(30):net.contextual_memory.update(str(i),2,True)
        self.assertEqual(len(net.contextual_memory.rows),8)
        before=copy.deepcopy(net.export());clone=evaluation_copy(net,learning=False)
        clone.decide([0]*15);clone.feedback(20)
        self.assertEqual(net.export(),before)
        self.assertEqual(clone.contextual_memory.export(),net.contextual_memory.export())

    def test_distilled_negative_consensus_can_reject_an_old_action(self):
        net=VectorBooleanNet();key=net.context_codec.encode(0,[])
        wrong=net.decide([0]*15);net.reset()
        for _ in range(3):net.contextual_memory.update(key,wrong,True)
        events=[]
        for source in ('A','B','C','D'):
            for _ in range(3):
                events.append(dict(source=source,context_key=key,observation=0,previous=0,
                                   valid=0,action=wrong,feedback=2,next=0,terminal_success=0))
        report=distill_into(net,events)
        self.assertEqual(report['conflicting_contexts'],1)
        self.assertNotEqual(net.decide([0]*15),wrong)

    def test_specialists_and_smith_remember_between_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            first,_=run(tmp,config(generations=2))
            old=load_profile(tmp);teacher=BooleanNet.restore(old['explorers'][0]['brain'])
            second,_=run(tmp,config(seed=618,generations=1))
            inherited=BooleanNet.restore(read_json(second/'brains/initial_explorers.json')[0])
            self.assertEqual(teacher.snapshot_id(),inherited.snapshot_id())
            state=load_profile(tmp)
            self.assertEqual(len(state['meta']['run_memory']),2)
            self.assertEqual(sum(v[0] for v in state['meta']['arms'].values()),3)
            self.assertEqual(state['roles']['population'],'explorers')
            for row in state['meta']['run_memory']:
                self.assertNotIn('holdout',json.dumps(row))
            last=read_json(second/'checkpoints/latest.json')['history'][-1]
            self.assertEqual(last['distillation']['explorers_observed'],4)
            self.assertTrue(last['distillation']['current_teachers'])

    def test_smith_learns_different_strategies_for_different_contexts(self):
        state={}
        for _ in range(5):
            smith.observe(state,'explore',True,context=0)
            smith.observe(state,'consolidate',False,context=0)
            smith.observe(state,'explore',False,context=1)
            smith.observe(state,'consolidate',True,context=1)
        smith.observe(state,'explore',True,context=0)
        smith.observe(state,'consolidate',True,context=1)
        for context in (0,1):
            for strategy in ('represent','compact'):smith.observe(state,strategy,False,context)
        self.assertEqual(smith.choose(state,0),'explore')
        self.assertEqual(smith.choose(state,1),'consolidate')


class RolesGUI(CallbackContracts):
    def test_single_eva_explorers_and_separate_smith_panel(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder,_=run(tmp,config(generations=1))
            actors=available_agent_specs(folder)
            self.assertEqual(sum(a['name'].startswith('EVA') for a in actors),1)
            self.assertEqual(sum(a['name'].startswith('Esploratore') for a in actors),4)
            self.assertFalse(any(a['name'].startswith('SMITH') for a in actors))
            def hook(app,ui):
                self.assertIn('supervisor',ui['panels'])
                ui['refresh_supervisor']()
                self.assertIn('run_memory',ui['supervisor_text'].text)
                self.assertIn('contesti',ui['eva_status'].get())
                ui['close_window']()
            self.harness(Path(tmp),hook)


if __name__=='__main__':unittest.main()
