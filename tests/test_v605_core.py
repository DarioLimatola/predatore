import itertools
import json
import random
import unittest

from boolean_engine import BinaryVectorNetwork, ContextCodec, distill_into
from boolean_engine.hdc import AssociativeMemory, majority
from boolean_engine.limits import CoreLimits as L
from boolean_engine.memory import ContextMemory, add, counter_value, increment
from boolean_engine.tensor import axis_roll, transpose, OPERATORS
from engine.challenges import training_generation, episode_specs


class Core605Tests(unittest.TestCase):
    def test_tensor_operations_match_coordinate_model(self):
        rng=random.Random(8)
        for shape in ((2,4),(4,4),(2,2,2)):
            coords=list(itertools.product(*(range(axis) for axis in shape)))
            v=rng.getrandbits(len(coords))
            for axis,length in enumerate(shape):
                for offset in (-1,0,1,4):
                    dest=[tuple((c+offset)%length if i==axis else c for i,c in enumerate(src)) for src in coords]
                    expected=sum(((v>>i)&1)<<coords.index(c) for i,c in enumerate(dest))
                    self.assertEqual(axis_roll(v,shape,axis,offset),expected)
            order=tuple(reversed(range(len(shape))))
            target=list(itertools.product(*(range(shape[i]) for i in order)))
            expected=sum(((v>>i)&1)<<target.index(tuple(c[j] for j in order)) for i,c in enumerate(coords))
            self.assertEqual(transpose(v,shape,order),expected)

    def test_sparse_matches_full_with_cycles_hold_and_delays(self):
        rng=random.Random(64)
        for seed in range(5):
            a=BinaryVectorNetwork(seed,neurons=24,shape=(4,4),internal_ticks=3)
            for node,rule in zip(a.nodes,a.rules):
                node.a=rng.randrange(a.sensors+20);node.b=rng.randrange(a.sensors+20)
                rule.update(hold=rng.randrange(-1,a.sensors+20),delay_a=rng.randrange(4),
                            operator_b=rng.choice(OPERATORS),write=rng.getrandbits(16),
                            lut=[rng.getrandbits(16) for _ in range(4)])
            a.outputs=[0,1,2];a.invalidate()
            b=BinaryVectorNetwork.restore(a.export());b.sparse_enabled=False
            self.assertLess(len(a._output_ancestors()),24)
            for i in range(24):
                obs=[rng.randrange(2) for _ in range(15)]
                self.assertEqual(a.decide(obs),b.decide(obs))
                for index in a.computed_nodes:self.assertEqual(a.vectors[index],b.vectors[index])
                for net in (a,b):net.transition(obs);net.feedback(i%3-1)
            self.assertEqual(a.contextual_memory.export(),b.contextual_memory.export())

    def test_json_resume_with_generic_actions_and_delays(self):
        a=BinaryVectorNetwork(4,sensors=5,action_bits=2,neurons=10,shape=(4,4),internal_ticks=2)
        for rule in a.rules:rule.update(delay_a=2,operator_b='neighbor_xor')
        for i in range(16):
            obs=[(i>>j)&1 for j in range(5)]
            self.assertIn(a.decide(obs),range(4));a.transition(obs);a.feedback(1)
        b=BinaryVectorNetwork.restore(json.loads(json.dumps(a.export())))
        for i in range(10):
            obs=[i%2]*5
            self.assertEqual(a.decide(obs),b.decide(obs))
            for net in (a,b):net.transition(obs);net.feedback(-1)
        self.assertEqual(a.snapshot_id(),b.snapshot_id())

    def test_majority_matches_scalar_and_all_zero_tie(self):
        self.assertEqual(majority([0,0],8,255),0)
        rng=random.Random(7)
        for n in range(1,11):
            for _ in range(10):
                vectors=[rng.getrandbits(16) for _ in range(n)]
                tie=rng.getrandbits(16)
                expected=0
                for bit in range(16):
                    votes=sum((v>>bit)&1 for v in vectors)
                    expected |= int(votes*2>n or votes*2==n and tie>>bit&1)<<bit
                self.assertEqual(majority(vectors,16,tie),expected)

    def test_associative_consensus_anchor_and_veto(self):
        codec=ContextCodec()
        mem=AssociativeMemory(codec,8)
        histories=[[(7,1),(3,1),(5,2)],[(6,1),(3,1),(5,2)],[(4,1),(3,1),(5,2)]]
        a,b,q=[codec.encode(9,h) for h in histories]
        mem.observe(a,4,0);self.assertIsNone(mem.choose(q,0))
        mem.observe(b,4,0);self.assertEqual(mem.choose(q,0),2)
        self.assertIsNone(mem.choose(q,0,rejected=4))
        self.assertIsNone(mem.choose(codec.encode(8,histories[-1]),0))

    def test_saturation_matches_scalar(self):
        rng=random.Random(2);mask=255
        def planes(values):return [sum(((v>>j)&1)<<i for i,v in enumerate(values)) for j in range(16)]
        for _ in range(32):
            a,b=[[rng.randrange(65536) for _ in range(8)] for _ in range(2)]
            result=add(planes(a),planes(b),mask)
            self.assertEqual([counter_value(result,j) for j in range(8)],[min(x+y,65535) for x,y in zip(a,b)])
        a=[mask]*16;increment(a,mask);self.assertEqual(a,[mask]*16)

    def test_teacher_disagreement_withdraws_consensus_but_not_personal(self):
        net=BinaryVectorNetwork()
        key=net.context_codec.encode(3,[])
        def event(i,feedback):
            return dict(source=f'actor{i}',event_id=str(i),observation=3,history=[],action=2,
                        feedback=feedback,terminal_success=0)
        net.contextual_memory.update(key,2,True)
        yes=[event(0,1),event(1,1)]
        distill_into(net,yes)
        self.assertEqual(net.contextual_memory.choose(key,0),2)
        both=yes+[event(2,2),event(3,2)]
        distill_into(net,both)
        self.assertIsNone(net.contextual_memory.choose(key,0))
        self.assertEqual(counter_value(net.contextual_memory.rows[key][0],2),1)
        self.assertEqual(distill_into(net,both*3)['contexts_taught'],0)

    def test_recoding_keeps_decayed_evidence_and_provenance(self):
        net=BinaryVectorNetwork(memory_capacity=8)
        key=net.context_codec.encode(2,[(3,0),(4,1),(5,2)])
        for _ in range(10):net.contextual_memory.update(key,3,True)
        net.contextual_memory.set_teacher(key,ContextMemory.empty(),['source'],'sig')
        net.contextual_memory.clock+=L.MEMORY_DECAY_INTERVAL
        net.change_context_history(1)
        new=net.context_codec.encode(2,[(5,2)])
        self.assertEqual(counter_value(net.contextual_memory.rows[new][0],3),5)
        self.assertEqual(net.contextual_memory.meta[new]['sources'],['source'])
        net.change_context_history(6)
        self.assertEqual(net.context_codec.decode(next(iter(net.contextual_memory.rows))),(2,[(5,2)]))

    def test_training_replay_only_uses_prior_train_and_is_deterministic(self):
        a,archive,replays=training_generation(8,42,1,2,[],'r')
        self.assertEqual(replays,0)
        b,retained,replays=training_generation(8,42,2,2,archive,'r')
        self.assertEqual(replays,2)
        self.assertEqual((b,retained,replays),training_generation(8,42,2,2,json.loads(json.dumps(archive)),'r'))
        seen={e['id'] for s in a for e in s['episode_specs']}
        for scenario in b:
            for e in scenario['episode_specs']:
                self.assertEqual(e['id'] in seen,e['replay'])
        with self.assertRaises(ValueError):episode_specs(b[0],42,'holdout',2)
