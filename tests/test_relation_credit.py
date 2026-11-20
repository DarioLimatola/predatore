import copy
import json
import unittest

from boolean_engine import BinaryVectorNetwork
from boolean_engine.relation_credit import RelationCredit


class RelationCreditTests(unittest.TestCase):
    def test_delayed_credit_is_bounded_and_opposing_feedback_cancels(self):
        earlier = ((1, 1),)
        later = ((2, 0),)
        for horizon in (1, 4):
            learner = RelationCredit(horizon)
            for repeat in range(2):
                learner.observe(earlier, 1, True)
                learner.feedback(0, 1, 8)
                learner.observe(later, 0, False)
                learner.feedback(1, 1, 8)
                learner.reset()
            self.assertEqual(learner.route(earlier, 1, 0), int(horizon == 4))
            self.assertFalse(any(key[0] == 2 for key in learner.weights))
            if horizon == 4:
                learner.observe(earlier, 1, True)
                learner.feedback(-1, 1, 8)
                self.assertEqual(learner.route(earlier, 1, 0), 0)

    def test_delayed_world_credits_contact_signal_only_with_long_trace(self):
        from engine.challenge_worlds import DelayedRewardWorld
        for horizon in (1, 8):
            net = BinaryVectorNetwork(42, neurons=4, shape=(8,))
            net.sensor_encoding = 'broadcast'
            net.associative_enabled = False
            net.outputs = [1, 2, 3]
            net.output_lanes = [0, 0, 0]
            # Node zero reads the below-prey sensor. The fixture sends DOWN
            # throughout so both variants see the same rare successful episode.
            for i, (node, rule) in enumerate(zip(net.nodes, net.rules)):
                node.a = 2 if i == 0 else net.sensors
                node.b = 0
                rule.update(hold=-1, shift_a=0, shift_b=0)
                rule['lut'] = ([0, 0, net.mask, net.mask] if i == 0 else
                               [net.mask if i == 1 else 0] * 4)
            net.invalidate()
            net.enable_relation_credit(horizon)
            for _ in range(2):
                world = DelayedRewardWorld({'size': 6, 'obstacles': []}, 1, 5, 4)
                world.predator, world.prey = (0, 0), (0, 1)
                net.reset()
                rewards = []
                for tick in range(5):
                    action = net.decide(world.observe())
                    self.assertEqual(action, 1)
                    reward, captured = world.step(action)
                    net.transition(world.observe(), captured)
                    net.feedback(reward)
                    rewards.append(reward)
                self.assertEqual(rewards, [0, 0, 0, 0, 20])
            counter = net.relation_credit.weights.get((0, 1, 0))
            self.assertEqual(counter.value if counter else 0, 2 if horizon == 8 else 0)

    def test_single_vote_per_reward_and_independent_motor_bits(self):
        model = RelationCredit(4)
        for _ in range(3):
            model.observe(((2, 1),), 2, True)
        model.feedback(20, 2, 8)
        self.assertEqual(model.weights[(2, 1, 0)].value, -1)
        self.assertEqual(model.weights[(2, 1, 1)].value, 1)
        model.observe(((2, 1),), 2, True)
        model.feedback(20, 2, 8)
        self.assertEqual(model.route(((2, 1),), 2, 1), 2)

    def test_no_cross_episode_credit(self):
        model = RelationCredit(4)
        model.observe(((1, 1),), 0, True)
        model.reset()
        model.feedback(-1, 1, 8)
        self.assertEqual(model.weights, {})

    def test_preserves_baseline_schema_and_roundtrips_mid_episode(self):
        net = BinaryVectorNetwork(42)
        ref = net.export()
        self.assertNotIn('relation_credit', ref)
        self.assertEqual(net.snapshot_id(), net.restore(ref).snapshot_id())
        net.enable_relation_credit(4)
        self.assertNotEqual(net.engine_id, ref['engine'])
        bad = net.export()
        bad['engine'] = ref['engine']
        with self.assertRaises(ValueError):
            net.restore(bad)
        obs = [0] * net.sensors
        net.decide(obs)
        net.feedback(-1)
        net.decide(obs)
        clone = net.restore(json.loads(json.dumps(net.export())))
        clone.feedback(2)
        net.feedback(2)
        self.assertEqual(net.relation_credit.export(), clone.relation_credit.export())
        self.assertEqual(net.decide(obs), clone.decide(obs))
        before = copy.deepcopy(clone.relation_credit.export())
        clone.plasticity_enabled = False
        for _ in range(8):
            clone.decide(obs)
            clone.feedback(1)
        self.assertEqual(clone.relation_credit.export(), before)
        self.assertEqual(net.export()['rules'], ref['rules'])
        self.assertIn('relation_credit', net.frame_data())

    def test_bad_weights_and_traces_are_rejected(self):
        net = BinaryVectorNetwork(1)
        net.enable_relation_credit()
        net.decide([0] * net.sensors)
        net.feedback(1)
        for corruption in ('index', 'duplicate', 'trace', 'horizon', 'width'):
            data = json.loads(json.dumps(net.export()))
            payload = data['relation_credit']
            if corruption == 'index':
                payload['weights'][0][0] = len(net.nodes) + 1
            elif corruption == 'duplicate':
                payload['weights'].append(payload['weights'][0])
            elif corruption == 'trace':
                data['vector_runtime']['relation_trace'][0][0][0][1] = 2
            elif corruption == 'horizon':
                payload['horizon'] = -1
            else:
                payload['weights'][0][-1]['width'] = 3
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                net.restore(data)

    def test_removal_does_not_reassign_evidence_to_different_module(self):
        net = BinaryVectorNetwork(111)
        net.enable_relation_credit()
        net.decide([0] * net.sensors)
        net.feedback(-1)
        self.assertTrue(net.relation_credit.weights)
        net.remove_neuron(0)
        self.assertFalse(net.relation_credit.weights)
        self.assertFalse(net.relation_credit.trace)

if __name__ == '__main__':
    unittest.main()
