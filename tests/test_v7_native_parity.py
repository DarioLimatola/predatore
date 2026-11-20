"""V7 gate: the native kernel reproduces the Python reference bit for bit.

Same brain twice (Python memories / native memories), same episodes on all seven
scenarios, several brain configurations, memory accumulating across episodes.
Compared: per-episode metrics, emitted experience events, and the full exported
brain (without the transient vector_runtime) after every episode.
"""
import copy
import random
import tempfile
import unittest

from boolean_engine import native_backend as nb
from boolean_engine.network import VectorBooleanNet
from engine.challenges import prepare_suites
from engine.evaluation import rollout


def _strip(data):
    data = copy.deepcopy(data)
    data.pop('vector_runtime', None)
    return data


def _configure(net, name):
    if name == 'wide':
        net.extend_width((16, 16))
    elif name == 'tensor':
        rng = random.Random(5)
        for i, rule in enumerate(net.rules):
            rule['operator_a'] = ('axis_roll', 'neighbor_or', 'neighbor_xor', 'transpose', 'rotate')[i % 5]
            rule['operator_b'] = ('transpose', 'neighbor_xor', 'rotate', 'axis_roll', 'neighbor_or')[i % 5]
            rule['axis_a'], rule['axis_b'] = rng.randrange(2), rng.randrange(2)
            rule['delay_a'], rule['delay_b'] = rng.randrange(4), rng.randrange(4)
        net.invalidate()
    elif name == 'ticks':
        net.internal_ticks = 3
        net.sensor_encoding = 'broadcast'
    elif name == 'nomemory':
        net.memory_enabled = False
        net.sparse_enabled = False
    elif name == 'history6':
        net.change_context_history(6)
    elif name == 'narrow':
        net.reshape((2, 4)) if net.width == 8 else None


def _pair(seed, name):
    with nb.python_reference():
        ref = VectorBooleanNet(seed, neurons=48 if name != 'narrow' else 24,
                               shape=(8,) if name == 'narrow' else (8, 8))
        _configure(ref, name)
    data = ref.export()
    native = VectorBooleanNet.restore(data)
    with nb.python_reference():
        ref = VectorBooleanNet.restore(data)
    return ref, native


@unittest.skipUnless(nb.AVAILABLE, 'native kernel not built')
class NativeParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.mkdtemp()
        cls.suites = prepare_suites(cls.folder, 8, 42, 'extended')['train']

    def run_pair(self, name, seed, episodes=4, steps=60, learn=True):
        ref, native = _pair(seed, name)
        self.assertFalse(nb.supports(ref))
        self.assertTrue(nb.supports(native), name)
        for net in (ref, native):
            net.collect_experience = learn
            net.events = []
            net.experience_sink = net.events.append
        for scenario in self.suites:
            for episode in range(episodes):
                s = seed * 1000 + episode
                with self.subTest(config=name, scenario=scenario['name'], episode=episode):
                    a, _ = rollout(ref, scenario, s, steps)
                    b, _ = rollout(native, scenario, s, steps)
                    self.assertEqual(a, b)
                    self.assertEqual(ref.events, native.events)
                    self.assertEqual(ref.export(), native.export())  # including working state
                    self.assertEqual(tuple(ref.computed_nodes), tuple(native.computed_nodes))
                    self.assertEqual(ref.update_count, native.update_count)
                    self.assertEqual(ref.decoder_counts, native.decoder_counts)
                    self.assertEqual(ref.snapshot_id(), native.snapshot_id())

    def test_default(self):
        for seed in (1, 42, 809):
            self.run_pair('default', seed)

    def test_configurations(self):
        for name in ('wide', 'tensor', 'ticks', 'nomemory', 'history6', 'narrow'):
            self.run_pair(name, 7, episodes=3)

    def test_without_experience(self):
        self.run_pair('default', 3, learn=False)

    def test_distillation_matches(self):
        from boolean_engine.distillation import distill_into
        ref, native = _pair(13, 'default')
        teacher_ref, teacher_native = _pair(14, 'default')
        for t in (teacher_ref, teacher_native):
            t.collect_experience = True
            t.events = []
            t.experience_sink = t.events.append
        for scenario in self.suites:
            for e in range(3):
                rollout(teacher_ref, scenario, e, 60)
                rollout(teacher_native, scenario, e, 60)
        events = [dict(ev, source='t', event_id=f't/{i}') for i, ev in enumerate(teacher_native.events)]
        self.assertEqual(teacher_ref.events, teacher_native.events)
        a = distill_into(ref, events)
        b = distill_into(native, events)
        self.assertEqual(a, b)
        self.assertEqual(_strip(ref.export()), _strip(native.export()))

    def test_clone_is_independent(self):
        _, native = _pair(11, 'default')
        rollout(native, self.suites[0], 5, 40)
        twin = native.clone()
        self.assertEqual(_strip(twin.export()), _strip(native.export()))
        rollout(twin, self.suites[1], 6, 40)
        self.assertNotEqual(_strip(twin.export()), _strip(native.export()))


if __name__ == '__main__':
    unittest.main()
