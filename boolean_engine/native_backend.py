"""Bridge to the V7 native kernel (`predatore_native`, Rust).

The Python code in this package is the reference implementation. When the
compiled module is importable, brains keep their memories in native objects with
the same Python API, and whole episodes run in Rust. Results are bit-identical to
the Python reference (tests/test_v7_native_parity.py), so the Python path can be
used for any single episode (live frames, recordings, relation credit).

PREDATORE_NATIVE=0 disables the kernel; nothing else changes.
"""
import importlib
import os
import sys
from pathlib import Path

from .limits import CoreLimits as L

_module = None
_error = None


def _load():
    global _module, _error
    if os.environ.get('PREDATORE_NATIVE', '1') == '0':
        _error = 'disabled by PREDATORE_NATIVE=0'
        return None
    root = Path(__file__).resolve().parents[1]
    for extra in (root / 'native' / 'dist', root):
        if extra.is_dir() and str(extra) not in sys.path:
            sys.path.append(str(extra))
    try:
        module = importlib.import_module('predatore_native')
    except ImportError as exc:
        _error = str(exc)
        return None
    if getattr(module, 'KERNEL_VERSION', '') != '7.0.0':
        _error = 'kernel version mismatch'
        return None
    _module = module
    return module


_load()
AVAILABLE = _module is not None


class python_reference:
    """Context manager: brains built inside use the pure-Python memories (parity tests)."""
    def __enter__(self):
        global AVAILABLE
        self._saved, AVAILABLE = AVAILABLE, False
        return self

    def __exit__(self, *exc):
        global AVAILABLE
        AVAILABLE = self._saved
        return False


def status():
    return {'native': AVAILABLE, 'kernel': getattr(_module, 'KERNEL_VERSION', None),
            'reason': None if AVAILABLE else _error}


# ------------------------------------------------------------------ memories

def make_context_memory(capacity, actions):
    if AVAILABLE and 2 <= actions <= 64:
        return _module.ContextMemory(capacity, actions)
    from .memory import ContextMemory
    return ContextMemory(capacity, actions)


def make_predictions():
    if AVAILABLE:
        return _module.PredictionTable()
    from collections import OrderedDict
    return OrderedDict()


def make_evidence(width):
    return _module.EvidenceTable(width) if AVAILABLE else {}


def is_native(obj):
    return AVAILABLE and type(obj).__module__ == 'predatore_native'


class _EntriesView:
    __slots__ = ('_core',)

    def __init__(self, core):
        self._core = core

    def __len__(self):
        return len(self._core)

    def get(self, key, default=None):
        found = self._core.entry(key)
        return default if found is None else found

    def __contains__(self, key):
        return self._core.entry(key) is not None

    def __getitem__(self, key):
        found = self._core.entry(key)
        if found is None:
            raise KeyError(key)
        return found

    def __iter__(self):
        return iter(self._core.keys())

    def keys(self):
        return self._core.keys()

    def items(self):
        return [(k, self._core.entry(k)) for k in self._core.keys()]


class _BucketsView:
    __slots__ = ('_core',)

    def __init__(self, core):
        self._core = core

    def get(self, anchor, default=None):
        keys = self._core.bucket_keys(anchor)
        return dict.fromkeys(keys) if keys else ({} if default is None else default)

    def __contains__(self, anchor):
        return bool(self._core.bucket_keys(anchor))

    def __len__(self):
        return len(self._core.bucket_anchors())


def _symbols(codec, width):
    from .hdc import BinarySpace
    space = BinarySpace(width)
    s = space.symbol
    return {'feature_role': [s(f'role:feature:{b}') for b in range(codec.sensors)],
            'value': [s('symbol:0'), s('symbol:1')],
            'action_role': s('role:action'),
            'action': [s(f'symbol:{a}') for a in range(codec.actions)],
            'hist_role': s('role:history_length'),
            'hist': [s(f'symbol:{h}') for h in range(codec.history + 1)],
            'tie': s('tie')}


class NativeAssociativeMemory:
    """Same API as hdc.AssociativeMemory, storage and search in Rust."""

    def __init__(self, codec, capacity=L.HDC_CAPACITY, width=L.HDC_WIDTH, sensor_mask=None, _core=None):
        from .hdc import BinarySpace
        if not L.HDC_MIN_WIDTH <= width <= L.HDC_MAX_WIDTH or capacity < 1:
            raise ValueError('Invalid associative budget')
        self.codec, self.capacity, self.width = codec, capacity, width
        self.space = BinarySpace(width)
        self.sensor_mask = (1 << codec.sensors)-1 if sensor_mask is None else sensor_mask
        if not 0 <= self.sensor_mask < (1 << codec.sensors):
            raise ValueError('Invalid sensor mask')
        self._core = _core or _module.AssociativeCoreBox(codec.sensors, codec.history, codec.action_bits,
                                                          capacity, width, self.sensor_mask, _symbols(codec, width))
        self.entries = _EntriesView(self._core)
        self.buckets = _BucketsView(self._core)

    @property
    def last_lookup(self):
        return self._core.last_lookup()

    def encode(self, key):
        return self._core.encode(key)

    def observe(self, key, positive, negative):
        self._core.observe(key, positive, negative)

    def bucket_tail(self, anchor):
        return self._core.bucket_keys(anchor)[-L.HDC_CANDIDATES:]

    def choose(self, key, proposed, rejected=0, allowed=None):
        return self._core.choose(key, proposed, rejected, allowed)

    def export(self):
        return {'width': self.width, 'capacity': self.capacity, 'sensor_mask': str(self.sensor_mask),
                'entries': self._core.export_entries()}

    def restore(self, data):
        if data['width'] != self.width or data['capacity'] != self.capacity or int(data['sensor_mask']) != self.sensor_mask:
            raise ValueError('Associative schema mismatch')
        self._core.restore_entries(data['entries'])

    def digest(self):
        return self._core.digest()

    def __deepcopy__(self, memo):
        return self.copy()

    def copy(self):
        return NativeAssociativeMemory(self.codec, self.capacity, self.width, self.sensor_mask, self._core.copy())


def make_associative(codec, capacity, width, sensor_mask=None):
    if AVAILABLE and codec.sensors <= 64 and codec.action_bits <= 6 and codec.bits <= 256:
        return NativeAssociativeMemory(codec, capacity, width, sensor_mask)
    from .hdc import AssociativeMemory
    return AssociativeMemory(codec, capacity, width, sensor_mask)


# ------------------------------------------------------------------ episodes

def circuit_spec(net):
    return {'sensors': net.sensors, 'action_bits': net.action_bits, 'width': net.width,
            'shape': list(net.shape), 'context_history': net.context_codec.history,
            'nodes': [(n.a, n.b, bool(n.active)) for n in net.nodes], 'rules': net.rules,
            'ancestors': list(net._output_ancestors()), 'outputs': list(net.outputs),
            'output_lanes': list(net.output_lanes), 'internal_ticks': net.internal_ticks,
            'trace_depth': net.trace_depth, 'memory_capacity': net.memory_capacity,
            'sensor_encoding': net.sensor_encoding}


def flags(net):
    return (bool(net.memory_enabled), bool(net.tensor_enabled), bool(net.sparse_enabled),
            bool(net.associative_enabled), bool(net.contextual_enabled), bool(net.hdc_enabled),
            bool(net.plasticity_enabled), bool(net.circuit_plasticity_enabled), bool(net.collect_experience))


def supports(net):
    """True when every part of this brain has a native twin."""
    return (AVAILABLE and getattr(net, 'relation_credit', None) is None
            and net.sensors <= 64 and net.action_bits <= 6 and net.context_codec.bits <= 256
            and is_native(net.contextual_memory) and is_native(net.short_memory)
            and isinstance(net.associative_memory, NativeAssociativeMemory)
            and is_native(net.predictions) and is_native(net.circuit_evidence)
            and net.sensor_encoding in ('mixed', 'broadcast'))


def compiled(net):
    """Compiled circuit, rebuilt when the structure fingerprint changes."""
    key = _structure_key(net)
    cache = getattr(net, '_native_circuit', None)
    if cache is None or cache[0] != key:
        cache = (key, _module.NativeCircuit(circuit_spec(net)))
        net._native_circuit = cache
    return cache[1]


def _structure_key(net):
    # Cheap and complete: every field the kernel reads from the structure.
    return (net.width, tuple(net.shape), net.internal_ticks, net.trace_depth, net.memory_capacity,
            net.sensor_encoding, net.context_codec.history, tuple(net.outputs), tuple(net.output_lanes),
            tuple((n.a, n.b, n.active) for n in net.nodes),
            tuple((r['shift_a'], r['shift_b'], r['hold'], r['write'], tuple(r['lut']),
                   r.get('operator_a', 'rotate'), r.get('operator_b', 'rotate'), r.get('axis_a', 0),
                   r.get('axis_b', 0), r.get('delay_a', 0), r.get('delay_b', 0), tuple(r.get('shape', net.shape)))
                  for r in net.rules))


_geometries = {}


def geometry(record):
    key = (record['size'], tuple(tuple(p) for p in record['obstacles']))
    found = _geometries.get(key)
    if found is None:
        if len(_geometries) > 256:
            _geometries.clear()
        found = _geometries[key] = _module.NativeGeometry(record['size'], [tuple(p) for p in record['obstacles']])
    return found


def _brain_args(net):
    return (compiled(net), flags(net), net.contextual_memory, net.short_memory,
            net.associative_memory._core, net.predictions, net.circuit_evidence)


def _write_back(net, rt):
    from collections import deque
    net.vectors = rt['vectors']
    net.state = [v & 1 for v in net.vectors]
    net.context, net.context_valid = rt['context'], rt['context_valid']
    net.update_count = rt['update_count']
    net.captured = rt['captured']
    net.next_observation = rt['next_observation']
    net.last_prediction_error = rt['last_prediction_error']
    net.current_context_key = rt['current_context_key']
    net.decoder_counts = rt['decoder_counts']
    if 'last_decoder' in rt:
        net.last_decoder = rt['last_decoder']
    # The complete working state of the finished episode, as the Python path leaves it.
    net.trace = deque(rt['trace'], maxlen=net.trace_depth)
    net.last = rt['last']
    net.observable_history = deque(rt['observable_history'], maxlen=L.MAX_CONTEXT_HISTORY)
    net.context_trace = deque(rt['context_trace'], maxlen=net.trace_depth)
    net.last_payload = rt['last_payload']
    net.input_history = deque(rt['input_history'], maxlen=L.MAX_DELAY)
    net.circuit_trace = deque(rt['circuit_trace'], maxlen=net.trace_depth)
    if net.last is not None:
        # Modules evaluated by the last decision (read by measurements and SMITH's context).
        net.computed_nodes = (net._output_ancestors() if net.sparse_enabled else
                              tuple(i for i, n in enumerate(net.nodes) if n.active))


_prey_templates = {}


def _prey_brain(scenario):
    """Native parts of a fresh prey for this episode, copied from a cached template.

    Equal to building the prey in Python for every episode (make_world does that):
    the template is never run, only its memories are copied.
    """
    from .network import VectorBooleanNet
    genome = scenario.get('prey_genome')
    key = ('genome', id(genome)) if genome is not None else ('seed', scenario.get('prey_seed', 641))
    template = _prey_templates.get(key)
    if template is None or (genome is not None and template[0] is not genome):
        net = (VectorBooleanNet.restore(genome) if genome is not None else
               VectorBooleanNet(scenario.get('prey_seed', 641), neurons=32, shape=(8,)))
        if net.sensors != 15 or net.action_bits != 3:
            raise ValueError('prey genome must support 15 sensors and 8 actions')
        if not supports(net):
            raise RuntimeError('prey brain has no native twin')
        if len(_prey_templates) > 32:
            _prey_templates.clear()
        template = _prey_templates[key] = (genome, net, compiled(net), flags(net))
    _, net, circuit, prey_flags = template
    return (circuit, prey_flags, net.contextual_memory.copy(), net.short_memory.copy(),
            net.associative_memory._core.copy(), net.predictions.copy(), net.circuit_evidence.copy())


def rollout(net, scenario, seed, steps, measure=False):
    """Native twin of engine.evaluation.rollout (record/observer handled by the caller)."""
    import time
    name = scenario.get('name', scenario.get('scenario', 'static'))
    kind, delay, pounce, prey = 'plain', 0, True, None
    prey_moves = bool(scenario.get('prey_moves', False))
    if name == 'delayed_reward':
        kind, delay = 'delayed', scenario.get('reward_delay', 4)
        if type(delay) is not int or not 0 <= delay <= 64:
            raise ValueError('reward_delay must be an integer between 0 and 64')
    elif name == 'intelligent_prey':
        kind = 'neural'
        prey = _prey_brain(scenario)
    elif name == 'maze_novel':
        pounce = False
    emit = bool(net.collect_experience and net.experience_sink is not None)
    begin = time.perf_counter_ns() if measure else 0
    metrics, events, rt = _module.run_episode(
        _brain_args(net), geometry(scenario['map']), kind, delay, pounce, seed, scenario['visibility'],
        prey_moves, steps, emit, prey)
    elapsed = time.perf_counter_ns() - begin if measure else 0
    _write_back(net, rt)
    if emit:
        for event in events:
            net.experience_sink(event)
    metrics.pop('predator'); metrics.pop('prey')
    metrics['reward'] = float(metrics['reward'])
    if measure:
        metrics['decision_ns'] = elapsed
    metrics['decoder_counts'] = dict(net.decoder_counts)
    return metrics


def _episode_spec(scenario, seed, steps):
    name = scenario.get('name', scenario.get('scenario', 'static'))
    kind, delay, pounce, prey = 'plain', 0, True, None
    if name == 'delayed_reward':
        kind, delay = 'delayed', scenario.get('reward_delay', 4)
        if type(delay) is not int or not 0 <= delay <= 64:
            raise ValueError('reward_delay must be an integer between 0 and 64')
    elif name == 'intelligent_prey':
        kind, prey = 'neural', _prey_brain(scenario)
    elif name == 'maze_novel':
        pounce = False
    return (geometry(scenario['map']), kind, delay, pounce, seed, scenario['visibility'],
            bool(scenario.get('prey_moves', False)), steps, prey)


def run_episodes(net, episodes, steps):
    """[(scenario, seed)] on the same brain, in order, in one native call.

    Equal to rollout() for each episode in turn (same rows, same experience events
    delivered to net.experience_sink in the same order).
    """
    emit = bool(net.collect_experience and net.experience_sink is not None)
    specs = [_episode_spec(scenario, seed, steps) for scenario, seed in episodes]
    results, rt = _module.run_batch(_brain_args(net), specs, emit)
    _write_back(net, rt)
    rows = []
    for metrics, events, counts in results:
        if emit:
            for event in events:
                net.experience_sink(event)
        metrics['reward'] = float(metrics['reward'])
        metrics['decoder_counts'] = counts
        rows.append(metrics)
    return rows
