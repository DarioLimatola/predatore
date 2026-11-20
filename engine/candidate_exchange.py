"""Detached exchange of circuit candidates and named training experiences.

The sender declares provenance. Local validation is repeated on disposable
controllers and never installs anything into EVA, SMITH or the source Vault.
"""
import hashlib
import json
import time
from copy import deepcopy

from boolean_engine import VectorBooleanNet
from boolean_engine.core_adapter import PACKING, encode_tensor, decode_tensor
from boolean_engine.distillation import distill_into
from boolean_engine.limits import CoreLimits as L
from .challenges import build_scenario, seeds_for
from .continual import evaluation_copy
from .evaluation import evaluate_suite, rollout
from .paired_evidence import paired_capture_evidence
from .world import ACTIONS

SCHEMA = 'predatore.candidate/1'
SENSORS = ('visible', 'prey_north', 'prey_south', 'prey_west', 'prey_east',
           'free_n1', 'free_s1', 'free_w1', 'free_e1',
           'free_n2', 'free_s2', 'free_w2', 'free_e2', 'visible_within_two', 'ever_collided')
BASE_SCENARIOS = ('static', 'moving', 'maze', 'hidden')
MAX_EVENTS = 16384


def digest(data):
    raw = json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _names(labels):
    if not isinstance(labels, list) or not labels or len(labels) != len(set(labels)):
        raise ValueError('Invalid or duplicate contract labels')
    return labels


def _contract(net):
    if net.sensors != len(SENSORS) or net.actions != len(ACTIONS):
        raise ValueError('Declare an explicit arena I/O mapping for nonstandard controllers')
    return {'sensors': list(SENSORS), 'actions': list(ACTIONS),
            'feedback': 'predatore.positive-negative-terminal/1',
            'engine': net.engine_id, 'codec': 'predatore.context/1',
            'hash': 'SHAKE-256', 'seed': net.associative_memory.space.seed,
            'hdc_width': net.associative_memory.width, 'sensor_mask': net.associative_memory.sensor_mask,
            'history': net.context_codec.history, 'sensor_encoding': net.sensor_encoding,
            'shape': list(net.shape), 'axes': _axis_names(net.shape), 'bits': net.width,
            'packing': PACKING, 'endian': 'little',
            'limits': {'neurons': L.MAX_NEURONS, 'state_bits': L.MAX_STATE_BITS,
                       'axes': L.MAX_AXES, 'delay': L.MAX_DELAY, 'ticks': L.MAX_TICKS}}


def _axis_names(shape):
    return [f'local_axis_{i}' for i in range(len(shape))]


def _package(net, kind, payload, source, license_name, origin):
    if not isinstance(source, str) or not source.strip() or not isinstance(license_name, str) or not license_name.strip():
        raise ValueError('Source and license/permissions declaration required')
    contract = _contract(net)
    record = {'schema_version': SCHEMA, 'kind': kind, 'status': 'candidate',
              'model_version': net.engine_id, 'source_model': net.snapshot_id(),
              'source': source, 'license': license_name, 'split': 'train',
              'contract': contract, 'contract_sha256': digest(contract), 'origin': origin,
              'payload': payload}
    return {**record, 'candidate_id': digest(record)}


def circuit_candidate(net, source, license_name, *, run, generation, training_seed):
    """Export circuit parameters only. No personal memory, runtime refs or private seed state."""
    if net.engine_id != L.ENGINE:
        raise ValueError('This circuit codec cannot preserve structural routing; exchange named experiences')
    data = net.export()
    rules = []
    for rule in data['rules']:
        shape = rule['shape']
        tensors = [encode_tensor(value, net.mask, shape, _axis_names(shape))
                   for value in (*rule['lut'], rule['write'])]
        rules.append({**{k: v for k, v in rule.items() if k not in ('write', 'lut')}, 'tensors': tensors})
    payload = {k: data[k] for k in ('nodes', 'outputs', 'output_lanes', 'internal_ticks', 'trace_depth')}
    payload['rules'] = rules
    return _package(net, 'boolean-vector.circuit/1', payload, source, license_name,
                    {'run': run, 'generation': generation, 'training_seed': training_seed})


def _observation(value):
    return [label for i, label in enumerate(SENSORS) if value & (1 << i)]


def _named_context(observation, history):
    return {'present': _observation(observation),
            'history': [[_observation(obs), ACTIONS[action]] for obs, action in history]}


def _experience(event, tick):
    return {'tick': tick, **_named_context(event['observation'], event['history']),
            'action': ACTIONS[event['action']], 'feedback': event['feedback'],
            'terminal_success': bool(event['terminal_success']),
            'terminal_trace': [{**_named_context(context['observation'], context['history']), 'action': ACTIONS[action]}
                               for context, action in event.get('eligible_contexts', [])]}


def experience_candidate(net, source, license_name, *, run, generation, scenario, map_seed, episode_seed, size, steps):
    """Collect one attributed training rollout; never pass a replay from validation as a teacher."""
    if scenario not in BASE_SCENARIOS or not 6 <= size <= 60 or not 1 <= steps <= MAX_EVENTS:
        raise ValueError('Invalid training arena or budget')
    task = build_scenario(scenario, size, map_seed)
    teacher = VectorBooleanNet.restore(net.export())
    teacher.collect_experience = True
    events = []
    teacher.experience_sink = lambda e: events.append(_experience(e, len(events) + 1))
    rollout(teacher, task, episode_seed, steps)
    origin = {'run': run, 'generation': generation, 'training_seed': map_seed,
              'episode_seed': episode_seed, 'scenario': scenario, 'size': size,
              'map_seed': task['map']['seed'], 'map_sha256': _geometry_id(task['map']),
              'visibility': task['visibility'], 'steps_budget': steps}
    return _package(net, 'arena.experience/1', events, source, license_name, origin)


def _geometry_id(record):
    return digest([record['size'], sorted(record['obstacles'])])


def _observation_bits(present):
    if not isinstance(present, list) or any(label not in SENSORS for label in present) or len(set(present)) != len(present):
        raise ValueError('Unrecognized or duplicated sensor')
    return sum(1 << SENSORS.index(label) for label in present)


def _mapped_context(event):
    history = event['history']
    if not isinstance(history, list) or len(history) > L.MAX_CONTEXT_HISTORY:
        raise ValueError('Invalid history budget')
    return {'observation': _observation_bits(event['present']),
            'history': [(_observation_bits(obs), ACTIONS.index(action)) for obs, action in history]}


def _import_experience(net, record):
    events = record['payload']
    if not isinstance(events, list) or not 0 < len(events) <= MAX_EVENTS:
        raise ValueError('Invalid training event budget')
    origin = record['origin']
    if (type(origin.get('episode_seed')) is not int or type(origin.get('map_seed')) is not int
            or not isinstance(origin.get('map_sha256'), str) or not origin.get('scenario')):
        raise ValueError('Missing training episode provenance')
    episode_id = digest([record['source'], record['source_model'], origin['episode_seed'],
                         origin['map_sha256'], origin['scenario']])
    decoded = []
    for tick, event in enumerate(events, 1):
        if (event.get('tick') != tick or type(event.get('feedback')) is not int or event['feedback'] not in (0, 1, 2)
                or type(event.get('terminal_success')) is not bool or event.get('action') not in ACTIONS):
            raise ValueError('Invalid, repeated or unordered transition')
        trace = event.get('terminal_trace', [])
        if (not isinstance(trace, list) or len(trace) > L.MAX_TRACE or (trace and not event['terminal_success'])
                or (event['terminal_success'] and tick != len(events))):
            raise ValueError('Invalid terminal trace')
        decoded.append({'source': record['source'], 'event_id': f'{episode_id}/{tick}', **_mapped_context(event),
                        'action': ACTIONS.index(event['action']), 'feedback': event['feedback'],
                        'terminal_success': event['terminal_success'],
                        'eligible_contexts': [(_mapped_context(item), ACTIONS.index(item['action'])) for item in trace]})
    candidate = VectorBooleanNet.restore(net.export())
    distill_into(candidate, decoded)
    return candidate


def _import_circuit(net, payload):
    if net.engine_id != L.ENGINE:
        raise ValueError('Unsupported receiver routing contract')
    source = deepcopy(payload)
    if len(source['nodes']) > L.MAX_NEURONS or len(source['rules']) != len(source['nodes']):
        raise ValueError('Unexpected circuit size')
    for rule in source['rules']:
        tensors = rule.pop('tensors')
        if len(tensors) != 5:
            raise ValueError('Expected four LUT planes and a write mask')
        lanes = []
        for tensor in tensors:
            value, known, shape, axes = decode_tensor(tensor)
            if known != net.mask or list(shape) != rule['shape'] or list(axes) != _axis_names(shape):
                raise ValueError('Incomplete or incompatible circuit tensor')
            lanes.append(value)
        rule['lut'], rule['write'] = lanes[:4], lanes[4]
    # All learned memories (including link-local evidence) start empty: indices
    # belonging to the receiver must never alias a different sender topology.
    fresh = VectorBooleanNet(neurons=0, sensors=net.sensors, action_bits=net.action_bits, shape=net.shape,
                             memory_capacity=net.memory_capacity, context_history=net.context_codec.history,
                             hdc_width=net.associative_memory.width, hdc_capacity=net.associative_memory.capacity)
    data = fresh.export()
    for key in ('nodes', 'rules', 'outputs', 'output_lanes', 'internal_ticks', 'trace_depth'):
        data[key] = source[key]
    data['sensor_encoding'] = net.sensor_encoding
    data['associative_memory']['sensor_mask'] = str(net.associative_memory.sensor_mask)
    data['vector_runtime'] = {}
    data['state'] = [0] * len(source['nodes'])
    return VectorBooleanNet.restore(data)


def import_candidate(net, record):
    """Decode to a disposable model. Named experiences can cross binary codecs."""
    if (not isinstance(record, dict) or record.get('schema_version') != SCHEMA or
            record.get('split') != 'train' or record.get('status') != 'candidate'):
        raise ValueError('Only unvalidated training candidates are importable')
    if record.get('candidate_id') != digest({k: v for k, v in record.items() if k != 'candidate_id'}):
        raise ValueError('Candidate integrity mismatch')
    source = record.get('contract')
    if not isinstance(source, dict) or digest(source) != record.get('contract_sha256'):
        raise ValueError('Contract fingerprint mismatch')
    origin = record.get('origin')
    if (not isinstance(origin, dict) or type(origin.get('generation')) is not int or origin['generation'] < 0
            or type(origin.get('training_seed')) is not int
            or any(not isinstance(s, str) or not s.strip() for s in (origin.get('run'), record.get('source'),
                    record.get('source_model'), record.get('model_version'), record.get('license')))):
        raise ValueError('Incomplete provenance')
    target = _contract(net)
    if record['kind'] == 'boolean-vector.circuit/1':
        if source != target or record.get('model_version') != target['engine']:
            raise ValueError('Raw circuit contract mismatch; exchange named experiences instead')
        return _import_circuit(net, record['payload'])
    if record['kind'] == 'arena.experience/1':
        if (set(_names(source.get('sensors'))) != set(SENSORS) or
                set(_names(source.get('actions'))) != set(ACTIONS) or source.get('feedback') != target['feedback']):
            raise ValueError('No semantic mapping for this sensor, action or feedback vocabulary')
        return _import_experience(net, record)
    raise ValueError('Unimplemented candidate type')


def evaluate_candidate(net, record, *, seed, episodes, steps, size=8):
    """Locally score a candidate, then return its *pre-validation* genome if accepted."""
    if not (1 <= episodes <= 128 and 1 <= steps <= 10000 and 6 <= size <= 60):
        raise ValueError('Invalid evaluation budget')
    origin = record.get('origin', {})
    if seed == origin.get('training_seed'):
        raise ValueError('Evaluation must use a fresh seed')
    started = time.perf_counter()
    candidate = import_candidate(net, record)
    scenes = [build_scenario(name, size, seed + 2000 + i * 197)
              for i, name in enumerate(BASE_SCENARIOS)]
    schedule = {s['name']: {'seeds': seeds_for(seed, 'validation', s['name'], episodes),
                            'map_sha256': _geometry_id(s['map']), 'map_seed': s['map']['seed']} for s in scenes}
    if any(origin.get('episode_seed') in row['seeds'] or origin.get('map_sha256') == row['map_sha256']
           for row in schedule.values()):
        raise ValueError('Candidate origin overlaps evaluation conditions')
    def measure(controller):
        frozen = evaluation_copy(controller, learning=False)
        return evaluate_suite(frozen, scenes, seed, 'validation', episodes, steps)
    baseline, measured = measure(net), measure(candidate)
    evidence = {name: paired_capture_evidence(baseline['by_scenario'][name], row)
                for name, row in measured['by_scenario'].items()}
    gains = sum(v['gained_captures'] for v in evidence.values())
    losses = sum(v['lost_captures'] for v in evidence.values())
    from math import comb
    discordant = gains + losses
    tail = sum(comb(discordant, i) for i in range(gains, discordant + 1))
    denominator = 1 << discordant
    accepted = gains > losses and tail * 20 <= denominator and all(
        row['gained_captures'] >= row['lost_captures'] for row in evidence.values())
    report = {'schema_version': 'predatore.candidate-evaluation/1', 'status': 'accepted' if accepted else 'rejected',
              'candidate_id': record['candidate_id'], 'rollback_id': net.snapshot_id(), 'model_id': candidate.snapshot_id(),
              'seed': seed, 'episodes_per_task': episodes, 'steps': steps, 'size': size, 'split': 'validation',
              'schedule': schedule, 'baseline': baseline, 'candidate': measured, 'paired_by_scenario': evidence,
              'gate': 'No net capture loss per scenario; positive net captures; pooled exact paired one-sided p <= 0.05.',
              'sign_tail': [tail, denominator], 'wall_seconds': time.perf_counter() - started}
    return report, candidate if accepted else None
