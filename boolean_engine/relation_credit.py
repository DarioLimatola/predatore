"""Experimental plastic readout from active circuit modules to motor bits."""
from collections import Counter, deque

from .core_adapter import StructuralCounter
from .limits import CoreLimits

SCHEMA = 'predatore.relation-credit/1'
MAX_SOURCES = 16
ROUTE_THRESHOLD = 2
FRAME_CHANGE_LIMIT = 8


class RelationCredit:
    """One signed counter per (module, observed lane state, output bit).

    Each active ancestor supplies one reproducible Boolean lane. The signed
    majority of sufficiently supported relations can replace an output bit.
    Neither the sensor encoding nor the LUTs change during this update.
    """
    def __init__(self, horizon):
        if type(horizon) is not int or not 1 <= horizon <= CoreLimits.MAX_TRACE:
            raise ValueError('Relation trace horizon outside resource bounds')
        self.horizon = horizon
        self.weights = {}
        self.trace = deque(maxlen=horizon)
        self.reset()

    def reset(self):
        self.trace.clear()
        self.changed = []
        self.updated_relations = 0

    def signals(self, vectors, width, ancestors):
        return tuple((node, (vectors[node] >> (node % width)) & 1)
                     for node in ancestors[:MAX_SOURCES])

    def route(self, signals, action_bits, proposed):
        routed = proposed
        for bit in range(action_bits):
            vote = 0
            for node, state in signals:
                counter = self.weights.get((node, state, bit))
                if counter and abs(counter.value) >= ROUTE_THRESHOLD:
                    vote += 1 if counter.value > 0 else -1
            if vote > 0:
                routed |= 1 << bit
            elif vote < 0:
                routed &= ~(1 << bit)
        return routed

    def observe(self, signals, action, owns_decision):
        # Empty entries age earlier decisions even after a memory override.
        self.trace.append((signals if owns_decision else (), action))

    def feedback(self, reward, action_bits, width):
        self.changed = []
        self.updated_relations = 0
        if not reward:
            return
        deltas = Counter()
        sign = 1 if reward > 0 else -1
        for signals, action in self.trace:
            for node, state in signals:
                for bit in range(action_bits):
                    key = (node, state, bit)
                    deltas[key] += sign if action & (1 << bit) else -sign
        for key, votes in deltas.items():
            if not votes:
                continue
            # One reward supplies at most one vote per relation, irrespective
            # of the number of repeated identical actions in its window.
            old = self.weights.get(key)
            value = (old or StructuralCounter()).updated(1 if votes > 0 else -1)
            if value.value:
                self.weights[key] = value
            else:
                self.weights.pop(key, None)
            self.updated_relations += 1
            if len(self.changed) < FRAME_CHANGE_LIMIT:
                node, state, bit = key
                self.changed.append({'node': node, 'lane': node % width, 'state': state,
                                     'action_bit': bit, 'before': old.value if old else 0,
                                     'after': value.value, 'width': value.width})

    def frame(self):
        return {'count': len(self.weights), 'horizon': self.horizon,
                'updated_relations': self.updated_relations,
                'sample_truncated': len(self.changed) < self.updated_relations,
                'changed': list(self.changed)}

    def export(self):
        return {'schema_version': SCHEMA, 'horizon': self.horizon,
                'weights': [[*key, counter.to_dict()]
                            for key, counter in sorted(self.weights.items())]}

    @classmethod
    def restore(cls, data, trace, ancestors, action_bits):
        if not isinstance(data, dict) or data.get('schema_version') != SCHEMA:
            raise ValueError('Unsupported relation-credit schema')
        result = cls(data.get('horizon'))
        allowed = set(ancestors[:MAX_SOURCES])
        weights = data.get('weights')
        if not isinstance(weights, list) or len(weights) > len(allowed) * 2 * action_bits:
            raise ValueError('Invalid number of relation counters')
        for row in weights:
            if not isinstance(row, (list, tuple)) or len(row) != 4:
                raise ValueError('Invalid relation counter')
            node, state, bit, raw = row
            key = (node, state, bit)
            if (any(type(v) is not int for v in key) or node not in allowed or
                    state not in (0, 1) or not 0 <= bit < action_bits or key in result.weights):
                raise ValueError('Invalid relation key')
            result.weights[key] = StructuralCounter.from_dict(raw)
        if not isinstance(trace, (list, tuple)) or len(trace) > result.horizon:
            raise ValueError('Invalid relation trace length')
        for item in trace:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError('Invalid relation trace item')
            signals, action = item
            if (type(action) is not int or not 0 <= action < 1 << action_bits or
                    not isinstance(signals, (list, tuple)) or len(signals) > len(allowed)):
                raise ValueError('Invalid action or signal count in relation trace')
            seen = set()
            for pair in signals:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError('Invalid relation trace signal')
                node, state = pair
                if (type(node) is not int or node not in allowed or node in seen or
                        type(state) is not int or state not in (0, 1)):
                    raise ValueError('Invalid signal in relation trace')
                seen.add(node)
            result.trace.append((tuple(tuple(pair) for pair in signals), action))
        return result
