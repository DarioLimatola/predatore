"""Boolean gates and the shared neuron structure used by the vector engine."""
import random
from dataclasses import dataclass
from .limits import CoreLimits as Limits


GATES = ('AND', 'OR', 'XOR', 'NAND', 'NOR', 'COPY', 'NOT', 'AND_NOT')


def gate(op, a, b):
    if op == 'AND':
        return a & b
    if op == 'OR':
        return a | b
    if op == 'XOR':
        return a ^ b
    if op == 'NAND':
        return 1-(a & b)
    if op == 'NOR':
        return 1-(a | b)
    if op == 'COPY':
        return a
    if op == 'NOT':
        return 1-a
    if op == 'AND_NOT':
        return a & (1-b)
    raise ValueError(f'Unknown Boolean gate: {op}')


@dataclass
class Neuron:
    a: int
    b: int
    op: str
    active: bool = True
    permanent: bool = False


class BooleanNet:
    """Base neuron graph; all executable decisions use the V6 vector subclass."""
    def __init__(self, seed=1, sensors=Limits.SENSOR_BITS, neurons=Limits.DEFAULT_BASE_NEURONS, action_bits=Limits.ACTION_BITS):
        self.sensors = sensors
        self.action_bits = action_bits
        self.actions = 1 << action_bits
        self.action_mask = (1 << self.actions) - 1
        rng = random.Random(seed)
        self.nodes = [Neuron(rng.randrange(sensors+i), rng.randrange(sensors+i),
                             rng.choice(GATES)) for i in range(neurons)]
        self.state = [0]*neurons
        self.outputs = [rng.randrange(neurons) for _ in range(action_bits)] if neurons else []
        self._compiled = None

    @classmethod
    def restore(cls, data):
        if data.get('engine') != Limits.ENGINE:
            raise ValueError('Unsupported brain format; start a new V6.05 profile')
        from .network import VectorBooleanNet
        return VectorBooleanNet.restore(data)
