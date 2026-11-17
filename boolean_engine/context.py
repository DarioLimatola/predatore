"""Exact observable addresses, independent of the world."""
from .limits import CoreLimits as L


class ContextCodec:
    def __init__(self, sensors=L.SENSOR_BITS, history=L.CONTEXT_HISTORY, action_bits=L.ACTION_BITS):
        if not (1 <= sensors <= L.MAX_SENSOR_BITS and 0 <= history <= L.MAX_CONTEXT_HISTORY and
                1 <= action_bits <= L.MAX_ACTION_BITS):
            raise ValueError('Invalid observation or action schema')
        self.sensors, self.history, self.action_bits = sensors, history, action_bits
        self.actions = 1 << action_bits
        self.action_mask = (1 << self.actions) - 1
        self.value_bits = sensors + history*(sensors+action_bits)
        self.bits = self.value_bits + history

    def encode(self, current, history):
        if not 0 <= current < (1 << self.sensors):
            raise ValueError('Observation outside context schema')
        past = list(history)[-self.history:] if self.history else []
        value, shift = current, self.sensors
        for obs, action in reversed(past):
            if not 0 <= obs < (1 << self.sensors) or not 0 <= action < self.actions:
                raise ValueError('Invalid observation/action history')
            value |= obs << shift
            shift += self.sensors
            value |= action << shift
            shift += self.action_bits
        return str(value | (((1 << len(past))-1) << self.value_bits))

    def validate(self, key):
        if not isinstance(key, str) or len(key) > self.bits or not key.isascii() or not key.isdecimal():
            return False
        value = int(key)
        valid = value >> self.value_bits
        used = self.sensors + valid.bit_count()*(self.sensors+self.action_bits)
        return (str(value)==key and value.bit_length() <= self.bits and not valid & (valid+1)
                and not ((value & ((1<<self.value_bits)-1)) >> used))

    def decode(self, key):
        if not self.validate(key):
            raise ValueError('Invalid context key')
        value = int(key)
        mask = (1 << self.sensors)-1
        current, count = value & mask, (value >> self.value_bits).bit_count()
        value >>= self.sensors
        history = []
        for _ in range(count):
            obs = value & mask
            value >>= self.sensors
            history.append((obs,value & (self.actions-1)))
            value >>= self.action_bits
        return current, history[::-1]
