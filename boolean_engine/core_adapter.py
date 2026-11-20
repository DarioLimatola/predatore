"""Versioned Predatore tensor wire format for Monade's public little-endian Words.

Logical tensor sizes may be smaller than one Word. Padding never becomes a
lane, a runtime reference or a Monade Size. No seed-generated space is assumed.
"""
from dataclasses import dataclass
from math import prod

from .tensor import validate_shape

SCHEMA = 'predatore.tensor/1'
PACKING = 'row-major/lsb0/le-u64'
CORE_REVISION = 'd6e1b8a99b408a288529d059c81551a9135a5ffc'


@dataclass(frozen=True)
class StructuralCounter:
    """Signed evidence on one relation. Width grows locally and never wraps."""
    value: int = 0
    width: int = 16

    def __post_init__(self):
        if type(self.width) is not int or self.width not in (16, 32, 64):
            raise ValueError('Unsupported signed counter width')
        half = 1 << (self.width - 1)
        if type(self.value) is not int or not -half <= self.value < half:
            raise ValueError('Counter value outside its signed width')

    def updated(self, delta):
        if type(delta) is not int:
            raise ValueError('Integer evidence required')
        result = self.value + delta
        width = self.width
        while width < 64 and not -(1 << (width - 1)) <= result < 1 << (width - 1):
            width *= 2
        half = 1 << (width - 1)
        return StructuralCounter(max(-half, min(result, half - 1)), width)

    def to_dict(self):
        return {'value': self.value, 'width': self.width}

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) != {'value', 'width'}:
            raise ValueError('Invalid structural counter')
        return cls(data['value'], data['width'])


def _axes(shape, axes):
    if not isinstance(axes, (list, tuple)) or len(axes) != len(shape):
        raise ValueError('One named axis per dimension is required')
    if any(not isinstance(name, str) or not 0 < len(name) <= 80 for name in axes):
        raise ValueError('Invalid axis name')
    if len(set(axes)) != len(axes):
        raise ValueError('Duplicate axis names')
    return list(axes)


def encode_tensor(value, known, shape, axes):
    shape = validate_shape(shape)
    width = prod(shape)
    if any(type(x) is not int or x < 0 or x.bit_length() > width for x in (value, known)):
        raise ValueError('Tensor value or known mask exceeds the logical width')
    nbytes = ((width + 63) // 64) * 8
    return {'schema_version': SCHEMA, 'packing': PACKING, 'shape': list(shape),
            'axes': _axes(shape, axes), 'bits': width,
            'value': value.to_bytes(nbytes, 'little').hex(),
            'known': known.to_bytes(nbytes, 'little').hex()}


def decode_tensor(data):
    if not isinstance(data, dict) or data.get('schema_version') != SCHEMA or data.get('packing') != PACKING:
        raise ValueError('Incompatible tensor schema or packing')
    shape = validate_shape(data.get('shape', ()))
    axes = _axes(shape, data.get('axes'))
    width = prod(shape)
    nchars = ((width + 63) // 64) * 16
    if type(data.get('bits')) is not int or data['bits'] != width:
        raise ValueError('Tensor descriptor has inconsistent bit length')
    decoded = []
    for key in ('value', 'known'):
        raw = data.get(key)
        if not isinstance(raw, str) or len(raw) != nchars or any(c not in '0123456789abcdef' for c in raw):
            raise ValueError('Invalid little-endian word sequence')
        value = int.from_bytes(bytes.fromhex(raw), 'little')
        if value.bit_length() > width:
            raise ValueError('Nonzero padded bits')
        decoded.append(value)
    return decoded[0], decoded[1], shape, tuple(axes)
