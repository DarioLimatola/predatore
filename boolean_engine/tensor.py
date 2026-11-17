"""Axis-dependent operations on packed Boolean tensors, in row-major order."""
from functools import lru_cache
from math import prod
from .limits import CoreLimits as L


def validate_shape(shape):
    shape = tuple(shape)
    if (not shape or len(shape) > L.MAX_AXES or
            any(type(x) is not int or x < 1 for x in shape) or
            not L.MIN_WIDTH <= prod(shape) <= L.MAX_WIDTH):
        raise ValueError('Invalid Boolean tensor shape')
    return shape


def rotate(value, shift, width):
    shift %= width
    mask = (1 << width) - 1
    return value & mask if not shift else ((value << shift) | (value >> (width-shift))) & mask


@lru_cache(maxsize=L.PRIMITIVE_CACHE_SIZE)
def _roll_plan(shape, axis, offset):
    shape = validate_shape(shape)
    if not 0 <= axis < len(shape):
        raise ValueError('Axis outside tensor')
    length = shape[axis]
    offset %= length
    stride = prod(shape[axis+1:])
    slab = length * stride
    low = 0
    for start in range(0, prod(shape), slab):
        low |= ((1 << ((length-offset)*stride))-1) << start
    return low, ((1 << prod(shape))-1) ^ low, offset*stride, (length-offset)*stride


def axis_roll(value, shape, axis, offset=1):
    low, high, forward, backward = _roll_plan(tuple(shape), axis, offset)
    return ((value & low) << forward) | ((value & high) >> backward)


@lru_cache(maxsize=L.PRIMITIVE_CACHE_SIZE)
def _transpose_plan(shape, order):
    shape = validate_shape(shape)
    if sorted(order) != list(range(len(shape))):
        raise ValueError('Axis order must be a permutation')
    target = tuple(shape[i] for i in order)
    groups = {}
    for old in range(prod(shape)):
        remainder = old
        coordinates = []
        for i in range(len(shape)):
            stride = prod(shape[i+1:])
            coordinate, remainder = divmod(remainder, stride)
            coordinates.append(coordinate)
        new = sum(coordinates[axis]*prod(target[i+1:]) for i,axis in enumerate(order))
        groups[new-old] = groups.get(new-old,0) | (1 << old)
    return tuple(sorted(groups.items()))


def transpose(value, shape, order):
    result = 0
    for shift, mask in _transpose_plan(tuple(shape), tuple(order)):
        result |= (value & mask) << shift if shift >= 0 else (value & mask) >> -shift
    return result


OPERATORS = ('rotate', 'axis_roll', 'neighbor_or', 'neighbor_xor', 'transpose')


def transform(value, shape, operator, axis, shift):
    axis %= len(shape)
    if operator == 'rotate':
        return rotate(value, shift, prod(shape))
    if operator == 'axis_roll':
        return axis_roll(value, shape, axis, shift)
    if operator in ('neighbor_or', 'neighbor_xor'):
        left = axis_roll(value, shape, axis, -1)
        right = axis_roll(value, shape, axis, 1)
        return value | left | right if operator == 'neighbor_or' else value ^ left ^ right
    if operator == 'transpose':
        order = list(range(len(shape)))
        other = (axis+1) % len(shape)
        order[axis],order[other] = order[other],order[axis]
        return transpose(value, shape, order)
    raise ValueError('Unknown tensor operator')


def mutate_shape(shape, rng):
    """Split, merge or exchange axes without changing the number of bits."""
    shape = list(validate_shape(shape))
    options = ['swap'] if len(shape)>1 else []
    if len(shape)>1: options.append('merge')
    divisors = [(i,d) for i,n in enumerate(shape) for d in range(2,int(n**.5)+1) if n%d==0]
    if divisors and len(shape)<L.MAX_AXES: options.append('split')
    if not options: return tuple(shape)
    kind = rng.choice(options)
    if kind == 'split':
        i,d = rng.choice(divisors)
        shape[i:i+1] = [d,shape[i]//d]
    elif kind == 'merge':
        i = rng.randrange(len(shape)-1)
        shape[i:i+2] = [shape[i]*shape[i+1]]
    else:
        a,b = rng.sample(range(len(shape)),2)
        shape[a],shape[b] = shape[b],shape[a]
    return tuple(shape)
