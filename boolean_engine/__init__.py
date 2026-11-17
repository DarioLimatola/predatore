"""Reusable Boolean vector engine."""
from .network import VectorBooleanNet, pack, unpack
from .context import ContextCodec
from .hdc import BinarySpace
from .distillation import distill_into

BinaryVectorNetwork = VectorBooleanNet
__version__ = '6.05'
