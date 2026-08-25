"""Legacy compatibility namespace for MLBricks Bolt attention.

New code should use ``mlbricks.bolt`` or ``from mlbricks import Bolt``.
"""
from .attention import Attention, Bolt, BoltAttention, GaussAttention

# Historical aliases kept for compatibility.
gaussAttention = Bolt
gauss_attention = Bolt

__all__ = [
    "Attention",
    "Bolt",
    "BoltAttention",
    "GaussAttention",
    "gaussAttention",
    "gauss_attention",
]
