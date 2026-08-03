"""Ensure spaces are accessible if you import whynot.gym as gym."""
from gymnasium.spaces import Space
from gymnasium.spaces import Box
from gymnasium.spaces import Discrete
from gymnasium.spaces import MultiDiscrete
from gymnasium.spaces import MultiBinary
from gymnasium.spaces import Tuple
from gymnasium.spaces import Dict

from gymnasium.spaces.utils import flatdim
from gymnasium.spaces.utils import flatten
from gymnasium.spaces.utils import unflatten

__all__ = [
    "Space",
    "Box",
    "Discrete",
    "MultiDiscrete",
    "MultiBinary",
    "Tuple",
    "Dict",
    "flatdim",
    "flatten",
    "unflatten",
]
