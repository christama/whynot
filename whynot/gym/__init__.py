"""Gym module following the Gymnasium API for RL environments.

Gymnasium is the maintained successor to OpenAI Gym, which was abandoned in
2022 and does not support NumPy 2. Environments are exposed through this
module rather than imported from Gymnasium directly, so ``import whynot.gym as
gym`` keeps working for downstream code.
"""

from gymnasium import error
from gymnasium import logger
from gymnasium.core import Env

from whynot.gym.envs import make, spec, register

__all__ = ["Env", "make", "spec", "register", "error", "logger"]
