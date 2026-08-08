"""whynot package initialization."""

import importlib

__version__ = "0.13.0"
from whynot.algorithms import *
from whynot import causal_graphs, dynamics, framework, simulators, traceable_numpy
from whynot.simulators import SIMULATORS
from whynot.dynamics import (
    DynamicsExperiment,
    Run,
)
from whynot.framework import (
    Dataset,
    InferenceResult,
    parameter,
)
from whynot import utils

#: Submodules exposed on demand alongside the simulators. Kept lazy so that
#: `import whynot` does not require the reinforcement learning dependencies.
_LAZY_SUBMODULES = ("gym",)


def __getattr__(name):
    """Expose simulators and lazy submodules as top-level attributes."""
    if name in SIMULATORS:
        return getattr(simulators, name)
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List simulators alongside the package's own attributes."""
    return sorted(set(globals()) | set(SIMULATORS) | set(_LAZY_SUBMODULES))
