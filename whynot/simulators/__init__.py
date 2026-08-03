"""Simulator initialization.

Simulators are imported lazily. Several simulators depend on heavy or
platform-specific third-party packages, and importing all of them eagerly means
a single unavailable dependency makes ``import whynot`` fail outright. With
lazy imports, an unavailable dependency only affects the simulator that needs
it.
"""
import importlib

#: Every simulator shipped with whynot.
SIMULATORS = (
    "civil_violence",
    "credit",
    "delayed_impact",
    "dice",
    "hiv",
    "incarceration",
    "lalonde",
    "lotka_volterra",
    "opioid",
    "schelling",
    "world2",
    "world3",
    "zika",
)

__all__ = list(SIMULATORS)


def __getattr__(name):
    """Import a simulator the first time it is accessed (PEP 562)."""
    if name not in SIMULATORS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{name}")
    # Cache on the module so subsequent lookups skip __getattr__ entirely.
    globals()[name] = module
    return module


def __dir__():
    """List simulators alongside the module's own attributes."""
    return sorted(set(globals()) | set(SIMULATORS))


def load_all():
    """Import every simulator, skipping those whose dependencies are missing.

    Returns
    -------
        A dict mapping simulator name to the exception it raised on import.
        Empty if every simulator loaded.

    """
    failures = {}
    for name in SIMULATORS:
        try:
            globals()[name] = importlib.import_module(f"{__name__}.{name}")
        except Exception as exc:  # pylint:disable-msg=broad-except
            failures[name] = exc
    return failures
