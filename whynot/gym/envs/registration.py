"""Registry of environments, kept separate from Gymnasium's global one.

WhyNot maintains its own registry so that importing whynot does not mutate
Gymnasium's global registry, and so environment ids cannot collide with those
of other packages.
"""

from gymnasium.envs.registration import EnvSpec, load_env_creator
from gymnasium.error import Error, NameNotFound
from gymnasium.wrappers import OrderEnforcing, TimeLimit


class EnvRegistry:
    """Map environment ids to the factories that build them."""

    def __init__(self):
        """Create an empty registry."""
        self.env_specs = {}
        self._simulators_loaded = False

    def _load_simulators(self):
        """Import simulators so their environments register themselves.

        Simulators are imported lazily, and an environment is only registered
        as a side effect of importing the simulator that defines it. Looking an
        environment up therefore has to force those imports first.
        """
        if self._simulators_loaded:
            return
        # Set before importing: the imports call back into register().
        self._simulators_loaded = True
        # Imported here rather than at module scope to avoid a circular import.
        from whynot import simulators  # pylint:disable-msg=import-outside-toplevel

        simulators.load_all()

    # pylint:disable-msg=redefined-builtin
    def register(
        self,
        id,
        entry_point,
        max_episode_steps=None,
        reward_threshold=None,
        kwargs=None,
    ):
        """Register an environment under the given id.

        Parameters
        ----------
            id: str
                The id used to look the environment up with `make`.
            entry_point: Callable or str
                A callable returning the environment, or a "module:attribute"
                string naming one.
            max_episode_steps: int
                (Optional) Number of steps after which an episode is truncated.
            reward_threshold: float
                (Optional) Return at which the environment is considered solved.
            kwargs: dict
                (Optional) Default arguments forwarded to the entry point.

        """
        if id in self.env_specs:
            raise Error(f"Cannot re-register id: {id}")
        self.env_specs[id] = EnvSpec(
            id=id,
            entry_point=entry_point,
            max_episode_steps=max_episode_steps,
            reward_threshold=reward_threshold,
            kwargs=dict(kwargs or {}),
            # WhyNot environments are built from pre-configured builders rather
            # than imported by name, and are covered by the whynot test suite.
            disable_env_checker=True,
        )
        return self.env_specs[id]

    def make(self, id, **kwargs):
        """Build the environment registered under id.

        Keyword arguments are forwarded to the entry point, overriding any
        defaults supplied at registration time.
        """
        spec = self.spec(id)

        entry_point = spec.entry_point
        if isinstance(entry_point, str):
            entry_point = load_env_creator(entry_point)

        env = entry_point(**{**spec.kwargs, **kwargs})
        env.unwrapped.spec = spec

        if spec.order_enforce:
            env = OrderEnforcing(env)
        if spec.max_episode_steps is not None:
            env = TimeLimit(env, spec.max_episode_steps)
        return env

    def spec(self, id):
        """View the spec for the environment."""
        if id not in self.env_specs:
            self._load_simulators()
        if id not in self.env_specs:
            raise NameNotFound(f"No registered env with id: {id}")
        return self.env_specs[id]

    def all(self):
        """Iterate over every registered spec."""
        self._load_simulators()
        return self.env_specs.values()


# Keep for consistency with original API
# pylint:disable-msg=invalid-name
# Have a global registry
registry = EnvRegistry()


# pylint:disable-msg=redefined-builtin
def register(id, **kwargs):
    """Register the environment."""
    return registry.register(id, **kwargs)


def make(id, **kwargs):
    """Build the environment."""
    return registry.make(id, **kwargs)


def spec(id):
    """View the spec for the environment."""
    return registry.spec(id)
