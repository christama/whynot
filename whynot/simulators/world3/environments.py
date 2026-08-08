"""Reinforcment learning for world3."""

import copy
from itertools import product

import numpy as np

from whynot.gym import Env, spaces
from whynot.gym.envs import register
from whynot.simulators.world3 import Config, Intervention, State
from whynot.simulators.world3.simulator import (
    read_state,
    set_parameter,
    start_engine,
    step_engine,
)


def get_intervention(action, time):
    """Return the intervention needed to take action in the simulator."""
    resource_usages = [0.8, 1.0, 1.2]
    pollution_generations = [0.8, 1.0, 1.2]
    action_list = list(product(resource_usages, pollution_generations))
    resource_use, pollution_gen = action_list[action]
    return Intervention(
        time,
        nonrenewable_resource_usage_factor=resource_use,
        persistent_pollution_generation_factor=pollution_gen,
    )


def get_reward(intervention, state):
    """Return the intervention needed to take action in the simulator."""
    # Constants are made up, but chosen so that all quantities have
    # the same magnitude for the initial state.

    # High population and capital are good, high pollution is bad.
    reward = 1e-11 * state.industrial_capital - 5e-7 * state.persistent_pollution
    reward += 2e-8 * state.total_population

    # Taking any action other than the "default" is costly
    resource_usage = intervention.updates["nonrenewable_resource_usage_factor"]
    pollution_generation = intervention.updates[
        "persistent_pollution_generation_factor"
    ]
    reward -= 0.3 * (1.0 - resource_usage) ** 2
    reward -= 0.5 * (1.0 - pollution_generation) ** 2

    return reward


def observation_space():
    """Return the model observation space."""
    num_states = State.num_variables()
    state_space_low = np.zeros(num_states)
    state_space_high = np.inf * np.ones(num_states)
    return spaces.Box(state_space_low, state_space_high, dtype=np.float64)


class World3Env(Env):
    """Sequential decision making on the world3 model.

    Keeps one engine alive for the episode and advances it in place, rather
    than re-simulating from the state as the ODE environments do. World3 holds
    internal state in its smoothed and delayed quantities that is established
    by a warmup and cannot be recovered from the twelve stocks, so restarting
    the engine each step would silently produce a different model.
    """

    def __init__(self, config=None, initial_state=None, timestep=1.0):
        """Initialize the environment.

        Parameters
        ----------
            config: whynot.simulators.world3.Config
                (Optional) Simulation parameters. A smaller delta_t than the
                simulator default improves numerical stability.
            initial_state: whynot.simulators.world3.State
                (Optional) State the episode starts from.
            timestep: float
                Time, in years, between successive observations.

        """
        self.config = copy.deepcopy(config) if config else Config(delta_t=0.5)
        self.initial_state = copy.deepcopy(initial_state) if initial_state else State()
        self.timestep = timestep

        # In this environment there are 9 actions defined by
        # nonrenewable_resource_usage and pollution_generation_factor.
        self.action_space = spaces.Discrete(9)
        self.observation_space = observation_space()

        self.start_time = self.config.start_time
        self.terminal_time = self.config.end_time
        self.time = self.start_time
        self.state = self.initial_state
        self.engine = None

    def reset(self, *, seed=None, options=None):
        """Start a new episode on a freshly warmed up engine."""
        super().reset(seed=seed)
        self.engine = start_engine(self.config, self.initial_state)
        self.time = self.start_time
        self.state = self.initial_state
        return self._get_observation(self.state), {}

    def step(self, action):
        """Apply an action for one timestep and advance the engine.

        Returns
        -------
            observation, reward, terminated, truncated, info.
            terminated is always False; the model has no absorbing state, and a
            run ends only by reaching the end of the simulated horizon, which is
            reported as truncation.

        """
        if not self.action_space.contains(action):
            raise ValueError("%r (%s) invalid" % (action, type(action)))
        if self.engine is None:
            raise RuntimeError("Cannot step before reset.")

        intervention = get_intervention(action, self.time)
        for parameter, value in intervention.updates.items():
            set_parameter(self.engine, parameter, value)

        for _ in range(self._steps_per_timestep()):
            step_engine(self.engine)
        self.time += self.timestep
        self.state = read_state(self.engine)

        truncated = bool(self.time >= self.terminal_time)
        reward = get_reward(intervention, self.state)
        return self._get_observation(self.state), reward, False, truncated, {}

    def render(self):
        """Render the environment, unused."""

    def close(self):
        """Release the engine backing the episode."""
        self.engine = None

    def _steps_per_timestep(self):
        """Return how many engine steps make up one environment timestep."""
        steps = round(self.timestep / self.config.delta_t)
        if steps < 1:
            raise ValueError(
                f"timestep {self.timestep} is smaller than the simulator's "
                f"delta_t {self.config.delta_t}."
            )
        return steps

    @staticmethod
    def _get_observation(state):
        """Convert a state to a numpy array observation."""
        return state.values()


register(
    id="world3-v0",
    entry_point=World3Env,
    max_episode_steps=400,
    reward_threshold=1e5,
)
