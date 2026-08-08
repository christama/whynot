"""Interface to the world 3 simulator."""

import dataclasses
import math
import os

import numpy as np
from py_mini_racer import MiniRacer

import whynot as wn
from whynot.dynamics import BaseConfig, BaseState, BaseIntervention

# Load javascript file for execution
DIR_NAME = os.path.dirname(__file__)
with open(os.path.join(DIR_NAME, "world3_app.js")) as handle:
    WORLD3_JS_CODE = handle.read()


# This used to subclass MiniRacer to suppress __del__, working around a deadlock
# in py_mini_racer 0.6 when many contexts executed concurrently. The maintained
# mini-racer fork reworked context lifetime management, so suppressing cleanup is
# no longer necessary and would now leak a V8 context per simulation.
PyMiniRacerContext = MiniRacer


@dataclasses.dataclass
class Config(BaseConfig):
    # pylint: disable-msg=too-few-public-methods
    """World3 simulation dynamics parameters.

    Default values correspond to the standard run of World3.

    Examples
    --------
    >>> # Configuration of a run from 1910-1950
    >>> world3.Config(start_time=1910, end_time=1950)

    .. note:
        There are many possible parameters to adjust in World3. This subset
        exposed in the Config correspond to variables that are scalar quantities
        rather than tabular functions for simplicity.

    """

    # Dynamics parameters. The #: ensures the attribute can be documented by Sphinx.
    #:
    industrial_capital_output_ratio: float = 3
    #:
    average_lifetime_of_industrial_capital: float = 14
    #:
    fraction_of_industrial_output_allocated_to_consumption_constant: float = 0.43
    #:
    average_lifetime_of_service_capital: float = 20
    #:
    service_capital_output_ratio: float = 1
    #:
    land_yield_factor: float = 1
    #:
    nonrenewable_resource_usage_factor: float = 1
    #:
    persistent_pollution_generation_factor: float = 1

    # Simulator parameters
    #: Year to start the simulation.
    start_time: float = 1900
    #: Year to end the simulation.
    end_time: float = 2100
    #: Granularity of the simulation (step size in the forward Euler method).
    delta_t: float = 1.0


class Intervention(BaseIntervention):
    # pylint: disable-msg=too-few-public-methods
    """Parameterization of an intervention in the World3 model.

    An intervention is a subset of the configuration variables,
    and only variables passed into the constructor are modified.

    Examples
    --------
    >>> # Semantics: Starting in year 1975, double land_yield. All
    >>> # other parameters are unchanged.
    >>> Intervention(time=1975, land_yield_factor=2)

    """

    def __init__(self, time=1975, **kwargs):
        """Construct an intervention object.

        Parameters
        ----------
            time: float
                time (in years) to intervene in the simulator.
            kwargs: dict
                Only valid keyword arguments are parameters of Config.

        """
        super(Intervention, self).__init__(Config, time, **kwargs)


@dataclasses.dataclass
class State(BaseState):
    # pylint: disable-msg=too-few-public-methods
    """World3 state.

    Default values correspond to the initial state in 1900 of the standard run.

    Examples
    --------
    >>> # Construct initial state with 800 land fertility.
    >>> world3.State(arable_land=800)

    """

    #:
    population_0_to_14: float = 6.5e8
    #:
    population_15_to_44: float = 7.0e8
    #:
    population_45_to_64: float = 1.9e8
    #:
    population_65_and_over: float = 6.0e7
    #:
    industrial_capital: float = 2.1e11
    #:
    service_capital: float = 1.44e11
    #:
    arable_land: float = 0.9e9
    #:
    potentially_arable_land: float = 2.3e9
    #:
    urban_industrial_land: float = 8.2e6
    #:
    land_fertility: float = 600.0
    #:
    nonrenewable_resources: float = 1.0e12
    #:
    persistent_pollution: float = 2.5e7

    @property
    def total_population(self):
        """Return the aggregate population."""
        return (
            self.population_0_to_14
            + self.population_15_to_44
            + self.population_45_to_64
            + self.population_65_and_over
        )


def to_camel_case(snake_str):
    """Convert snake_str to snakeStr (camel case)."""
    components = snake_str.split("_")
    # We capitalize the first letter of each component except the first one
    # with the 'title' method and join them together.
    return components[0] + "".join(x.title() for x in components[1:])


def to_js_number(value):
    """Render a Python number as a JavaScript numeric literal.

    Values are passed to the engine by interpolating them into JavaScript
    source, and Python spells the non-finite floats ``nan``, ``inf``, and
    ``-inf``, none of which are valid JavaScript.
    """
    number = float(value)
    if math.isnan(number):
        return "NaN"
    if math.isinf(number):
        return "Infinity" if number > 0 else "-Infinity"
    return repr(number)


def set_state(js_context, initial_state):
    """Set the state of the world3 simulator.

    Raises
    ------
        ValueError
            If any state variable is not finite. World3 cannot be meaningfully
            resumed from such a state, and continuing would silently produce
            non-finite observations.

    """
    stocks = dataclasses.asdict(initial_state)

    diverged = sorted(
        name for name, value in stocks.items() if not math.isfinite(value)
    )
    if diverged:
        raise ValueError(
            "Cannot set world3 state: "
            + ", ".join(f"{name}={stocks[name]}" for name in diverged)
            + ". The simulation has diverged."
        )

    for stock_name, value in stocks.items():
        js_context.eval(f"{to_camel_case(stock_name)}.initVal = {to_js_number(value)}")

    # special case for resources
    js_context.eval(
        "nonrenewableResourcesInitialK = "
        f"{to_js_number(initial_state.nonrenewable_resources)}"
    )
    js_context.eval("resetModel()")


def decode_states(js_context):
    """Read out the sequence of states from the world3 engine.

    Parameters
    ----------
        js_context: PyMiniRacerContext
            Context containing a completed execution of world3.

    Returns
    -------
        (states, sample_times): list, np.ndarray
            The recorded state values and the time each value was sampled during the run.

    """

    def unpack(values):
        """Parse list of [{"x": time, "y":value}] into [times], [values]."""
        return list(zip(*[(value["x"], value["y"]) for value in values]))

    state_names = [f.name for f in dataclasses.fields(State)]
    data = {}
    sample_times = None
    for idx, state_name in enumerate(state_names):
        state_data = js_context.eval(f"{to_camel_case(state_name)}.data")
        times, values = unpack(state_data)
        if idx == 0:
            sample_times = times
        data[state_name] = values

    states = [State(*vals) for vals in zip(*data.values())]

    return states, np.array(sample_times)


def set_config(js_context, config, intervention):
    """Set the non-state variables of the world3 simulator."""
    # Set global simulator parameters
    js_context.eval(f"startTime = {to_js_number(config.start_time)}")
    js_context.eval(f"stopTime = {to_js_number(config.end_time)}")
    js_context.eval(f"dt = {to_js_number(config.delta_t)}")

    if intervention:
        intervention_config = config.update(intervention)
        js_context.eval(f"policyYear = {to_js_number(intervention.time)}")
    else:
        intervention_config = config
        js_context.eval(f"policyYear = {to_js_number(config.end_time)}")

    intervention_config = dataclasses.asdict(intervention_config)
    for parameter, before in dataclasses.asdict(config).items():
        if parameter in ["policy_year", "start_time", "end_time", "delta_t"]:
            continue
        after = intervention_config[parameter]
        js_context.eval(f"{to_camel_case(parameter)}.before = {to_js_number(before)}")
        js_context.eval(f"{to_camel_case(parameter)}.after = {to_js_number(after)}")
    js_context.eval("resetModel()")


#: Iterations fastRun uses to settle the smoothed and delayed quantities before
#: advancing time. Reproduced here so a stepped engine starts identically.
WARMUP_ITERATIONS = 100


def start_engine(config, initial_state, intervention=None):
    """Return an engine initialized and warmed up, ready to be stepped.

    World3 carries internal state beyond the twelve stocks, in its smoothed and
    delayed quantities. That state is built by the warmup below and is *not*
    recoverable from the stocks, so an engine that is to be advanced
    incrementally must be initialized once and then kept alive.

    Returns
    -------
        js_context: PyMiniRacerContext
            An engine sitting at ``config.start_time``.

    """
    js_context = PyMiniRacerContext()
    js_context.eval(WORLD3_JS_CODE)
    set_state(js_context, initial_state)
    set_config(js_context, config, intervention)
    js_context.eval("resetModel(); initModel();")
    js_context.eval(
        f"for (var i = 1; i <= {WARMUP_ITERATIONS}; i++)"
        " { warmupAuxen(); warmupRates(); tock(); }"
    )
    return js_context


def step_engine(js_context):
    """Advance a live engine by a single delta_t."""
    js_context.eval("timeStep()")


def read_state(js_context):
    """Read the current state out of a live engine."""
    return State(
        **{
            name: js_context.eval(f"{to_camel_case(name)}.k")
            for name in State.variable_names()
        }
    )


def set_parameter(js_context, parameter, value):
    """Set a config parameter on a live engine, effective immediately.

    Parameters switch from ``before`` to ``after`` at ``policyYear``. Setting
    both makes the value take effect whatever the current time is.
    """
    name = to_camel_case(parameter)
    js_context.eval(f"{name}.before = {to_js_number(value)}")
    js_context.eval(f"{name}.after = {to_js_number(value)}")


def simulate(initial_state, config, intervention=None, seed=None):
    """Run the world3 simulation for the specified initial state and configuration.

    Uses the forward Euler method internally to simulate world3 from start_time
    to end_time (specified in config) with a step size of delta_t (in Config).

    Parameters
    ----------
        initial_state: whynot.simulators.world3.State
            Initial state of the dynamical system
        config: whynot.simulators.world3.Config
            Configuraton parameters to control simulator dynamics.
        intervention: whynot.simulators.world3.Intervention
            (Optional) Intervention, if any, to perform during simulator execution.
        seed: int
            (Optional) Ignored since the simulator is deterministic.

    Returns
    -------
        run: whynot.dynamics.Run
            A single rollout from the simulator.

    """
    # pylint:disable-msg=unused-argument
    # Initialize a separate context separately for each simulator rollout.
    # This may be potentially less efficient that a single global context,
    # but it ensures that parallel executions will work without issue.
    ctx = PyMiniRacerContext()

    # Load the world3 simulation code
    ctx.eval(WORLD3_JS_CODE)

    # Initialize the similuator
    set_state(ctx, initial_state)
    set_config(ctx, config, intervention)

    # Single simulator rollout
    ctx.eval("fastRun()")

    # Read out the results
    states, times = decode_states(ctx)
    # Ensure the user passed initial state appears in the run
    states = [initial_state] + states[1:]
    return wn.dynamics.Run(states=states, times=times)
