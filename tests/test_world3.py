"""Unit tests for world 3 model."""

import math

import numpy as np

import whynot as wn
from whynot.simulators.world3.simulator import *


def get_world3_context():
    """Load a javascript context for world3"""
    ctx = PyMiniRacerContext()
    ctx.eval(WORLD3_JS_CODE)
    return ctx


def test_set_state():
    """Check setting initial_state."""
    ctx = get_world3_context()
    initial_state = State(land_fertility=3434)

    set_state(ctx, initial_state)
    assert ctx.eval(f"landFertility.initVal") == 3434

    # Run the model and check if land fertility is correctly used.
    ctx.eval("fastRun()")
    states, _ = decode_states(ctx)
    assert states[0].land_fertility == 3434


def test_set_config():
    """Check setting config + intervention works as expected."""
    ctx = get_world3_context()
    config = Config(service_capital_output_ratio=23)
    intervention = Intervention(nonrenewable_resource_usage_factor=2)

    set_config(ctx, config, intervention)

    for param in dataclasses.asdict(config):
        if param in ["policy_year", "start_time", "end_time", "delta_t"]:
            continue

        if param == "service_capital_output_ratio":
            assert ctx.eval("serviceCapitalOutputRatio.before") == 23
            assert ctx.eval("serviceCapitalOutputRatio.after") == 23
        elif param == "nonrenewable_resource_usage_factor":
            assert (
                ctx.eval("nonrenewableResourceUsageFactor.before")
                == config.nonrenewable_resource_usage_factor
            )
            assert ctx.eval("nonrenewableResourceUsageFactor.after") == 2
        else:
            assert ctx.eval(f"{to_camel_case(param)}.before") == ctx.eval(
                f"{to_camel_case(param)}.after"
            )


def test_setup():
    """Test setup of world3 simulator."""
    for idx in range(3):
        ctx = get_world3_context()
        initial_state = State(land_fertility=idx * 1111)
        config = Config(service_capital_output_ratio=idx * 22)
        intervention = Intervention(nonrenewable_resource_usage_factor=idx * 22)

        set_state(ctx, initial_state)
        set_config(ctx, config, intervention)

        assert ctx.eval(f"landFertility.initVal") == idx * 1111
        assert ctx.eval("serviceCapitalOutputRatio.before") == idx * 22
        assert ctx.eval("serviceCapitalOutputRatio.after") == idx * 22
        assert (
            ctx.eval("nonrenewableResourceUsageFactor.before")
            == config.nonrenewable_resource_usage_factor
        )
        assert ctx.eval("nonrenewableResourceUsageFactor.after") == idx * 22


def test_stepped_engine_matches_continuous_run():
    """Advancing an engine step by step must reproduce a continuous run.

    World3 keeps internal state in its smoothed and delayed quantities that is
    established by the warmup, not derived from the twelve stocks. Restarting
    the engine from the stocks each step silently produces a different model.
    """
    config = Config(delta_t=0.5)
    reference = wn.world3.simulate(State(), config)
    expected = {
        time: np.asarray(state.values(), dtype=float)
        for time, state in zip(reference.times, reference.states)
    }

    engine = start_engine(config, State())
    compared = 0
    while engine.eval("t") < config.end_time:
        step_engine(engine)
        time = engine.eval("t")
        if not math.isclose(time, round(time), abs_tol=1e-9):
            continue
        if round(time) not in expected:
            continue
        actual = np.asarray(read_state(engine).values(), dtype=float)
        assert np.allclose(actual, expected[round(time)], rtol=0, atol=0)
        compared += 1

    assert compared > 100


def test_environment_tracks_the_model():
    """The neutral action must leave world3 running its standard trajectory.

    Action 4 is (1.0, 1.0), which changes no parameter, so the environment
    should follow the same trajectory as an unintervened continuous run and
    stay inside its declared observation space.
    """
    config = Config(delta_t=0.5)
    reference = wn.world3.simulate(State(), config)
    expected = dict(zip(reference.times, reference.states))

    env = wn.gym.make("world3-v0")
    observation, _ = env.reset(seed=0)
    compared = 0
    while True:
        observation, _reward, terminated, truncated, _info = env.step(4)
        assert env.observation_space.contains(np.asarray(observation))

        time = env.unwrapped.time
        if time in expected:
            assert np.allclose(
                np.asarray(observation, dtype=float),
                np.asarray(expected[time].values(), dtype=float),
                rtol=0,
                atol=0,
            )
            compared += 1
        if terminated or truncated:
            break
    env.close()

    # Guard against the comparison silently matching no timesteps at all.
    assert compared > 100
