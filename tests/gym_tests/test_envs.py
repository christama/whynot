"""Smoke tests for the registered WhyNot environments."""

import dataclasses

import numpy as np
import pytest

import whynot as wn
from whynot.gym import envs
from spec_list import spec_list


# This runs a smoketest on each official registered env. We may want
# to try also running environments which are not officially registered
# envs.
@pytest.mark.parametrize("spec", spec_list, ids=lambda spec: spec.id)
def test_env(spec):
    """Each registered environment resets and steps within its declared spaces."""
    env = envs.make(spec.id)

    ob_space = env.observation_space
    act_space = env.action_space

    ob, info = env.reset()
    assert ob_space.contains(ob), "Reset observation: {!r} not in space".format(ob)
    assert isinstance(info, dict)

    action = act_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    assert ob_space.contains(observation), "Step observation: {!r} not in space".format(
        observation
    )
    assert np.isscalar(reward), "{} is not a scalar for {}".format(reward, env)
    assert isinstance(terminated, bool), "Expected {} to be a boolean".format(
        terminated
    )
    assert isinstance(truncated, bool), "Expected {} to be a boolean".format(truncated)
    assert isinstance(info, dict)

    for _ in env.metadata.get("render_modes", []):
        env.render()

    env.close()


# Run a longer rollout on some environments
@pytest.mark.parametrize("spec_id", ["HIV-v0", "world3-v0", "opioid-v0"])
def test_random_rollout(spec_id):
    """Rolling out an environment stays inside its declared spaces."""
    for env in [envs.make(spec_id), envs.make(spec_id), envs.make(spec_id)]:

        def agent(ob):
            # pylint:disable-msg=cell-var-from-loop
            return env.action_space.sample()

        ob, _info = env.reset()
        for _ in range(10):
            assert env.observation_space.contains(ob)
            action = agent(ob)
            assert env.action_space.contains(action)
            ob, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break
        env.close()


@pytest.mark.parametrize("spec_id", ["HIV-v0", "world3-v0", "opioid-v0", "Zika-v0"])
def test_config(spec_id):
    """Test setting simulator config via gym.make"""
    base_env = envs.make(spec_id)
    base_config = base_env.unwrapped.config
    new_config = dataclasses.replace(base_config, delta_t=-100)
    new_env = envs.make(spec_id, config=new_config)
    assert base_env.unwrapped.config.delta_t == base_config.delta_t
    assert new_env.unwrapped.config.delta_t == new_config.delta_t
    base_env.close()
    new_env.close()


def test_credit_config():
    """Set simulator config for Credit sim via gym.make"""
    base_features = wn.credit.Config().changeable_features
    new_features = np.array([0, 1, 2])
    base_env = envs.make("Credit-v0")
    config = wn.credit.Config(changeable_features=new_features)
    env = envs.make("Credit-v0", config=config)
    assert np.allclose(base_env.unwrapped.config.changeable_features, base_features)
    assert np.allclose(env.unwrapped.config.changeable_features, new_features)
    base_env.close()
    env.close()


def test_credit_initial_state():
    """Test initial state for Credit sim via gym.make"""
    base_env = envs.make("Credit-v0")
    original_state, _info = base_env.reset()
    features, labels = original_state["features"], original_state["labels"]
    subsampled_state = wn.credit.State(features[:100], labels[:100])

    env = envs.make("Credit-v0", initial_state=subsampled_state)

    assert np.allclose(env.unwrapped.initial_state.features, subsampled_state.features)
    assert np.allclose(env.unwrapped.initial_state.labels, subsampled_state.labels)

    ob, _info = env.reset()
    assert np.allclose(ob["features"], subsampled_state.features)
    assert np.allclose(ob["labels"], subsampled_state.labels)
    for _ in range(10):
        assert env.observation_space.contains(ob)
        action = env.action_space.sample()
        assert env.action_space.contains(action)
        ob, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            break
    ob, _info = env.reset()
    assert np.allclose(ob["features"], subsampled_state.features)
    assert np.allclose(ob["labels"], subsampled_state.labels)
    base_env.close()
    env.close()
