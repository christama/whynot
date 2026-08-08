# -*- coding: utf-8 -*-
import pytest

import whynot.gym as gym
from whynot.gym import error, envs
from whynot.gym.envs import registration
import whynot.simulators as simulators


class ArgumentEnv(gym.Env):
    def __init__(self, arg1, arg2, arg3):
        self.arg1 = arg1
        self.arg2 = arg2
        self.arg3 = arg3


gym.register(
    id="test.ArgumentEnv-v0",
    entry_point="test_registration:ArgumentEnv",
    kwargs={
        "arg1": "arg1",
        "arg2": "arg2",
    },
)


def test_make():
    env = envs.make("HIV-v0")
    assert env.spec.id == "HIV-v0"
    assert isinstance(env.unwrapped, envs.ODEEnvBuilder)


def test_make_with_kwargs():
    env = envs.make("test.ArgumentEnv-v0", arg2="override_arg2", arg3="override_arg3")
    assert env.spec.id == "test.ArgumentEnv-v0"
    assert isinstance(env.unwrapped, ArgumentEnv)
    assert env.unwrapped.arg1 == "arg1"
    assert env.unwrapped.arg2 == "override_arg2"
    assert env.unwrapped.arg3 == "override_arg3"


def test_spec():
    spec = envs.spec("HIV-v0")
    assert spec.id == "HIV-v0"


def test_missing_lookup():
    registry = registration.EnvRegistry()
    registry.register(id="Test-v1", entry_point=None)
    with pytest.raises(error.NameNotFound):
        registry.spec("Unknown-v1")


def test_malformed_lookup():
    registry = registration.EnvRegistry()
    with pytest.raises(error.NameNotFound):
        registry.spec("“Breakout-v0”")


def test_no_reregistration():
    registry = registration.EnvRegistry()
    registry.register(id="Test-v1", entry_point=None)
    with pytest.raises(error.Error):
        registry.register(id="Test-v1", entry_point=None)
