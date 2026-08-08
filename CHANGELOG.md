# Changelog

This file starts at 0.13.0 and does not cover earlier releases.

## [0.13.0] - 2026-08-04

WhyNot had stopped installing: `gym==0.21.0` cannot be built by modern
setuptools. This release gets the package running on current Python and
dependencies. It contains breaking changes.

### Breaking

- **Environments follow the Gymnasium API.** OpenAI Gym was abandoned in 2022
  and does not support NumPy 2. `step()` now returns
  `(observation, reward, terminated, truncated, info)` rather than
  `(observation, reward, done, info)`; reaching the end of the simulation
  horizon is reported as truncation, since it is a time limit and not an
  absorbing state. `reset()` is keyword-only, takes `seed` and `options`, and
  returns `(observation, info)`. `env.seed()` is gone; pass `reset(seed=...)`.
  `render()` no longer takes a `mode`. `import whynot.gym as gym` still works.

  ```python
  # before
  observation = env.reset()
  env.seed(1)
  observation, reward, done, info = env.step(action)

  # after
  observation, info = env.reset(seed=1)
  observation, reward, terminated, truncated, info = env.step(action)
  ```

  Gymnasium also no longer forwards attribute access through wrappers, so
  `env.config` and `env.initial_state` become `env.unwrapped.config` and
  `env.unwrapped.initial_state`.

- **Python 3.12 or newer is required**, raised from 3.8. This is the floor the
  dependencies impose rather than a preference: current NumPy and SciPy require
  3.12. Declaring anything lower would let pip resolve an older, untested
  scientific stack instead of failing.

- **`world3-v0` produces different trajectories.** They are now the model's:
  see Fixed below. Results generated with earlier versions of this environment
  do not reproduce against it.

- **`setup.py` is replaced by `pyproject.toml`.** Installing from a checkout is
  unchanged (`pip install -e .`), but development extras are now
  `pip install -e '.[dev]'`.

### Fixed

- `world3-v0` was not simulating world3. It restarted the engine from the twelve
  stocks on every step, discarding the smoothed and delayed quantities that the
  engine builds during warmup and cannot recover from the stocks. Measured
  against a continuous run, and taking the action that changes nothing, it
  deviated 32% within ten steps and 55x by step 200, drove
  `nonrenewable_resources` negative, and left its own observation space; seven
  of eight random rollouts reached a non-finite state. The environment now keeps
  one engine alive for an episode and advances it in place, which reproduces a
  continuous run exactly.
- `Credit-v0` could not be constructed. The Credit state's default dataset had
  been replaced with empty arrays to fix a mutable default, so building the
  environment raised `IndexError`.
- World3 emitted invalid JavaScript for non-finite values, since Python spells
  them `nan` and `inf`. It now raises a clear error rather than a
  `ReferenceError` from inside the engine.
- `whynot.simulators.delayed_impact` used `Index.get_loc(method=...)`, removed
  in pandas 2.
- Agent-based simulators passed NumPy integers as model seeds, which
  `random.seed` has rejected since Python 3.11.
- Removed an `import distutils.version`, which fails on Python 3.12.
- The HIV example notebook could not run its final cell, which neither imported
  `whynot` nor passed `env` to `plot_sample_trajectory`.
- The documentation did not build; `sphinxcontrib-bibtex` 2.0 requires
  `bibtex_bibfiles`.

### Changed

- Depends on `gymnasium` instead of `gym`, and on the maintained `mini-racer`
  fork instead of the abandoned `py_mini_racer`, which ships no aarch64 wheels.
  Apple Silicon is now supported.
- Simulators are imported lazily, so a simulator whose dependencies are
  unavailable no longer prevents `import whynot`.
- DICE prefers an IPOPT found on `PATH` over the binaries bundled with whynot,
  which are x86-64 only. Its tests skip, with a reason, when neither is usable.

### Added

- GitHub Actions CI on Python 3.12, 3.13 and 3.14 across Linux and macOS,
  replacing Travis.
- Tests that `world3-v0` and a stepped engine both reproduce a continuous run.
