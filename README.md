# CS 4701 Project: Reinforcement Learning for Super Mario Bros 1

Training RL agents (PPO, then DQN) to play Super Mario Bros level 1-1 using
`gym-super-mario-bros` and PyTorch.

## Setup

Requires Python 3.9+ (developed on 3.12) and a C++ compiler, because `nes-py`
builds the emulator on install. On macOS, run `xcode-select --install` first.

```bash
bash build.sh                 # creates .venv and installs requirements.txt
source .venv/bin/activate
python -c "import gym_super_mario_bros; print('emulator ok')"
```

The Gym / NumPy / step-API warnings printed on import are expected and harmless
(`gym-super-mario-bros` requires the old Gym API, so we pin `gym==0.25.1`).

## Running

All commands are run from the repo root. `run.sh` activates `.venv` and sets
`PYTHONPATH` for you.

| Command | What it does |
|---|---|
| `bash run.sh test` | Run the full test suite |
| `bash run.sh eval --episodes 20 --timeout 2000` | Evaluate the baseline policies and save `results/baselines.json` |
| `bash run.sh play` | Watch a random policy play with the game window open |

`play` doesn't forward arguments. To choose a policy or episode count, call the
script directly:

```bash
source .venv/bin/activate
PYTHONPATH="$PWD" python scripts/play_mario.py --policy right --episodes 2
```

`--policy` is `random` or `right`. The window runs as fast as the emulator
allows, so it plays much faster than real time.

## Tests

```bash
bash run.sh test
```

Expected: all 39 tests pass. Most tests use a fake environment
(`tests/fakes.py`), so they run without the emulator. `test_environment.py` and
`test_runner.py` use the real game and open a window briefly.

To run only the emulator-free tests:

```bash
export PYTHONPATH="$PWD"
pytest tests/test_preprocessing.py tests/test_reward.py tests/test_rollout_buffer.py \
       tests/test_logger.py tests/test_evaluation.py tests/test_checkpoint.py
```

## Baseline results (level 1-1, 20 episodes)

| Policy | Mean distance | Best distance | Completion |
|---|---|---|---|
| random | 611.2 | 1122 | 0% |
| always right | 296.0 | 296 | 0% |

A trained agent should beat these. "Always right" dies at the first goomba
(x = 296) every time.

## Project layout

```
src/
  mario_environment.py   Environment wrapper (frame skip, preprocessing, reward shaping)
  preprocessing.py       Grayscale, resize to 84x84, 4-frame stacking
  reward.py              Shaped reward (progress, time penalty, death/flag)
  rollout_buffer.py      Trajectory storage for on-policy training
  runner.py              Runs episodes and collects rollouts
  logger.py              Per-step records and per-episode statistics
  checkpoint.py          Save/load models, keep last N plus best
  evaluation.py          Shared evaluation for every agent
  policies.py            Random and constant baseline policies
scripts/
  play_mario.py          Watch a baseline policy play
  evaluate_baseline.py   Evaluate baselines with the shared harness
tests/                   pytest suite (fakes.py is an emulator-free fake environment)
```

## Using the environment

```python
from src.mario_environment import make_training_environment

env = make_training_environment()   # frame skip 4, (4, 84, 84) observations, shaped reward
result = env.reset()
result = env.step(1)                # dict: observation, reward, raw_reward, episode_ended, info
env.close()
```

`MarioEnvironment()` with no arguments gives raw frames and the built-in reward.

## Status

- Milestone 0 (environment setup, dummy policy, logging): done
- Milestone 1 (preprocessing, reward, rollouts, checkpointing, evaluation): done
- Milestone 2 (PPO): next