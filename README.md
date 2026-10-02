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

On Ubuntu/WSL, the emulator may require the GLU library:

```bash
sudo apt update
sudo apt install -y libglu1-mesa
```

## Running

All commands are run from the repo root. `run.sh` activates `.venv` and sets
`PYTHONPATH` for you.

| Command | What it does |
|---|---|
| `bash run.sh test` | Run the full test suite |
| `bash run.sh train` | Train PPO |
| `bash run.sh eval --episodes 20 --timeout 2000` | Evaluate the baseline policies |
| `python scripts/analyze_results.py` | Calculate baseline statistics with pandas |
| `python scripts/plot_results.py` | Generate plots from the evaluation results |
| `bash run.sh play` | Watch a policy play with the game window open |

The baseline evaluation currently supports three simple policies:

- `random`: selects actions randomly
- `always_right`: always selects action `1`
- `zero`: always selects action `0`

Run the baseline evaluation with:

```bash
bash run.sh eval --episodes 20 --timeout 2000
```

`play` doesn't forward arguments. To choose a policy or episode count, call the
script directly:

```bash
source .venv/bin/activate
PYTHONPATH="$PWD" python scripts/play_mario.py --policy right --episodes 2
```

`--policy` is `random`, `right`, or `ppo`. The window runs as fast as the
emulator allows, so it plays much faster than real time.

To watch a trained agent (see [Training](#training)):

```bash
PYTHONPATH="$PWD" python scripts/play_mario.py --policy ppo --checkpoint data/checkpoints/<run_id>/best.pt --episodes 3
```

## Training

```bash
bash run.sh train                    # 1000 updates, rollout 2048, all defaults
bash run.sh train --config configs/ppo.yaml --updates 500   # custom run
bash run.sh train --resume           # continue from the latest checkpoint
```

Hyperparameters live in `configs/ppo.yaml`; any value can be overridden from
the command line. Each update collects `rollout_size` transitions across
episode boundaries, computes GAE advantages, and runs `epochs` minibatch PPO
updates. Progress (logged per update) includes policy/value loss and episode
statistics over the last 10 episodes.

Every run is self-contained in `data/runs/ppo_<timestamp>/` (gitignored):
`config.yaml` (resolved config), `run.json` (git commit, seed, start time),
`metrics.jsonl` (one line per update) and `episodes.json`.

Checkpoints are written to `data/checkpoints/<run_id>/` every `--checkpoint-every`
updates (and at the end). While training: `latest.pt` (most recent, for `--resume`),
`best.pt` (best mean distance so far), and periodic `checkpoint_<step>.pt` capped at
the top 3 by score. When training finishes, `latest.pt` is deleted and only `best.pt`
plus the top-3 periodic checkpoints remain. Checkpoints store the full agent —
policy and value models plus both optimizers — so `--resume` continues exactly
where the run left off.

## Tests

```bash
bash run.sh test
```

Expected: all 60 tests pass. Most tests use a fake environment
(`tests/fakes.py`), so they run without the emulator. `test_environment.py` and
`test_runner.py` use the real game and open a window briefly.

To run only the emulator-free tests:

```bash
export PYTHONPATH="$PWD"
pytest tests/test_preprocessing.py tests/test_reward.py tests/test_rollout_buffer.py \
       tests/test_logger.py tests/test_evaluation.py tests/test_checkpoint.py \
       tests/test_ppo.py tests/test_trainer.py tests/test_policies.py tests/test_config.py
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
  environment/
    mario_environment.py   Environment wrapper (frame skip, preprocessing, reward shaping)
    preprocessing.py       Grayscale, resize to 84x84, 4-frame stacking
    reward.py              Shaped reward (progress, time penalty, death/flag)
  algorithms/
    cnn.py                 CNN policy/value network
    ppo.py                 PPO agent (act, GAE, clipped minibatch updates)
  training/
    rollout_buffer.py      Trajectory storage for on-policy training
    runner.py              Runs episodes and collects rollouts
    trainer.py             PPO training loop (rollout -> GAE -> update -> checkpoint)
    logger.py              Per-step records and per-episode statistics
    checkpoint.py          Save/load models, keep last N plus best
    config.py              YAML config load/save
    reproducibility.py     Seeding and git-commit tracking
  policies/
    baselines.py           Random and constant baseline policies
    ppo_policy.py          Loads a trained checkpoint for play/eval
  evaluation/
    evaluation.py          Shared evaluation for every agent
scripts/
  train.py               Train PPO (./run.sh train)
  play_mario.py          Watch a policy play (random/right/ppo 
  --checkpoint ...)
  evaluate_baseline.py   Evaluate baselines with the shared harness
  analyze_results.py     Calculate statistics with pandas
  plot_results.py        Generate plots from saved results
  push_weights.py        Upload checkpoints to Hugging Face Hub
  pull_weights.py        Download checkpoints from Hugging Face Hub
configs/
  ppo.yaml               Default training configuration
tests/                   pytest suite (fakes.py is an emulator-free fake environment)
```

## Using the environment

```python
from src.environment.mario_environment import make_training_environment

env = make_training_environment()   # frame skip 4, (4, 84, 84) observations, shaped reward
result = env.reset()
result = env.step(1)                # dict: observation, reward, raw_reward, episode_ended, info
env.close()
```

`MarioEnvironment()` with no arguments gives raw frames and the built-in reward.

## Status

- Milestone 0 (environment setup, dummy policy, logging): done
- Milestone 1 (preprocessing, reward, rollouts, checkpointing, evaluation): done
- Milestone 2 (PPO): done — `src/algorithms/ppo.py` + `src/training/trainer.py`, run via `./run.sh train`

## Sharing model weights

Checkpoints live in `data/checkpoints/` (gitignored, `*.pt`). To share them,
publish to Hugging Face Hub:

```bash
python scripts/push_weights.py --repo-id <user>/mario-ppo --run-id <run_id>
python scripts/pull_weights.py --repo-id <user>/mario-ppo --run-id <run_id>
```

Hub layout: `<run_id>/config.yaml`, `<run_id>/best.pt`, `<run_id>/checkpoint_*.pt`.

