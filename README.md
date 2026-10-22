# CS 4701 Project: Reinforcement Learning for Super Mario Bros 1

Training RL agents (PPO, then DQN) to play Super Mario Bros level 1-1 using
`gym-super-mario-bros` and PyTorch.

**Current result:** PPO beats every baseline. The trained agent finishes the
level in 82% of episodes when sampling actions and in 100% of episodes when
playing greedily; random, always-right and zero never finish (see
[Results](#results-level-1-1-100-episodes)).

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
`PYTHONPATH` for you. When calling a script directly, do this first:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD"
```

| Command | What it does |
|---|---|
| `bash run.sh test` | Run the full test suite |
| `bash run.sh train` | Train PPO (also starts the dashboard) |
| `bash run.sh train_dqn` | Train DQN (also starts the dashboard) |
| `bash run.sh eval --episodes 100 --timeout 2000` | Evaluate the baseline policies |
| `bash run.sh eval_dqn --run <run_dir> --episodes 100` | Evaluate a trained DQN run and add it to the comparison |
| `python scripts/evaluate_ppo.py --run <run_dir>` | Evaluate a trained PPO run and add it to the comparison |
| `python scripts/analyze_results.py` | Calculate statistics with pandas |
| `python scripts/plot_results.py` | Generate comparison plots |
| `bash run.sh analysis` | Baselines end to end: eval, analyze, plot |
| `bash run.sh play` | Watch a baseline policy play with the game window open |

`play` doesn't forward arguments. To choose a policy or episode count, call the
script directly:

```bash
python scripts/play_mario.py --policy right --episodes 2
```

`--policy` is `random`, `right`, `ppo` or `dqn`. The window runs as fast as the
emulator allows unless you pass `--fps`.

To watch a trained agent:

```bash
python scripts/play_mario.py --policy ppo --episodes 3 --fps 15 \
  --checkpoint "data/runs/<run_id>/checkpoints/best.pt"
```

The network size and frame settings are read automatically from
`data/runs/<run_id>/config.yaml`, so the checkpoint always loads with the
architecture it was trained with. Add `--greedy` to play the most likely
action instead of sampling. For DQN that means epsilon = 0; without it the
standard evaluation epsilon of 0.05 is used.

## Training

```bash
bash run.sh train                                          # 1000 updates, all defaults
bash run.sh train --config configs/ppo.yaml --updates 500  # custom run
```

Hyperparameters live in `configs/ppo.yaml`; any value can be overridden from
the command line. Each update collects `rollout_size` transitions across
episode boundaries, computes GAE advantages, and runs `epochs` minibatch PPO
updates.

Each run lives in `data/runs/<run_id>/`: `config.yaml` (resolved config),
`run.json` (git commit, seed, start time), `metrics.jsonl` (one line per
update), `episodes.json` (written when training finishes), `checkpoints/`
and `plots/`.

### What makes PPO train here

An earlier version of the agent did not learn at all (the mean distance stayed
at about 800 for 2M steps and the critic loss never decreased). The current
version trains reliably because of these changes:

- **Normalized value targets.** The critic trains on returns normalized with
  running mean/std, while `act()` and `value_of()` still return raw reward
  units so GAE is unaffected. This is the key fix.
- **Orthogonal initialization**, with a small output gain on the policy head
  so the policy starts near uniform.
- **Entropy bonus** (`ent_coef`) and **gradient clipping** (`max_grad_norm`).
- **Learning-rate annealing** (`lr_anneal: true`): linear decay to 0 over the run.
- **Larger learning rate** (2.5e-4) and `reward_scale: 1.0`.

### Training diagnostics

The log line and `metrics.jsonl` include these per update. Check them in the
first ~50 updates of any new run:

| Metric | Healthy behavior |
|---|---|
| `value_loss` | Falls well below its starting value |
| `explained_variance` | Rises above 0 (reached ~0.97 in a good run) |
| `entropy` | Starts near 1.95 (ln 7) and declines gradually; collapsing to ~0 early means too little exploration |
| `approx_kl` | Stays small, roughly under 0.05 |
| `clip_fraction` | Around 0.1-0.2 |
| `mean_distance` | Trends upward |

### Checkpoints and resuming

Checkpoints are written to `data/runs/<run_id>/checkpoints/` every
`--checkpoint-every` updates (and at the end). During training: `latest.pt`
(for `--resume`), `best.pt` (best mean distance over a 10-episode window so
far) and periodic `checkpoint_<step>.pt`, capped at the top 3 by score. When
training finishes, `latest.pt` is deleted. Checkpoints store the full agent
(policy and value models, both optimizers and the return statistics).

Each save writes two full checkpoints, so make sure you have free disk space
(`df -h .`), or save less often with `--checkpoint-every 25`.

**To resume an interrupted run you must point at its existing folder.** A plain
`--resume` would start a new timestamped run with no checkpoints:

```bash
RUN="data/runs/ppo_<run_id>"
rm -f "$RUN"/checkpoints/*.tmp          # remove any half-written save

bash run.sh train \
  --config "$RUN/config.yaml" \
  --run-dir "$RUN" \
  --checkpoint-dir "$RUN/checkpoints" \
  --resume
```

Notes: updates since the last checkpoint are repeated, so a few lines in
`metrics.jsonl` will appear twice (dedupe by `update` when plotting).
`episodes.json` is only written when training finishes, so episodes from
before an interruption are not in it; `metrics.jsonl` still has the full
training curve.

## Training with DQN

```bash
bash run.sh train_dqn                                            # configs/dqn.yaml
bash run.sh train_dqn --config configs/dqn.yaml --total-steps 50000
```

DQN is an off-policy alternative to PPO, implemented in
`src/algorithms/dqn.py` with the training loop in
`src/training/dqn_trainer.py`. The differences that matter:

- **One gradient step per environment step** after a burn-in
  (`train_start`), instead of `epochs` passes over a fresh rollout. It is
  much slower per step: about 12-20 steps/s on CPU, so a 300k-step run
  takes ~6 hours. Watch the `steps/s` field in the log line.
- **A replay buffer** (`src/training/replay_buffer.py`) stores the last
  `buffer_size` transitions and samples mini-batches uniformly at random.
  Consecutive Mario frames are highly correlated; shuffling decorrelates
  them and lets every collected transition be reused.
- **A target network** that is resynced every `target_update_freq` gradient
  steps, so the TD target stays fixed while the online network regresses on
  it. The target is `r + gamma * (1 - done) * max_a' Q_target(s', a')` and
  the loss is Huber.
- **Epsilon-greedy exploration**, annealed linearly from `epsilon_start` to
  `epsilon_end` over `epsilon_decay_steps`.
- The network is the standard Nature-DQN architecture: 32/64/64 conv
  channels and a 512-unit dense layer, with no softmax (the output is a
  Q-value per action, so the greedy action is the argmax).

Runs are written to `data/runs/dqn_<timestamp>/` with the same layout as PPO.
`metrics.jsonl` records `loss`, `td_error`, `q_mean`, `epsilon` and the
episode statistics.

**Memory.** The buffer stores two stacked `(4, 84, 84)` uint8 observations
per transition, about 56 KB each. The default `buffer_size: 30000` therefore
needs ~1.7 GB of RAM. Lower it if you are short on memory.

**Reward shaping.** The dense part of the reward (progress minus time
penalty) is clipped to `[-dense_clip, dense_clip]` and the flag bonus and
death penalty are added *after* the clip. Clipping the total reward would
flatten the terminal bonus into the same range as a single step and make
"survive longer" the objective.

### DQN training diagnostics

| Metric | Healthy behavior |
|---|---|
| `loss` | Falls well below its starting value, then stays low and noisy |
| `td_error` | Shrinks over the first few thousand steps |
| `q_mean` | Rises away from 0 as the agent learns what is worth |
| `epsilon` | Decreases linearly from 1.0 towards `epsilon_end` |
| `mean_distance` | Trends upward |
| `eval_mean_distance` | Only on evaluation rows (`eval_every`): the greedy score. Once the agent is competent it should beat the sampled score; while it is still near-random it often does not |

Evaluation rows also carry `eval_completion_rate`. Both fields are absent from
rows where no evaluation ran.

### Evaluating DQN

```bash
bash run.sh eval_dqn --run data/runs/dqn_<timestamp> --episodes 100
```

This evaluates the checkpoint twice and merges both into
`data/results/baselines.json` so the analysis scripts and the dashboard
compare them against the baselines and PPO:

- `dqn_greedy`: always the argmax action (epsilon = 0).
- `dqn_eps005`: epsilon = 0.05, the standard DQN evaluation protocol and the
  honest estimate, because the agent is trained with exploration.

### Resuming a DQN run

Same as PPO, but point at the existing run directory:

```bash
RUN="data/runs/dqn_<timestamp>"
bash run.sh train_dqn --config "$RUN/config.yaml" --run-dir "$RUN" \
  --checkpoint-dir "$RUN/checkpoints" --resume
```

The checkpoint stores the target network and the epsilon schedule as well, so
a resumed run continues with the same exploration state instead of starting
over.

**The replay buffer is not checkpointed.** A resumed run starts with an empty
buffer and pauses gradient updates for a few dozen steps while it refills.
That is deliberate: the buffer is large (~1.7 GB at the default size) and the
old transitions are only mildly useful. The network, target network, optimizer
and epsilon all resume exactly where they left off.

If `--resume` cannot find anything it fails with a `FileNotFoundError` naming
the directory it looked in. That usually means `--run-dir` points somewhere
other than the directory the training run actually used.

### Verifying the DQN implementation

Unit tests cover the components in isolation, but these check the pieces
against the real emulator:

```bash
python scripts/verify_dqn.py      # env + agent + buffer + a gradient step, ~1 min
python scripts/verify_dashboard.py  # dashboard dispatch + template rendering
```

`verify_dqn.py` ends by overfitting a single batch: the loss should fall from
roughly 7 to below 1. If it does not, the learning machinery is broken (as
opposed to merely untrained). For a learning-dynamics check, run a few
thousand steps and confirm `q_mean` rises, `epsilon` falls linearly and
`mean_distance` trends upward — see the diagnostics table above.

## Evaluating and comparing against the baselines

All policies (baselines and trained PPO/DQN) are evaluated through the same
harness (`src/evaluation/evaluation.py`), so the numbers are directly
comparable. Do not compare against the training `episodes.json`: training
episodes use sampled actions and are not a clean evaluation.

**1. List the baselines to run** in `configs/baselines.yaml`:

```yaml
policies:
  - random
  - always_right
  - zero
```

- `random`: selects actions randomly
- `always_right`: always selects action `1`
- `zero`: always selects action `0` (never moves)

**2. Evaluate the baselines** (100 episodes gives much less noisy numbers than 20):

```bash
bash run.sh eval --episodes 100 --timeout 2000
```

**3. Evaluate the trained PPO model** with the same episode count and timeout:

```bash
python scripts/evaluate_ppo.py --run "data/runs/<run_id>" --episodes 100
```

This evaluates `best.pt` twice and adds both to `data/results/baselines.json`
next to the baselines:

- `ppo_sampled`: samples actions, as during training. **This is the honest
  performance estimate.**
- `ppo_greedy`: always takes the most likely action. The emulator and policy
  are both deterministic, so all episodes are essentially the same run; treat
  it as a demonstration, not a statistical result.

It also prints a 95% confidence interval on the completion rate. Useful options:

- `--checkpoint <path>`: evaluate a different checkpoint (e.g. the last
  `checkpoint_<step>.pt`) instead of `best.pt`
- `--name <label>`: label prefix, so several checkpoints can be kept side by
  side (e.g. `--name ppo_final` gives `ppo_final_sampled`/`ppo_final_greedy`)
- `--device cuda|mps|cpu`, `--timeout`, `--output`

The network architecture and environment settings are read from the run's
`config.yaml`, so they always match the checkpoint.

**4. Generate the statistics table and plots:**

```bash
python scripts/analyze_results.py     # -> data/results/statistics.json
python scripts/plot_results.py        # -> data/plots/baseline_<timestamp>/
```

The dashboard's **Results** page shows the same statistics and plots, and
lets you overlay episode curves of any policies/runs.

Notes on comparing fairly:

- Compare **mean distance** and **completion rate**. Mean reward is only
  comparable between policies evaluated with the same reward settings.
- `best.pt` is selected on the best 10-episode window during training, which
  partly selects on luck. If its sampled completion rate comes out lower than
  the late-training rate, also evaluate a late periodic checkpoint
  (`--checkpoint ... --name ppo_final`).

## Results (level 1-1, 100 episodes)

| Policy | Mean distance | Completion | Mean episode length |
|---|---|---|---|
| zero | 40 | 0% | 2000 (hits timeout) |
| always_right | 296 | 0% | 40 |
| random | 663 | 0% | 314 |
| **PPO (sampled)** | **2889** | **82%** | 288 |
| **PPO (greedy)** | **3161** | **100%** | 283 |

The flag is at x = 3161. "Always right" dies at the first goomba (x = 296)
every time. The 82% sampled completion rate has a 95% confidence interval of
roughly 73-88%; its failures are scattered across the level rather than at a
single obstacle. The training curve (mean distance rising from ~800 to
~3000 over the run) is in each run's `metrics.jsonl` and on the dashboard run
page.

DQN is implemented and verified end to end but has not been trained to
completion (a 300k-step run is ~6 hours on CPU), so it has no row in this
table yet. Evaluate it with `bash run.sh eval_dqn --run <run_dir>` once it has
trained; the results merge into the same `baselines.json`.

## Tests

```bash
bash run.sh test
```

Expected: all tests pass. Most tests use a fake environment (`tests/fakes.py`),
so they run without the emulator. `test_environment.py` and `test_runner.py`
use the real game and open a window briefly.

To run only the emulator-free tests:

```bash
export PYTHONPATH="$PWD"
pytest tests/test_preprocessing.py tests/test_reward.py tests/test_rollout_buffer.py \
       tests/test_logger.py tests/test_evaluation.py tests/test_checkpoint.py \
       tests/test_ppo.py tests/test_trainer.py tests/test_policies.py tests/test_config.py \
       tests/test_replay_buffer.py tests/test_dqn.py tests/test_dqn_trainer.py
```

The DQN tests are emulator-free too, and can also be run on their own with
`pytest -k dqn`.

## Dashboard

```bash
bash scripts/start_dashboard.sh       # http://localhost:8000
```

Pages: **Overview** (all runs), **Train** (launch/stop training), **Grid
Search** (hyper-parameter sweeps), **Play** (live game view of a checkpoint)
and **Results** (baseline statistics, plots and policy comparison).

The **Train** page has an algorithm selector: choosing DQN shows the DQN
hyperparameters (total steps, buffer size, batch size, epsilon schedule,
target update frequency) and launches `scripts/train_dqn.py` instead of
`scripts/train.py`. The run pages read `loss`/`td_error` for DQN runs and
`policy_loss`/`value_loss` for PPO runs.

## Project layout

```
src/
  paths.py              Data directory layout (runs/, results/, plots/)
  environment/
    mario_environment.py   Environment wrapper (frame skip, preprocessing, reward shaping)
    preprocessing.py       Grayscale, resize, 4-frame stacking
    reward.py              Shaped reward (progress, time penalty, death/flag)
  algorithms/
    cnn.py                 CNN policy/value network
    ppo.py                 PPO agent (act, GAE, normalized-value critic, clipped minibatch updates)
    dqn.py                 DQN agent (epsilon-greedy, target network, Huber TD loss)
  training/
    rollout_buffer.py      Trajectory storage for on-policy training
    replay_buffer.py       Circular transition buffer for off-policy (DQN) training
    runner.py              Runs episodes and collects rollouts
    trainer.py             PPO training loop (rollout -> GAE -> update -> checkpoint)
    dqn_trainer.py         DQN training loop (act -> store -> sample -> gradient step)
    logger.py              Per-step records and per-episode statistics
    checkpoint.py          Save/load models, keep top-k by score plus best
    config.py              YAML config load/save, seeding, device resolution
  policies/
    baselines.py           Random and constant baseline policies
    ppo_policy.py          Loads a trained checkpoint for play/eval (sampled or greedy)
    dqn_policy.py          Loads a trained checkpoint for play/eval (epsilon-greedy)
  evaluation/
    evaluation.py          Shared evaluation for every agent
scripts/
  train.py               Train PPO (./run.sh train)
  train_dqn.py           Train DQN (./run.sh train_dqn)
  play_mario.py          Watch a policy play (random/right/ppo/dqn --checkpoint ...)
  evaluate_baseline.py   Evaluate baselines with the shared harness
  evaluate_ppo.py        Evaluate a trained PPO run and merge into the comparison
  evaluate_dqn.py        Evaluate a trained DQN run (greedy + eps=0.05) and merge
  verify_dqn.py          Smoke test: env + agent + buffer + a gradient step
  verify_dashboard.py    Check the dashboard dispatch and template rendering
  analyze_results.py     Calculate statistics with pandas
  plot_results.py        Generate plots from saved results
  full_analysis.py       End-to-end baseline analysis (eval + analyze + plot)
  grid_search.py         Hyper-parameter sweep runner (N processes at a time)
  push_weights.py        Upload checkpoints to Hugging Face Hub
  pull_weights.py        Download checkpoints from Hugging Face Hub
  start_dashboard.sh     Start the dashboard in the background
  train_daemon.sh        Daemonized training that auto-resumes on crash
configs/
  ppo.yaml               Default PPO training configuration
  dqn.yaml               Default DQN training configuration
  baselines.yaml         Baseline policies for evaluation
tests/                   pytest suite (fakes.py is an emulator-free fake environment)
dashboard/
  server/                FastAPI app: runs, results, training, play, grid search APIs
  templates/             Jinja2 pages
  static/                CSS
data/
  runs/<run_id>/        One training run: config, metrics.jsonl, episodes.json,
                        checkpoints/, plots/            (gitignored)
  results/              baselines.json + statistics.json (gitignored)
  plots/                baseline_<ts>/ and compare/ PNGs (tracked in git)
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
- Milestone 2 (PPO): done. PPO trains and clears the level (82% sampled /
  100% greedy completion), via `src/algorithms/ppo.py` + `src/training/trainer.py`,
  run with `./run.sh train`
- Milestone 3 (DQN): implemented and verified end to end, via
  `src/algorithms/dqn.py` + `src/training/dqn_trainer.py`, run with
  `./run.sh train_dqn`. The implementation is complete (replay buffer, target
  network, epsilon annealing, checkpoint/resume, evaluation harness, dashboard
  integration) and the full test suite passes. It has **not** been trained to
  completion yet, so there are no DQN results to report: a full 300k-step run
  takes roughly 6 hours on CPU at ~12-20 steps/s.

## Sharing model weights

Checkpoints live in `data/runs/<run_id>/checkpoints/` (gitignored, `*.pt`). To share them,
publish to Hugging Face Hub:

```bash
python scripts/push_weights.py --repo-id <user>/mario-ppo --run-id <run_id>
python scripts/pull_weights.py --repo-id <user>/mario-ppo --run-id <run_id>
```

Hub layout: `<run_id>/config.yaml`, `<run_id>/run.json`, `<run_id>/checkpoints/best.pt`.
By default only the **top 10 runs by best distance** are pushed, and for each run
only `best.pt` (+ `latest.pt` while training is unfinished). Pass `--include-top-k`
to also push the top-3 periodic checkpoints, and `--max-runs` to change the cap.
