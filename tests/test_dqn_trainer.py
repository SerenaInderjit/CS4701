import json
import os

import numpy as np
import pytest

from src.algorithms.dqn import DQN
from src.training.dqn_trainer import DQNTrainer
from src.training.replay_buffer import ReplayBuffer
from tests.fakes import FakeEnvironment

OBS_SHAPE = (4, 40, 40)


def make_trainer(tmp_path, checkpoint_every=10, train_start=4, log_every=5, eval_env=None,
                 run_dir=None, eval_every=0, eval_episodes=5):
    env = FakeEnvironment(episode_length=5, observation_shape=OBS_SHAPE)
    agent = DQN(
        observation_shape=OBS_SHAPE,
        num_actions=7,
        batch_size=4,
        hidden_dim=32,
        epsilon_decay_steps=50,
        device="cpu",
    )
    buffer = ReplayBuffer(capacity=64, observation_shape=OBS_SHAPE)
    trainer = DQNTrainer(
        env=env,
        agent=agent,
        buffer=buffer,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        checkpoint_every=checkpoint_every,
        train_start=train_start,
        log_every=log_every,
        eval_every=eval_every,
        eval_episodes=eval_episodes,
        eval_env=eval_env,
        run_dir=str(run_dir) if run_dir else None,
    )
    return trainer


def test_train_fills_the_replay_buffer(tmp_path):
    trainer = make_trainer(tmp_path)

    trainer.train(20)

    assert len(trainer.buffer) == 20


def test_train_performs_gradient_updates(tmp_path):
    trainer = make_trainer(tmp_path)

    trainer.train(20)

    assert trainer.agent.updates > 0


def test_burn_in_delays_the_first_gradient_step(tmp_path):
    trainer = make_trainer(tmp_path, train_start=1000)

    trainer.train(20)

    # Everything was collected, but nothing was learned yet.
    assert len(trainer.buffer) == 20
    assert trainer.agent.updates == 0


def test_epsilon_is_annealed_once_per_environment_step(tmp_path):
    trainer = make_trainer(tmp_path)

    trainer.train(10)

    assert trainer.agent.epsilon_steps == 10


def test_train_writes_run_tracking_files(tmp_path):
    run_dir = tmp_path / "run"
    trainer = make_trainer(tmp_path, run_dir=run_dir)

    trainer.train(10)

    lines = (run_dir / "metrics.jsonl").read_text().strip().splitlines()
    # log_every=5 over 10 steps -> logs at step 5 and step 10.
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["step"] == 5
    assert "loss" in first and "td_error" in first
    assert "epsilon" in first
    assert (run_dir / "episodes.json").exists()


def test_train_saves_checkpoints(tmp_path):
    trainer = make_trainer(tmp_path, checkpoint_every=10)

    trainer.train(20)

    saved = [os.path.basename(p) for p in trainer.checkpoints._periodic_paths()]
    assert len(saved) == 2


def test_resume_restores_the_agent(tmp_path):
    checkpoint_dir = str(tmp_path / "ckpts")
    trainer = make_trainer(tmp_path, checkpoint_every=5)
    trainer.checkpoints.directory = checkpoint_dir
    trainer.train(10)
    updates_before = trainer.agent.updates

    env = FakeEnvironment(episode_length=5, observation_shape=OBS_SHAPE)
    agent = DQN(
        observation_shape=OBS_SHAPE,
        num_actions=7,
        batch_size=4,
        hidden_dim=32,
        epsilon_decay_steps=50,
        device="cpu",
    )
    buffer = ReplayBuffer(capacity=64, observation_shape=OBS_SHAPE)
    resumed = DQNTrainer(
        env=env,
        agent=agent,
        buffer=buffer,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=5,
        train_start=4,
    )

    step = resumed.resume()

    assert step == 10
    assert resumed.agent.updates == updates_before
    assert resumed.agent.epsilon_steps == trainer.agent.epsilon_steps


def test_resume_without_checkpoint_raises(tmp_path):
    trainer = make_trainer(tmp_path)

    with pytest.raises(FileNotFoundError):
        trainer.resume()


def test_evaluation_runs_greedy_episodes(tmp_path):
    eval_env = FakeEnvironment(episode_length=5, observation_shape=OBS_SHAPE)
    trainer = make_trainer(tmp_path, eval_env=eval_env)

    result = trainer._evaluate()

    assert result is not None
    assert result["num_episodes"] == 5
    assert "mean_distance" in result


def test_a_step_that_both_logs_and_evaluates_is_written_once(tmp_path):
    run_dir = tmp_path / "run"
    eval_env = FakeEnvironment(episode_length=5, observation_shape=OBS_SHAPE)
    trainer = make_trainer(
        tmp_path, log_every=5, eval_every=5, eval_env=eval_env, run_dir=run_dir
    )

    trainer.train(10)

    lines = (run_dir / "metrics.jsonl").read_text().strip().splitlines()
    steps = [json.loads(line)["step"] for line in lines]
    assert steps == sorted(steps), "metric rows must be in increasing step order"
    assert len(steps) == len(set(steps)), f"duplicate metric rows for steps {steps}"


def test_only_the_evaluated_row_carries_the_evaluation_numbers(tmp_path):
    run_dir = tmp_path / "run"
    eval_env = FakeEnvironment(episode_length=5, observation_shape=OBS_SHAPE)
    trainer = make_trainer(
        tmp_path, log_every=5, eval_every=10, eval_env=eval_env, run_dir=run_dir
    )

    trainer.train(20)

    records = [json.loads(line) for line in
               (run_dir / "metrics.jsonl").read_text().strip().splitlines()]
    with_eval = [r for r in records if r.get("eval_mean_distance") is not None]
    # eval_every=10 over 20 steps evaluates at 10 and 20; only those rows
    # carry the evaluation numbers.
    assert [r["step"] for r in with_eval] == [10, 20]
    assert [r["step"] for r in records] == [5, 10, 15, 20]
