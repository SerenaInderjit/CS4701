import json
import os

import numpy as np
import pytest
import torch

from src.training.rollout_buffer import RolloutBuffer
from src.training.trainer import Trainer
from tests.fakes import FakeEnvironment


class FakeAgent:
    """Implements the PPO interface the Trainer relies on, without a CNN."""

    def __init__(self, num_actions=3):
        self.num_actions = num_actions
        self.update_calls = 0
        # The Trainer checkpoints the policy model/optimizer, so use real
        # (tiny) torch objects to exercise the real save path.
        self.policy_model = torch.nn.Linear(4, num_actions)
        self.policy_optimizer = torch.optim.Adam(self.policy_model.parameters())

    def act(self, observation):
        return 0, {"log_prob": -1.0, "value": 0.5}

    def value_of(self, observation):
        return 0.5

    def compute_gae(self, rewards, dones, values, last_value):
        return rewards.copy(), rewards + values

    def update(self, observations, actions, old_log_probs, advantages, returns):
        self.update_calls += 1
        return {"policy_loss": 0.1, "value_loss": 0.2}

    def state_dicts(self):
        return {}

    def load_state_dicts(self, state):
        pass


def make_trainer(tmp_path, updates_buffer=20, checkpoint_every=1):
    env = FakeEnvironment(episode_length=5, observation_shape=(4, 8, 8))
    agent = FakeAgent()
    buffer = RolloutBuffer(capacity=updates_buffer, observation_shape=(4, 8, 8))
    trainer = Trainer(
        env=env,
        agent=agent,
        buffer=buffer,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        checkpoint_every=checkpoint_every,
    )
    return trainer


def test_update_collects_rollout_and_clears_buffer(tmp_path):
    trainer = make_trainer(tmp_path)

    trainer._update()

    assert trainer.agent.update_calls == 1
    assert len(trainer.buffer) == 0  # cleared for the next rollout


def test_train_runs_requested_updates(tmp_path):
    trainer = make_trainer(tmp_path)

    trainer.train(3)

    assert trainer.agent.update_calls == 3


def test_train_saves_checkpoint_every_and_final(tmp_path):
    trainer = make_trainer(tmp_path, checkpoint_every=2)

    trainer.train(3)

    checkpoints = trainer.checkpoints
    assert checkpoints.latest_path() is not None
    # update 2 (periodic) and update 3 (final) are saved.
    saved = [os.path.basename(p) for p in checkpoints._periodic_paths()]
    assert len(saved) == 2


def test_train_carries_episode_across_rollouts(tmp_path):
    # episode_length=5 with capacity=20: each rollout spans multiple episodes
    # and the in-progress episode is carried over, so no reset is wasted.
    trainer = make_trainer(tmp_path)

    trainer.train(2)

    # 2 rollouts x 20 steps = 40 steps; episodes are 5 steps each, so at most
    # 8 episodes plus at most 1 carried-over partial episode.
    assert trainer.env.num_resets <= 9


def test_train_writes_run_tracking_files(tmp_path):
    run_dir = tmp_path / "run"
    env = FakeEnvironment(episode_length=5, observation_shape=(4, 8, 8))
    agent = FakeAgent()
    buffer = RolloutBuffer(capacity=20, observation_shape=(4, 8, 8))
    trainer = Trainer(env, agent, buffer, checkpoint_dir=str(tmp_path / "ckpts"),
                      checkpoint_every=1, run_dir=str(run_dir))

    trainer.train(2)

    metrics_lines = (run_dir / "metrics.jsonl").read_text().strip().splitlines()
    assert len(metrics_lines) == 2
    first = json.loads(metrics_lines[0])
    assert first["update"] == 1
    assert "policy_loss" in first and "value_loss" in first
    assert (run_dir / "episodes.json").exists()


def test_resume_restores_latest_checkpoint(tmp_path):
    checkpoint_dir = str(tmp_path / "ckpts")
    trainer = make_trainer(tmp_path, checkpoint_every=1)
    trainer.checkpoints.directory = checkpoint_dir
    trainer.train(2)  # saves checkpoints at updates 1 and 2

    # A fresh trainer on the same checkpoint dir resumes from update 2.
    env = FakeEnvironment(episode_length=5, observation_shape=(4, 8, 8))
    agent = FakeAgent()
    buffer = RolloutBuffer(capacity=20, observation_shape=(4, 8, 8))
    resumed = Trainer(env, agent, buffer, checkpoint_dir=checkpoint_dir, checkpoint_every=1)

    step = resumed.resume()

    assert step == 2
    resumed.train(3, start=step + 1)
    assert agent.update_calls == 1  # only update 3 ran


def test_resume_without_checkpoint_raises(tmp_path):
    trainer = make_trainer(tmp_path)  # nothing saved yet

    with pytest.raises(FileNotFoundError):
        trainer.resume()
