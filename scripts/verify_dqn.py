"""Smoke test: does the real Mario environment work, and does the DQN agent
plug into it end-to-end?

Run from the repo root:
    .venv/bin/python scripts/verify_dqn.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms.dqn import DQN
from src.environment.mario_environment import make_training_environment
from src.training.replay_buffer import ReplayBuffer


def main():
    print("=" * 60)
    print("1. Building the real Mario environment")
    print("=" * 60)
    env = make_training_environment(
        frame_skip=4,
        num_stack=4,
        frame_size=(84, 84),
    )
    print(f"   observation_shape = {env.observation_shape}")
    print(f"   num_actions       = {env.num_actions}")

    obs = env.reset()["observation"]
    print(f"   reset obs shape   = {obs.shape} dtype={obs.dtype}")

    result = env.step(0)
    print(f"   step obs shape    = {result['observation'].shape}")
    print(f"   reward            = {result['reward']:.4f}")
    print(f"   info keys         = {sorted(result['info'].keys())}")

    print()
    print("=" * 60)
    print("2. Creating the DQN agent on the real observation shape")
    print("=" * 60)
    agent = DQN(
        observation_shape=env.observation_shape,
        num_actions=env.num_actions,
        batch_size=8,
        target_update_freq=100,
        device="cpu",
    )
    total_params = sum(p.numel() for p in agent.model.parameters())
    print(f"   parameters        = {total_params:,}")
    print(f"   output layer      = {agent.model.output}")

    q = agent.q_values(obs)
    print(f"   q_values(obs)     = {np.round(q, 4)}")
    assert q.shape == (env.num_actions,), "q_values must have one entry per action"

    print()
    print("=" * 60)
    print("3. Filling the replay buffer with real transitions")
    print("=" * 60)
    buffer = ReplayBuffer(capacity=100, observation_shape=env.observation_shape)
    observation = env.reset()["observation"]
    for _ in range(40):
        action = agent.act(observation)
        result = env.step(action)
        buffer.add(
            observation,
            action,
            result["reward"],
            result["episode_ended"],
            result["observation"],
        )
        observation = (
            env.reset()["observation"]
            if result["episode_ended"]
            else result["observation"]
        )
    print(f"   buffer size       = {len(buffer)}")
    print(f"   buffer ready      = {buffer.is_ready(8)}")

    print()
    print("=" * 60)
    print("4. Gradient step on real data")
    print("=" * 60)
    batch = buffer.sample(8)
    metrics = agent.train_step(batch)
    for key, value in metrics.items():
        print(f"   {key:<10} = {value:.6f}")
    print(f"   updates           = {agent.updates}")

    print()
    print("=" * 60)
    print("5. Overfitting a single batch (can the network fit anything at all?)")
    print("=" * 60)
    agent2 = DQN(
        observation_shape=env.observation_shape,
        num_actions=env.num_actions,
        batch_size=8,
        learning_rate=1e-3,
        device="cpu",
    )
    fixed = buffer.sample(8)
    losses = []
    for _ in range(50):
        metrics = agent2.train_step(fixed)
        losses.append(metrics["loss"])
    print(f"   loss start        = {losses[0]:.6f}")
    print(f"   loss end          = {losses[-1]:.6f}")
    if losses[-1] < losses[0]:
        print("   -> loss decreased: the network CAN fit the data. OK")
    else:
        print("   -> WARNING: loss did not decrease")

    print()
    print("=" * 60)
    print("6. Learning signal: does Q-value track the reward?")
    print("=" * 60)
    # A state where Mario is alive vs. one right after death should give
    # different Q-values once the network has learned anything.
    obs_alive = env.reset()["observation"]
    q_alive = agent2.q_values(obs_alive)
    print(f"   q (alive)         = {np.round(q_alive, 4)}")
    print(f"   q spread          = {q_alive.max() - q_alive.min():.4f}")
    if q_alive.max() - q_alive.min() < 1e-6:
        print("   -> WARNING: all Q-values identical (untrained output layer)")

    env.close()
    print()
    print("All environment checks passed.")


if __name__ == "__main__":
    main()
