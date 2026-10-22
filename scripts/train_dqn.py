"""Train DQN on Super Mario Bros 1-1.

Usage: ./run.sh train_dqn  (or: python scripts/train_dqn.py --config configs/dqn.yaml)

Hyperparameters come from the YAML config file; any of them can be overridden
from the command line, e.g. ./run.sh train_dqn --total-steps 100000 --device cpu.
Each run writes its resolved config, per-interval metrics and episode log to
data/runs/dqn_<timestamp>/ (gitignored).

DQN is off-policy and does one gradient step per environment step, so it is
much slower per step than PPO. The log line reports the throughput in
steps/s; use that to decide how many --total-steps you can afford.
"""
import argparse
import json
import logging
import os
from datetime import datetime

from src.algorithms.dqn import DQN
from src.environment.mario_environment import make_training_environment
from src.environment.reward import reward_config_from_dict
from src.paths import RUNS_DIR
from src.training.config import load_config, save_config
from src.training.config import git_commit_hash, resolve_device, seed_everything
from src.training.dqn_trainer import DQNTrainer
from src.training.replay_buffer import ReplayBuffer


def main():
    parser = argparse.ArgumentParser(description="Train a DQN agent on Super Mario Bros 1-1")
    parser.add_argument("--config", type=str, default="configs/dqn.yaml")

    # Defaults are None so an explicit CLI arg always wins over the config file.
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--target-update-freq", type=int, default=None)
    parser.add_argument("--epsilon-start", type=float, default=None)
    parser.add_argument("--epsilon-end", type=float, default=None)
    parser.add_argument("--epsilon-decay-steps", type=int, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--train-start", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="continue from the latest checkpoint in --checkpoint-dir")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="where to write config/metrics (default data/runs/dqn_<timestamp>)")
    args = parser.parse_args()

    config = load_config(args.config)
    overrides = {k: v for k, v in vars(args).items()
                 if v is not None and k not in ("config", "resume", "run_dir")}
    config.update(overrides)
    config["device"] = resolve_device(config.get("device", "auto"))
    config["algorithm"] = "dqn"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    seed_everything(config["seed"])

    run_dir = args.run_dir or os.path.join(RUNS_DIR, f"dqn_{datetime.now():%Y%m%d_%H%M%S}")
    checkpoint_dir = (
        args.checkpoint_dir
        or os.path.join(run_dir, config.get("checkpoint_dir", "checkpoints"))
    )
    config["checkpoint_dir"] = checkpoint_dir
    os.makedirs(run_dir, exist_ok=True)
    save_config(config, os.path.join(run_dir, "config.yaml"))
    with open(os.path.join(run_dir, "run.json"), "w") as f:
        json.dump({
            "git_commit": git_commit_hash(),
            "seed": config["seed"],
            "started_at": datetime.now().isoformat(),
        }, f, indent=2)
    logging.info(f"Run directory: {run_dir}")

    reward_config = reward_config_from_dict(config)

    env = make_training_environment(
        frame_skip=config["frame_skip"],
        num_stack=config["num_stack"],
        frame_size=(config["frame_size"], config["frame_size"]),
        reward_config=reward_config,
    )

    # A second environment for the periodic greedy evaluation, so evaluating
    # does not interrupt the episode the training loop is in the middle of.
    eval_env = None
    if config.get("eval_every", 0):
        eval_env = make_training_environment(
            frame_skip=config["frame_skip"],
            num_stack=config["num_stack"],
            frame_size=(config["frame_size"], config["frame_size"]),
            reward_config=reward_config,
        )

    agent = DQN(
        observation_shape=env.observation_shape,
        num_actions=env.num_actions,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        batch_size=config["batch_size"],
        target_update_freq=config["target_update_freq"],
        epsilon_start=config["epsilon_start"],
        epsilon_end=config["epsilon_end"],
        epsilon_decay_steps=config["epsilon_decay_steps"],
        conv_channels=tuple(config.get("conv_channels", [32, 64, 64])),
        hidden_dim=config.get("hidden_dim", 512),
        device=config["device"],
        max_grad_norm=config.get("max_grad_norm", 10.0),
    )

    buffer = ReplayBuffer(
        capacity=config["buffer_size"],
        observation_shape=env.observation_shape,
    )

    trainer = DQNTrainer(
        env=env,
        agent=agent,
        buffer=buffer,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=config["checkpoint_every"],
        run_dir=run_dir,
        lr_anneal=bool(config.get("lr_anneal", False)),
        train_start=config.get("train_start", 2000),
        log_every=config.get("log_every", 1000),
        eval_every=config.get("eval_every", 0),
        eval_episodes=config.get("eval_episodes", 3),
        eval_env=eval_env,
        eval_timeout=config.get("eval_timeout", 2000),
    )

    try:
        start = 1
        if args.resume:
            start = trainer.resume() + 1
            if start > config["total_steps"]:
                logging.info("Already trained past --total-steps; nothing to do")
                return
        trainer.train(config["total_steps"], start=start)
    finally:
        env.close()
        if eval_env is not None:
            eval_env.close()


if __name__ == "__main__":
    main()
