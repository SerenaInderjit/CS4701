"""Train PPO on Super Mario Bros 1-1.

Usage: ./run.sh train  (or: python scripts/train.py --config configs/ppo.yaml)

Hyperparameters come from the YAML config file; any of them can be overridden
from the command line, e.g. ./run.sh train --updates 500 --device cpu.
Each run writes its resolved config, per-update metrics and episode log to
runs/ppo_<timestamp>/ (gitignored).
"""
import argparse
import json
import logging
import os
from datetime import datetime

from src.algorithms.ppo import PPO
from src.environment.mario_environment import make_training_environment
from src.training.config import load_config, save_config
from src.training.device import resolve_device
from src.training.reproducibility import git_commit_hash, seed_everything
from src.training.rollout_buffer import RolloutBuffer
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train a PPO agent on Super Mario Bros 1-1")
    parser.add_argument("--config", type=str, default="configs/ppo.yaml")

    # Defaults are None so an explicit CLI arg always wins over the config file.
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--rollout-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=None, help="PPO clip range")
    parser.add_argument("--epochs", type=int, default=None, help="optimization epochs per rollout")
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--frame-skip", type=int, default=None)
    parser.add_argument("--num-stack", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="continue from the latest checkpoint in --checkpoint-dir")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="where to write config/metrics (default runs/ppo_<timestamp>)")
    args = parser.parse_args()

    config = load_config(args.config)
    overrides = {k: v for k, v in vars(args).items()
                 if v is not None and k not in ("config", "resume", "run_dir")}
    config.update(overrides)
    config["device"] = resolve_device(config.get("device", "auto"))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    seed_everything(config["seed"])

    run_dir = args.run_dir or os.path.join("runs", f"ppo_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(run_dir, exist_ok=True)
    save_config(config, os.path.join(run_dir, "config.yaml"))
    with open(os.path.join(run_dir, "run.json"), "w") as f:
        json.dump({
            "git_commit": git_commit_hash(),
            "seed": config["seed"],
            "started_at": datetime.now().isoformat(),
        }, f, indent=2)
    logging.info(f"Run directory: {run_dir}")

    env = make_training_environment(
        frame_skip=config["frame_skip"],
        num_stack=config["num_stack"],
        frame_size=(config["frame_size"], config["frame_size"]),
    )

    agent = PPO(
        observation_shape=env.observation_shape,
        num_actions=env.num_actions,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        epsilon=config["epsilon"],
        epochs=config["epochs"],
        minibatch_size=config["minibatch_size"],
        device=config["device"],
        conv_channels=tuple(config.get("conv_channels", [64, 128, 128])),
        hidden_dim=config.get("hidden_dim", 1024),
    )

    buffer = RolloutBuffer(
        capacity=config["rollout_size"],
        observation_shape=env.observation_shape,
    )

    trainer = Trainer(
        env=env,
        agent=agent,
        buffer=buffer,
        checkpoint_dir=config["checkpoint_dir"],
        checkpoint_every=config["checkpoint_every"],
        run_dir=run_dir,
    )

    try:
        start = 1
        if args.resume:
            start = trainer.resume() + 1
            if start > config["updates"]:
                logging.info("Already trained past --updates; nothing to do")
                return
        trainer.train(config["updates"], start=start)
    finally:
        env.close()


if __name__ == "__main__":
    main()
