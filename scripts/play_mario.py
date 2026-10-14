"""Watch a policy play Mario.

Usage: ./run.sh play  (or: python scripts/play_mario.py --policy right --episodes 3)

--policy is 'random', 'right', or 'ppo'. 'ppo' loads a trained checkpoint
(produced by scripts/train.py) and needs the preprocessed training
observations, so it runs in a make_training_environment instead of the raw
one the baselines use. The network architecture and frame settings are read
from the config.yaml of the run the checkpoint belongs to
(<run>/checkpoints/best.pt -> <run>/config.yaml).

--greedy plays the most likely action instead of sampling.

--fps controls playback speed (agent steps per second). Default 15 is
approximately real-time for frame_skip=4 on the NES (60 FPS / 4).

--save-frames saves each rendered frame as a PNG to the given directory
(used by the dashboard live view).
"""
import argparse
import os
import time

from src.algorithms.cnn import ConvolutionalNeuralNetwork
from src.environment.mario_environment import MarioEnvironment, make_training_environment
from src.policies.baselines import ConstantPolicy, RandomPolicy
from src.policies.ppo_policy import PPOPolicy
from src.training.config import load_config
from src.training.logger import Logger
from src.training.runner import Runner


def run_config_for(checkpoint):
    """config.yaml of the run a checkpoint belongs to, or {} if not found."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(checkpoint))), "config.yaml")
    return load_config(path) if os.path.exists(path) else {}


def make_policy(name, num_actions, checkpoint=None, device="cpu", config=None, deterministic=False):
    if name == "ppo":
        config = config or {}
        model = ConvolutionalNeuralNetwork(
            output_dim=num_actions,
            conv_channels=tuple(config.get("conv_channels", [64, 128, 128])),
            hidden_dim=config.get("hidden_dim", 1024),
            frame_size=config.get("frame_size", 84),
        )
        return PPOPolicy(model, checkpoint_path=checkpoint, device=device, deterministic=deterministic)
    if name == "random":
        return RandomPolicy(num_actions=num_actions)
    return ConstantPolicy(action=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["random", "right", "ppo"], default="random")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="trained policy checkpoint (required for --policy ppo)")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=2000, help="max steps per episode")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--greedy", action="store_true", help="play argmax actions instead of sampling")
    parser.add_argument("--fps", type=int, default=15,
                        help="agent steps per second (15 ≈ real-time for frame_skip=4)")
    parser.add_argument("--save-frames", type=str, default=None,
                        help="save rendered frames as PNGs to this directory")
    args = parser.parse_args()

    if args.policy == "ppo" and args.checkpoint is None:
        parser.error("--policy ppo requires --checkpoint")

    # PPO acts on preprocessed stacks, so it needs the training env, built with
    # the same frame settings the checkpoint was trained with.
    config = {}
    if args.policy == "ppo":
        config = run_config_for(args.checkpoint)
        environment = make_training_environment(
            frame_skip=config.get("frame_skip", 4),
            num_stack=config.get("num_stack", 4),
            frame_size=(config.get("frame_size", 84), config.get("frame_size", 84)),
        )
    else:
        environment = MarioEnvironment()

    policy = make_policy(args.policy, environment.num_actions, args.checkpoint, args.device,
                         config=config, deterministic=args.greedy)

    frame_dir = None
    if args.save_frames:
        os.makedirs(args.save_frames, exist_ok=True)
        frame_dir = args.save_frames

    logger = Logger()
    step_delay = 1.0 / args.fps if args.fps > 0 else 0.0

    frame_count = 0

    def save_frame():
        nonlocal frame_count
        if frame_dir is None:
            return
        try:
            frame = environment.env.render(mode="rgb_array")
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.imsave(os.path.join(frame_dir, f"frame_{frame_count:06d}.png"), frame)
            frame_count += 1
        except Exception:
            pass

    runner = Runner(environment, policy, logger, render=True, on_step=save_frame)

    try:
        for _ in range(args.episodes):
            runner.run_episode(timeout=args.timeout)
            if step_delay > 0:
                time.sleep(step_delay)
    finally:
        environment.close()

    for episode in logger.episodes:
        print(
            f"episode {episode['episode']}: distance={episode['max_x_pos']} "
            f"steps={episode['length']} reward={episode['total_reward']:.1f}"
        )


if __name__ == "__main__":
    main()
