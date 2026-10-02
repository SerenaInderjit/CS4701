"""Watch a policy play Mario.

Usage: ./run.sh play  (or: python scripts/play_mario.py --policy right --episodes 3)

--policy is 'random', 'right', or 'ppo'. 'ppo' loads a trained checkpoint
(produced by scripts/train.py) and needs the preprocessed training
observations, so it runs in a make_training_environment instead of the raw
one the baselines use.

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
from src.training.logger import Logger
from src.training.runner import Runner


def make_policy(name, num_actions, checkpoint=None, device="cpu"):
    if name == "ppo":
        model = ConvolutionalNeuralNetwork(output_dim=num_actions)
        return PPOPolicy(model, checkpoint_path=checkpoint, device=device)
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
    parser.add_argument("--fps", type=int, default=15,
                        help="agent steps per second (15 ≈ real-time for frame_skip=4)")
    parser.add_argument("--save-frames", type=str, default=None,
                        help="save rendered frames as PNGs to this directory")
    args = parser.parse_args()

    if args.policy == "ppo" and args.checkpoint is None:
        parser.error("--policy ppo requires --checkpoint")

    # PPO acts on preprocessed (4, 84, 84) stacks, so it needs the training env.
    if args.policy == "ppo":
        environment = make_training_environment()
    else:
        environment = MarioEnvironment()

    policy = make_policy(args.policy, environment.num_actions, args.checkpoint, args.device)

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
