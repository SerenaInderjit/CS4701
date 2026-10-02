import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd


RAW_FILE = "data/results/baselines.json"
STATS_FILE = "data/results/statistics.json"
PLOTS_DIR = "data/plots"

EPISODE_COLUMNS = {"policy", "episode", "max_x_pos"}
STATS_COLUMNS = {"policy", "mean_reward", "mean_distance", "mean_episode_length", "completion_rate"}


def load_data(raw_file=RAW_FILE, stats_file=STATS_FILE):
    try:
        with open(raw_file, "r") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        sys.exit(f"Error: {raw_file} not found. Run `bash run.sh eval` first.")
    except json.JSONDecodeError:
        sys.exit(f"Error: {raw_file} is not valid JSON.")

    if not os.path.exists(stats_file):
        sys.exit(f"Error: {stats_file} not found. Run `python scripts/analyze_results.py` first.")
    try:
        statistics = pd.read_json(stats_file)
    except ValueError:
        sys.exit(f"Error: {stats_file} is not valid JSON.")

    episodes = pd.DataFrame(raw_data)
    if episodes.empty:
        sys.exit(f"Error: {raw_file} contains no episodes.")
    missing = EPISODE_COLUMNS - set(episodes.columns)
    if missing:
        sys.exit(
            f"Error: {raw_file} is missing columns {sorted(missing)}. "
            "It may be in the old format — re-run `bash run.sh eval` to regenerate it."
        )
    missing = STATS_COLUMNS - set(statistics.columns)
    if missing:
        sys.exit(
            f"Error: {stats_file} is missing columns {sorted(missing)}. "
            "Re-run `python scripts/analyze_results.py` to regenerate it."
        )

    return episodes, statistics


def plot_summary(statistics, plots_dir=PLOTS_DIR):
    os.makedirs(plots_dir, exist_ok=True)

    statistics.plot(
        x="policy",
        y="mean_reward",
        kind="bar",
        legend=False,
    )
    plt.ylabel("Mean Reward")
    plt.title("Mean Reward by Policy")
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/mean_reward.png")
    plt.close()

    statistics.plot(
        x="policy",
        y="mean_distance",
        kind="bar",
        legend=False,
    )
    plt.ylabel("Mean Distance")
    plt.title("Mean Distance by Policy")
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/mean_distance.png")
    plt.close()

    statistics.plot(
        x="policy",
        y="mean_episode_length",
        kind="bar",
        legend=False,
    )
    plt.ylabel("Mean Episode Length")
    plt.title("Mean Episode Length by Policy")
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/mean_episode_length.png")
    plt.close()

    statistics.plot(
        x="policy",
        y="completion_rate",
        kind="bar",
        legend=False,
    )
    plt.ylabel("Completion Rate")
    plt.title("Completion Rate by Policy")
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/completion_rate.png")
    plt.close()


def plot_distance_by_episode(episodes, plots_dir=PLOTS_DIR):
    for policy in episodes["policy"].unique():
        policy_data = episodes[episodes["policy"] == policy]

        plt.plot(
            policy_data["episode"],
            policy_data["max_x_pos"],
            label=policy,
        )

    plt.xlabel("Episode")
    plt.ylabel("Maximum X Position")
    plt.title("Distance by Episode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/distance_by_episode.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=RAW_FILE)
    parser.add_argument("--stats", default=STATS_FILE)
    parser.add_argument("--plots-dir", default=PLOTS_DIR)
    args = parser.parse_args()

    raw_file, stats_file, plots_dir = args.raw, args.stats, args.plots_dir

    os.makedirs(plots_dir, exist_ok=True)

    episodes, statistics = load_data(raw_file, stats_file)

    plot_summary(statistics, plots_dir)
    plot_distance_by_episode(episodes, plots_dir)

    print(f"Plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()