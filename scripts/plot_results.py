import argparse
import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd


RAW_FILE = "data/results/baselines.json"
STATS_FILE = "data/results/statistics.json"
PLOTS_DIR = None  # defaults to data/plots/baseline_<timestamp> in main()

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


def _bar_with_labels(statistics, y, ylabel, title, out, fmt):
    ax = statistics.plot(x="policy", y=y, kind="bar", legend=False)
    plt.ylabel(ylabel)
    plt.title(title)
    for rect, val in zip(ax.patches, statistics[y]):
        ax.annotate(fmt.format(val), (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_summary(statistics, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)

    _bar_with_labels(statistics, "mean_reward", "Mean Reward", "Mean Reward by Policy",
                     f"{plots_dir}/mean_reward.png", "{:.1f}")
    _bar_with_labels(statistics, "mean_distance", "Mean Distance", "Mean Distance by Policy",
                     f"{plots_dir}/mean_distance.png", "{:.1f}")
    _bar_with_labels(statistics, "mean_episode_length", "Mean Episode Length", "Mean Episode Length by Policy",
                     f"{plots_dir}/mean_episode_length.png", "{:.1f}")
    _bar_with_labels(statistics, "completion_rate", "Completion Rate", "Completion Rate by Policy",
                     f"{plots_dir}/completion_rate.png", "{:.0%}")


def plot_distance_by_episode(episodes, plots_dir):
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
    parser.add_argument("--plots-dir", default=None)
    args = parser.parse_args()

    raw_file, stats_file = args.raw, args.stats
    plots_dir = args.plots_dir or os.path.join(
        "data", "plots", f"baseline_{datetime.now():%Y%m%d_%H%M%S}"
    )

    os.makedirs(plots_dir, exist_ok=True)

    episodes, statistics = load_data(raw_file, stats_file)

    plot_summary(statistics, plots_dir)
    plot_distance_by_episode(episodes, plots_dir)

    print(f"Plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()