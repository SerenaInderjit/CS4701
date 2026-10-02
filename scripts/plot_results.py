import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd


RAW_FILE = "results/baselines.json"
STATS_FILE = "results/statistics.json"
PLOTS_DIR = "plots"

EPISODE_COLUMNS = {"policy", "episode", "max_x_pos"}
STATS_COLUMNS = {"policy", "mean_reward", "mean_distance", "mean_episode_length", "completion_rate"}


def load_data():
    try:
        with open(RAW_FILE, "r") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        sys.exit(f"Error: {RAW_FILE} not found. Run `bash run.sh eval` first.")
    except json.JSONDecodeError:
        sys.exit(f"Error: {RAW_FILE} is not valid JSON.")

    if not os.path.exists(STATS_FILE):
        sys.exit(f"Error: {STATS_FILE} not found. Run `python scripts/analyze_results.py` first.")
    try:
        statistics = pd.read_json(STATS_FILE)
    except ValueError:
        sys.exit(f"Error: {STATS_FILE} is not valid JSON.")

    episodes = pd.DataFrame(raw_data)
    if episodes.empty:
        sys.exit(f"Error: {RAW_FILE} contains no episodes.")
    missing = EPISODE_COLUMNS - set(episodes.columns)
    if missing:
        sys.exit(
            f"Error: {RAW_FILE} is missing columns {sorted(missing)}. "
            "It may be in the old format — re-run `bash run.sh eval` to regenerate it."
        )
    missing = STATS_COLUMNS - set(statistics.columns)
    if missing:
        sys.exit(
            f"Error: {STATS_FILE} is missing columns {sorted(missing)}. "
            "Re-run `python scripts/analyze_results.py` to regenerate it."
        )

    return episodes, statistics


def plot_summary(statistics):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    statistics.plot(
        x="policy",
        y="mean_reward",
        kind="bar",
        legend=False,
    )
    plt.ylabel("Mean Reward")
    plt.title("Mean Reward by Policy")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/mean_reward.png")
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
    plt.savefig(f"{PLOTS_DIR}/mean_distance.png")
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
    plt.savefig(f"{PLOTS_DIR}/mean_episode_length.png")
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
    plt.savefig(f"{PLOTS_DIR}/completion_rate.png")
    plt.close()


def plot_distance_by_episode(episodes):
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
    plt.savefig(f"{PLOTS_DIR}/distance_by_episode.png")
    plt.close()


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    episodes, statistics = load_data()

    plot_summary(statistics)
    plot_distance_by_episode(episodes)

    print(f"Plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()