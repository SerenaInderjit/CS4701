import json
import os

import matplotlib.pyplot as plt
import pandas as pd


RAW_FILE = "results/baselines.json"
STATS_FILE = "results/statistics.json"
PLOTS_DIR = "plots"


def load_data():
    with open(RAW_FILE, "r") as f:
        raw_data = json.load(f)

    episodes = pd.DataFrame(raw_data)
    statistics = pd.read_json(STATS_FILE)

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