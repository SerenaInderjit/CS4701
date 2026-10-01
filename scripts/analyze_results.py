import json

import pandas as pd


INPUT_FILE = "results/baselines.json"
OUTPUT_FILE = "results/statistics.json"


def load_results(path):
    with open(path, "r") as f:
        data = json.load(f)

    return pd.DataFrame(data)


def calculate_statistics(df):
    return (
        df.groupby("policy")
        .agg(
            num_episodes=("episode", "count"),
            mean_reward=("total_reward", "mean"),
            mean_distance=("max_x_pos", "mean"),
            best_distance=("max_x_pos", "max"),
            completion_rate=("flag_get", "mean"),
            mean_episode_length=("length", "mean"),
        )
        .reset_index()
    )


def main():
    df = load_results(INPUT_FILE)

    statistics = calculate_statistics(df)

    print("\nEpisode data:")
    print(df)

    print("\nStatistics:")
    print(statistics.to_string(index=False))

    statistics.to_json(
        OUTPUT_FILE,
        orient="records",
        indent=2,
    )

    print(f"\nSaved statistics to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()