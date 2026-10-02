import json
import sys

import pandas as pd


INPUT_FILE = "results/baselines.json"
OUTPUT_FILE = "results/statistics.json"

REQUIRED_COLUMNS = {"policy", "episode", "total_reward", "max_x_pos", "flag_get", "length"}


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
    try:
        df = load_results(INPUT_FILE)
    except FileNotFoundError:
        sys.exit(f"Error: {INPUT_FILE} not found. Run `bash run.sh eval` first.")
    except json.JSONDecodeError:
        sys.exit(f"Error: {INPUT_FILE} is not valid JSON.")

    if df.empty:
        sys.exit(f"Error: {INPUT_FILE} contains no episodes.")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        sys.exit(
            f"Error: {INPUT_FILE} is missing columns {sorted(missing)}. "
            "It may be in the old format — re-run `bash run.sh eval` to regenerate it."
        )

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