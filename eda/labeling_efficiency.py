import os
import pandas as pd
from test import trading
import config


def simulate_max_gain_on_folder(folder):
    all_results = []

    csv_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")]
    for file in csv_files:
        df = pd.read_csv(file)

        timestamps = df["Date"].values
        closes = df["Close"].values
        predictions = df["Label"].astype(int).tolist()

        result, trades_df = trading(timestamps, closes, predictions)
        print(f"\n{os.path.basename(file)}")
        for k, v in result.items():
            print(f"{k}: {v:.4f}")
        all_results.append(result)

    if all_results:
        print("\nAverage Max Gain Over Test Set:")
        keys = all_results[0].keys()
        avg_results = {k: sum(d[k] for d in all_results) / len(all_results) for k in keys}
        for k, v in avg_results.items():
            print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    simulate_max_gain_on_folder(config.test_dir)
