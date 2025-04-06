from datetime import datetime
import numpy as np
import pandas as pd


def trading(timestamps, closes, predictions, initial_fund=100, commission=0.01):
    # Convert UNIX milliseconds to datetime if needed
    if isinstance(timestamps[0], (int, float, np.integer, np.floating)):
        timestamps = [datetime.utcfromtimestamp(int(ts) / 1000) for ts in timestamps]

    trades = []
    capital = initial_fund
    shares = 0
    last_buy_price = None
    last_buy_date = None
    prev_label = None
    daily_values = []
    last_close = closes[-1]

    for ts, close, pred in zip(timestamps, closes, predictions):
        daily_values.append(capital if shares == 0 else capital + shares * close)

        if pred == prev_label:
            continue

        if pred == 1 and shares == 0:  # BUY
            num_shares = (capital - commission) / close
            shares = num_shares
            capital = 0
            last_buy_price = close
            last_buy_date = ts

        elif pred == 2 and shares > 0:  # SELL
            gross = shares * close
            net = gross - commission
            profit_pct = ((close - last_buy_price) / last_buy_price) * 100
            length = (ts - last_buy_date).days
            trades.append({
                'buy_date': last_buy_date,
                'sell_date': ts,
                'buy_price': last_buy_price,
                'sell_price': close,
                'profit_pct': profit_pct,
                'length': length
            })
            capital = net
            shares = 0
            last_buy_price = None
            last_buy_date = None

        prev_label = pred

    final_value = capital if shares == 0 else capital + shares * last_close
    buy_and_hold = (initial_fund / closes[0]) * closes[-1]

    # Trade stats
    trades_performed = len(trades)
    trades_won = sum(t['profit_pct'] > 0 for t in trades)
    win_rate = (trades_won / trades_performed) * 100 if trades_performed > 0 else 0
    avg_profit = np.mean([t['profit_pct'] for t in trades]) if trades else 0
    avg_length = np.mean([t['length'] for t in trades]) if trades else 0
    max_profit = max([t['profit_pct'] for t in trades], default=0)
    max_loss = min([t['profit_pct'] for t in trades], default=0)

    # Annualized return
    total_days = (timestamps[-1] - timestamps[0]).days
    ann_return = (final_value / initial_fund) ** (365 / total_days) - 1 if total_days > 0 else 0

    # Sharpe ratio (daily)
    equity_series = pd.Series(daily_values)
    daily_returns = equity_series.pct_change().fillna(0)
    sharpe_ratio = daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0

    # Idle ratio
    idle_ratio = predictions.count(0) / len(predictions) * 100

    # Collect results
    result = {
        'Final Value': final_value,
        'Buy & Hold': buy_and_hold,
        'Annualized Return (%)': ann_return * 100,
        'Sharpe Ratio (daily)': sharpe_ratio,
        'Trades Performed': trades_performed,
        'Trades Won': trades_won,
        'Percent of Success': win_rate,
        'Avg Profit per Trade (%)': avg_profit,
        'Avg Trade Length (days)': avg_length,
        'Max Profit/Trade (%)': max_profit,
        'Max Loss/Trade (%)': max_loss,
        'Idle Ratio (%)': idle_ratio
    }

    return result, pd.DataFrame(trades)


if __name__ == "__main__":
    import torch
    import config
    import os
    from model.focal_loss import FocalLoss
    from dataset import FinanceImageDataset
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = config.model(apply_bn=config.apply_bn, window_length=config.window_length).to(device)

    checkpoint_path = r"C:\Users\KemalUtkuLekesiz\Documents\Kod\Tez\CnnTezV3\pythonProject\checkpoints\DOW-2025_04_04_14_45_checkpoint_epoch_199.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    config.test_dir = r"C:\Users\KemalUtkuLekesiz\Documents\Kod\Tez\CnnTezV3\pythonProject\data\test\1d"
    test_files = [os.path.join(config.test_dir, f) for f in os.listdir(config.test_dir)]

    cumulative_result = {}
    num_tests = len(test_files)

    for tf in test_files:
        dataset = FinanceImageDataset(tf, config.indicators, index_dir=config.test_index_dir,
                                      use_index=config.use_index)
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

        all_predictions = []
        all_timestamps = []
        all_closes = []

        with torch.no_grad():
            for ts, closes, images, _ in loader:
                images = images.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=-1).cpu().numpy().flatten()
                all_predictions.extend(preds)
                all_timestamps.extend(ts.flatten().cpu().numpy())
                all_closes.extend(closes.flatten().cpu().numpy())

        result, trades_df = trading(all_timestamps, all_closes, all_predictions)
        print(f"\nResults for {os.path.basename(tf)}:")
        for key, value in result.items():
            print(f"{key}: {value:.2f}")

        save_dir = os.path.join(config.results_dir, os.path.basename(tf.replace(".csv", "_trades.csv")))
        trades_df.to_csv(save_dir, index=False)

        for key, value in result.items():
            cumulative_result[key] = cumulative_result.get(key, 0) + value

    # Average results across all test files
    if num_tests > 0:
        print("\nAverage Results Across All Test Files:")
        for key, value in cumulative_result.items():
            avg = value / num_tests
            print(f"{key}: {avg:.2f}")
