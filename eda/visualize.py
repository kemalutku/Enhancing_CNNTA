import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

def plot_gt_labels_with_trades(gt_csv, trades_csv=None):
    # Load GT data
    gt_df = pd.read_csv(gt_csv)
    gt_df['Date'] = pd.to_datetime(gt_df['Date'], unit='ms')

    if trades_csv is not None:
        # Load trade predictions
        trades_df = pd.read_csv(trades_csv)
        trades_df['buy_date'] = pd.to_datetime(trades_df['buy_date'])
        trades_df['sell_date'] = pd.to_datetime(trades_df['sell_date'])

    # Define GT label colors
    label_colors = {0: 'blue', 1: 'green', 2: 'red'}
    # label_colors = {1: 'green', 2: 'red'}

    # Plot close price
    plt.figure(figsize=(14, 7))
    plt.plot(gt_df['Date'], gt_df['Close'], color='black', label='Close Price')

    # Plot GT labels
    for label, color in label_colors.items():
        subset = gt_df[gt_df['Label'] == label]
        plt.scatter(subset['Date'], subset['Close'],
                    color=color, label=f'GT Label {label}',
                    alpha=0.6, edgecolors='k')

    # Plot buy signals
    if trades_csv is not None:
        for _, trade in trades_df.iterrows():
            buy_point = gt_df[gt_df['Date'] == trade['buy_date']]
            sell_point = gt_df[gt_df['Date'] == trade['sell_date']]

            if not buy_point.empty:
                plt.scatter(buy_point['Date'], buy_point['Close'],
                            color='green', marker='^', s=100, label='Buy Signal')

            if not sell_point.empty:
                plt.scatter(sell_point['Date'], sell_point['Close'],
                            color='red', marker='v', s=100, label='Sell Signal')

    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('GT Labels with Buy/Sell Trades')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Avoid duplicate legends
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Example usage:
plot_gt_labels_with_trades(
    gt_csv=r'C:\Users\KemalUtkuLekesiz\Documents\Kod\Tez\CnnTezV3\pythonProject\data\test\1d\AAPL_2017_2017_test.csv',
    # trades_csv=r'C:\Users\KemalUtkuLekesiz\Documents\Kod\Tez\CnnTezV3\pythonProject\results\AAPL_2020_2024_test_trades.csv'
)
