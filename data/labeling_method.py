import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def sliding_window_labeling(data, window_size=11, relabel_range=0):
    def relabel_labels(labels, relabel_range=0):
        labels_copy = labels.copy()
        n = len(labels)

        for i in range(n):
            if labels[i] in [1, 2]:
                target = labels[i]

                for j in range(1, relabel_range + 1):
                    if i + j < n and labels_copy[i + j] == 0:
                        labels_copy[i + j] = target

                # Spread label backward within range
                for j in range(1, relabel_range + 1):
                    if i - j >= 0 and labels_copy[i - j] == 0:
                        labels_copy[i - j] = target

        return labels_copy

    half_window = window_size // 2
    labels = []

    for i in range(len(data)):
        if i < half_window or i >= len(data) - half_window:
            labels.append(0)
            continue

        window = data['Close'][i - half_window: i + half_window + 1]
        middle_price = data['Close'].values[i]

        if middle_price == window.min():
            labels.append(1)  # Buy
        elif middle_price == window.max():
            labels.append(2)  # Sell
        else:
            labels.append(0)  # Hold

    if relabel_range > 0:
        labels = relabel_labels(labels, relabel_range)

    data['Label'] = labels
    return data


def sliding_window_with_midpoint_labeling(data, window_size=11):
    half_window = window_size // 2
    labels = []

    for i in range(len(data)):
        if i < half_window or i >= len(data) - half_window:
            labels.append(0)
            continue

        window = data['Close'][i - half_window: i + half_window + 1]
        middle_price = data['Close'].values[i]

        if middle_price == window.min():
            labels.append(1)  # Buy
        elif middle_price == window.max():
            labels.append(2)  # Sell
        else:
            labels.append(0)  # Hold

    # Insert opposite label between same-type non-zero labels
    new_labels = labels.copy()
    i = 0
    while i < len(labels):
        current = labels[i]
        if current == 0:
            i += 1
            continue

        # Find next non-zero label
        j = i + 1
        while j < len(labels) and labels[j] == 0:
            j += 1

        if j < len(labels) and labels[j] == current:
            mid = (i + j) // 2
            if new_labels[mid] == 0:
                price_mid = data['Close'].iloc[mid]
                price_next = data['Close'].iloc[j]

                if current == 1 and price_mid > price_next:
                    new_labels[mid] = 2  # Insert sell
                elif current == 2 and price_mid < price_next:
                    new_labels[mid] = 1  # Insert buy

        i = j  # move to the next non-zero label

    data['Label'] = new_labels
    return data


def fixed_time_horizon_labeling(data, horizon=5, up_threshold=0.02, down_threshold=-0.02):
    """
    Labels each point in the time series based on fixed time horizon future return.
    - Buy (1): if return > up_threshold
    - Sell (2): if return < down_threshold
    - Hold (0): otherwise

    Parameters:
    - data: DataFrame with at least a 'Close' column
    - horizon: number of days to look ahead
    - up_threshold: minimum return to consider as Buy
    - down_threshold: maximum return to consider as Sell

    Returns:
    - DataFrame with new 'Label' column
    """
    labels = []
    close_prices = data['Close'].values

    for i in range(len(close_prices)):
        if i + horizon >= len(close_prices):
            labels.append(0)
            continue

        future_return = (close_prices[i + horizon] - close_prices[i]) / close_prices[i]

        if future_return > up_threshold:
            labels.append(1)  # Buy
        elif future_return < down_threshold:
            labels.append(2)  # Sell
        else:
            labels.append(0)  # Hold

    data['Label'] = labels
    return data


def tn_rsi_labeling(data, rsi_window=14, trend_window=14, buy_threshold=25, sell_threshold=65, relabel_range=0):
    import numpy as np
    import pandas as pd

    def detrend_prices(prices, window):
        detrended = [np.nan] * len(prices)
        prices_array = prices.values
        for i in range(window, len(prices)):
            y = prices_array[i - window:i]
            if len(y) < window:
                continue
            x = np.arange(window)
            A = np.vstack([x, np.ones(len(x))]).T
            m, b = np.linalg.lstsq(A, y, rcond=None)[0]
            trend = m * (window - 1) + b
            detrended[i] = prices_array[i] - trend
        return pd.Series(detrended, index=prices.index)

    def calculate_rsi(prices, window):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def relabel_labels(labels, relabel_range=0):
        labels_copy = labels.copy()
        n = len(labels)
        for i in range(n):
            if labels[i] in [1, 2]:
                target = labels[i]
                for j in range(1, relabel_range + 1):
                    if i + j < n and labels_copy[i + j] == 0:
                        labels_copy[i + j] = target
                    if i - j >= 0 and labels_copy[i - j] == 0:
                        labels_copy[i - j] = target
        return labels_copy

    # Detrend and calculate TN-RSI
    detrended = detrend_prices(data['Close'], trend_window)
    tn_rsi = calculate_rsi(detrended, rsi_window)

    # Labeling
    labels = []
    for val in tn_rsi:
        if np.isnan(val):
            labels.append(0)
        elif val < buy_threshold:
            labels.append(1)
        elif val > sell_threshold:
            labels.append(2)
        else:
            labels.append(0)

    data = data.copy()
    data['Label'] = labels

    if relabel_range > 0:
        data['Label'] = relabel_labels(data['Label'], relabel_range)

    return data
