import os
import pandas as pd
from labeling_method import sliding_window_labeling as label_function
# from labeling_method import fixed_time_horizon_labeling as label_function
# from labeling_method import sliding_window_with_midpoint_labeling as label_function
# from labeling_method import tn_rsi_labeling as label_function
import pandas_ta as ta
import pandas_ta.volume as tav
from sklearn.preprocessing import MinMaxScaler
from custom_indicators import compute_tn_rsi
import joblib


def preprocess():
    file_list = os.listdir(RAW_DATA_DIR)
    print("Number of files found:", len(file_list))

    all_data = []  # Store all data to compute global min/max

    # First Pass: Collect Data for Normalization
    for i, file in enumerate(file_list):
        print(f"\rCollecting {i + 1}/{len(file_list)}: {file}", end="")

        file_path = os.path.join(RAW_DATA_DIR, file)
        raw_df = pd.read_csv(file_path, index_col=0)

        raw_df = label_function(raw_df, window_size=window_size, relabel_range=relabel_range)
        # raw_df = label_function(raw_df)

        close = raw_df["Close"]
        high = raw_df["High"]
        low = raw_df["Low"]
        volume = raw_df["Volume"]

        ind_df = pd.DataFrame({
            "RSI": ta.rsi(close),
            "WIL": ta.willr(high, low, close),
            "WMA": ta.wma(close),
            "EMA": ta.ema(close),
            "SMA": ta.sma(close),
            "HMA": ta.hma(close),
            "TMA": ta.tema(close),
            "CCI": ta.cci(high, low, close),
            "CMO": ta.cmo(close),
            "MCD": ta.macd(close)["MACD_12_26_9"],
            "PPO": ta.ppo(close)["PPO_12_26_9"],
            "ROC": ta.roc(close),
            "CMF": tav.cmf(high, low, close, volume),
            "ADX": ta.adx(high, low, close)["ADX_14"],
            "TN_RSI": compute_tn_rsi(raw_df),
            "PSA": ta.psar(high, low)["PSARaf_0.02_0.2"],
            "STOCHRSI": ta.stochrsi(close)["STOCHRSIk_14_14_3_3"],
            "UO": ta.uo(high, low, close),
            "MOM": ta.mom(close),
            "TSI": ta.tsi(close)["TSI_13_25_13"],
            "AO": ta.ao(high, low),
            "KDJ": ta.kdj(high, low, close)["J_9_3"],
            "RVI": ta.rvi(close, high, low),
            "VIDYA": ta.vidya(close),
            "ZLEMA": ta.zlma(close),
            "DEMA": ta.dema(close),
            "T3": ta.t3(close),
            "ATR": ta.atr(high, low, close),
            "BBW": ta.bbands(close)["BBB_5_2.0"],
            "DONCH": ta.donchian(high, low)["DCL_20_20"],
            "TRIX": ta.trix(close=close)["TRIXs_30_9"],
            "OBV": ta.obv(close=close, volume=volume),
            "MFI": ta.mfi(high, low, close, volume),
            "Label": raw_df["Label"]
        })

        ind_df = ind_df[60:]
        all_data.append(ind_df)

    print("\nData Collection Complete")

    # Concatenate all data to compute global min/max
    combined_data = pd.concat(all_data)

    # Normalize using Min-Max Scaling (0 to 1)
    scaler = MinMaxScaler()
    combined_data.iloc[:, :-1] = scaler.fit_transform(combined_data.iloc[:, :-1])

    os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    # Collect all training data
    all_train_dfs = []

    # Second Pass: Normalize Each File and Save
    for i, file in enumerate(file_list):
        print(f"\rProcessing {i + 1}/{len(file_list)}: {file}", end="")
        symbol = file[:-4]
        file_path = os.path.join(RAW_DATA_DIR, file)
        raw_df = pd.read_csv(file_path, index_col=0)

        raw_df = label_function(raw_df, window_size=window_size, relabel_range=relabel_range)
        # raw_df = label_function(raw_df)

        close = raw_df["Close"]
        high = raw_df["High"]
        low = raw_df["Low"]
        volume = raw_df["Volume"]

        ind_df = pd.DataFrame({
            "RSI": ta.rsi(close),
            "WIL": ta.willr(high, low, close),
            "WMA": ta.wma(close),
            "EMA": ta.ema(close),
            "SMA": ta.sma(close),
            "HMA": ta.hma(close),
            "TMA": ta.tema(close),
            "CCI": ta.cci(high, low, close),
            "CMO": ta.cmo(close),
            "MCD": ta.macd(close)["MACD_12_26_9"],
            "PPO": ta.ppo(close)["PPO_12_26_9"],
            "ROC": ta.roc(close),
            "CMF": tav.cmf(high, low, close, volume),
            "ADX": ta.adx(high, low, close)["ADX_14"],
            "TN_RSI": compute_tn_rsi(raw_df),
            "PSA": ta.psar(high, low)["PSARaf_0.02_0.2"],
            "STOCHRSI": ta.stochrsi(close)["STOCHRSIk_14_14_3_3"],
            "UO": ta.uo(high, low, close),
            "MOM": ta.mom(close),
            "TSI": ta.tsi(close)["TSI_13_25_13"],
            "AO": ta.ao(high, low),
            "KDJ": ta.kdj(high, low, close)["J_9_3"],
            "RVI": ta.rvi(close, high, low),
            "VIDYA": ta.vidya(close),
            "ZLEMA": ta.zlma(close),
            "DEMA": ta.dema(close),
            "T3": ta.t3(close),
            "ATR": ta.atr(high, low, close),
            "BBW": ta.bbands(close)["BBB_5_2.0"],
            "DONCH": ta.donchian(high, low)["DCL_20_20"],
            "TRIX": ta.trix(close=close)["TRIXs_30_9"],
            "OBV": ta.obv(close=close, volume=volume),
            "MFI": ta.mfi(high, low, close, volume),
            "Label": raw_df["Label"]
        })

        ind_df = ind_df[60:]
        ind_df.iloc[:, :-1] = scaler.transform(ind_df.iloc[:, :-1])
        ind_df["Close"] = raw_df["Close"]
        ind_df["Symbol"] = symbol  # Optional: helps track original file

        # Convert timestamps for filtering
        train_start = int(pd.Timestamp(f"{START_YEAR_TRAIN}-01-01").timestamp() * 1000)
        train_end = int(pd.Timestamp(f"{END_YEAR_TRAIN}-12-31").timestamp() * 1000)
        test_start = int(pd.Timestamp(f"{START_YEAR_TEST}-01-01").timestamp() * 1000)
        test_end = int(pd.Timestamp(f"{END_YEAR_TEST}-12-31").timestamp() * 1000)

        train_df = ind_df[(ind_df.index >= train_start) & (ind_df.index <= train_end)]
        test_df = ind_df[(ind_df.index >= test_start) & (ind_df.index <= test_end)]

        all_train_dfs.append(train_df)

        # Save test file individually
        test_file_path = os.path.join(TEST_DATA_DIR, f"{symbol}_{START_YEAR_TEST}_{END_YEAR_TEST}_test.csv")
        test_df.to_csv(test_file_path)

    # Save all training data into a single file
    final_train_df = pd.concat(all_train_dfs)
    final_train_path = os.path.join(TRAIN_DATA_DIR, f"all_train_{START_YEAR_TRAIN}_{END_YEAR_TRAIN}.csv")
    final_train_df.to_csv(final_train_path)

    joblib.dump(scaler, os.path.join(TRAIN_DATA_DIR, "scaler.pkl"))
    print("\nPreprocessing completed.")


if __name__ == '__main__':
    FREQUENCY = "1d"

    RAW_DATA_DIR = os.path.join(r"C:\Users\KemalUtkuLekesiz\Documents\Kod\Tez\CnnTezV3\pythonProject\data\raw",
                                FREQUENCY
                                )
    TRAIN_DATA_DIR = os.path.join(r"C:\Users\KemalUtkuLekesiz\Documents\Kod\Tez\CnnTezV3\pythonProject\data\train",
                                  FREQUENCY)
    TEST_DATA_DIR = os.path.join(r"C:\Users\KemalUtkuLekesiz\Documents\Kod\Tez\CnnTezV3\pythonProject\data\test",
                                 FREQUENCY)

    START_YEAR_TRAIN = 2002
    END_YEAR_TRAIN = 2016
    START_YEAR_TEST = 2017
    END_YEAR_TEST = 2017

    relabel_range = 1
    window_size = 11

    preprocess()
