import torch
from torch.utils.data import Dataset
import pandas as pd
import pandas_ta as ta
import pandas_ta.volume as tav
from sklearn.preprocessing import MinMaxScaler
from data.labeling_method import sliding_window_labeling as label_function
from custom_indicators import compute_tn_rsi


class RawFinanceDataset(Dataset):
    def __init__(self, csv_path, feature_cols, window_size=11, relabel_range=0, use_fitted_scaler=False,
                 fitted_scaler=None, start_year=None, end_year=None):
        self.sequence_length = len(feature_cols)
        self.feature_cols = feature_cols
        self.label_col = 'Label'
        self.num_classes = 3

        df = pd.read_csv(csv_path)
        df = label_function(df, window_size=window_size, relabel_range=relabel_range)

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        # Construct indicator dataframe
        ind_df = pd.DataFrame({
            "Date": df["Date"],
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
            "TN_RSI": compute_tn_rsi(df),
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
            "Label": df["Label"]
        })

        # Drop initial rows with NaNs
        ind_df = ind_df[60:].copy()

        # Scale features
        # if use_fitted_scaler and fitted_scaler is not None:
        #     ind_df.iloc[:, 1:-1] = fitted_scaler.transform(ind_df.iloc[:, 1:-1])
        # else:
        #     scaler = MinMaxScaler()
        #     ind_df[self.feature_cols] = scaler.fit_transform(ind_df[self.feature_cols])

        ind_df["Close"] = df["Close"].iloc[60:].values

        if start_year is not None and end_year is not None:
            train_start = int(pd.Timestamp(f"{start_year}-01-01").timestamp() * 1000)
            train_end = int(pd.Timestamp(f"{end_year}-12-31").timestamp() * 1000)
            ind_df = ind_df[(ind_df["Date"] >= train_start) & (ind_df["Date"] <= train_end)]

        self.data = ind_df
        self.data_array = self.data[self.feature_cols].values
        self.label_array = self.data["Label"].values
        self.timestamps = self.data["Date"].values
        self.closes = self.data["Close"].values
        self.data_length = len(self.data_array)

    def __len__(self):
        return self.data_length - self.sequence_length

    def __getitem__(self, idx):
        x = self.data_array[idx:idx + self.sequence_length]
        y = self.label_array[idx + self.sequence_length]

        x = torch.tensor(x, dtype=torch.float32).reshape(-1, self.sequence_length, len(self.feature_cols))
        y = torch.nn.functional.one_hot(torch.tensor(y, dtype=torch.long), num_classes=self.num_classes).float()

        ts = torch.tensor(self.timestamps[idx + self.sequence_length])
        close = torch.tensor(self.closes[idx + self.sequence_length])

        return ts, close, x, y
