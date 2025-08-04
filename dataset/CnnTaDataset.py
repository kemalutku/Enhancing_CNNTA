from pathlib import Path
from bisect import bisect_right

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FinanceImageDataset(Dataset):
    """Dataset returning image tensors for multiple timeframes.

    Parameters
    ----------
    roots : dict[str, str | Path]
        Mapping from timeframe label (e.g. ``"1d"``) to directory containing
        the preprocessed CSV files for that frequency.  All directories must
        contain the same set of files.
    feature_cols : list[str]
        Column names of technical indicators to include in the image.
    sequence_len : int
        Number of periods to include in each input window.
    """

    def __init__(
        self,
        roots: dict[str, str | Path],
        feature_cols: list[str],
        sort_by_date: bool = True,
        return_symbol: bool = False,
        sequence_len: int = 15,
    ):

        self.roots = {k: Path(v) for k, v in roots.items()}
        self.feature_cols = feature_cols
        self.sequence_len = sequence_len
        self.num_classes = 3
        self.return_symbol = return_symbol

        self._symbols = []

        # ── discover source files from the daily directory ──────────
        base_dir = self.roots["1d"]
        files = sorted(base_dir.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {base_dir}")

        # ── load them once into contiguous arrays ───────────────────
        self._arrays: list[np.ndarray] = []  # shape: (C, N, F)
        self._labels: list[np.ndarray] = []
        self._timestamps: list[np.ndarray] = []
        self._closes: list[np.ndarray] = []
        self._lengths: list[int] = []  # usable sample count per file

        for csv_path in files:
            symbol = csv_path.stem
            self._symbols.append(symbol)

            dfs = {}
            for tf, root in self.roots.items():
                df = pd.read_csv(root / csv_path.name)
                if sort_by_date and "Date" in df.columns:
                    df = df.sort_values("Date")
                df = df.iloc[self.sequence_len:]  # drop warm-up rows
                dfs[tf] = df

            # ensure equal length across timeframes
            min_len = min(len(df) for df in dfs.values())
            usable = min_len - self.sequence_len
            if usable <= 0:
                continue
            for tf in dfs:
                dfs[tf] = dfs[tf].iloc[: usable + self.sequence_len]

            arrays = [
                dfs[tf][self.feature_cols].to_numpy(dtype=np.float32)
                for tf in ["1d", "1wk", "1mo"]
            ]
            stacked = np.stack(arrays, axis=0)  # (C, N, F)

            self._arrays.append(stacked)
            self._labels.append(dfs["1d"]["Label"].to_numpy(dtype=np.int64))
            self._timestamps.append(dfs["1d"]["Date"].to_numpy())
            self._closes.append(dfs["1d"]["Close"].to_numpy(dtype=np.float32))
            self._lengths.append(usable)

        self._cum = np.cumsum(self._lengths).tolist()  # cumulative lengths

    # ───────────────────────── Dataset API ───────────────────────── #

    def __len__(self) -> int:
        return self._cum[-1] if self._cum else 0

    def __getitem__(self, idx: int):
        file_idx = bisect_right(self._cum, idx)
        offset = idx - (self._cum[file_idx - 1] if file_idx else 0)

        x_arr = self._arrays[file_idx]
        y_arr = self._labels[file_idx]
        t_arr = self._timestamps[file_idx]
        c_arr = self._closes[file_idx]

        start, end = offset, offset + self.sequence_len

        x = torch.from_numpy(x_arr[:, start:end])
        y = torch.nn.functional.one_hot(
            torch.tensor(y_arr[end], dtype=torch.long),
            num_classes=self.num_classes
        ).float()

        timestamp = torch.tensor(t_arr[end])
        close = torch.tensor(c_arr[end])

        if self.return_symbol:
            return self._symbols[file_idx], timestamp, close, x, y

        return timestamp, close, x, y
