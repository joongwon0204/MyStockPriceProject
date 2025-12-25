from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class StockDataset(Dataset):
    def __init__(self, data_dir, tickers, L=500, K=5, stride=5, start_date=None, end_date=None):
        self.data = {}
        self.samples = []
        self.L = L
        self.K = K
        self.stride = stride

        cols = ["Open","High","Low","Close","Adj Close","Volume"]
        for ticker in tickers:
            df = pd.read_parquet(Path(data_dir) / f"{ticker}.parquet").sort_index()

            if start_date is not None:
                df = df.loc[df.index >= pd.Timestamp(start_date)]
            if end_date is not None:
                df = df.loc[df.index <= pd.Timestamp(end_date)]


            df = df[cols].dropna()
            if len(df) <= L + K: continue
            self.data[ticker] = df

            for end_idx in range(L - 1, len(df) - K, self.stride):
                self.samples.append((ticker, end_idx))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ticker, end_idx = self.samples[idx]
        df = self.data[ticker]

        window = df.iloc[end_idx - self.L + 1 : end_idx + 1].to_numpy(np.float32)  # (L, 6)
        window = (window - window.mean(axis=0, keepdims=True)) / (window.std(axis=0, keepdims=True) + 1e-8)

        adj_now = df["Adj Close"].iloc[end_idx].item()
        adj_fut = df["Adj Close"].iloc[end_idx + self.K].item()
        y_ret = np.log(adj_fut / adj_now).astype(np.float32)

        y = np.float32(1.0 if y_ret > 0 else 0.0)

        x = torch.from_numpy(window).T  # (6, L)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y
    