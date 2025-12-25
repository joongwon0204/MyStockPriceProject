# minimal_skeleton.py
# pip install yfinance pandas pyarrow torch

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# 1) Dataset (raw parquet -> window)
# -----------------------------
class StockWindowDataset(Dataset):
    def __init__(self, raw_dir="data/raw", tickers=None, L=120, K=5):
        self.raw_dir = Path(raw_dir)
        self.L = L
        self.K = K

        if tickers is None:
            tickers = [p.stem for p in self.raw_dir.glob("*.parquet")]
        self.tickers = tickers

        self.data = {}
        self.samples = []  # (ticker, end_idx)

        cols = ["Open","High","Low","Close","Adj Close","Volume"]

        for t in self.tickers:
            df = pd.read_parquet(self.raw_dir / f"{t}.parquet").sort_index()
            df = df[cols].dropna()
            if len(df) < L + K + 1:
                continue
            self.data[t] = df

            # 가능한 모든 윈도우 인덱스 만들기
            for end_idx in range(L-1, len(df)-K):
                self.samples.append((t, end_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        t, end_idx = self.samples[i]
        df = self.data[t]

        w = df.iloc[end_idx-self.L+1:end_idx+1].to_numpy(np.float32)  # (L,6)

        # (선택) 아주 간단 윈도우 표준화
        w = (w - w.mean(axis=0, keepdims=True)) / (w.std(axis=0, keepdims=True) + 1e-8)

        # 라벨: K일 후 수익률 방향 (분류)
        adj_now = float(df.iloc[end_idx]["Adj Close"])
        adj_fut = float(df.iloc[end_idx + self.K]["Adj Close"])
        y = 1.0 if np.log(adj_fut/adj_now) > 0 else 0.0

        x = torch.from_numpy(w).T           # (C=6, L)  Conv1d input
        y = torch.tensor(y, dtype=torch.float32)
        return x, y


# -----------------------------
# 2) Model: Multi-kernel Conv -> RNN
# -----------------------------
class ConvRNN(nn.Module):
    def __init__(self, in_ch=6, conv_ch=32, rnn_h=64, kernels=(3,5,10,20)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_ch, conv_ch, k, padding=k//2) for k in kernels
        ])
        self.mix = nn.Conv1d(conv_ch * len(kernels), conv_ch, 1)

        self.rnn = nn.GRU(input_size=conv_ch, hidden_size=rnn_h, batch_first=True)
        self.head = nn.Linear(rnn_h, 1)

    def forward(self, x):              # x: (B, C, L)
        feats = [torch.relu(c(x)) for c in self.convs]    # each: (B, conv_ch, L)
        z = torch.cat(feats, dim=1)                       # (B, conv_ch*K, L)
        z = torch.relu(self.mix(z))                       # (B, conv_ch, L)

        z = z.permute(0, 2, 1)                            # (B, L, conv_ch)
        out, _ = self.rnn(z)
        last = out[:, -1, :]                              # (B, rnn_h)
        logits = self.head(last).squeeze(-1)              # (B,)
        return logits


# -----------------------------
# 3) Train loop (minimal)
# -----------------------------
def main():
    ds = StockWindowDataset(raw_dir="data/raw", L=120, K=5)
    dl = DataLoader(ds, batch_size=128, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConvRNN().to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(3):
        model.train()
        total = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * x.size(0)

        print("epoch", epoch, "loss", total / len(ds))

if __name__ == "__main__":
    main()
