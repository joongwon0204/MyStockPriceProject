import torch
from torch.utils.data import DataLoader

from src.datasets.StockDataset import StockDataset
from src.models.StockPredictor import StockPredictor
from src.train.trainer import train_one_epoch, eval_one_epoch

DATA_DIR = "./data/raw"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tickers = [
        # Communication Services
        "META", "GOOGL", "NFLX", "DIS",
        # Consumer Discretionary
        "AMZN", "TSLA", "HD", "MCD", "BKNG",
        # Consumer Staples
        "WMT", "COST", "PG", "KO",
        # Energy
        "XOM", "CVX", "COP", "SLB", "EOG",
        # Financials
        "BRK-B", "JPM", "V", "MA", "BAC",
        # Health Care
        "LLY", "JNJ", "ABBV", "UNH", "MRK",
        # Industrials
        "GE", "CAT", "RTX", "UNP", "HON",
        # Information Technology
        "NVDA", "AAPL", "MSFT", "AVGO", "AMD",
        # Materials
        "LIN", "NEM", "SHW", "FCX",
        # Real Estate
        "WELL", "PLD", "AMT", "EQIX",
        # Utilities
        "NEE", "CEG", "SO", "DUK",
    ]

    train_ds = StockDataset(DATA_DIR, tickers, L=500, K=20, stride=5,
                            start_date="2000-01-01", end_date="2018-12-31")
    val_ds   = StockDataset(DATA_DIR, tickers, L=500, K=20, stride=5,
                            start_date="2019-01-01", end_date="2022-12-31")
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    model = StockPredictor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(1, 51):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, opt, device=device)
        va_loss, va_acc = eval_one_epoch(model, val_dl, device=device)
        print(f"epoch {ep} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")

if __name__ == "__main__":
    main()
