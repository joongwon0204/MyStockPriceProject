import yfinance as yf
from pathlib import Path

def download_and_save(
    tickers,
    start="2000-01-01",
    auto_adjust=False,
    out_dir="raw",
): 
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Saving to:", out_dir)

    for t in tickers:
        df = yf.download(t, start=start, auto_adjust=auto_adjust, progress=False)
        df = df.dropna()
        if len(df) == 0:
            print(f"[SKIP] {t}: empty")
            continue
        df.to_parquet(out_dir / f"{t}.parquet")
        print(f"[OK] {t}: {len(df)} rows -> {out_dir / f'{t}.parquet'}")

if __name__ == "__main__":
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
    download_and_save(tickers, start="2000-01-01", auto_adjust=False)
