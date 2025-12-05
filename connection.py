import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time

def fetch_and_format_mt5_data(symbol: str, timeframe: str = "M1", limit: int = 1000, retry_wait: int = 5) -> pd.DataFrame:
    """
    Fetches and formats live candle data from MetaTrader 5.
    Returns clean DataFrame with ['time', 'open', 'high', 'low', 'close', 'tickvol', 'volume', 'spread'].
    """
    # Initialize MT5
    if not mt5.initialize():
        print("❌ MT5 initialization failed:", mt5.last_error())
        acc = mt5.account_info()
        if acc:
            print(f"✅ Logged in to account {acc.login}, balance={acc.balance}")
        else:
            raise SystemExit("❌ Not logged in to MT5. Please log in first.")
    
    # Resolve timeframe
    try:
        tf = getattr(mt5, f"TIMEFRAME_{timeframe}")
    except AttributeError:
        raise ValueError(f"Invalid timeframe: {timeframe}")

    # Fetch data with retry
    while True:
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, limit)
        if rates is None or len(rates) == 0:
            print(f"[WARN] No data for {symbol}-{timeframe}. Retrying in {retry_wait}s...")
            time.sleep(retry_wait)
            continue
        break

    # Convert to DataFrame
    if getattr(rates, "dtype", None) and rates.dtype.names:
        df = pd.DataFrame(rates)
    else:
        df = pd.DataFrame(list(rates), columns=[
            "time", "open", "high", "low", "close", "tickvol", "volume", "spread"
        ])

    df["time"] = pd.to_datetime(df["time"], unit="s", errors="coerce")
    df.rename(columns={"tick_volume": "tickvol", "real_volume": "volume", "vol": "volume"}, inplace=True)

    # Ensure all columns exist
    for c in ["time", "open", "high", "low", "close", "tickvol", "volume", "spread"]:
        if c not in df.columns:
            df[c] = np.nan

    # Convert numeric
    numeric_cols = ["open", "high", "low", "close", "tickvol", "volume", "spread"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Drop NaNs
    df = df.dropna(subset=["time", "close"]).reset_index(drop=True)

    return df[["time", "open", "high", "low", "close", "tickvol", "volume", "spread"]]
