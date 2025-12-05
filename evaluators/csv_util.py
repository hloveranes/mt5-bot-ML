import os
import pandas as pd

def load_mt5_csv(path: str, debug: bool = False) -> pd.DataFrame:
    """
    Loads and cleans MT5-exported CSV (tab-separated, quoted).
    Columns: <DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    df_raw = pd.read_csv(
        path,
        sep="\t",
        engine="python",
        quotechar='"',
        names=["date", "time", "open", "high", "low", "close", "tickvol", "vol", "spread"],
        header=0,
        dtype=str,
        encoding="utf-8",
    )

    if debug:
        print("📂 Raw columns:", df_raw.columns.tolist())
        print(df_raw.head(3))

    # Combine date and time
    df_raw["time"] = pd.to_datetime(
        df_raw["date"].str.replace(".", "-") + " " + df_raw["time"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    # Convert numerics
    for col in ["open", "high", "low", "close", "tickvol", "vol", "spread"]:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    before = len(df_raw)
    df = df_raw.dropna(subset=["time", "close"]).rename(columns={"vol": "volume"})
    after = len(df)

    if debug:
        print(f"✅ Cleaned rows: {before} → {after}")

    if after == 0:
        raise ValueError("Parsed 0 valid rows — check CSV format or missing data!")

    return df[["time", "open", "high", "low", "close", "tickvol", "volume", "spread"]]
