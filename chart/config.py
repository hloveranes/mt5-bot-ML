CONFIG = {
    # === CORE TRADING SETTINGS ===
    "symbol": "XAUUSD",  # used for chart visualization
    "num_candles": 5000,  # used in chart.fetch_m5_data

    # === VOLUME PROFILE / SIGNAL GENERATION ===
    "base_r_multiple": 1.5,  # default R target for compute_volume_profile logic
    "r_multiple_overlap_boost": 0.5,  # optional performance tuning

    # === LOGGING / DATA PATHS ===
    "volume_profile_export": "vol_profile.xlsx",
    "trade_log_path": "trades/live_log.jsonl",

    # === CHART SETTINGS ===
    "chart_ref_interval": 5000,  # GUI refresh interval (ms)
}
