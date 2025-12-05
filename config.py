CONFIG = {
    # === CORE TRADING SETTINGS ===
    "symbols": ["XAUUSD"],
    # "symbols": ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
    "timeframe": "M1",
    "timeframes_per_symbol": {
        "XAUUSD-VIP": ["M1", "M5", "H1"],
        # "EURUSD": ["M1", "M5", "H1"],
        # "GBPUSD": ["M1", "M5", "H1"],
        # "USDJPY": ["M1", "M5", "H1"],
        # "AUDUSD": ["M1", "M5", "H1"],
    },
    "num_candles": 3000,  # how many bars to load in memory

    # === LIVE LOOP ===
    "sleep_interval": 5,          # wait between ticks (seconds)

    # === META TRADER SETTINGS ===
    "magic_number": 987654,
    "deviation": 10,              # slippage tolerance

    # === RISK MANAGEMENT ===
    "initial_balance": 10000.0,
    "lot_size": 0.01,                 # fallback lot
    "atr_period": 14,
    "atr_mult_sl": 0.85,              # stop-loss = ATR × multiplier

    # === LIVE TRAILING LOOP SETTINGS ===
    "trail_update_interval_secs": 1,    # evaluate every second
    "min_sl_change_points": 3,          # avoid micro updates
    "atr_multiplier_trail": 0.5,        # trailing distance = 0.5 × ATR
    "trail_round_to_tick": True,        # round SL to nearest tick
    "trail_always_on": True,            # always trailing, not conditional

    # === CONFLUENCE / DECISION ===
    "prob_gate_neutral": 0.55,  # threshold for flat bias
    "prob_gate_biased": 0.51,   # mild directional bias

    # === LOGGING ===
    "trade_log_path": "logs/live_trades.jsonl",
    "heartbeat_interval": 60,  # seconds

    # === PROFIT TARGET BEHAVIOR ===
    "base_r_multiple": 1.5,          # base R-multiple for targets
}
