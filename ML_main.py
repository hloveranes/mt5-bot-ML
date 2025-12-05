"""
Main Entry for Live Institutional Trading Engine
------------------------------------------------
- Loads ALL pretrained models from:
    model/joblib/*.joblib
    model/pytorch/*.pt
    model/tensorflow/*.keras
- No training or saving (read-only use).
- Combines model probabilities for a consensus signal.
"""

import os
import time
import pandas as pd
import numpy as np
import MetaTrader5 as mt5

from config import CONFIG
from connection import fetch_and_format_mt5_data
from institutional_trade_manager import InstitutionalTradeManager
from institutional_confluence_logger import InstitutionalConfluenceLogger

# Evaluators (READ-ONLY usage)
from evaluators.joblib_runner import JoblibRunner
from evaluators.pytorch_evaluator import PyTorchEvaluator
from evaluators.tensorflow_evaluator import TensorFlowEvaluator


def initialize_models():
    """
    Discover and load all models. Read-only. Never writes or saves.

    Returns two dicts:
        models = {"joblib": {method: runner}, "pytorch": {...}, "tensorflow": {...}}
        meta   = {"joblib": {method: val_acc}, "pytorch": {...}, "tensorflow": {...}}
    """
    models = {"joblib": {}, "pytorch": {}, "tensorflow": {}}
    meta = {"joblib": {}, "pytorch": {}, "tensorflow": {}}

    try:
        # --- Joblib ---
        jdir = os.path.join("model", "joblib")
        if os.path.isdir(jdir):
            jrunner = JoblibRunner(model_dir=jdir)
            for f in os.listdir(jdir):
                if f.endswith(".joblib"):
                    mname = f[:-7]
                    fpath = os.path.join(jdir, f)
                    try:
                        blob = joblib.load(fpath)
                        val_acc = blob.get("val_acc", 0.5)
                    except Exception:
                        val_acc = 0.5
                    models["joblib"][mname] = jrunner
                    meta["joblib"][mname] = val_acc
            print(f"✅ Loaded {len(models['joblib'])} Joblib models.")

        # --- PyTorch ---
        tdir = os.path.join("model", "pytorch")
        if os.path.isdir(tdir):
            trunner = PyTorchEvaluator(model_dir=tdir)
            for f in os.listdir(tdir):
                if f.endswith(".pt"):
                    mname = f[:-3]
                    fpath = os.path.join(tdir, f)
                    try:
                        blob = torch.load(fpath, map_location="cpu")
                        val_acc = blob.get("val_acc", 0.5)
                    except Exception:
                        val_acc = 0.5
                    models["pytorch"][mname] = trunner
                    meta["pytorch"][mname] = val_acc
            print(f"✅ Loaded {len(models['pytorch'])} PyTorch models.")

        # --- TensorFlow ---
        tfdir = os.path.join("model", "tensorflow")
        if os.path.isdir(tfdir):
            tfrunner = TensorFlowEvaluator(model_dir=tfdir)
            for f in os.listdir(tfdir):
                if f.endswith(".keras"):
                    mname = f[:-3]
                    fpath = os.path.join(tfdir, f)
                    try:
                        import json
                        meta_path = fpath.replace(".keras", ".json")
                        val_acc = 0.5
                        if os.path.exists(meta_path):
                            with open(meta_path, "r") as j:
                                info = json.load(j)
                                val_acc = info.get("val_acc", 0.5)
                    except Exception:
                        val_acc = 0.5
                    models["tensorflow"][mname] = tfrunner
                    meta["tensorflow"][mname] = val_acc
            print(f"✅ Loaded {len(models['tensorflow'])} TensorFlow models.")

    except Exception as e:
        print(f"❌ Error initializing models: {e}")

    return models, meta


def get_combined_signal(df: pd.DataFrame, models: dict, meta: dict, symbol: str):
    """
    Run all loaded models. Each evaluator returns a float probability for BUY.
    We combine them using accuracy-weighted averaging for a smarter consensus.
    """
    model_outputs = {}
    flat_probs, weights = [], []

    for framework, group in models.items():
        model_outputs[framework] = {}
        for name, runner in group.items():
            try:
                prob = runner.predict(df, name)
                if prob is not None and pd.notna(prob):
                    val_acc = meta.get(framework, {}).get(name, 0.5)
                    model_outputs[framework][name] = {
                        "prob": float(prob),
                        "val_acc": float(val_acc)
                    }
                    flat_probs.append(float(prob))
                    weights.append(float(val_acc))
                else:
                    model_outputs[framework][name] = {"prob": None, "val_acc": None}
            except Exception as e:
                model_outputs[framework][name] = {"prob": None, "val_acc": None}
                print(f"⚠️ {framework}/{name} predict failed: {e}")

    if not flat_probs:
        return None, 0.0, model_outputs

    # Weighted average (if all weights = 0, fallback to mean)
    if any(weights):
        avg_prob = np.average(flat_probs, weights=weights)
    else:
        avg_prob = sum(flat_probs) / len(flat_probs)

    signal = "BUY" if avg_prob >= CONFIG["prob_gate_neutral"] else "SELL"

    # Log confluence
    logger = InstitutionalConfluenceLogger()
    logger.logs.append({
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "symbol": symbol,
        "signal": signal,
        "avg_prob": avg_prob,
        "model_outputs": model_outputs,
    })

    return signal, avg_prob, model_outputs


def live_trading_loop():
    print("🚀 Starting Institutional Live Trading Engine ...")
    models, meta = initialize_models()
    manager = InstitutionalTradeManager()

    last_heartbeat = 0
    heartbeat_interval = CONFIG.get("heartbeat_interval", 60)
    trail_interval = CONFIG.get("trail_update_interval_secs", 1)
    last_trail_ts = 0

    while True:
        try:
            for symbol in CONFIG["symbols"]:
                df = fetch_and_format_mt5_data(symbol, CONFIG["timeframe"], CONFIG["num_candles"], CONFIG["sleep_interval"])
                if not isinstance(df, pd.DataFrame) or df.empty or len(df) < 60:
                    print(f"⚠️ Skipping {symbol}: invalid or insufficient data.")
                    continue

                signal, confidence, _ = get_combined_signal(df, models, meta, symbol)
                if not signal:
                    continue

                print(f"📈 {symbol}: {signal} ({confidence:.2f})")

                # --- Use live price instead of last close
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    print(f"❌ No live tick for {symbol}, skipping.")
                    continue

                entry = float(tick.ask) if signal == "BUY" else float(tick.bid)

                # --- ATR for SL sizing (keep your quick ATR)
                atr = (df["high"].tail(14).max() - df["low"].tail(14).min()) / 2.0

                # --- Model-based TP
                targets = manager.compute_targets(df, side=signal)
                tp = float(targets["TP"])

                # --- Initial SL from ATR
                sl = entry - (atr * CONFIG["atr_mult_sl"]) if signal == "BUY" else entry + (atr * CONFIG["atr_mult_sl"])

                # --- ✅ Make SL/TP broker-valid (fixes MT5 10016 'Invalid stops')
                sl, tp = manager.ensure_valid_stops(symbol, entry, sl, tp, signal)

                # --- Place order + log locally
                manager.record_entry(symbol, signal, entry, sl, tp)

                # --- Start dynamic management off this bar
                manager.manage_trade(df, symbol, entry, signal, atr)

                # --- Always-on trailing (evaluates every second)
                if time.time() - last_trail_ts >= trail_interval:
                    manager._maybe_trail_once(symbol, df)



            # ⏰ Periodic heartbeat
            if time.time() - last_heartbeat > heartbeat_interval:
                manager.heartbeat()
                last_heartbeat = time.time()

            time.sleep(CONFIG["sleep_interval"])

        except KeyboardInterrupt:
            print("🛑 Stopped manually.")
            break
        except Exception as e:
            print(f"⚠️ Loop error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    live_trading_loop()
