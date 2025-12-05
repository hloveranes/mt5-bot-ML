"""
Simulated Institutional Trading Engine
--------------------------------------
Runs full trading logic using the local simulated market API instead of MT5.
Includes live performance tracking (PnL, win rate, Sharpe) for evaluation.
"""

import os
import time
import numpy as np
import pandas as pd

from config import CONFIG
from institutional_trade_manager import InstitutionalTradeManager
from institutional_confluence_logger import InstitutionalConfluenceLogger
from market_simulator.simulated_broker import SimulatedBroker

# Evaluators (read-only)
from evaluators.joblib_runner import JoblibRunner
from evaluators.pytorch_evaluator import PyTorchEvaluator
from evaluators.tensorflow_evaluator import TensorFlowEvaluator


# ============================================================
# 🔹 Model Initialization (shared with live engine)
# ============================================================
def initialize_models():
    """Load all pretrained models with their validation accuracies."""
    models = {"joblib": {}, "pytorch": {}, "tensorflow": {}}
    meta = {"joblib": {}, "pytorch": {}, "tensorflow": {}}

    try:
        for name, (runner_class, ext, key) in {
            "joblib": (JoblibRunner, ".joblib", "joblib"),
            "pytorch": (PyTorchEvaluator, ".pt", "pytorch"),
            "tensorflow": (TensorFlowEvaluator, ".keras", "tensorflow"),
        }.items():
            model_dir = os.path.join("model", key)
            if not os.path.isdir(model_dir):
                continue

            runner = runner_class(model_dir=model_dir)
            for f in os.listdir(model_dir):
                if f.endswith(ext):
                    mname = f[:-len(ext)]
                    models[name][mname] = runner
                    # Try reading val_acc metadata
                    try:
                        if name == "joblib":
                            import joblib
                            blob = joblib.load(os.path.join(model_dir, f))
                            meta[name][mname] = blob.get("val_acc", 0.5)
                        elif name == "pytorch":
                            import torch
                            blob = torch.load(os.path.join(model_dir, f), map_location="cpu")
                            meta[name][mname] = blob.get("val_acc", 0.5)
                        else:
                            meta[name][mname] = 0.5
                    except Exception:
                        meta[name][mname] = 0.5

            print(f"✅ Loaded {len(models[name])} {name.capitalize()} models.")
    except Exception as e:
        print(f"❌ Error loading models: {e}")

    return models, meta


# ============================================================
# 🔹 Confluence Logic (weighted averaging)
# ============================================================
def get_combined_signal(df: pd.DataFrame, models: dict, meta: dict, symbol: str):
    """Compute weighted consensus signal (BUY/SELL) using model accuracies."""
    model_outputs = {}
    flat_probs, weights = [], []

    for framework, group in models.items():
        model_outputs[framework] = {}
        for name, runner in group.items():
            try:
                prob = runner.predict(df, name)
                if prob is not None and pd.notna(prob):
                    val_acc = meta.get(framework, {}).get(name, 0.5)
                    model_outputs[framework][name] = {"prob": float(prob), "val_acc": float(val_acc)}
                    flat_probs.append(float(prob))
                    weights.append(float(val_acc))
                else:
                    model_outputs[framework][name] = {"prob": None, "val_acc": None}
            except Exception as e:
                model_outputs[framework][name] = {"prob": None, "val_acc": None}
                print(f"⚠️ {framework}/{name} predict failed: {e}")

    if not flat_probs:
        return None, 0.0, model_outputs

    avg_prob = np.average(flat_probs, weights=weights) if any(weights) else np.mean(flat_probs)
    signal = "BUY" if avg_prob >= CONFIG["prob_gate_neutral"] else "SELL"

    InstitutionalConfluenceLogger().logs.append({
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "symbol": symbol,
        "signal": signal,
        "avg_prob": avg_prob,
        "model_outputs": model_outputs,
    })
    return signal, avg_prob, model_outputs


# ============================================================
# 🔹 Live Simulation Loop with Analytics
# ============================================================
def simulated_trading_loop():
    print("🚀 Starting Institutional Simulated Trading Engine ...")
    models, meta = initialize_models()
    manager = InstitutionalTradeManager()
    broker = SimulatedBroker(symbol="XAUUSD", base_url="http://127.0.0.1:8000")

    warmup_threshold = 60
    loop_count = 0
    analytics_interval = 100

    # Rolling metrics
    equity_curve = []
    win_trades, total_trades = 0, 0

    while True:
        try:
            for symbol in CONFIG["symbols"]:
                df = broker.get_bars(limit=min(CONFIG.get("num_candles", 10000), 5000))

                if df is None or df.empty or len(df) < warmup_threshold:
                    print(f"⏳ Waiting for simulator warm-up ({len(df) if df is not None else 0}/{warmup_threshold} bars)...")
                    time.sleep(2)
                    continue

                signal, confidence, _ = get_combined_signal(df, models, meta, symbol)
                if not signal:
                    continue

                atr = (df["high"].tail(14).max() - df["low"].tail(14).min()) / 2.0
                entry = df["close"].iloc[-1]

                targets = manager.compute_targets(df, side=signal)
                tp = targets["TP"]
                sl = entry - (atr * CONFIG["atr_mult_sl"]) if signal == "BUY" else entry + (atr * CONFIG["atr_mult_sl"])

                # Simulated order execution
                result = broker.order_send(signal, entry, sl, tp, volume=0.1)
                pnl = result.get("pnl", 0.0) if isinstance(result, dict) else 0.0
                equity_curve.append(pnl)
                total_trades += 1
                if pnl > 0:
                    win_trades += 1

                manager.record_entry(symbol, signal, entry, sl, tp)
                manager.manage_trade(df, symbol, entry, signal, atr)

                loop_count += 1

                # Log performance every N iterations
                if loop_count % analytics_interval == 0 and equity_curve:
                    pnl_total = np.sum(equity_curve)
                    avg_pnl = np.mean(equity_curve)
                    wr = win_trades / total_trades if total_trades > 0 else 0
                    sharpe = (np.mean(equity_curve) / (np.std(equity_curve) + 1e-9)) * np.sqrt(252)
                    print(
                        f"\n📊 [Performance Update #{loop_count}]"
                        f"\n   Trades: {total_trades}"
                        f"\n   Win Rate: {wr*100:.2f}%"
                        f"\n   Total PnL: {pnl_total:.2f}"
                        f"\n   Avg Trade PnL: {avg_pnl:.4f}"
                        f"\n   Sharpe (dailyized): {sharpe:.3f}\n"
                    )

            time.sleep(CONFIG["sleep_interval"])

        except KeyboardInterrupt:
            print("🛑 Simulation stopped manually.")
            break
        except Exception as e:
            print(f"⚠️ Loop error: {e}")
            time.sleep(3)


# ============================================================
# 🔹 Entry Point
# ============================================================
if __name__ == "__main__":
    simulated_trading_loop()
