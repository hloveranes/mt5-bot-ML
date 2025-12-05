"""
probability_bot.py — Unified Probability Bot Trainer & Evaluator
---------------------------------------------------------------
- Loads historical CSV via csv_util.load_mt5_csv
- Discovers trading methods from method_library.py
- Trains each backend (joblib, pytorch, tensorflow) automatically
- Each method model is saved under:
      models/{backend}/{method}.{joblib|pt|keras}
- Then evaluates all trained methods and logs results

Usage:
  python probability_bot.py --csv data/historical/XAUUSD_M1_1970_2025.csv
"""
from __future__ import annotations
import os, sys, json, itertools, argparse, importlib, datetime as dt
import numpy as np
import pandas as pd

# Core imports
from evaluators.csv_util import load_mt5_csv

# Evaluator imports
from evaluators.pytorch_evaluator import PyTorchEvaluator
from evaluators.tensorflow_evaluator import TensorFlowEvaluator
from evaluators.joblib_runner import JoblibRunner


# ============================================================
# 🔹 Method Discovery
# ============================================================
def discover_methods():
    """Discovers trading methods from method_library.py."""
    try:
        from method_library import METHODS as methods
        if isinstance(methods, dict) and methods:
            return methods
    except ImportError:
        pass

    # fallback dynamic discovery
    mod = importlib.import_module("method_library")
    methods = {}
    for k, v in mod.__dict__.items():
        if callable(v) and k.startswith("method_"):
            methods[k] = v

    if not methods:
        raise RuntimeError("❌ No trading methods found in method_library.py")
    return methods


# ============================================================
# 🔹 Backend Selection
# ============================================================
def get_backends():
    """Return all supported backends for auto-training."""
    return {
        "joblib": JoblibRunner(),
        "pytorch": PyTorchEvaluator(),
        "tensorflow": TensorFlowEvaluator(),
    }


# ============================================================
# 🔹 Helper Utilities
# ============================================================
def method_combinations(methods: dict, size: int):
    keys = list(methods.keys())
    for combo in itertools.combinations(keys, size):
        yield list(combo)

def pick_methods(methods: dict, names: list[str]):
    return {k: methods[k] for k in names if k in methods}

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)

def sanitize_filename(name: str) -> str:
    """
    Replace invalid characters in filenames for Windows, Linux, macOS.
    Keeps names readable while preventing OSError [Errno 22].
    """
    invalid_chars = '<>:"/\\|?*='
    safe = name
    for ch in invalid_chars:
        safe = safe.replace(ch, "_")
    # Collapse double underscores and strip spaces
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip().rstrip(".")

# ============================================================
# 🔹 Orchestrator
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="Unified Probability Bot Trainer & Evaluator")
    ap.add_argument("--csv", type=str, default="data/historical/XAUUSD_M1_4000.csv", help="Path to MT5-style CSV.")
    ap.add_argument("--combo", type=int, default=1, help="Method combination size (1 = each method alone)")
    ap.add_argument("--max-combos", type=int, default=0, help="Limit number of combos evaluated (0 = no limit)")
    ap.add_argument("--threshold", type=float, default=0.55, help="Entry probability threshold (for ML backends)")
    ap.add_argument("--sl-atr", type=float, default=0.8, help="Stop-loss in ATR multiples")
    ap.add_argument("--tp-atr", type=float, default=3.0, help="Take-profit in ATR multiples")
    ap.add_argument("--time-stop", type=int, default=0, help="Max bars to hold (0 = disable)")
    ap.add_argument("--outdir", type=str, default="data/results", help="Output directory for results")
    ap.add_argument("--sample", type=int, default=0, help="Use only first N rows for quick runs")
    args = ap.parse_args()

    # ---------------------------
    # Load CSV(s)
    # ---------------------------
    dfs = []
    if args.csv and os.path.exists(args.csv):
        df = load_mt5_csv(args.csv)
        dfs.append((os.path.basename(args.csv), df))
    else:
        ddir = os.path.join(os.path.dirname(__file__), "data", "historical")
        if not os.path.isdir(ddir):
            raise FileNotFoundError("No --csv given and data/historical missing")
        for fn in os.listdir(ddir):
            if fn.lower().endswith(".csv"):
                df = load_mt5_csv(os.path.join(ddir, fn))
                dfs.append((fn, df))
    if not dfs:
        raise RuntimeError("❌ No datasets found")

    # ---------------------------
    # Discover Methods
    # ---------------------------
    methods = discover_methods()
    print(f"✅ Discovered {len(methods)} methods: {list(methods.keys())[:8]}{'...' if len(methods)>8 else ''}")

    # ---------------------------
    # Shared Config
    # ---------------------------
    cfg = dict(
        threshold=args.threshold,
        sl_atr=args.sl_atr,
        tp_atr=args.tp_atr,
        time_stop=args.time_stop,
    )

    all_results = []
    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # ---------------------------
    # Loop through datasets
    # ---------------------------
    for ds_name, df in dfs:
        if args.sample > 0:
            df = df.head(args.sample).copy()
        print(f"\n[DATA] {ds_name}: {len(df)} rows")

        # ---------------------------
        # Auto-train & evaluate across all backends
        # ---------------------------
        for backend_name, evaluator in get_backends().items():
            print(f"\n🚀 Training & evaluating backend: {backend_name}")

            # Train all models first
            if hasattr(evaluator, "train_and_save"):
                evaluator.train_and_save(df.copy(), methods, cfg)

            # Evaluate
            combo_iter = method_combinations(methods, max(1, args.combo))
            if args.max_combos > 0:
                combo_iter = itertools.islice(combo_iter, args.max_combos)

            for names in combo_iter:
                subset = pick_methods(methods, names)
                print(f"  → Evaluating combo: {names}")
                metrics, trades = evaluator.run(df.copy(), subset, cfg)
                rec = dict(dataset=ds_name, backend=backend_name, methods=names, metrics=metrics, n_trades=len(trades))
                all_results.append(rec)

                safe_combo = sanitize_filename('_'.join(names))
                out_trades = os.path.join(args.outdir, f"trades_{backend_name}_{ds_name}_{safe_combo}_{ts}.csv")

                os.makedirs(os.path.dirname(out_trades), exist_ok=True)
                pd.DataFrame(trades).to_csv(out_trades, index=False)

    # ---------------------------
    # Save summary
    # ---------------------------
    out_summary = os.path.join(args.outdir, f"summary_all_backends_{ts}.json")
    save_json(all_results, out_summary)
    print(f"\n💾 Saved summary to: {out_summary}")

    if all_results:
        df_sum = pd.DataFrame([dict(methods='|'.join(r['methods']), dataset=r['dataset'], backend=r['backend'], **r['metrics']) for r in all_results])
        print("\n🏆 Top 10 by win_rate (all backends):")
        print(df_sum.sort_values("win_rate", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
