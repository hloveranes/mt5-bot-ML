# evaluators/joblib_runner.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Callable, Tuple, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import time

# -----------------------
# Small utilities
# -----------------------
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

def ensure_dir(path: str):
    """
    Safely create a directory if it doesn't exist.
    Works even if a file with the same name exists.
    """
    if os.path.exists(path):
        if not os.path.isdir(path):
            # A file with this name exists; remove or rename it
            os.remove(path)
            os.makedirs(path, exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    # pandas deprecates fillna(method="bfill") -> use bfill()
    return tr.rolling(period).mean().bfill()

def method_signal_series(df: pd.DataFrame, fn: Callable) -> pd.Series:
    """
    Run method function row-by-row, returning a numeric series:
      BUY=+1, SELL=-1, None=0
    Method signature: fn(df_slice) -> {"signal": "BUY"/"SELL"/None, "name": "..."}
    """
    sig = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        sub = df.iloc[: i + 1]
        try:
            out = fn(sub) or {}
            s = out.get("signal")
            if s == "BUY":
                sig[i] = 1
            elif s == "SELL":
                sig[i] = -1
        except Exception:
            sig[i] = 0
    return pd.Series(sig, index=df.index)

def build_features_labels(df: pd.DataFrame, method_fn: Callable) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Features: OHLCV-derived + ATR + method_signal
    Target: next-candle direction (1 if close_{t+1} > close_t else 0)
    """
    X = pd.DataFrame(index=df.index)
    X["ret1"] = df["close"].pct_change().fillna(0.0)
    X["hl_range"] = (df["high"] - df["low"]).fillna(0.0)
    X["oc_range"] = (df["close"] - df["open"]).fillna(0.0)
    X["vol"] = df["volume"].fillna(0.0)
    X["atr14"] = compute_atr(df, 14).fillna(0.0)
    X["method_sig"] = method_signal_series(df, method_fn).astype(float)

    y = (df["close"].shift(-1) > df["close"]).astype(int)  # 1=up next bar
    X = X.iloc[:-1].copy()
    y = y.iloc[:-1].copy()
    return X, y

def simple_train_val_split(X: pd.DataFrame, y: pd.Series, split: float = 0.8):
    n = len(X)
    k = int(n * split)
    return (X.iloc[:k], X.iloc[k:], y.iloc[:k], y.iloc[k:])

def _safe_series(x):
    """Replace NaN/Inf with finite numbers."""
    return pd.Series(x).replace([np.inf, -np.inf], np.nan).fillna(0.0)

# -----------------------
# Backtest engine (exit: opposite signal or SL/TP/time)
# -----------------------
@dataclass
class TradeCfg:
    sl_atr: float = 0.8
    tp_atr: float = 3.0
    time_stop: int = 0  # 0 disables
    threshold: float = 0.55  # used by ML backends; here unused but kept for API parity

def simulate_method(df: pd.DataFrame, method_fn: Callable, cfg: TradeCfg) -> Tuple[Dict, List[Dict]]:
    atr = compute_atr(df, 14).fillna(0.0)
    sig = method_signal_series(df, method_fn).values

    balance = 0.0
    trades = []
    open_trade = None

    for i in range(1, len(df)):
        price = df["close"].iloc[i]
        high = df["high"].iloc[i]
        low  = df["low"].iloc[i]
        s = sig[i]  # +1 buy, -1 sell, 0 flat

        # manage open trade
        if open_trade is not None:
            # SL/TP
            if open_trade["type"] == "BUY":
                if low <= open_trade["sl"]:
                    pnl = open_trade["sl"] - open_trade["entry"]
                    balance += pnl
                    open_trade.update(exit_index=i, exit=open_trade["sl"], profit=pnl)
                    trades.append(open_trade); open_trade = None
                elif high >= open_trade["tp"]:
                    pnl = open_trade["tp"] - open_trade["entry"]
                    balance += pnl
                    open_trade.update(exit_index=i, exit=open_trade["tp"], profit=pnl)
                    trades.append(open_trade); open_trade = None
            else:  # SELL
                if high >= open_trade["sl"]:
                    pnl = open_trade["entry"] - open_trade["sl"]
                    balance += pnl
                    open_trade.update(exit_index=i, exit=open_trade["sl"], profit=pnl)
                    trades.append(open_trade); open_trade = None
                elif low <= open_trade["tp"]:
                    pnl = open_trade["entry"] - open_trade["tp"]
                    balance += pnl
                    open_trade.update(exit_index=i, exit=open_trade["tp"], profit=pnl)
                    trades.append(open_trade); open_trade = None

            # time stop
            if open_trade is not None and cfg.time_stop > 0:
                if (i - open_trade["entry_index"]) >= cfg.time_stop:
                    # exit at market
                    m_exit = price
                    pnl = (m_exit - open_trade["entry"]) if open_trade["type"] == "BUY" \
                        else (open_trade["entry"] - m_exit)
                    balance += pnl
                    open_trade.update(exit_index=i, exit=m_exit, profit=pnl)
                    trades.append(open_trade); open_trade = None

            # opposite signal exit
            if open_trade is not None:
                if (open_trade["type"] == "BUY" and s == -1) or (open_trade["type"] == "SELL" and s == 1):
                    m_exit = price
                    pnl = (m_exit - open_trade["entry"]) if open_trade["type"] == "BUY" \
                        else (open_trade["entry"] - m_exit)
                    balance += pnl
                    open_trade.update(exit_index=i, exit=m_exit, profit=pnl)
                    trades.append(open_trade); open_trade = None

        # flat -> open on signal
        if open_trade is None and s != 0 and not np.isnan(atr.iloc[i]):
            entry = price
            a = atr.iloc[i]
            if s == 1:
                sl = entry - cfg.sl_atr * a if cfg.sl_atr > 0 else -np.inf
                tp = entry + cfg.tp_atr * a if cfg.tp_atr > 0 else np.inf
                open_trade = dict(type="BUY", entry=entry, sl=sl, tp=tp, entry_index=i, method=getattr(method_fn, "__name__", "method"))
            elif s == -1:
                sl = entry + cfg.sl_atr * a if cfg.sl_atr > 0 else np.inf
                tp = entry - cfg.tp_atr * a if cfg.tp_atr > 0 else -np.inf
                open_trade = dict(type="SELL", entry=entry, sl=sl, tp=tp, entry_index=i, method=getattr(method_fn, "__name__", "method"))

    # metrics
    if len(trades) == 0:
        return dict(win_rate=0.0, pnl=0.0, n_trades=0, avg_profit=0.0), trades
    pnl = sum(t["profit"] for t in trades)
    wins = sum(1 for t in trades if t["profit"] > 0)
    wr = wins / len(trades)
    avgp = pnl / len(trades)
    return dict(win_rate=wr, pnl=pnl, n_trades=len(trades), avg_profit=avgp), trades

# -----------------------
# Evaluator
# -----------------------
class JoblibRunner:
    """
    Trains a simple RandomForest per method and saves to model/joblib/*.joblib
    Also supports rule-only backtest via `run`.
    """

    def __init__(self, model_dir: str = "model/joblib"):
        self.model_dir = model_dir
        ensure_dir(self.model_dir)

    def train_and_save(self, df: pd.DataFrame, methods: Dict[str, Callable], cfg: Dict) -> Dict[str, str]:
        # 🧹 Sanitize numeric columns to avoid NaN/Inf in training
        for col in df.select_dtypes(include=["float64", "int64"]).columns:
            df[col] = _safe_series(df[col])

        total = len(methods)
        paths = {}
        start_time = time.time()

        print(f"🧠 Starting Joblib training for {total} methods...")

        for idx, (name, fn) in enumerate(methods.items(), 1):
            try:
                print(f"\n[{idx}/{total}] ⚙️ Training method: {name} ...")
                method_start = time.time()

                X, y = build_features_labels(df, fn)
                if X is None or y is None or len(X) == 0:
                    print(f"⚠️ Skipping {name}: empty dataset or bad feature output.")
                    continue

                Xtr, Xva, ytr, yva = simple_train_val_split(X, y, 0.8)
                clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
                clf.fit(Xtr, ytr)
                acc = accuracy_score(yva, clf.predict(Xva))

                # --- Dual-safe filename ---
                safe_name = sanitize_filename(name)
                out = os.path.join(self.model_dir, f"{safe_name}.joblib")

                # If old raw file exists, remove to avoid confusion
                raw_old = os.path.join(self.model_dir, f"{name}.joblib")
                if os.path.exists(raw_old):
                    os.remove(raw_old)

                joblib.dump(dict(model=clf, feature_cols=list(X.columns), val_acc=float(acc)), out)
                paths[name] = out

                elapsed = time.time() - method_start
                print(f"✅ [joblib] saved {name} -> {out} (val_acc={acc:.3f}) | took {elapsed:.2f}s")

            except Exception as e:
                print(f"❌ Error training {name}: {e}")
                continue

        print(f"\n🏁 Finished training {len(paths)}/{total} methods in {(time.time() - start_time)/60:.2f} minutes.")
        return paths



    def run(self, df: pd.DataFrame, methods: Dict[str, Callable], cfg: Dict) -> Tuple[Dict, List[Dict]]:
        """
        For probability bot API — if multiple methods passed, we simulate the first one.
        (Your orchestration runs each method individually anyway.)
        """
        # 🧹 Ensure numeric safety before simulation
        for col in df.select_dtypes(include=["float64", "int64"]).columns:
            df[col] = _safe_series(df[col])

        name, fn = next(iter(methods.items()))
        tcfg = TradeCfg(
            sl_atr=float(cfg.get("sl_atr", 0.8)),
            tp_atr=float(cfg.get("tp_atr", 3.0)),
            time_stop=int(cfg.get("time_stop", 0)),
            threshold=float(cfg.get("threshold", 0.55)),
        )
        return simulate_method(df, fn, tcfg)

    def predict(self, df: pd.DataFrame, method_name: str) -> float:
        """
        Predict the probability of upward movement (class=1) using a saved joblib model.
        Returns a float between 0 and 1.
        """
        safe_name = sanitize_filename(method_name)
        model_path = os.path.join(self.model_dir, f"{safe_name}.joblib")

        # Fallback to raw name
        if not os.path.exists(model_path):
            raw_path = os.path.join(self.model_dir, f"{method_name}.joblib")
            if os.path.exists(raw_path):
                model_path = raw_path
            else:
                raise FileNotFoundError(f"Model not found: {model_path}")

        blob = joblib.load(model_path)
        model = blob["model"]
        feat_cols = blob["feature_cols"]

        for col in df.select_dtypes(include=["float64", "int64"]).columns:
            df[col] = _safe_series(df[col])

        X, _ = build_features_labels(df, lambda _: None)
        X_last = X[feat_cols].iloc[[-1]]
        prob = model.predict_proba(X_last)[0, 1]
        return float(prob)



