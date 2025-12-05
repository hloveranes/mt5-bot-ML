# evaluators/pytorch_evaluator.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Dict, Callable, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Reuse some helpers (duplicated for isolation)
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
    return tr.rolling(period).mean().bfill()

def method_signal_series(df: pd.DataFrame, fn: Callable) -> pd.Series:
    sig = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        try:
            out = fn(df.iloc[: i + 1]) or {}
            s = out.get("signal")
            sig[i] = 1 if s == "BUY" else (-1 if s == "SELL" else 0)
        except Exception:
            sig[i] = 0
    return pd.Series(sig, index=df.index)

def build_features_labels(df: pd.DataFrame, method_fn: Callable) -> Tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(index=df.index)
    X["ret1"] = df["close"].pct_change().fillna(0.0)
    X["hl_range"] = (df["high"] - df["low"]).fillna(0.0)
    X["oc_range"] = (df["close"] - df["open"]).fillna(0.0)
    X["vol"] = df["volume"].fillna(0.0)
    X["atr14"] = compute_atr(df, 14).fillna(0.0)
    X["method_sig"] = method_signal_series(df, method_fn).astype(float)

    y = (df["close"].shift(-1) > df["close"]).astype(int)
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
# Tiny MLP
# -----------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)

# -----------------------
# Backtest (same logic as joblib, but we’ll gate entries by model prob if desired)
# -----------------------
def simulate_with_model(df: pd.DataFrame, method_fn: Callable, model: nn.Module, feat_cols: List[str],
                        threshold: float, sl_atr: float, tp_atr: float, time_stop: int) -> Tuple[Dict, List[Dict]]:
    atr = compute_atr(df, 14).fillna(0.0)
    sig = method_signal_series(df, method_fn).values

    balance = 0.0
    trades = []
    open_trade = None

    # build features incrementally for live-like inference
    X_full, _ = build_features_labels(df, method_fn)
    # pad last row to align loop length
    last = X_full.iloc[[-1]].copy()
    X_stream = pd.concat([X_full, last], axis=0).reset_index(drop=True)

    for i in range(1, len(df)):
        price = df["close"].iloc[i]
        high = df["high"].iloc[i]
        low  = df["low"].iloc[i]
        s = sig[i]

        # model prob (next candle up)
        x = torch.tensor(X_stream.loc[i, feat_cols].values, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            prob_up = torch.softmax(logits, dim=1)[0, 1].item()

        # manage open trade
        if open_trade is not None:
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
            else:
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

            if open_trade is not None and time_stop > 0:
                if (i - open_trade["entry_index"]) >= time_stop:
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

        # entry: require method signal and model confirmation
        if open_trade is None and s != 0 and not np.isnan(atr.iloc[i]):
            # BUY needs prob_up >= threshold; SELL needs (1-prob_up) >= threshold
            if (s == 1 and prob_up >= threshold) or (s == -1 and (1 - prob_up) >= threshold):
                entry = price
                a = atr.iloc[i]
                if s == 1:
                    sl = entry - sl_atr * a if sl_atr > 0 else -np.inf
                    tp = entry + tp_atr * a if tp_atr > 0 else np.inf
                    open_trade = dict(type="BUY", entry=entry, sl=sl, tp=tp, entry_index=i,
                                      method=getattr(method_fn, "__name__", "method"), prob_up=prob_up)
                else:
                    sl = entry + sl_atr * a if sl_atr > 0 else np.inf
                    tp = entry - tp_atr * a if tp_atr > 0 else -np.inf
                    open_trade = dict(type="SELL", entry=entry, sl=sl, tp=tp, entry_index=i,
                                      method=getattr(method_fn, "__name__", "method"), prob_up=prob_up)

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
class PyTorchEvaluator:
    """
    Trains a small MLP per method and saves to model/pytorch/*.pt
    """

    def __init__(self, model_dir: str = "model/pytorch", epochs: int = 5, lr: float = 1e-3, batch: int = 512):
        self.model_dir = model_dir
        self.epochs = epochs
        self.lr = lr
        self.batch = batch
        ensure_dir(self.model_dir)

    def train_and_save(self, df: pd.DataFrame, methods: Dict[str, Callable], cfg: Dict) -> Dict[str, str]:
        for col in df.select_dtypes(include=["float64", "int64"]).columns:
            df[col] = _safe_series(df[col])

        paths = {}
        device = "cuda" if torch.cuda.is_available() else "cpu"

        for name, fn in methods.items():
            X, y = build_features_labels(df, fn)
            Xtr, Xva, ytr, yva = simple_train_val_split(X, y, 0.8)

            model = MLP(in_dim=X.shape[1]).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=self.lr)
            loss_fn = nn.CrossEntropyLoss()

            ds_tr = TensorDataset(torch.tensor(Xtr.values, dtype=torch.float32),
                                torch.tensor(ytr.values, dtype=torch.long))
            dl_tr = DataLoader(ds_tr, batch_size=self.batch, shuffle=True)

            for _ in range(int(cfg.get("epochs", self.epochs))):
                model.train()
                for xb, yb in dl_tr:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    out = model(xb)
                    loss = loss_fn(out, yb)
                    loss.backward()
                    opt.step()

            model.eval()
            with torch.no_grad():
                logits = model(torch.tensor(Xva.values, dtype=torch.float32).to(device))
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                acc = (pred == yva.values).mean()

            # --- Dual-safe filename ---
            safe_name = sanitize_filename(name)
            out = os.path.join(self.model_dir, f"{safe_name}.pt")

            raw_old = os.path.join(self.model_dir, f"{name}.pt")
            if os.path.exists(raw_old):
                os.remove(raw_old)

            torch.save(dict(state_dict=model.state_dict(), feature_cols=list(X.columns)), out)
            paths[name] = out
            print(f"[pytorch] saved {name} -> {out} (val_acc={acc:.3f})")

        return paths

    def run(self, df: pd.DataFrame, methods: Dict[str, Callable], cfg: Dict) -> Tuple[Dict, List[Dict]]:
        for col in df.select_dtypes(include=["float64", "int64"]).columns:
            df[col] = _safe_series(df[col])

        name, fn = next(iter(methods.items()))
        # if a saved model exists, load & use; else do a quick train-on-the-fly for the run
        device = "cuda" if torch.cuda.is_available() else "cpu"
        path = os.path.join(self.model_dir, f"{name}.pt")
        X, _ = build_features_labels(df, fn)
        feat_cols = list(X.columns)

        model = MLP(in_dim=X.shape[1]).to(device)
        if os.path.exists(path):
            blob = torch.load(path, map_location=device)
            model.load_state_dict(blob["state_dict"])
            feat_cols = blob.get("feature_cols", feat_cols)
        else:
            # quick small train to enable a usable run
            self.train_and_save(df, {name: fn}, cfg)

        return simulate_with_model(
            df, fn, model, feat_cols,
            threshold=float(cfg.get("threshold", 0.55)),
            sl_atr=float(cfg.get("sl_atr", 0.8)),
            tp_atr=float(cfg.get("tp_atr", 3.0)),
            time_stop=int(cfg.get("time_stop", 0)),
        )

    def predict(self, df: pd.DataFrame, method_name: str) -> float:
        """
        Predict the probability of upward movement (class=1) using a saved PyTorch model.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        safe_name = sanitize_filename(method_name)
        model_path = os.path.join(self.model_dir, f"{safe_name}.pt")

        # Fallback to raw name
        if not os.path.exists(model_path):
            raw_path = os.path.join(self.model_dir, f"{method_name}.pt")
            if os.path.exists(raw_path):
                model_path = raw_path
            else:
                raise FileNotFoundError(f"Model not found: {model_path}")

        blob = torch.load(model_path, map_location=device)
        feat_cols = blob["feature_cols"]

        model = MLP(in_dim=len(feat_cols)).to(device)
        model.load_state_dict(blob["state_dict"])
        model.eval()

        for col in df.select_dtypes(include=["float64", "int64"]).columns:
            df[col] = _safe_series(df[col])

        X, _ = build_features_labels(df, lambda _: None)
        X_last = torch.tensor(X[feat_cols].iloc[[-1]].values, dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = model(X_last)
            prob_up = torch.softmax(logits, dim=1)[0, 1].item()

        return float(prob_up)

