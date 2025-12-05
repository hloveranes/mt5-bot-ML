# patterns/exhaustion_patterns.py
"""
Exhaustion / end-of-trend patterns (both sides) in 1–5 bars.
Buying/Selling Climax, Final Push, Spike & Reverse, Long Wick Rejection.

Exports:
- PATTERNS
- method_exhaustion(df)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Callable, Optional

def _body(o,c): return np.abs(c-o)
def _upper(h,o,c): return h - np.maximum(o,c)
def _lower(l,o,c): return np.minimum(o,c) - l
def _rng(h,l): return h-l
def _bull(o,c): return c>o
def _bear(o,c): return c<o

def _buying_climax(df, wick_mult=1.8, body_mult=1.3) -> bool:
    if len(df) < 2: return False
    c = df.iloc[-1]; p = df.iloc[-2]
    u = _upper(c.high, c.open, c.close)
    b = _body(c.open, c.close) + 1e-12
    return _bull(c.open, c.close) and u >= wick_mult*b and b >= body_mult*_body(p.open, p.close)

def _selling_climax(df, wick_mult=1.8, body_mult=1.3) -> bool:
    if len(df) < 2: return False
    c = df.iloc[-1]; p = df.iloc[-2]
    l = _lower(c.low, c.open, c.close)
    b = _body(c.open, c.close) + 1e-12
    return _bear(c.open, c.close) and l >= wick_mult*b and b >= body_mult*_body(p.open, p.close)

def _spike_reverse(df, side="buy", wick_mult=2.0) -> bool:
    if len(df) < 1: return False
    c = df.iloc[-1]
    b = _body(c.open, c.close) + 1e-12
    if side == "buy":
        return _lower(c.low, c.open, c.close) >= wick_mult*b and _bull(c.open, c.close)
    else:
        return _upper(c.high, c.open, c.close) >= wick_mult*b and _bear(c.open, c.close)

def _final_push(df, k=3, side="buy") -> bool:
    if len(df) < k: return False
    sub = df.iloc[-k:]
    if side == "buy":
        return np.all(sub.close.diff().fillna(0) > 0)
    else:
        return np.all(sub.close.diff().fillna(0) < 0)

PATTERNS: Dict[str, Callable[[pd.DataFrame], bool]] = {}
for wm in [1.5, 1.8, 2.0]:
    for bm in [1.1, 1.3, 1.5]:
        PATTERNS[f"BuyingClimax[w>{wm},b>{bm}]"] = (lambda df, w=wm, b=bm: _buying_climax(df, w, b))
        PATTERNS[f"SellingClimax[w>{wm},b>{bm}]"] = (lambda df, w=wm, b=bm: _selling_climax(df, w, b))
for side in ["buy","sell"]:
    for wm in [1.8,2.0,2.2]:
        PATTERNS[f"SpikeReverse[{side},w>{wm}]"] = (lambda df, s=side, w=wm: _spike_reverse(df, s, w))
for side in ["buy","sell"]:
    for k in [3,4,5]:
        PATTERNS[f"FinalPush[{side},{k}]"] = (lambda df, s=side, kk=k: _final_push(df, kk, s))

def method_exhaustion(df: pd.DataFrame) -> Optional[dict]:
    if len(df) < 2: return None
    win = df.iloc[-5:].copy()
    for name, fn in PATTERNS.items():
        try:
            if fn(win):
                side = "BUY" if ("Buying" in name or "[buy" in name) else "SELL"
                return {"signal": side, "name": name}
        except Exception:
            continue
    return None
