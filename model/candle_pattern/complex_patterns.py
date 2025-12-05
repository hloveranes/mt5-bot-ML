# patterns/bearish.py
"""
Bearish candle-pattern detectors (bearish-only).
Fast, vectorized, 1–5 candle lookback, with many auto-generated variants.

Exports:
- PATTERNS: dict[str, callable(df) -> bool]
- method_bearish_catalog(df): returns {"signal": "SELL", "name": <pattern>} or None
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Callable, Optional, List, Tuple

# ---------- primitive geometry ----------
def _body(o, c): return np.abs(c - o)
def _upper(h, o, c): return h - np.maximum(o, c)
def _lower(l, o, c): return np.minimum(o, c) - l
def _bear(o, c): return c < o
def _bull(o, c): return c > o

def _last(df: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    n = max(1, min(n, len(df)))
    return df.iloc[-n:]

def _vol_z(df: pd.DataFrame, win: int = 50) -> pd.Series:
    v = df["volume"].astype(float)
    mu = v.rolling(win, min_periods=5).mean()
    sd = v.rolling(win, min_periods=5).std()
    return (v - mu) / (sd.replace(0, np.nan))

# ---------- base checks ----------
def _bearish_engulfing(df, min_ratio=1.0) -> bool:
    if len(df) < 2: return False
    p = df.iloc[-2]; c = df.iloc[-1]
    return (_bull(p.open, p.close) and _bear(c.open, c.close)
            and c.open >= p.close and c.close <= p.open
            and _body(c.open, c.close) >= min_ratio * _body(p.open, p.close))

def _dark_cloud_cover(df, pen_ratio=0.5) -> bool:
    if len(df) < 2: return False
    p = df.iloc[-2]; c = df.iloc[-1]
    mid = (p.open + p.close) / 2.0
    return (_bull(p.open, p.close) and _bear(c.open, c.close)
            and c.open > p.high and c.close < (p.open * (1 - pen_ratio) + p.close * pen_ratio))

def _shooting_star(df, wick_mult=2.0, body_max_mult=0.7) -> bool:
    c = df.iloc[-1]
    b = _body(c.open, c.close) + 1e-12
    u = _upper(c.high, c.open, c.close)
    l = _lower(c.low, c.open, c.close)
    return (u >= wick_mult * b) and (l <= body_max_mult * b) and _bear(c.open, c.close)

def _evening_star(df, small_ratio=0.6) -> bool:
    if len(df) < 3: return False
    a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    return (_bull(a.open, a.close) and (_body(b.open, b.close) < small_ratio * _body(a.open, a.close))
            and _bear(c.open, c.close) and c.close < (a.open + a.close) / 2)

def _three_black_crows(df) -> bool:
    if len(df) < 3: return False
    a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    return _bear(a.open, a.close) and _bear(b.open, b.close) and _bear(c.open, c.close)

def _hanging_man(df, wick_mult=2.5, upper_max=0.6) -> bool:
    c = df.iloc[-1]
    b = _body(c.open, c.close) + 1e-12
    u = _upper(c.high, c.open, c.close)
    l = _lower(c.low, c.open, c.close)
    return (l >= wick_mult * b) and (u <= upper_max * b) and _bear(c.open, c.close)

def _bear_breakout_volume(df, vol_z_thresh=1.5) -> bool:
    if len(df) < 2: return False
    z = _vol_z(df)
    return _bear(df.iloc[-1].open, df.iloc[-1].close) and (z.iloc[-1] >= vol_z_thresh)

# ---------- parameter grids -> many variants ----------
def _grid(values: List, name: str, base_fn) -> Dict[str, Callable]:
    out = {}
    for v in values:
        pat_name = f"{name}[{v}]"
        out[pat_name] = (lambda df, vv=v: base_fn(df, vv))  # bind vv
    return out

PATTERNS: Dict[str, Callable[[pd.DataFrame], bool]] = {}

# Engulfing variants
PATTERNS |= _grid([0.8, 1.0, 1.2, 1.5], "BearishEngulfing", _bearish_engulfing)
# Dark Cloud variants (penetration ratio)
PATTERNS |= _grid([0.4, 0.5, 0.6, 0.7], "DarkCloudCover", _dark_cloud_cover)
# Shooting Star variants
PATTERNS |= _grid([1.8, 2.0, 2.5, 3.0], "ShootingStar", _shooting_star)
# Hanging Man variants
PATTERNS |= _grid([2.0, 2.5, 3.0], "HangingMan", _hanging_man)
# Evening Star fixed + variants
PATTERNS["EveningStar"] = _evening_star
PATTERNS |= _grid([0.5, 0.6, 0.7], "EveningStarCompact", _evening_star)
# Three Black Crows
PATTERNS["ThreeBlackCrows"] = _three_black_crows
# Bearish Volume Breakdown variants
PATTERNS |= _grid([1.0, 1.5, 2.0], "HighVolumeBreakdown", _bear_breakout_volume)

# Expand to ~200+ by composing simple ANDs (programmatic meta-patterns)
def _and(a_fn, b_fn):
    return lambda df: a_fn(df) and b_fn(df)

BASES = list(PATTERNS.items())
for (n1, f1) in BASES:
    for (n2, f2) in BASES:
        if n1 == n2: continue
        combo_name = f"{n1} + {n2}"
        PATTERNS[combo_name] = _and(f1, f2)

# ---------- flat method for library ----------
def method_bearish_catalog(df: pd.DataFrame) -> Optional[dict]:
    """Returns the first bearish hit across the catalog as a SELL signal."""
    # Only inspect last up to 5 bars for speed
    if len(df) < 1: return None
    window = df.iloc[-5:].copy()
    for name, fn in PATTERNS.items():
        try:
            if fn(window):
                return {"signal": "SELL", "name": name}
        except Exception:
            continue
    return None
