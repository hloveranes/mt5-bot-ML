# patterns/bullish.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Callable, Optional

"""
Bullish price-action patterns (vectorized).
Each function inspects the last 1–5 candles and returns:
    {"signal": "BUY" | None, "name": "<PatternName>"}

Exported:
    PATTERNS: Dict[str, Callable[[pd.DataFrame], dict]]
"""

# --------------------
# Core helpers (vectorized)
# --------------------
def _last(df: pd.DataFrame, n: int = 1):
    return df.iloc[-n:] if len(df) >= n else df.iloc[0:0]

def _body(o, c):
    return np.abs(c - o)

def _is_bull(o, c):
    return c > o

def _is_bear(o, c):
    return o > c

def _upper(h, o, c):
    return h - np.maximum(o, c)

def _lower(l, o, c):
    return np.minimum(o, c) - l

def _avg_body(df: pd.DataFrame, n: int = 10):
    if len(df) < n:
        n = len(df)
    if n <= 0:
        return np.nan
    o = df["open"].iloc[-n:].values
    c = df["close"].iloc[-n:].values
    return np.nanmean(np.abs(c - o))

# --------------------
# Single-candle factories
# --------------------
def make_hammer(name: str, wick_ratio: float = 2.0, max_upper_body_ratio: float = 0.6):
    """
    Hammer: long lower shadow (>= wick_ratio * body), small upper shadow.
    BUY bias if close > open (bull hammer).
    """
    def fn(df: pd.DataFrame):
        if len(df) < 1:
            return {"signal": None, "name": name}
        r = df.iloc[-1]
        o, h, l, c = r["open"], r["high"], r["low"], r["close"]
        b = abs(c - o) + 1e-12
        lower = min(c, o) - l
        upper = h - max(c, o)
        cond = (lower >= wick_ratio * b) and (upper <= max_upper_body_ratio * b) and (c > o)
        return {"signal": "BUY" if cond else None, "name": name}
    return fn

def make_bullish_marubozu(name: str, body_min_frac_of_range: float = 0.85):
    """
    Bullish Marubozu: large full body near range extremes.
    """
    def fn(df: pd.DataFrame):
        if len(df) < 1:
            return {"signal": None, "name": name}
        r = df.iloc[-1]
        o, h, l, c = r["open"], r["high"], r["low"], r["close"]
        rng = max(h - l, 1e-12)
        b = abs(c - o)
        cond = (c > o) and (b / rng >= body_min_frac_of_range)
        return {"signal": "BUY" if cond else None, "name": name}
    return fn

def make_bullish_engulfing(name: str, min_body_ratio: float = 1.0):
    """
    Bullish Engulfing (2-candle): candle 2 is bullish and its body engulfs candle 1's body by ratio.
    """
    def fn(df: pd.DataFrame):
        if len(df) < 2:
            return {"signal": None, "name": name}
        a, b = df.iloc[-2], df.iloc[-1]
        oa, ca = a["open"], a["close"]
        ob, cb = b["open"], b["close"]
        body_a = abs(ca - oa) + 1e-12
        body_b = abs(cb - ob)
        cond = _is_bear(oa, ca) and _is_bull(ob, cb) and (ob <= ca) and (cb >= oa) and (body_b >= min_body_ratio * body_a)
        return {"signal": "BUY" if cond else None, "name": name}
    return fn

def make_bullish_harami(name: str, max_child_ratio: float = 0.9):
    """
    Bullish Harami (2-candle): small bullish body fully inside prior bearish body.
    """
    def fn(df: pd.DataFrame):
        if len(df) < 2:
            return {"signal": None, "name": name}
        a, b = df.iloc[-2], df.iloc[-1]
        oa, ca = a["open"], a["close"]
        ob, cb = b["open"], b["close"]
        body_a = abs(ca - oa) + 1e-12
        body_b = abs(cb - ob)
        inside = (min(ob, cb) > min(oa, ca)) and (max(ob, cb) < max(oa, ca))
        cond = _is_bear(oa, ca) and _is_bull(ob, cb) and inside and (body_b <= max_child_ratio * body_a)
        return {"signal": "BUY" if cond else None, "name": name}
    return fn

def make_piercing(name: str, min_penetration: float = 0.5):
    """
    Piercing Pattern (2-candle): bearish then bullish closing above the mid of prior body.
    """
    def fn(df: pd.DataFrame):
        if len(df) < 2:
            return {"signal": None, "name": name}
        a, b = df.iloc[-2], df.iloc[-1]
        oa, ca = a["open"], a["close"]
        ob, cb = b["open"], b["close"]
        mid_prev = (oa + ca) / 2.0
        cond = _is_bear(oa, ca) and (ob < min(oa, ca)) and _is_bull(ob, cb) and (cb > mid_prev + (min_penetration - 0.5) * abs(oa - ca))
        return {"signal": "BUY" if cond else None, "name": name}
    return fn

# --------------------
# Multi-candle factories
# --------------------
def make_morning_star(name: str, mid_close_bias: float = 0.5, small_middle_ratio: float = 0.6):
    """
    Morning Star (3-candle): big red, small indecision, big green closing above mid of first.
    """
    def fn(df: pd.DataFrame):
        if len(df) < 3:
            return {"signal": None, "name": name}
        a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        body_a = abs(a["close"] - a["open"])
        body_b = abs(b["close"] - b["open"])
        body_c = abs(c["close"] - c["open"])
        mid_a = (a["open"] + a["close"]) / 2.0
        cond = (
            _is_bear(a["open"], a["close"])
            and (body_b <= small_middle_ratio * body_a)
            and _is_bull(c["open"], c["close"])
            and (c["close"] > mid_a + (mid_close_bias - 0.5) * body_a)
        )
        return {"signal": "BUY" if cond else None, "name": name}
    return fn

def make_three_white_soldiers(name: str):
    """
    Three White Soldiers (3-candle): three consecutive bullish bodies.
    """
    def fn(df: pd.DataFrame):
        if len(df) < 3:
            return {"signal": None, "name": name}
        a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        cond = _is_bull(a["open"], a["close"]) and _is_bull(b["open"], b["close"]) and _is_bull(c["open"], c["close"])
        return {"signal": 'BUY' if cond else None, "name": name}
    return fn

def make_bullish_inside_break(name: str):
    """
    Inside bar then bullish breakout (2+ candles).
    """
    def fn(df: pd.DataFrame):
        if len(df) < 3:
            return {"signal": None, "name": name}
        a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        inside = (b["high"] <= a["high"]) and (b["low"] >= a["low"])
        breakout_up = c["close"] > a["high"]
        cond = inside and breakout_up
        return {"signal": "BUY" if cond else None, "name": name}
    return fn

# --------------------
# Build a large registry via parameter grids
# --------------------
PATTERNS: Dict[str, Callable] = {}

# Hammers with multiple thresholds
for wr in (1.8, 2.0, 2.5, 3.0, 4.0):
    for ub in (0.4, 0.6, 0.8):
        nm = f"bullish_hammer_wr{wr}_ub{ub}"
        PATTERNS[nm] = make_hammer(nm, wick_ratio=wr, max_upper_body_ratio=ub)

# Bullish engulfing variations
for ratio in (0.8, 1.0, 1.2, 1.5):
    nm = f"bullish_engulfing_x{ratio}"
    PATTERNS[nm] = make_bullish_engulfing(nm, min_body_ratio=ratio)

# Bullish harami variations
for child in (0.5, 0.7, 0.9):
    nm = f"bullish_harami_child<=x{child}"
    PATTERNS[nm] = make_bullish_harami(nm, max_child_ratio=child)

# Morning star variations
for mid_bias in (0.5, 0.6, 0.7):
    for small_ratio in (0.4, 0.6):
        nm = f"morning_star_mid{mid_bias}_small{small_ratio}"
        PATTERNS[nm] = make_morning_star(nm, mid_close_bias=mid_bias, small_middle_ratio=small_ratio)

# Three white soldiers (single)
PATTERNS["three_white_soldiers"] = make_three_white_soldiers("three_white_soldiers")

# Piercing variants
for pen in (0.5, 0.6, 0.7):
    nm = f"piercing_pen{pen}"
    PATTERNS[nm] = make_piercing(nm, min_penetration=pen)

# Inside-break bullish
PATTERNS["bullish_inside_break"] = make_bullish_inside_break("bullish_inside_break")

# Marubozu variants
for frac in (0.8, 0.85, 0.9, 0.95):
    nm = f"bullish_marubozu_body>=x{frac}"
    PATTERNS[nm] = make_bullish_marubozu(nm, body_min_frac_of_range=frac)
