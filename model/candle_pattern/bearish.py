# patterns/bearish.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Callable

"""
Bearish price-action patterns (vectorized).
Returns {"signal": "SELL" | None, "name": "<PatternName>"}.
"""

def _body(o, c): return np.abs(c - o)
def _is_bear(o, c): return o > c
def _is_bull(o, c): return c > o
def _upper(h, o, c): return h - np.maximum(o, c)
def _lower(l, o, c): return np.minimum(o, c) - l

# -------- factories --------
def make_shooting_star(name: str, wick_ratio: float = 2.0, max_lower_body_ratio: float = 0.6):
    def fn(df: pd.DataFrame):
        if len(df) < 1:
            return {"signal": None, "name": name}
        r = df.iloc[-1]
        o, h, l, c = r["open"], r["high"], r["low"], r["close"]
        b = _body(o, c) + 1e-12
        upper = h - max(o, c)
        lower = min(o, c) - l
        cond = (upper >= wick_ratio * b) and (lower <= max_lower_body_ratio * b) and _is_bear(o, c)
        return {"signal": "SELL" if cond else None, "name": name}
    return fn

def make_bearish_engulfing(name: str, min_body_ratio: float = 1.0):
    def fn(df: pd.DataFrame):
        if len(df) < 2:
            return {"signal": None, "name": name}
        a, b = df.iloc[-2], df.iloc[-1]
        oa, ca = a["open"], a["close"]
        ob, cb = b["open"], b["close"]
        body_a = _body(oa, ca) + 1e-12
        body_b = _body(ob, cb)
        cond = _is_bull(oa, ca) and _is_bear(ob, cb) and (ob >= ca) and (cb <= oa) and (body_b >= min_body_ratio * body_a)
        return {"signal": "SELL" if cond else None, "name": name}
    return fn

def make_bearish_harami(name: str, max_child_ratio: float = 0.9):
    def fn(df: pd.DataFrame):
        if len(df) < 2:
            return {"signal": None, "name": name}
        a, b = df.iloc[-2], df.iloc[-1]
        oa, ca = a["open"], a["close"]
        ob, cb = b["open"], b["close"]
        body_a = _body(oa, ca) + 1e-12
        body_b = _body(ob, cb)
        inside = (min(ob, cb) > min(oa, ca)) and (max(ob, cb) < max(oa, ca))
        cond = _is_bull(oa, ca) and _is_bear(ob, cb) and inside and (body_b <= max_child_ratio * body_a)
        return {"signal": "SELL" if cond else None, "name": name}
    return fn

def make_dark_cloud_cover(name: str, min_penetration: float = 0.5):
    def fn(df: pd.DataFrame):
        if len(df) < 2:
            return {"signal": None, "name": name}
        a, b = df.iloc[-2], df.iloc[-1]
        oa, ca = a["open"], a["close"]
        ob, cb = b["open"], b["close"]
        mid_prev = (oa + ca) / 2.0
        cond = _is_bull(oa, ca) and (ob > max(oa, ca)) and _is_bear(ob, cb) and (cb < mid_prev - (min_penetration - 0.5) * _body(oa, ca))
        return {"signal": "SELL" if cond else None, "name": name}
    return fn

def make_evening_star(name: str, mid_close_bias: float = 0.5, small_middle_ratio: float = 0.6):
    def fn(df: pd.DataFrame):
        if len(df) < 3:
            return {"signal": None, "name": name}
        a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        body_a = _body(a["open"], a["close"])
        body_b = _body(b["open"], b["close"])
        mid_a = (a["open"] + a["close"]) / 2.0
        cond = (
            _is_bull(a["open"], a["close"])
            and (body_b <= small_middle_ratio * body_a)
            and _is_bear(c["open"], c["close"])
            and (c["close"] < mid_a - (mid_close_bias - 0.5) * body_a)
        )
        return {"signal": "SELL" if cond else None, "name": name}
    return fn

def make_three_black_crows(name: str):
    def fn(df: pd.DataFrame):
        if len(df) < 3:
            return {"signal": None, "name": name}
        a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        cond = _is_bear(a["open"], a["close"]) and _is_bear(b["open"], b["close"]) and _is_bear(c["open"], c["close"])
        return {"signal": "SELL" if cond else None, "name": name}
    return fn

def make_bearish_inside_break(name: str):
    def fn(df: pd.DataFrame):
        if len(df) < 3:
            return {"signal": None, "name": name}
        a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        inside = (b["high"] <= a["high"]) and (b["low"] >= a["low"])
        breakout_dn = c["close"] < a["low"]
        return {"signal": "SELL" if (inside and breakout_dn) else None, "name": name}
    return fn

def make_bearish_marubozu(name: str, body_min_frac_of_range: float = 0.85):
    def fn(df: pd.DataFrame):
        if len(df) < 1:
            return {"signal": None, "name": name}
        r = df.iloc[-1]
        o, h, l, c = r["open"], r["high"], r["low"], r["close"]
        rng = max(h - l, 1e-12)
        b = _body(o, c)
        cond = (o > c) and (b / rng >= body_min_frac_of_range)
        return {"signal": "SELL" if cond else None, "name": name}
    return fn

# -------- registry via parameter grids --------
PATTERNS: Dict[str, Callable] = {}

for wr in (1.8, 2.0, 2.5, 3.0, 4.0):
    for lb in (0.4, 0.6, 0.8):
        nm = f"shooting_star_wr{wr}_lb{lb}"
        PATTERNS[nm] = make_shooting_star(nm, wick_ratio=wr, max_lower_body_ratio=lb)

for ratio in (0.8, 1.0, 1.2, 1.5):
    nm = f"bearish_engulfing_x{ratio}"
    PATTERNS[nm] = make_bearish_engulfing(nm, min_body_ratio=ratio)

for child in (0.5, 0.7, 0.9):
    nm = f"bearish_harami_child<=x{child}"
    PATTERNS[nm] = make_bearish_harami(nm, max_child_ratio=child)

for pen in (0.5, 0.6, 0.7):
    nm = f"dark_cloud_cover_pen{pen}"
    PATTERNS[nm] = make_dark_cloud_cover(nm, min_penetration=pen)

for mid in (0.5, 0.6, 0.7):
    for small in (0.4, 0.6):
        nm = f"evening_star_mid{mid}_small{small}"
        PATTERNS[nm] = make_evening_star(nm, mid_close_bias=mid, small_middle_ratio=small)

PATTERNS["three_black_crows"] = make_three_black_crows("three_black_crows")
PATTERNS["bearish_inside_break"] = make_bearish_inside_break("bearish_inside_break")

for frac in (0.8, 0.85, 0.9, 0.95):
    nm = f"bearish_marubozu_body>=x{frac}"
    PATTERNS[nm] = make_bearish_marubozu(nm, body_min_frac_of_range=frac)
