# patterns/volume_patterns.py
"""
Pure volume/price interaction patterns for 1–5 candles.
Includes: High Volume Breakout/Breakdown, Low Volume Reversal, Climax Candle, Dry-Up, Accum/Distrib.

Exports:
- PATTERNS
- method_volume_patterns(df) -> {"signal": "BUY"/"SELL", "name": ...} or None
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Callable, Optional

def _vol_z(df: pd.DataFrame, win: int = 50) -> pd.Series:
    v = df["volume"].astype(float)
    mu = v.rolling(win, min_periods=5).mean()
    sd = v.rolling(win, min_periods=5).std()
    return (v - mu) / (sd.replace(0, np.nan))

def _body(o,c): return np.abs(c-o)
def _range(h,l): return h-l
def _bull(o,c): return c>o
def _bear(o,c): return c<o

def _high_vol_breakout(df, zt=1.5) -> bool:
    if len(df) < 3: return False
    z = _vol_z(df)
    sub = df.iloc[-3:]
    return (_bull(sub.iloc[-1].open, sub.iloc[-1].close)
            and z.iloc[-1] >= zt
            and sub.iloc[-1].close > sub.high[:-1].max())

def _high_vol_breakdown(df, zt=1.5) -> bool:
    if len(df) < 3: return False
    z = _vol_z(df)
    sub = df.iloc[-3:]
    return (_bear(sub.iloc[-1].open, sub.iloc[-1].close)
            and z.iloc[-1] >= zt
            and sub.iloc[-1].close < sub.low[:-1].min())

def _low_vol_reversal(df, zt=-0.8) -> bool:
    if len(df) < 3: return False
    z = _vol_z(df)
    sub = df.iloc[-3:]
    # quiet candle at end, after move
    return (z.iloc[-1] <= zt and _range(sub.iloc[-1].high, sub.iloc[-1].low) < 0.6*np.mean(_range(sub.high, sub.low)))

def _climax_candle(df, zt=2.0, body_mult=1.5) -> bool:
    if len(df) < 2: return False
    z = _vol_z(df)
    c = df.iloc[-1]; p = df.iloc[-2]
    return (z.iloc[-1] >= zt and _body(c.open, c.close) >= body_mult * _body(p.open, p.close))

def _dryup_then_break(df, zt=-0.7, side="buy") -> bool:
    if len(df) < 4: return False
    z = _vol_z(df)
    if not (z.iloc[-2] <= zt): return False
    last = df.iloc[-1]
    if side == "buy":
        return last.close > df.iloc[-2].high
    else:
        return last.close < df.iloc[-2].low

PATTERNS: Dict[str, Callable[[pd.DataFrame], bool]] = {}
for z in [1.2, 1.5, 2.0]:
    PATTERNS[f"HighVolumeBreakout[z>{z}]"] = (lambda df, zz=z: _high_vol_breakout(df, zz))
    PATTERNS[f"HighVolumeBreakdown[z>{z}]"] = (lambda df, zz=z: _high_vol_breakdown(df, zz))
for z in [-0.6, -0.8, -1.0]:
    PATTERNS[f"LowVolumeReversal[z<{z}]"] = (lambda df, zz=z: _low_vol_reversal(df, zz))
for z in [1.8, 2.0, 2.5]:
    for bm in [1.2, 1.5, 2.0]:
        PATTERNS[f"ClimaxCandle[z>{z},body>{bm}x]"] = (lambda df, zz=z, bb=bm: _climax_candle(df, zz, bb))
for side in ["buy","sell"]:
    for z in [-0.5,-0.7,-0.9]:
        PATTERNS[f"DryUpThenBreak[{side},z<{z}]"] = (lambda df, s=side, zz=z: _dryup_then_break(df, zz, s))

def method_volume_patterns(df: pd.DataFrame) -> Optional[dict]:
    if len(df) < 3: return None
    win = df.iloc[-5:].copy()
    for name, fn in PATTERNS.items():
        try:
            if fn(win):
                sig = "BUY" if ("Breakout" in name or "[buy" in name) else "SELL" if ("Breakdown" in name or "[sell" in name) else None
                if sig is None:
                    # generic patterns (Climax / LowVolumeReversal) — infer from last candle color
                    last = win.iloc[-1]
                    sig = "BUY" if last.close > last.open else "SELL"
                return {"signal": sig, "name": name}
        except Exception:
            continue
    return None
