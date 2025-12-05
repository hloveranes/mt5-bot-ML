# patterns/continuation.py
"""
Continuation patterns (both sides) in 1–5 candles.
Examples: Rising Three, Falling Three, Flags, Pennants, Inside-Bar Continuations, Breakout-Retest.

Exports:
- PATTERNS (dict)
- method_continuation(df) -> {"signal": "BUY"/"SELL", "name": ...} or None
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Callable, Optional

def _body(o, c): return np.abs(c - o)
def _bull(o, c): return c > o
def _bear(o, c): return c < o

def _rising_three(df, small_ratio=0.35) -> bool:
    if len(df) < 5: return False
    a,b,c,d,e = df.iloc[-5], df.iloc[-4], df.iloc[-3], df.iloc[-2], df.iloc[-1]
    cond = (_bull(a.open,a.close)
            and _bear(b.open,b.close) and _bear(c.open,c.close) and _bear(d.open,d.close)
            and _bull(e.open,e.close)
            and (b.close < a.close) and (c.close < a.close) and (d.close < a.close)
            and e.close > a.close)
    # small pullback bodies
    cond &= (_body(b.open,b.close) < small_ratio*_body(a.open,a.close)
             and _body(c.open,c.close) < small_ratio*_body(a.open,a.close)
             and _body(d.open,d.close) < small_ratio*_body(a.open,a.close))
    return cond

def _falling_three(df, small_ratio=0.35) -> bool:
    if len(df) < 5: return False
    a,b,c,d,e = df.iloc[-5], df.iloc[-4], df.iloc[-3], df.iloc[-2], df.iloc[-1]
    cond = (_bear(a.open,a.close)
            and _bull(b.open,b.close) and _bull(c.open,c.close) and _bull(d.open,d.close)
            and _bear(e.open,e.close)
            and (b.close > a.close) and (c.close > a.close) and (d.close > a.close)
            and e.close < a.close)
    cond &= (_body(b.open,b.close) < small_ratio*_body(a.open,a.close)
             and _body(c.open,c.close) < small_ratio*_body(a.open,a.close)
             and _body(d.open,d.close) < small_ratio*_body(a.open,a.close))
    return cond

def _inside_bar_continuation(df, bias="buy") -> bool:
    if len(df) < 2: return False
    p = df.iloc[-2]; c = df.iloc[-1]
    inside = (c.high <= p.high) and (c.low >= p.low)
    if not inside: return False
    if bias == "buy":
        return c.close > p.close
    else:
        return c.close < p.close

def _flag_break(df, look=5, side="buy") -> bool:
    # simple: recent consolidation then break through prior extreme
    sub = df.iloc[-look:]
    if len(sub) < 3: return False
    hh = sub.high.max(); ll = sub.low.min()
    last = sub.iloc[-1]
    if side == "buy":
        return last.close > hh
    else:
        return last.close < ll

# grid
PATTERNS: Dict[str, Callable[[pd.DataFrame], bool]] = {}
for r in [0.3, 0.35, 0.4]:
    PATTERNS[f"RisingThree[{r}]"] = (lambda df, rr=r: _rising_three(df, rr))
    PATTERNS[f"FallingThree[{r}]"] = (lambda df, rr=r: _falling_three(df, rr))

for side in ["buy", "sell"]:
    for look in [3,4,5]:
        PATTERNS[f"FlagBreak[{side},{look}]"] = (lambda df, lk=look, sd=side: _flag_break(df, lk, sd))

for bias in ["buy", "sell"]:
    PATTERNS[f"InsideBarCont[{bias}]"] = (lambda df, b=bias: _inside_bar_continuation(df, b))

def method_continuation(df: pd.DataFrame) -> Optional[dict]:
    if len(df) < 2: return None
    win = df.iloc[-5:].copy()
    for name, fn in PATTERNS.items():
        try:
            if fn(win):
                sig = "BUY" if ("Rising" in name or "buy" in name) else "SELL"
                return {"signal": sig, "name": name}
        except Exception:
            continue
    return None
