"""
Trading Methods Library (Institutional + Retail Classics)
---------------------------------------------------------
A single-file collection of common, practical trading methods you can call
independently. Each function returns a tiny dict like:

    {"signal": "BUY" | "SELL" | None, "name": "<Method Name>"}

Data expectation:
    df: pandas.DataFrame with at least columns
        ["open", "high", "low", "close", "volume"] indexed in time order.

Notes:
- Methods are intentionally simple but battle‑tested variations.
- Keep your own risk, position sizing, and conflict resolution separate.
- All thresholds are configurable via parameters.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict


# =============================================================
# Candle Pattern Imports
# =============================================================
from model.candle_pattern.bearish import PATTERNS as BEARISH
from model.candle_pattern.bullish import PATTERNS as BULLISH  
from model.candle_pattern.complex_patterns import PATTERNS as COMPLEX
from model.candle_pattern.continuation import PATTERNS as CONT
from model.candle_pattern.exhaustion_patterns import PATTERNS as EXH
from model.candle_pattern.volume_patterns import PATTERNS as VOL

# =============================================================
# Core helpers (minimal, dependency‑free)
# =============================================================

def _ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()


def _sma(x: pd.Series, length: int) -> pd.Series:
    return x.rolling(length).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    return tr.rolling(period).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow
    sig = _ema(macd, signal)
    hist = macd - sig
    return macd, sig, hist


def _bb(close: pd.Series, length: int = 20, mult: float = 2.0):
    ma = _sma(close, length)
    std = close.rolling(length).std()
    upper = ma + mult * std
    lower = ma - mult * std
    return ma, upper, lower


def _stoch(df: pd.DataFrame, k: int = 14, d: int = 3):
    low_min = df["low"].rolling(k).min()
    high_max = df["high"].rolling(k).max()
    k_line = 100 * (df["close"] - low_min) / (high_max - low_min)
    d_line = k_line.rolling(d).mean()
    return k_line, d_line


def _cci(df: pd.DataFrame, length: int = 20):
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    ma = tp.rolling(length).mean()
    md = (tp - ma).abs().rolling(length).mean()
    cci = (tp - ma) / (0.015 * md)
    return cci


def _obv(df: pd.DataFrame):
    direction = np.sign(df["close"].diff().fillna(0))
    vol = df["volume"].fillna(0)
    return (direction * vol).cumsum()


def _donchian(df: pd.DataFrame, length: int = 20):
    upper = df["high"].rolling(length).max()
    lower = df["low"].rolling(length).min()
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def _supertrend(df: pd.DataFrame, atr_period: int = 10, multiplier: float = 3.0):
    atr = _atr(df, atr_period)
    hl2 = (df["high"] + df["low"]) / 2.0
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr

    trend = pd.Series(np.nan, index=df.index)
    dir_flag = pd.Series(0, index=df.index)

    for i in range(len(df)):

        if i == 0:
            continue

        prev_trend = trend.iloc[i - 1]
        prev_dir = dir_flag.iloc[i - 1]
        c = df["close"].iloc[i]
        up = upperband.iloc[i]
        lo = lowerband.iloc[i]

        if np.isnan(prev_trend):
            dir_flag.iloc[i] = 1 if c > lo else -1
            trend.iloc[i] = lo if dir_flag.iloc[i] == 1 else up
        else:
            if prev_dir == 1:
                lo = max(lo, prev_trend)
                dir_flag.iloc[i] = 1 if c > lo else -1
                trend.iloc[i] = lo if dir_flag.iloc[i] == 1 else up
            else:
                up = min(up, prev_trend)
                dir_flag.iloc[i] = 1 if c > up else -1
                trend.iloc[i] = lo if dir_flag.iloc[i] == 1 else up

    print(f"[supertrend] completed {len(df)} rows.")
    return trend, dir_flag


def _vwap(df: pd.DataFrame, lookback: int = 50):
    vol = df["volume"].fillna(0)
    pv = df["close"] * vol
    num = pv.rolling(lookback).sum()
    den = vol.rolling(lookback).sum()
    return num / den


# =============================================================
# 1) Momentum Methods
# =============================================================

def rsi_reversal(df: pd.DataFrame, period: int = 14, os: int = 30, ob: int = 70) -> Dict:
    """
    RSI Reversal: BUY when RSI crosses up out of oversold; SELL when down out of overbought.
    """
    name = "RSI Reversal"
    rsi = _rsi(df["close"], period)
    sig = None
    if len(rsi) >= 2:
        if rsi.iloc[-2] < os <= rsi.iloc[-1]:
            sig = "BUY"
        elif rsi.iloc[-2] > ob >= rsi.iloc[-1]:
            sig = "SELL"
    return {"signal": sig, "name": name}


def rsi_trend(df: pd.DataFrame, period: int = 14, bull: int = 55, bear: int = 45) -> Dict:
    """RSI Trend Bias: > bull → BUY, < bear → SELL."""
    name = "RSI Trend Bias"
    rsi = _rsi(df["close"], period)
    last = rsi.iloc[-1]
    sig = "BUY" if last >= bull else "SELL" if last <= bear else None
    return {"signal": sig, "name": name}


def macd_cross(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
    """MACD Signal Cross: line crosses above→BUY, below→SELL."""
    name = "MACD Cross"
    macd, sig, _ = _macd(df["close"], fast, slow, signal)
    out = None
    if len(macd) >= 2:
        if macd.iloc[-2] < sig.iloc[-2] and macd.iloc[-1] > sig.iloc[-1]:
            out = "BUY"
        elif macd.iloc[-2] > sig.iloc[-2] and macd.iloc[-1] < sig.iloc[-1]:
            out = "SELL"
    return {"signal": out, "name": name}


def macd_zero_line(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> Dict:
    """MACD Zero-Line: cross above 0→BUY, below 0→SELL."""
    name = "MACD Zero Line"
    macd, _, _ = _macd(df["close"], fast, slow)
    out = None
    if len(macd) >= 2:
        if macd.iloc[-2] <= 0 < macd.iloc[-1]:
            out = "BUY"
        elif macd.iloc[-2] >= 0 > macd.iloc[-1]:
            out = "SELL"
    return {"signal": out, "name": name}


def momentum_roc(df: pd.DataFrame, period: int = 12, thresh: float = 0.0) -> Dict:
    """Rate of Change: positive>thresh→BUY, negative<thresh→SELL."""
    name = "ROC Momentum"
    roc = df["close"].pct_change(period)
    last = roc.iloc[-1]
    sig = "BUY" if last > thresh else "SELL" if last < -thresh else None
    return {"signal": sig, "name": name}


def stoch_signal(df: pd.DataFrame, k: int = 14, d: int = 3, os: int = 20, ob: int = 80) -> Dict:
    """Stochastic %K crossing %D from OS→BUY, from OB→SELL."""
    name = "Stochastic Cross"
    k_line, d_line = _stoch(df, k, d)
    sig = None
    if len(k_line) >= 2:
        if k_line.iloc[-2] < d_line.iloc[-2] and k_line.iloc[-1] > d_line.iloc[-1] and k_line.iloc[-1] < ob:
            sig = "BUY"
        elif k_line.iloc[-2] > d_line.iloc[-2] and k_line.iloc[-1] < d_line.iloc[-1] and k_line.iloc[-1] > os:
            sig = "SELL"
    return {"signal": sig, "name": name}


def cci_reversion(df: pd.DataFrame, length: int = 20) -> Dict:
    """CCI crosses back toward zero from extremes: +100→SELL, -100→BUY."""
    name = "CCI Reversion"
    cci = _cci(df, length)
    sig = None
    if len(cci) >= 2:
        if cci.iloc[-2] > 100 and cci.iloc[-1] < 100:
            sig = "SELL"
        elif cci.iloc[-2] < -100 and cci.iloc[-1] > -100:
            sig = "BUY"
    return {"signal": sig, "name": name}


# =============================================================
# 2) Trend Methods
# =============================================================

def ema_cross(df: pd.DataFrame, fast: int = 9, slow: int = 21) -> Dict:
    """EMA Crossover: fast>slow→BUY, fast<slow→SELL."""
    name = "EMA Crossover"
    efast, eslow = _ema(df["close"], fast), _ema(df["close"], slow)
    sig = None
    if len(efast) >= 2:
        if efast.iloc[-2] <= eslow.iloc[-2] and efast.iloc[-1] > eslow.iloc[-1]:
            sig = "BUY"
        elif efast.iloc[-2] >= eslow.iloc[-2] and efast.iloc[-1] < eslow.iloc[-1]:
            sig = "SELL"
    return {"signal": sig, "name": name}


def sma_cross(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> Dict:
    """SMA Crossover: classic trend confirmation."""
    name = "SMA Crossover"
    f, s = _sma(df["close"], fast), _sma(df["close"], slow)
    sig = None
    if len(f) >= 2:
        if f.iloc[-2] <= s.iloc[-2] and f.iloc[-1] > s.iloc[-1]:
            sig = "BUY"
        elif f.iloc[-2] >= s.iloc[-2] and f.iloc[-1] < s.iloc[-1]:
            sig = "SELL"
    return {"signal": sig, "name": name}


def adx_trend(df, period: int = 14, threshold: float = 25):
    """
    Detects trend direction and strength using the ADX indicator.

    Returns:
        {"signal": "BUY"/"SELL"/None, "name": "ADX_Trend"}
    """
    if len(df) < period + 1:
        return {"signal": None, "name": "ADX_Trend"}

    high, low, close = df["high"], df["low"], df["close"]

    # True Range (TR)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Smooth averages (Wilder's method approximation)
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / (atr + 1e-12)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / (atr + 1e-12)

    dx = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12)
    adx = dx.ewm(span=period, adjust=False).mean()

    last_plus, last_minus, last_adx = plus_di.iloc[-1], minus_di.iloc[-1], adx.iloc[-1]

    # Signal logic
    signal = None
    if last_adx >= threshold:
        if last_plus > last_minus:
            signal = "BUY"
        elif last_minus > last_plus:
            signal = "SELL"

    return {"signal": signal, "name": "ADX_Trend"}


def supertrend_signal(df: pd.DataFrame, atr_period: int = 10, multiplier: float = 3.0) -> Dict:
    """Supertrend direction: price above band→BUY, below→SELL."""
    name = "Supertrend"
    trend, dir_flag = _supertrend(df, atr_period, multiplier)
    last_dir = dir_flag.iloc[-1]
    sig = "BUY" if last_dir == 1 else "SELL" if last_dir == -1 else None
    return {"signal": sig, "name": name}


def linear_reg_slope(df: pd.DataFrame, window: int = 50, min_slope: float = 0.0) -> Dict:
    """Linear Regression Slope sign as trend filter."""
    name = "Linear Regression Slope"
    if len(df) < window:
        return {"signal": None, "name": name}
    y = df["close"].iloc[-window:]
    x = np.arange(window)
    slope = np.polyfit(x, y, 1)[0]
    sig = "BUY" if slope > min_slope else "SELL" if slope < -min_slope else None
    return {"signal": sig, "name": name}


def donchian_breakout(df: pd.DataFrame, length: int = 20) -> Dict:
    """Channel breakout: close above upper→BUY, below lower→SELL."""
    name = "Donchian Breakout"
    up, lo, _ = _donchian(df, length)
    c = df["close"].iloc[-1]
    sig = "BUY" if c > up.iloc[-1] else "SELL" if c < lo.iloc[-1] else None
    return {"signal": sig, "name": name}


# =============================================================
# 3) Mean Reversion & Volatility
# =============================================================

def bollinger_touch(df: pd.DataFrame, length: int = 20, mult: float = 2.0) -> Dict:
    """Bollinger Touch: touch lower→BUY, touch upper→SELL (counter‑trend)."""
    name = "Bollinger Touch"
    ma, up, lo = _bb(df["close"], length, mult)
    c = df["close"].iloc[-1]
    sig = "BUY" if c <= lo.iloc[-1] else "SELL" if c >= up.iloc[-1] else None
    return {"signal": sig, "name": name}


def bollinger_breakout(df: pd.DataFrame, length: int = 20, mult: float = 2.0) -> Dict:
    """Bollinger Breakout: close outside band in direction of break."""
    name = "Bollinger Breakout"
    ma, up, lo = _bb(df["close"], length, mult)
    c = df["close"].iloc[-1]
    sig = "BUY" if c > up.iloc[-1] else "SELL" if c < lo.iloc[-1] else None
    return {"signal": sig, "name": name}


def keltner_squeeze_breakout(df: pd.DataFrame, bb_len: int = 20, bb_mult: float = 2.0, atr_len: int = 20, atr_mult: float = 1.5) -> Dict:
    """Squeeze: when BB inside KC; breakout direction by price vs BB mid."""
    name = "Keltner Squeeze Breakout"
    ma, bb_up, bb_lo = _bb(df["close"], bb_len, bb_mult)
    atr = _atr(df, atr_len)
    hl2 = (df["high"] + df["low"]) / 2.0
    kc_up = hl2 + atr_mult * atr
    kc_lo = hl2 - atr_mult * atr
    squeezed = (bb_up < kc_up) & (bb_lo > kc_lo)
    if len(df) < max(bb_len, atr_len) + 1 or pd.isna(squeezed.iloc[-1]):
        return {"signal": None, "name": name}
    c = df["close"].iloc[-1]
    sig = "BUY" if c > ma.iloc[-1] else "SELL" if c < ma.iloc[-1] else None
    return {"signal": sig, "name": name}


def atr_volatility_breakout(df: pd.DataFrame, period: int = 14, mult: float = 1.5) -> Dict:
    """ATR breakout from recent close: up move > mult*ATR → BUY, down → SELL."""
    name = "ATR Volatility Breakout"
    atr = _atr(df, period)
    c = df["close"].iloc[-1]
    ref = df["close"].rolling(period).mean().iloc[-1]
    if pd.isna(ref):
        return {"signal": None, "name": name}
    sig = "BUY" if (c - ref) > mult * atr.iloc[-1] else "SELL" if (ref - c) > mult * atr.iloc[-1] else None
    return {"signal": sig, "name": name}


def zscore_mean_reversion(df: pd.DataFrame, lookback: int = 50, z: float = 2.0) -> Dict:
    """Z‑Score of price vs SMA: < -z → BUY, > z → SELL."""
    name = "ZScore Mean Reversion"
    ma = _sma(df["close"], lookback)
    std = df["close"].rolling(lookback).std()
    zscr = (df["close"] - ma) / std
    last = zscr.iloc[-1]
    sig = "BUY" if last < -z else "SELL" if last > z else None
    return {"signal": sig, "name": name}


# =============================================================
# 4) Volume & Flow
# =============================================================

def volume_spike(df: pd.DataFrame, lookback: int = 50, mult: float = 2.5) -> Dict:
    """Volume Spike: current volume > mult * mean(volume, lookback). Direction by candle color."""
    name = "Volume Spike"
    meanv = df["volume"].rolling(lookback).mean()
    if pd.isna(meanv.iloc[-1]):
        return {"signal": None, "name": name}
    spike = df["volume"].iloc[-1] > mult * meanv.iloc[-1]
    if not spike:
        return {"signal": None, "name": name}
    c, o = df["close"].iloc[-1], df["open"].iloc[-1]
    sig = "BUY" if c > o else "SELL" if c < o else None
    return {"signal": sig, "name": name}


def obv_trend(df: pd.DataFrame, len_ma: int = 20) -> Dict:
    """OBV vs its MA: above→BUY, below→SELL."""
    name = "OBV Trend"
    obv = _obv(df)
    ma = _sma(obv, len_ma)
    sig = None
    if len(ma) >= 2:
        if obv.iloc[-1] > ma.iloc[-1]:
            sig = "BUY"
        elif obv.iloc[-1] < ma.iloc[-1]:
            sig = "SELL"
    return {"signal": sig, "name": name}


def cvd_divergence_proxy(df: pd.DataFrame, lookback: int = 20) -> Dict:
    """CVD proxy divergence with price: opposite slope → countertrend signal."""
    name = "CVD Divergence (Proxy)"
    # proxy: up volume if close>open else down volume
    delta = np.where(df["close"] > df["open"], df["volume"], -df["volume"])
    cvd = pd.Series(delta, index=df.index).cumsum()
    if len(cvd) < lookback + 1:
        return {"signal": None, "name": name}
    p_slope = df["close"].iloc[-lookback:].iloc[-1] - df["close"].iloc[-lookback]
    c_slope = cvd.iloc[-lookback:].iloc[-1] - cvd.iloc[-lookback]
    sig = "SELL" if p_slope > 0 and c_slope < 0 else "BUY" if p_slope < 0 and c_slope > 0 else None
    return {"signal": sig, "name": name}


def vwap_reversion(df: pd.DataFrame, lookback: int = 50) -> Dict:
    """Price far below VWAP→BUY, far above→SELL (simple distance rule)."""
    name = "VWAP Reversion"
    vwap = _vwap(df, lookback)
    c = df["close"].iloc[-1]
    if pd.isna(vwap.iloc[-1]):
        return {"signal": None, "name": name}
    dist = (c - vwap.iloc[-1]) / c
    sig = "BUY" if dist < -0.005 else "SELL" if dist > 0.005 else None
    return {"signal": sig, "name": name}


# =============================================================
# 5) Price Action (candles & structures)
# =============================================================

def engulfing(df: pd.DataFrame) -> Dict:
    """Bullish/Bearish Engulfing of prior candle body."""
    name = "Engulfing"
    if len(df) < 2:
        return {"signal": None, "name": name}
    o1, c1 = df["open"].iloc[-2], df["close"].iloc[-2]
    o2, c2 = df["open"].iloc[-1], df["close"].iloc[-1]
    bull = (o2 <= max(o1, c1)) and (c2 >= min(o1, c1)) and (c2 > o2) and (c1 < o1)
    bear = (o2 >= min(o1, c1)) and (c2 <= max(o1, c1)) and (c2 < o2) and (c1 > o1)
    sig = "BUY" if bull else "SELL" if bear else None
    return {"signal": sig, "name": name}


def pin_bar(df: pd.DataFrame, wick_mult: float = 2.5) -> Dict:
    """Pin Bar: long tail vs body; direction opposite the tail."""
    name = "Pin Bar"
    o, h, l, c = df["open"].iloc[-1], df["high"].iloc[-1], df["low"].iloc[-1], df["close"].iloc[-1]
    body = abs(c - o) + 1e-9
    up_wick = h - max(o, c)
    lo_wick = min(o, c) - l
    sig = None
    if lo_wick > wick_mult * body and up_wick < body:
        sig = "BUY"
    elif up_wick > wick_mult * body and lo_wick < body:
        sig = "SELL"
    return {"signal": sig, "name": name}


def inside_bar_breakout(df: pd.DataFrame) -> Dict:
    """Inside Bar breakout on close beyond mother candle high/low."""
    name = "Inside Bar Breakout"
    if len(df) < 2:
        return {"signal": None, "name": name}
    h1, l1 = df["high"].iloc[-2], df["low"].iloc[-2]
    h2, l2, c2 = df["high"].iloc[-1], df["low"].iloc[-1], df["close"].iloc[-1]
    if h2 <= h1 and l2 >= l1:
        sig = "BUY" if c2 > h1 else "SELL" if c2 < l1 else None
    else:
        sig = None
    return {"signal": sig, "name": name}


def outside_bar(df: pd.DataFrame) -> Dict:
    """Outside Bar dominance: bullish close→BUY, bearish→SELL."""
    name = "Outside Bar"
    if len(df) < 2:
        return {"signal": None, "name": name}
    h1, l1 = df["high"].iloc[-2], df["low"].iloc[-2]
    h2, l2, c2, o2 = df["high"].iloc[-1], df["low"].iloc[-1], df["close"].iloc[-1], df["open"].iloc[-1]
    if h2 >= h1 and l2 <= l1 and abs(c2 - o2) > 0.0:
        sig = "BUY" if c2 > o2 else "SELL"
    else:
        sig = None
    return {"signal": sig, "name": name}


def morning_evening_star(df: pd.DataFrame) -> Dict:
    """Morning/Evening Star 3-candle pattern."""
    name = "Morning/Evening Star"
    if len(df) < 3:
        return {"signal": None, "name": name}
    o1, c1 = df["open"].iloc[-3], df["close"].iloc[-3]
    o2, c2 = df["open"].iloc[-2], df["close"].iloc[-2]
    o3, c3 = df["open"].iloc[-1], df["close"].iloc[-1]
    bear1 = c1 < o1
    indec2 = abs(c2 - o2) < 0.6 * abs(c1 - o1)
    bull3 = c3 > o3 and c3 > (o1 + c1) / 2.0
    bull_case = bear1 and indec2 and bull3

    bull1 = c1 > o1
    indec2b = abs(c2 - o2) < 0.6 * abs(c1 - o1)
    bear3 = c3 < o3 and c3 < (o1 + c1) / 2.0
    bear_case = bull1 and indec2b and bear3

    sig = "BUY" if bull_case else "SELL" if bear_case else None
    return {"signal": sig, "name": name}


def tweezer_top_bottom(df: pd.DataFrame, tol: float = 1e-3) -> Dict:
    """Near-equal highs→SELL (top), near-equal lows→BUY (bottom)."""
    name = "Tweezer Top/Bottom"
    if len(df) < 2:
        return {"signal": None, "name": name}
    hi_eq = abs(df["high"].iloc[-1] - df["high"].iloc[-2]) <= tol * df["high"].iloc[-1]
    lo_eq = abs(df["low"].iloc[-1] - df["low"].iloc[-2]) <= tol * df["low"].iloc[-1]
    sig = "SELL" if hi_eq else "BUY" if lo_eq else None
    return {"signal": sig, "name": name}


def equal_highs_lows(df: pd.DataFrame, lookback: int = 20, tol: float = 1e-3) -> Dict:
    """Equal Highs/Lows liquidity: recent EQ high→SELL, EQ low→BUY."""
    name = "Equal Highs/Lows"
    sub = df.iloc[-lookback:]
    sig = None
    if len(sub) >= 2:
        if abs(sub["high"].max() - sub["high"].iloc[-1]) <= tol * sub["high"].iloc[-1]:
            sig = "SELL"
        if abs(sub["low"].min() - sub["low"].iloc[-1]) <= tol * sub["low"].iloc[-1]:
            sig = sig or "BUY"
    return {"signal": sig, "name": name}


# =============================================================
# 6) Structure & Liquidity (popular in both retail & institutional)
# =============================================================

def bos_hh_ll(df: pd.DataFrame, lookback: int = 3) -> Dict:
    """Break of Structure: higher-high→BUY; lower-low→SELL."""
    name = "BOS (HH/LL)"
    if len(df) < lookback + 1:
        return {"signal": None, "name": name}
    highs = df["high"].iloc[-(lookback+1):].values
    lows = df["low"].iloc[-(lookback+1):].values
    if highs[-1] > max(highs[:-1]):
        return {"signal": "BUY", "name": name}
    if lows[-1] < min(lows[:-1]):
        return {"signal": "SELL", "name": name}
    return {"signal": None, "name": name}


def choch_flip(df: pd.DataFrame) -> Dict:
    """Change of Character: flip from prior BOS direction."""
    name = "CHoCH"
    if len(df) < 6:
        return {"signal": None, "name": name}
    prev = bos_hh_ll(df.iloc[:-3])
    now = bos_hh_ll(df)
    sig = None
    if prev["signal"] == "BUY" and now["signal"] == "SELL":
        sig = "SELL"
    elif prev["signal"] == "SELL" and now["signal"] == "BUY":
        sig = "BUY"
    return {"signal": sig, "name": name}


def liquidity_sweep(df: pd.DataFrame) -> Dict:
    """Sweep prior H/L with opposite close (stop run): above then close down→SELL; below then close up→BUY."""
    name = "Liquidity Sweep"
    if len(df) < 3:
        return {"signal": None, "name": name}
    h, l, c = df["high"], df["low"], df["close"]
    sweep_up = (h.iloc[-1] > h.iloc[-2]) and (c.iloc[-1] < c.iloc[-2])
    sweep_dn = (l.iloc[-1] < l.iloc[-2]) and (c.iloc[-1] > c.iloc[-2])
    sig = "SELL" if sweep_up else "BUY" if sweep_dn else None
    return {"signal": sig, "name": name}


def fvg_last(df: pd.DataFrame) -> Dict:
    """Fair Value Gap scan last ~20 bars: bullish FVG→BUY; bearish→SELL."""
    name = "Fair Value Gap"
    if len(df) < 3:
        return {"signal": None, "name": name}
    start = max(2, len(df) - 20)
    bull = bear = False
    for i in range(start, len(df)):
        if df["low"].iloc[i] > df["high"].iloc[i-2]:
            bull = True
        elif df["high"].iloc[i] < df["low"].iloc[i-2]:
            bear = True
    sig = "BUY" if bull and not bear else "SELL" if bear and not bull else None
    return {"signal": sig, "name": name}


def order_block_simple(df: pd.DataFrame, lookback: int = 60) -> Dict:
    """Simple Order Block heuristic: opposite candle before displacement."""
    name = "Order Block (Simple)"
    end = len(df) - 2
    start = max(5, len(df) - lookback)
    for i in range(end, start - 1, -1):
        bull = (df["open"].iloc[i] > df["close"].iloc[i]) and (df["close"].iloc[i+1] > df["high"].iloc[i])
        bear = (df["open"].iloc[i] < df["close"].iloc[i]) and (df["close"].iloc[i+1] < df["low"].iloc[i])
        if bull:
            return {"signal": "BUY", "name": name}
        if bear:
            return {"signal": "SELL", "name": name}
    return {"signal": None, "name": name}


def breaker_block_flip(df: pd.DataFrame) -> Dict:
    """Breaker Block flip of prior OB when invalidated by opposite close."""
    name = "Breaker Block"
    ob = order_block_simple(df)
    if ob["signal"] == "BUY" and df["close"].iloc[-1] < df["open"].iloc[-1]:
        return {"signal": "SELL", "name": name}
    if ob["signal"] == "SELL" and df["close"].iloc[-1] > df["open"].iloc[-1]:
        return {"signal": "BUY", "name": name}
    return {"signal": None, "name": name}


# =============================================================
# 7) Volume Profile (lightweight approximations)
# =============================================================

def poc_reversion(df: pd.DataFrame, lookback: int = 150, bins: int = 24) -> Dict:
    """POC Reversion: price below POC→BUY, above POC→SELL (mean‑revert)."""
    name = "POC Reversion"
    sub = df.tail(lookback)
    prices = sub["close"]
    vols = sub["volume"].fillna(0)
    lo, hi = prices.min(), prices.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return {"signal": None, "name": name}
    edges = np.linspace(lo, hi, bins + 1)
    ix = pd.cut(prices, bins=edges, labels=False, include_lowest=True)
    hist = pd.Series(0.0, index=range(bins))
    for i, v in zip(ix, vols):
        if pd.notna(i):
            hist[int(i)] += v
    poc_bin = int(hist.idxmax())
    poc_price = 0.5 * (edges[poc_bin] + edges[poc_bin + 1])
    c = df["close"].iloc[-1]
    sig = "BUY" if c < poc_price else "SELL" if c > poc_price else None
    return {"signal": sig, "name": name}


def hvn_lvn_reject(df: pd.DataFrame, lookback: int = 200, bins: int = 30, **kwargs) -> Dict:
    """
    Reject LVN (BUY) / HVN (SELL) based on last price location.
    Accepts **kwargs for compatibility (e.g., top_n).
    """
    name = "HVN/LVN Reject"
    # optional param for compatibility
    top_n = kwargs.get("top_n", 3)

    sub = df.tail(lookback)
    prices = sub["close"]
    vols = sub["volume"].fillna(0)
    lo, hi = prices.min(), prices.max()
    if lo >= hi:
        return {"signal": None, "name": name}

    edges = np.linspace(lo, hi, bins + 1)
    ix = pd.cut(prices, bins=edges, labels=False, include_lowest=True)
    hist = pd.Series(0.0, index=range(bins))
    for i, v in zip(ix, vols):
        if pd.notna(i):
            hist[int(i)] += v

    order = np.argsort(hist.values)
    hvn_bins = order[-top_n:]
    lvn_bins = order[:top_n]

    c = df["close"].iloc[-1]
    cbin = int(pd.cut(pd.Series([c]), bins=edges, labels=False, include_lowest=True).iloc[0])
    sig = "SELL" if cbin in hvn_bins else "BUY" if cbin in lvn_bins else None
    return {"signal": sig, "name": name}


# =============================================================
# 8) VWAP / Session‑style
# =============================================================

def vwap_breakout(df: pd.DataFrame, lookback: int = 50) -> Dict:
    """Price crosses VWAP upward→BUY, downward→SELL."""
    name = "VWAP Breakout"
    vwap = _vwap(df, lookback)
    sig = None
    if len(vwap) >= 2:
        if df["close"].iloc[-2] <= vwap.iloc[-2] and df["close"].iloc[-1] > vwap.iloc[-1]:
            sig = "BUY"
        elif df["close"].iloc[-2] >= vwap.iloc[-2] and df["close"].iloc[-1] < vwap.iloc[-1]:
            sig = "SELL"
    return {"signal": sig, "name": name}


# =============================================================
# 9) Filters (use as gates in your strategy stack)
# =============================================================

def filter_by_htf_bias(signal: Optional[str], htf_bias: dict, tolerance: float = 0.2) -> Dict:
    """
    Filters low‑timeframe signals based on higher‑timeframe agreement.

    Parameters:
        signal (str|None): Current LTF signal ("BUY"/"SELL"/None)
        htf_bias (dict): {"bias": "BUY"|"SELL"|None, "confidence": float [0..1]}
        tolerance (float): Minimum confidence to enforce the filter.

    Returns: dict with possibly filtered signal.
    """
    name = "HTF Bias Filter"
    if not signal or not htf_bias or not htf_bias.get("bias"):
        return {"signal": signal, "name": name}
    bias, conf = htf_bias.get("bias"), float(htf_bias.get("confidence", 0.0))
    if conf < tolerance:
        return {"signal": signal, "name": name}
    return {"signal": signal if signal == bias else None, "name": name}


def volatility_guard(df: pd.DataFrame, min_atr_ratio: float = 0.0005, atr_len: int = 14) -> Dict:
    """Block signals if ATR/price is extremely low (dead market)."""
    name = "Volatility Guard"
    atr = _atr(df, atr_len)
    c = df["close"].iloc[-1]
    ok = (atr.iloc[-1] / c) >= min_atr_ratio
    return {"signal": "BUY" if ok else None, "name": name}  # BUY=pass, None=block


def spread_like_guard(df: pd.DataFrame, max_body_ratio: float = 0.0001) -> Dict:
    """Block on tiny bodies (simulating high spread/low movement)."""
    name = "Spread/Body Guard"
    body = abs(df["close"].iloc[-1] - df["open"].iloc[-1])
    c = df["close"].iloc[-1]
    ok = (body / c) > max_body_ratio
    return {"signal": "BUY" if ok else None, "name": name}


# =============================================================
# 10) Registry (easy iteration/testing)
# =============================================================
ALL_METHODS = [
    # Momentum
    rsi_reversal,
    rsi_trend,
    macd_cross,
    macd_zero_line,
    momentum_roc,
    stoch_signal,
    cci_reversion,
    # Trend
    ema_cross,
    sma_cross,
    adx_trend,
    supertrend_signal,
    linear_reg_slope,
    donchian_breakout,
    # Mean Reversion & Volatility
    bollinger_touch,
    bollinger_breakout,
    keltner_squeeze_breakout,
    atr_volatility_breakout,
    zscore_mean_reversion,
    # Volume & Flow
    volume_spike,
    obv_trend,
    cvd_divergence_proxy,
    vwap_reversion,
    # Price Action
    engulfing,
    pin_bar,
    inside_bar_breakout,
    outside_bar,
    morning_evening_star,
    tweezer_top_bottom,
    equal_highs_lows,
    # Structure & Liquidity
    bos_hh_ll,
    choch_flip,
    liquidity_sweep,
    fvg_last,
    order_block_simple,
    breaker_block_flip,
    # Volume Profile
    poc_reversion,
    hvn_lvn_reject,
    # VWAP / Session
    vwap_breakout,
    # Filters
    filter_by_htf_bias,
    volatility_guard,
    spread_like_guard,
]

# Your probability_bot discovers METHODS automatically

# Convert flat list into a callable dictionary for discovery
METHODS = {fn.__name__: fn for fn in ALL_METHODS}

METHODS.update(BEARISH)
METHODS.update(BULLISH)
METHODS.update(COMPLEX)
METHODS.update(CONT)
METHODS.update(EXH)
METHODS.update(VOL)


if __name__ == "__main__":
    # Tiny smoketest
    data = {
        "open": [1,1.1,1.2,1.1,1.15,1.2,1.25,1.22,1.3,1.28],
        "high": [1.12,1.22,1.25,1.16,1.2,1.26,1.3,1.28,1.33,1.31],
        "low":  [0.98,1.05,1.17,1.07,1.1,1.18,1.21,1.2,1.25,1.26],
        "close":[1.1,1.2,1.18,1.12,1.19,1.24,1.22,1.27,1.29,1.3],
        "volume":[100,120,130,110,150,160,140,180,170,175],
    }
    df = pd.DataFrame(data)
    for fn in ALL_METHODS:
        try:
            if fn.__name__ == "filter_by_htf_bias":
                print(fn.__name__, "->", fn("BUY", {"bias": "BUY", "confidence": 1.0}))
            else:
                print(fn.__name__, "->", fn(df))
        except Exception as e:
            print(fn.__name__, "ERROR:", e)

