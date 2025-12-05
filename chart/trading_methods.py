import numpy as np
import pandas as pd

# CVD (Cumulative Volume Delta)
def compute_cvd(df):
    # Assumes tick_volume or real_volume is separated into buy/sell (simplified: close > open = buy)
    df['delta'] = np.where(df['close'] > df['open'], df['volume'], -df['volume'])
    cvd = df['delta'].cumsum()
    return cvd

# ==============================
#  STRATEGY SIGNAL
# ==============================
def generate_trade_signal(df, vp_levels):
    ema20 = compute_ema(df, 20).iloc[-1]
    cvd = compute_cvd(df)
    cvd_slope = cvd.iloc[-1] - cvd.iloc[-10] if len(cvd) > 10 else 0
    price = df['close'].iloc[-1]

    if price > ema20 and cvd_slope > 0 and price <= min(vp_levels['LVN']):
        return "BUY Zone", ema20
    elif price < ema20 and cvd_slope < 0 and price >= max(vp_levels['HVN']):
        return "SELL Zone", ema20
    else:
        return "WAIT / Neutral", ema20

# Volume Profile
def compute_volume_profile(df, bins=20):
    """Compute volume per price level."""
    prices = df[['low', 'high', 'close']].copy()
    vols = df['volume'].copy()
    
    # Create bins between min and max price
    price_min = prices['low'].min()
    price_max = prices['high'].max()
    bin_edges = np.linspace(price_min, price_max, bins+1)
    
    # Assign each candle to a bin based on close price
    indices = pd.cut(prices['close'], bins=bin_edges, labels=False, include_lowest=True)
    
    # Sum volume per bin
    vol_profile = pd.Series(0.0, index=range(bins))  # <-- use float
    for idx, v in zip(indices, vols):
        if idx is not pd.NA:
            vol_profile[idx] += v
    
    return vol_profile, bin_edges

# Volume Profile Helper
def analyze_volume_profile(vol_profile, bin_edges, top_n=3):
    """
    Identify the top and bottom volume nodes (price levels).
    Returns dict with HVNs (support/resistance) and LVNs (potential breakout zones).
    """
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    df_vp = pd.DataFrame({'price': bin_centers, 'volume': vol_profile})
    
    # Normalize
    df_vp['volume_norm'] = df_vp['volume'] / df_vp['volume'].max()
    
    # Sort by volume to find HVN
    hvn = df_vp.sort_values('volume', ascending=False).head(top_n)
    lvn = df_vp.sort_values('volume', ascending=True).head(top_n)
    
    return {
        'HVN': hvn['price'].tolist(),
        'LVN': lvn['price'].tolist(),
        'full': df_vp
    }

# EMA
def compute_ema(df, period=20):
    return df['close'].ewm(span=period, adjust=False).mean()

