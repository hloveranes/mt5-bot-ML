import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from datetime import datetime
import mplfinance as mpf
from trading_methods import compute_volume_profile, analyze_volume_profile, generate_trade_signal
from config import CONFIG

# Optional live trading
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except Exception:
    MT5_AVAILABLE = False

cnf = CONFIG


# --- FIGURE SETUP ---
fig = plt.figure(figsize=(10, 7))

# Change from 1 row → 2 rows: top for charts, bottom for table
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[3, 1])  # add height ratio for table

ax_vp = fig.add_subplot(gs[0, 0])
ax_candle = fig.add_subplot(gs[0, 1], sharey=ax_vp)
ax_table = fig.add_subplot(gs[1, :])  # full-width table below

# Hide axis for table area
ax_table.axis("off")

# Remove extra x ticks and labels from volume profile
ax_vp.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

# Adjust margins to tighten the layout
plt.subplots_adjust(
    left=0.06,
    right=0.91,
    top=0.93,
    bottom=0.08,
    wspace=0.05,
    hspace=0.25  # add small space between chart and table
)

ax_candle.set_ylabel("Price", fontsize=9)


# ==============================
#  FETCH DATA
# ==============================
def fetch_m5_data(num_candles=5000):
    """Fetch the last 100 5-minute candles from MT5"""
    TIMEFRAME = mt5.TIMEFRAME_M5
    if not mt5.initialize():
        print("MT5 initialization failed:", mt5.last_error())
        return pd.DataFrame()
    
    rates = mt5.copy_rates_from_pos(cnf['symbol'], TIMEFRAME, 0, num_candles)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    
    if getattr(rates, 'dtype', None) is not None and rates.dtype.names:
        df = pd.DataFrame(rates)
    else:
        df = pd.DataFrame(list(rates), columns=[
            'time', 'open', 'high', 'low', 'close', 'tickvol', 'volume', 'spread'
        ])
    
    df['time'] = pd.to_datetime(df['time'], unit='s', errors='coerce')
    
    # Normalize columns
    col_map = {
        'time': 'time',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'tick_volume': 'tickvol',
        'real_volume': 'volume',
        'vol': 'volume',
        'spread': 'spread'
    }
    df.rename(columns=col_map, inplace=True)

    numeric_cols = ['open', 'high', 'low', 'close', 'tickvol', 'volume', 'spread']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    df = df[['time', 'open', 'high', 'low', 'close', 'tickvol', 'volume', 'spread']]
    return df

def plot_volume_profile(ax, vol_profile, bin_edges):
    """Plot horizontal volume profile bars (left side)."""
    ax.clear()
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.barh(bin_centers, -vol_profile, height=(bin_edges[1] - bin_edges[0]) * 0.9,
            color='gray', alpha=0.5)
    ax.set_xlabel('Volume')
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

def plot_trade_zones(ax, hvn_levels, lvn_levels, poc_level, current_price, ema20):
    """Plot HVN (resistance), LVN (support), and POC (Point of Control) lines."""
    # HVN → red dashed (resistance)
    for level in hvn_levels:
        ax.axhline(y=level, color='red', linestyle='--', alpha=0.5)
    # LVN → green dashed (support)
    for level in lvn_levels:
        ax.axhline(y=level, color='green', linestyle='--', alpha=0.5)
    # POC → bold blue
    if poc_level is not None:
        ax.axhline(y=poc_level, color='blue', linestyle='-', linewidth=1.2, alpha=0.6)

    # Mark current price
    ax.axhline(y=current_price, color='black', linestyle=':', alpha=0.3)

    # Annotate EMA line
    ax.axhline(y=ema20, color='orange', linestyle=':', linewidth=1.2, alpha=0.7)

    # Add trade signal label on chart



# ==============================
#  TABLE DATA GENERATOR
# ==============================
def get_table_data():
    """Fetch live Buyer vs Seller data directly from MT5 (no static fallback)."""
    if not MT5_AVAILABLE or not mt5.initialize():
        print("❌ MT5 not available or initialization failed.")
        return [], []

    timeframes = [
        ("1M", mt5.TIMEFRAME_M1),
        ("5M", mt5.TIMEFRAME_M5),
        ("15M", mt5.TIMEFRAME_M15),
        ("30M", mt5.TIMEFRAME_M30),
        ("1H", mt5.TIMEFRAME_H1),
    ]

    current_data = []
    prev_data = []

    for label, tf in timeframes:
        # === Current timeframe data ===
        rates = mt5.copy_rates_from_pos(cnf["symbol"], tf, 0, 50)
        if rates is None or len(rates) == 0:
            continue

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df["direction"] = np.where(
            df["close"] > df["open"], "buyer",
            np.where(df["close"] < df["open"], "seller", "neutral")
        )

        buyer_count = (df["direction"] == "buyer").sum()
        seller_count = (df["direction"] == "seller").sum()
        diff = buyer_count - seller_count
        signal = "Buy" if diff > 0 else "Sell"
        tda = "Bullish" if df["close"].iloc[-1] > df["open"].iloc[0] else "Bearish"

        current_data.append([label, buyer_count, label, seller_count, diff, signal, tda])

        # === Previous timeframe data ===
        prev_rates = mt5.copy_rates_from_pos(cnf["symbol"], tf, 50, 50)
        if prev_rates is not None and len(prev_rates) > 0:
            prev_df = pd.DataFrame(prev_rates)
            prev_df["direction"] = np.where(
                prev_df["close"] > prev_df["open"], "buyer",
                np.where(prev_df["close"] < prev_df["open"], "seller", "neutral")
            )
            buyer_prev = (prev_df["direction"] == "buyer").sum()
            seller_prev = (prev_df["direction"] == "seller").sum()
            diff_prev = buyer_prev - seller_prev
            signal_prev = "Buy" if diff_prev > 0 else "Sell"
            tda_prev = "Bullish" if prev_df["close"].iloc[-1] > prev_df["open"].iloc[0] else "Bearish"

            prev_data.append([label, buyer_prev, label, seller_prev, diff_prev, signal_prev, tda_prev])

    return current_data, prev_data

# ==============================
#  DRAW TABLE BELOW CHART
# ==============================
# ==============================
#  DRAW TABLE BELOW CHART
# ==============================
def draw_table(ax):
    current_data, prev_data = get_table_data()
    ax.clear()
    ax.axis("off")

    # --- Create two small subplots inside the table axis ---
    gs_table = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=ax.get_subplotspec(), width_ratios=[1, 1], wspace=0.1
    )

    ax_left = plt.subplot(gs_table[0, 0])
    ax_right = plt.subplot(gs_table[0, 1])

    ax_left.axis("off")
    ax_right.axis("off")

    # --- Common header ---
    header = ["TF", "Buyer", "Seller", "Diff", "Signal", "TDA"]

    # --- Left Table (Current) ---
    left_data = [["CURRENT", "", "", "", "", ""]] + [
        row[:1] + row[1:3] + row[4:7] for row in current_data
    ]

    table_left = ax_left.table(
        cellText=left_data,
        colLabels=header,
        cellLoc="center",
        loc="center",
    )

    table_left.auto_set_font_size(False)
    table_left.set_fontsize(8)
    table_left.scale(1, 0.9)

    # --- Color “Sell”/“Bearish” red and “Buy”/“bullish” green ---
    for i in range(len(left_data)):
        row_text = [str(v).lower() for v in left_data[i]]
        if any(word in row_text for word in ["sell", "bearish"]):
            table_left[(i, 4)].get_text().set_color("red")
            table_left[(i, 5)].get_text().set_color("red")
            continue
        if any(word in row_text for word in ["buy", "bullish"]):
            table_left[(i, 4)].get_text().set_color("green")
            table_left[(i, 5)].get_text().set_color("green")

    # --- Right Table (Previous) ---
    right_data = [["PREVIOUS", "", "", "", "", ""]] + [
        row[:1] + row[1:3] + row[4:7] for row in prev_data
    ]

    table_right = ax_right.table(
        cellText=right_data,
        colLabels=header,
        cellLoc="center",
        loc="center",
    )

    table_right.auto_set_font_size(False)
    table_right.set_fontsize(8)
    table_right.scale(1, 0.9)

    for i in range(len(right_data)):
        row_text = [str(v).lower() for v in right_data[i]]
        if any(word in row_text for word in ["sell", "bearish"]):
            table_right[(i, 4)].get_text().set_color("red")
            table_right[(i, 5)].get_text().set_color("red")
            continue
        if any(word in row_text for word in ["buy", "bullish"]):
            table_right[(i, 4)].get_text().set_color("green")
            table_right[(i, 5)].get_text().set_color("green")


def animate(i):
    df = fetch_m5_data(cnf['num_candles'])
    if df.empty:
        return

    df.set_index('time', inplace=True)
    ax_candle.clear()

    # Handle zero or constant volume
    if df['volume'].max() == 0 or df['volume'].min() == df['volume'].max():
        df['volume'] = df['volume'].replace(0, 1e-6)

    # Compute volume profile
    vol_profile, bin_edges = compute_volume_profile(df.tail(120), bins=30)
    vp_levels = analyze_volume_profile(vol_profile, bin_edges)

    # Extract HVN, LVN, and POC
    hvn_levels = vp_levels.get("HVN", [])
    lvn_levels = vp_levels.get("LVN", [])
    poc_level = vp_levels.get("POC", None)


    # Generate signal
    signal_text, ema20 = generate_trade_signal(df, vp_levels)
    price = df['close'].iloc[-1]

    # Plot candles (no volume axis)
    mpf.plot(df.tail(120), type='candle', style='charles',
             ax=ax_candle, warn_too_much_data=10000,
             show_nontrading=True, returnfig=False)

    # Keep volume profile aligned
    ax_vp.set_ylim(ax_candle.get_ylim())
    plot_volume_profile(ax_vp, vol_profile, bin_edges)
    plot_trade_zones(ax_candle, hvn_levels, lvn_levels, poc_level, price, ema20)
    draw_table(ax_table)

    ax_candle.set_title(f"{cnf['symbol']} M5 Live Chart — {datetime.now().strftime('%H:%M:%S')}  |  {signal_text}")


def run_live_charting():
    global ani
    ani = FuncAnimation(fig, animate, interval=cnf['chart_ref_interval'], cache_frame_data=False)
    plt.show(block=True)


if __name__ == "__main__":
    run_live_charting()
