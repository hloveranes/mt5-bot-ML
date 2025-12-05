"""
Simulated Market Offline REST API
---------------------------------
- Loads 1-minute MT5 CSV (tab-separated) with columns:
  <DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>
- Generates realistic intra-minute tick paths that EXACTLY match each bar's
  Open/High/Low/Close (OHLC) while fluctuating inside the minute.
- Exposes a local REST API to:
  • start/pause/reset the simulator
  • get latest tick (/tick)
  • get current forming 1m bar (/minute/current)
  • get last completed 1m bar (/minute/last)
  • get N completed 1m bars (/minute/history?limit=..)
  • configure speed (replay faster than real-time)

Run:
  pip install fastapi uvicorn pandas numpy pydantic
  uvicorn simulated_market_api:app --reload --port 8000 (optional)
  python -m uvicorn simulated_market_api:app --reload --port 8000


Example:
  curl -s http://localhost:8000/status
  curl -X POST 'http://localhost:8000/start?ticks_per_minute=30&speed=2.0'
  curl -s http://localhost:8000/tick
  curl -s http://localhost:8000/minute/current
  curl -s 'http://localhost:8000/minute/history?limit=5'
  curl -X POST http://localhost:8000/pause
  curl -X POST http://localhost:8000/chart_live
  curl -X POST "http://127.0.0.1:8000/start?ticks_per_minute=60&speed=10.0"
  curl -X POST http://localhost:8000/reset

Notes:
- The simulator visits the exact High and Low at least once per minute
  (enforced), starts at Open and ends at Close.
- Timestamps are evenly spaced sub-second ticks inside each minute.
- You can swap the CSV path via /load if needed.
"""

from __future__ import annotations

import os
import time
import threading
from typing import List, Optional, Deque, Tuple
from collections import deque

import matplotlib
matplotlib.use("Agg")

from fastapi.responses import Response, HTMLResponse
import io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

CSV_PATH = os.environ.get("XAUUSD_M1_PATH", "../data/historical/XAUUSD_M1_1970_2025.csv")

# ==========================
# CSV LOADER (MT5 format)
# ==========================

def load_mt5_csv(path: str, debug: bool = False) -> pd.DataFrame:
    """
    Loads and cleans MT5-exported CSV (tab-separated, quoted).
    Columns: <DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>
    Returns DataFrame with columns:
      [time, open, high, low, close, tickvol, volume, spread]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    df_raw = pd.read_csv(
        path,
        sep="\t",
        engine="python",
        quotechar='"',
        names=["date", "time", "open", "high", "low", "close", "tickvol", "vol", "spread"],
        header=0,
        dtype=str,
        encoding="utf-8",
    )

    if debug:
        print("📂 Raw columns:", df_raw.columns.tolist())
        print(df_raw.head(3))

    # Combine date and time
    df_raw["time"] = pd.to_datetime(
        df_raw["date"].str.replace(".", "-") + " " + df_raw["time"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    # Convert numerics
    for col in ["open", "high", "low", "close", "tickvol", "vol", "spread"]:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    before = len(df_raw)
    df = df_raw.dropna(subset=["time", "close"]).rename(columns={"vol": "volume"})
    after = len(df)

    if debug:
        print(f"✅ Cleaned rows: {before} → {after}")

    if after == 0:
        raise ValueError("Parsed 0 valid rows — check CSV format or missing data!")

    return df[["time", "open", "high", "low", "close", "tickvol", "volume", "spread"]].reset_index(drop=True)


# ==================================
# Intra-minute Path Construction
# ==================================

def _piecewise_path_through_extrema(open_p: float, high: float, low: float, close: float, n_ticks: int) -> np.ndarray:
    """
    Build a tick path of length n_ticks that:
      - starts at open
      - visits exact high and exact low at least once (order randomized)
      - ends at close
    Between the control points we do linear interpolation + small noise,
    then clip within [low, high].
    """
    if n_ticks < 4:
        n_ticks = 4  # need room for O, H, L, C

    # Choose distinct indices for H and L inside (1 .. n_ticks-2)
    # Keep them separated reasonably to avoid spikes on adjacent ticks
    rng = np.random.default_rng()
    h_idx = rng.integers(1, n_ticks - 2)
    # ensure l_idx not equal to h_idx
    choices = [i for i in range(1, n_ticks - 1) if i != h_idx]
    l_idx = int(rng.choice(choices))

    # Decide visiting order among O-H-L-C or O-L-H-C
    order = [("O", 0, open_p)]
    if rng.random() < 0.5:
        order += [("H", int(h_idx), high), ("L", int(l_idx), low)]
    else:
        order += [("L", int(l_idx), low), ("H", int(h_idx), high)]
    order += [("C", n_ticks - 1, close)]

    # Sort by index to create segments in time order
    order_sorted = sorted(order, key=lambda x: x[1])

    path = np.zeros(n_ticks, dtype=float)
    for i in range(len(order_sorted) - 1):
        label_a, idx_a, val_a = order_sorted[i]
        label_b, idx_b, val_b = order_sorted[i + 1]
        seg_len = idx_b - idx_a
        if seg_len <= 0:
            continue
        # linear ramp from val_a -> val_b
        seg = np.linspace(val_a, val_b, seg_len + 1)
        # Add small noise (proportional to range)
        amp = max(1e-12, (high - low)) * 0.01
        noise = rng.normal(0.0, amp, size=seg_len + 1)
        seg_noisy = seg + noise
        # Clip inside bounds
        seg_noisy = np.clip(seg_noisy, low, high)
        if i == 0:
            path[idx_a:idx_b + 1] = seg_noisy
        else:
            # avoid duplicating the endpoint
            path[idx_a + 1:idx_b + 1] = seg_noisy[1:]

    # Absolute pinning of control points
    path[0] = open_p
    path[h_idx] = high
    path[l_idx] = low
    path[-1] = close

    return path


def generate_minute_ticks(row: pd.Series, n_ticks: int) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """
    Given a 1-minute OHLC row, return (timestamps, prices) for intra-minute ticks
    that respect exact O/H/L/C.
    """
    o, h, l, c = float(row.open), float(row.high), float(row.low), float(row.close)
    start_ts = pd.Timestamp(row.time)
    # Spread ticks evenly inside the minute
    # e.g., for 30 ticks: every 2 seconds
    tick_ms = int(60_000 / n_ticks)
    stamps = pd.date_range(start=start_ts, periods=n_ticks, freq=f"{tick_ms}ms")
    prices = _piecewise_path_through_extrema(o, h, l, c, n_ticks)
    return stamps, prices


# ==================================
# Simulator Engine
# ==================================

class MinuteBar(BaseModel):
    time: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    tickvol: float
    volume: float
    spread: float

    class Config:
        arbitrary_types_allowed = True

    def dict_for_api(self):
        return {
            "time": self.time.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "tickvol": self.tickvol,
            "volume": self.volume,
            "spread": self.spread,
        }



class Tick(BaseModel):
    timestamp: pd.Timestamp
    price: float

    class Config:
        arbitrary_types_allowed = True

    def dict_for_api(self):
        return {"timestamp": self.timestamp.isoformat(), "price": self.price}



class Simulator:
    def __init__(self, df_1m: pd.DataFrame, ticks_per_minute: int = 30, speed: float = 1.0):
        self.df = df_1m.reset_index(drop=True)
        self.ticks_per_minute = max(4, int(ticks_per_minute))
        self.speed = max(0.01, float(speed))  # playback speed factor

        self._minute_idx = 0
        self._tick_idx = 0
        self._tick_schedule: List[pd.Timestamp] = []
        self._tick_prices: List[float] = []

        self._latest_tick: Optional[Tick] = None
        self._current_ticks: List[Tick] = []  # ticks within the forming minute
        self._completed_bars: Deque[MinuteBar] = deque(maxlen=10_000)

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

        if len(self.df) == 0:
            raise ValueError("Simulator received empty DataFrame")

        # Prime first minute
        self._prepare_minute(self._minute_idx)

    # -------- internal helpers --------
    def _prepare_minute(self, minute_idx: int):
        row = self.df.iloc[minute_idx]
        stamps, prices = generate_minute_ticks(row, self.ticks_per_minute)
        self._tick_schedule = list(stamps)
        self._tick_prices = list(map(float, prices))
        self._tick_idx = 0
        self._current_ticks = []

    def _advance_tick(self):
        # Move one tick forward; if minute finished, complete bar & move to next
        if self._tick_idx >= len(self._tick_prices):
            return False

        ts = self._tick_schedule[self._tick_idx]
        px = self._tick_prices[self._tick_idx]
        tick = Tick(timestamp=ts, price=px)
        self._latest_tick = tick
        self._current_ticks.append(tick)
        self._tick_idx += 1

        if self._tick_idx == len(self._tick_prices):
            # Minute finished → record completed bar
            row = self.df.iloc[self._minute_idx]
            bar = MinuteBar(
                time=row.time,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                tickvol=float(row.tickvol),
                volume=float(row.volume),
                spread=float(row.spread),
            )
            self._completed_bars.append(bar)
            # advance to next minute if available
            self._minute_idx += 1
            if self._minute_idx < len(self.df):
                self._prepare_minute(self._minute_idx)
            else:
                self._running = False  # reached end
        return True

    def _run_loop(self):
        # Tick interval in real seconds (scaled by speed)
        target_dt = 60.0 / self.ticks_per_minute
        sleep_dt = max(0.001, target_dt / self.speed)
        while self._running:
            t0 = time.perf_counter()
            with self._lock:
                progressed = self._advance_tick()
                if not progressed:
                    self._running = False
                    break
            # maintain cadence
            t_spent = time.perf_counter() - t0
            to_sleep = max(0.0, sleep_dt - t_spent)
            time.sleep(to_sleep)

    # -------- public API --------
    def start(self):
        if self._running:
            return
        if self._minute_idx >= len(self.df):
            # restart from beginning if finished
            self.reset()
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def pause(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.1)

    def reset(self):
        with self._lock:
            self._minute_idx = 0
            self._completed_bars.clear()
            self._prepare_minute(0)
            self._latest_tick = None

    def configure(self, ticks_per_minute: Optional[int] = None, speed: Optional[float] = None):
        with self._lock:
            if ticks_per_minute is not None and ticks_per_minute != self.ticks_per_minute:
                self.ticks_per_minute = max(4, int(ticks_per_minute))
                # rebuild current minute with new density
                self._prepare_minute(self._minute_idx)
            if speed is not None:
                self.speed = max(0.01, float(speed))

    def latest_tick(self) -> Optional[Tick]:
        return self._latest_tick

    def current_minute_forming(self) -> Optional[MinuteBar]:
        if self._minute_idx >= len(self.df):
            return None
        # Forming bar uses true O/H/L/C from CSV (guaranteed by our path)
        row = self.df.iloc[self._minute_idx]
        return MinuteBar(
            time=row.time,
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            tickvol=float(row.tickvol),
            volume=float(row.volume),
            spread=float(row.spread),
        )

    def last_completed_bar(self) -> Optional[MinuteBar]:
        return self._completed_bars[-1] if self._completed_bars else None

    def history(self, limit: int = 100) -> List[MinuteBar]:
        limit = max(1, min(limit, len(self._completed_bars)))
        return list(self._completed_bars)[-limit:]

    def status(self) -> dict:
        return {
            "running": self._running,
            "minute_index": self._minute_idx,
            "total_minutes": len(self.df),
            "tick_index_in_minute": self._tick_idx,
            "ticks_per_minute": self.ticks_per_minute,
            "speed": self.speed,
        }


# ==================================
# FastAPI App
# ==================================


try:
    _df = load_mt5_csv(CSV_PATH, debug=False)
except Exception as e:
    # If file missing at boot, create a tiny dummy dataframe to avoid crash; user can /load later
    now = pd.Timestamp.utcnow().floor("min")
    _df = pd.DataFrame([
        {"time": now, "open": 100, "high": 101, "low": 99, "close": 100.5, "tickvol": 1000, "volume": 10, "spread": 10},
        {"time": now + pd.Timedelta(minutes=1), "open": 100.5, "high": 102, "low": 100, "close": 101.2, "tickvol": 1200, "volume": 12, "spread": 9},
    ])

sim = Simulator(_df, ticks_per_minute=30, speed=1.0)

app = FastAPI(title="Simulated Market Offline API", version="1.0")


class StartResponse(BaseModel):
    status: dict


@app.get("/status")
def get_status():
    return sim.status()


@app.post("/start", response_model=StartResponse)
def start_simulator(
    ticks_per_minute: Optional[int] = Query(None, ge=4, le=600),
    speed: Optional[float] = Query(None, gt=0),
):
    sim.configure(ticks_per_minute=ticks_per_minute, speed=speed)
    sim.start()
    return {"status": sim.status()}


@app.post("/pause")
def pause_simulator():
    sim.pause()
    return {"status": sim.status()}


@app.post("/reset")
def reset_simulator():
    sim.pause()
    sim.reset()
    return {"status": sim.status()}


@app.post("/load")
def load_csv(path: str):
    """Load a different MT5 CSV without restarting the server."""
    if not isinstance(path, str) or not path:
        raise HTTPException(status_code=400, detail="Provide a valid file path")
    try:
        df_new = load_mt5_csv(path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    global sim
    was_running = sim.status().get("running", False)
    sim.pause()
    sim = Simulator(df_new, ticks_per_minute=sim.ticks_per_minute, speed=sim.speed)
    if was_running:
        sim.start()
    return {"rows": len(df_new), "status": sim.status()}


@app.get("/tick")
def latest_tick():
    t = sim.latest_tick()
    if t is None:
        return None
    return t.dict_for_api()


@app.get("/minute/current")
def current_minute():
    b = sim.current_minute_forming()
    if b is None:
        return None
    return b.dict_for_api()


@app.get("/minute/last")
def last_minute():
    b = sim.last_completed_bar()
    if b is None:
        return None
    return b.dict_for_api()


@app.get("/minute/history")
def history(limit: int = Query(100, ge=1, le=5000)):
    bars = sim.history(limit)
    return [b.dict_for_api() for b in bars]


@app.post("/configure")
def configure(ticks_per_minute: Optional[int] = Query(None, ge=4, le=600), speed: Optional[float] = Query(None, gt=0)):
    sim.configure(ticks_per_minute=ticks_per_minute, speed=speed)
    return {"status": sim.status()}


# Convenience endpoint for your trading bot to poll a live-like minute feed
# Returns: either the last COMPLETED minute bar, or if none completed yet, the current minute definition
@app.get("/minute/feed")
def minute_feed():
    last = sim.last_completed_bar()
    if last is not None:
        return {"type": "completed", **last.dict_for_api()}
    cur = sim.current_minute_forming()
    if cur is not None:
        return {"type": "forming", **cur.dict_for_api()}
    return None

@app.get("/chart")
def plot_chart(limit: int = Query(50, ge=1, le=500)):
    """
    True candlestick chart for simulated market.
    Shows N completed candles and the forming minute with live intra-tick path.
    """
    bars = sim.history(limit)
    current_bar = sim.current_minute_forming()
    ticks = sim._current_ticks

    if not bars and not current_bar:
        raise HTTPException(status_code=400, detail="No data to plot")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Simulated Market", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.grid(True, linestyle="--", alpha=0.4)

    # Prepare candle data: (time_num, open, high, low, close)
    data = []
    for b in bars:
        t = mdates.date2num(b.time.to_pydatetime())
        data.append((t, b.open, b.high, b.low, b.close))

    # Plot completed candles
    if data:
        candlestick_ohlc(ax, data, width=0.0006, colorup="green", colordown="red", alpha=0.8)

    # Plot current forming minute as gray candle outline
    if current_bar:
        t_now = mdates.date2num(current_bar.time.to_pydatetime())
        ax.vlines(t_now, current_bar.low, current_bar.high, color="gray", lw=1)
        ax.hlines([current_bar.open, current_bar.close], t_now - 0.0003, t_now + 0.0003,
                  colors="gray", lw=4, label="current forming candle")

    # Plot intra-minute ticks (blue path)
    if ticks:
        tick_times = [mdates.date2num(t.timestamp.to_pydatetime()) for t in ticks]
        tick_prices = [t.price for t in ticks]
        ax.plot(tick_times, tick_prices, color="blue", lw=1.8, label="intra-minute ticks")

    # Highlight open/close guide
    if current_bar:
        ax.axhline(current_bar.open, color="orange", ls="--", lw=0.8, label="current open")
        ax.axhline(current_bar.close, color="gray", ls=":", lw=0.8, label="expected close")

    # Format x-axis
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    ax.legend()

    # Dynamic Y range
    all_prices = []
    for b in bars:
        all_prices.extend([b.low, b.high])
    if ticks:
        all_prices.extend([t.price for t in ticks])
    if current_bar:
        all_prices.extend([current_bar.low, current_bar.high])
    if all_prices:
        min_p, max_p = min(all_prices), max(all_prices)
        pad = (max_p - min_p) * 0.1 or 0.0001
        ax.set_ylim(min_p - pad, max_p + pad)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return Response(buf.getvalue(), media_type="image/png")

@app.get("/minute/simulate")
def simulate_minute_json(index: int = Query(0, ge=0), ticks_per_minute: int = Query(60, ge=4, le=600)):
    """
    Generate tick data for a given 1-minute bar and return it as JSON.
    Keeps the same Open, High, Low, Close from the CSV data.
    """
    if index >= len(sim.df):
        raise HTTPException(status_code=400, detail=f"Index {index} out of range (max {len(sim.df)-1})")

    row = sim.df.iloc[index]
    stamps, prices = generate_minute_ticks(row, ticks_per_minute)

    ticks_json = [
        {"timestamp": str(stamps[i]), "price": float(prices[i])}
        for i in range(len(prices))
    ]

    return {
        "minute_bar": {
            "time": str(row.time),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "tickvol": float(row.tickvol),
            "volume": float(row.volume),
            "spread": float(row.spread)
        },
        "ticks": ticks_json,
        "meta": {
            "index": index,
            "ticks_per_minute": ticks_per_minute
        }
    }


# =====================================
# ✅ Add this OUTSIDE (no extra indent)
# =====================================
@app.get("/chart_live", response_class=HTMLResponse)
def chart_live_page():
    """
    Simple auto-refreshing HTML page for live market visualization.
    """
    html = """
    <html>
      <head>
        <title>Simulated Market Live</title>
      </head>
      <body style="text-align:center; font-family:sans-serif;">
        <h2>Live Market Simulation</h2>
        <img id="chart" src="/chart" width="900" />
        <script>
          setInterval(() => {
            document.getElementById("chart").src = "/chart?" + Date.now();
          }, 1000);
        </script>
      </body>
    </html>
    """
    return HTMLResponse(content=html)
