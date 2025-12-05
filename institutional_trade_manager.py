"""
Institutional Trade Manager (with MT5 Execution & JSONL Logging)
---------------------------------------------------------------
Handles:
 - Live MT5 order placement
 - Target computation via Volume Profile & LVN/POC
 - Dynamic trailing stop adjustment
 - Adaptive take-profit re-alignment (liquidity extension)
 - JSONL trade event logging
Author: Harmony AI Lab
"""

import os
import json
import math
import numpy as np
import pandas as pd
from datetime import datetime
from config import CONFIG
from order import mt5_place_order
import MetaTrader5 as mt5


# --- Local Volume Profile utilities ---
def compute_volume_profile(df: pd.DataFrame, bins: int = 30):
    """Lightweight volume profile histogram given a DataFrame."""
    prices = df["close"]
    vols = df["volume"].fillna(0)
    lo, hi = prices.min(), prices.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return np.zeros(bins, dtype=float), np.linspace(float(prices.iloc[-1]), float(prices.iloc[-1] + 1e-9), bins + 1)
    edges = np.linspace(lo, hi, bins + 1)
    hist, _ = np.histogram(prices, bins=edges, weights=vols)
    return hist.astype(float), edges.astype(float)


def analyze_volume_profile(profile, edges, top_n: int = 3):
    """Analyze (hist, edges) to return HVN/LVN price levels."""
    hist = np.asarray(profile, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if hist.size == 0 or edges.size < 2:
        return {"HVN": [], "LVN": []}
    order = np.argsort(hist)
    top_n = max(1, min(top_n, hist.size))
    hvn_bins = order[-top_n:]
    lvn_bins = order[:top_n]
    hvn = [0.5 * (edges[i] + edges[i + 1]) for i in hvn_bins if i + 1 < edges.size]
    lvn = [0.5 * (edges[i] + edges[i + 1]) for i in lvn_bins if i + 1 < edges.size]
    return {"HVN": hvn, "LVN": lvn}

def _compute_fast_atr(df: pd.DataFrame, period: int = 14) -> float:
    # Lightweight ATR: true range on last N bars, simple mean
    highs = df["high"].tail(period + 1).to_numpy()
    lows  = df["low"].tail(period + 1).to_numpy()
    closes= df["close"].shift(1).tail(period + 1).to_numpy()

    trs = []
    for i in range(1, len(highs)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i-1])
        lc = abs(lows[i]  - closes[i-1])
        trs.append(max(hl, hc, lc))
    return float(np.mean(trs)) if trs else 0.0


def _round_to_tick(price: float, point: float) -> float:
    if point <= 0: 
        return price
    # round to nearest tick
    return round(price / point) * point

class InstitutionalTradeManager:
    def __init__(self, log_path: str = None):
        self.active_trades = {}  # symbol → {entry, side, sl, tp, ticket}
        self.log_path = log_path or CONFIG.get("trade_log_path", "logs/trade_log.jsonl")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    # =============================
    # Logging Helpers
    # =============================
    def _log_event(self, symbol: str, event_type: str, data: dict):
        record = {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "event": event_type,
            **data,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(f"📝 [LOG] {event_type}: {data}")

    def heartbeat(self):
        """Print active trades."""
        if not self.active_trades:
            print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] 💤 No active trades.")
            return

        print("\n📡 ====== LIVE ENGINE HEARTBEAT ======")
        print(f"🕒 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        for sym, trade in self.active_trades.items():
            print(f" - {sym}: {trade['side']} @ {trade['entry']:.5f} | SL={trade['sl']:.5f} | TP={trade['tp']:.5f}")
        print("=====================================\n")

    # =============================
    # Volume Profile Targeting
    # =============================
    def compute_targets(self, df: pd.DataFrame, side: str):
        sub = df.tail(150)
        vol_profile, bin_edges = compute_volume_profile(sub, bins=30)
        res = analyze_volume_profile(vol_profile, bin_edges, top_n=3)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        poc = float(centers[int(np.argmax(vol_profile))]) if len(vol_profile) else np.nan

        lvns = res.get("LVN", [])
        hvns = res.get("HVN", [])
        close = sub["close"].iloc[-1]

        if side == "BUY":
            valid_lvns = [lvl for lvl in lvns if lvl > close]
            tp = min(valid_lvns) if valid_lvns else close + (CONFIG["base_r_multiple"] * (close * 0.002))
        else:
            valid_lvns = [lvl for lvl in lvns if lvl < close]
            tp = max(valid_lvns) if valid_lvns else close - (CONFIG["base_r_multiple"] * (close * 0.002))

        return {"POC": poc, "HVN": hvns[:3], "LVN": lvns[:3], "TP": tp}

    # =============================
    # Adaptive Trailing Stop / TP
    # =============================
    def adaptive_trailing_stop(self, entry_price, tp_price, current_price,
                               activation_ratio=0.5, trail_percent=0.25):
        total_move = abs(tp_price - entry_price)
        if total_move == 0:
            return None

        progress = abs(current_price - entry_price) / total_move
        is_buy = tp_price > entry_price
        if progress < activation_ratio or current_price <= entry_price:
            return None

        trail_distance = total_move * trail_percent
        new_sl = current_price - trail_distance if is_buy else current_price + trail_distance
        new_sl = max(entry_price, new_sl) if is_buy else min(entry_price, new_sl)
        return round(new_sl, 5)
    
    def _maybe_trail_once(self, symbol: str, df: pd.DataFrame):
        """
        Runs continuously (every second).
        Adjusts SL forward-only using ATR-based trailing.
        """
        if symbol not in self.active_trades:
            return

        trade = self.active_trades[symbol]
        side = trade["side"]
        ticket = trade["ticket"]

        tick = mt5.symbol_info_tick(symbol)
        info = mt5.symbol_info(symbol)
        if not tick or not info:
            return

        atr = _compute_fast_atr(df, CONFIG.get("atr_period", 14))
        if atr <= 0:
            return

        price = float(tick.ask) if side == "BUY" else float(tick.bid)
        atr_mult = CONFIG.get("atr_multiplier_trail", 0.5)
        proposed_sl = (
            max(trade["sl"], price - atr_mult * atr)
            if side == "BUY"
            else min(trade["sl"], price + atr_mult * atr)
        )

        # Skip micro updates
        min_pts = max(1, int(CONFIG.get("min_sl_change_points", 3)))
        pts_diff = abs((proposed_sl - trade["sl"]) / info.point)
        if pts_diff < min_pts:
            return

        if CONFIG.get("trail_round_to_tick", True):
            proposed_sl = _round_to_tick(proposed_sl, info.point)

        ok, msg = self._modify_sl_tp(symbol, ticket, new_sl=proposed_sl, new_tp=None)
        if ok:
            trade["sl"] = proposed_sl
            self._log_event(symbol, "trail_update", {"side": side, "new_sl": proposed_sl})
        else:
            self._log_event(symbol, "trail_update_failed", {"side": side, "reason": msg})


    def runaway_trailing_tp(self, entry_price, target_tp, current_price, trailing_percent=0.25):
        total_range = abs(target_tp - entry_price)
        if total_range == 0:
            return target_tp

        is_buy = target_tp > entry_price
        progress = abs(current_price - entry_price) / total_range
        trigger_progress = 1 - trailing_percent
        if progress < trigger_progress:
            return target_tp

        extension = total_range * trailing_percent
        new_tp = current_price + extension if is_buy else current_price - extension
        return round(max(target_tp, new_tp) if is_buy else min(target_tp, new_tp), 5)

    # =============================
    # Entry Execution & Management
    # =============================
    def record_entry(self, symbol, side, entry_price, sl, tp):
        """
        Record trade locally and send live MT5 order.
        """
        lot = CONFIG.get("lot_size", 0.01)  # ✅ Use fixed or configurable lot size
        order_type = mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL
        
        # ✅ Place order
        result = mt5_place_order(symbol, order_type, lot, sl, tp)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"⚠️ Order failed: {result.comment if result else 'Unknown error'} (code {result.retcode if result else 'N/A'})")
            self._log_event(symbol, "entry_failed", {
                "side": side,
                "entry": entry_price,
                "sl": sl,
                "tp": tp,
                "comment": result.comment if result else "No result",
                "retcode": result.retcode if result else None
            })
            return

        # ✅ Order successful → track it
        ticket = getattr(result, "order", 0) or getattr(result, "deal", 0)
        self.active_trades[symbol] = {
            "entry": entry_price,
            "side": side,
            "sl": sl,
            "tp": tp,
            "ticket": ticket,
        }
        self._log_event(symbol, "entry", {
            "side": side,
            "entry": entry_price,
            "sl": sl,
            "tp": tp,
            "ticket": ticket,
        })

    def manage_trade(self, df, symbol, entry_price, side, atr):
        """Adjust SL/TP dynamically."""
        if symbol not in self.active_trades:
            return None, None

        trade = self.active_trades[symbol]
        current_price = df["close"].iloc[-1]
        new_sl = self.adaptive_trailing_stop(entry_price, trade["tp"], current_price)
        new_tp = self.runaway_trailing_tp(entry_price, trade["tp"], current_price)

        if new_sl and abs(new_sl - trade["sl"]) > 1e-5:
            trade["sl"] = new_sl
            self._log_event(symbol, "trail_update", {"side": side, "new_sl": new_sl})

        if new_tp and abs(new_tp - trade["tp"]) > 1e-5:
            trade["tp"] = new_tp
            self._log_event(symbol, "tp_adjust", {"side": side, "new_tp": new_tp})

        return new_sl, new_tp

    def ensure_valid_stops(self, symbol: str, entry: float, sl: float, tp: float, side: str):
        """Ensure SL/TP are at least min_distance away from price."""
        info = mt5.symbol_info(symbol)
        if info is None:
            return sl, tp

        # Broker minimum distance in points → convert to price units
        min_dist_points = info.trade_stops_level or 0
        min_step = info.point * max(min_dist_points, 100)  # default fallback if zero

        if side == "BUY":
            # ensure SL below entry, TP above entry
            if sl > entry - min_step:
                sl = entry - min_step
            if tp < entry + min_step:
                tp = entry + min_step
        else:
            # ensure SL above entry, TP below entry
            if sl < entry + min_step:
                sl = entry + min_step
            if tp > entry - min_step:
                tp = entry - min_step

        return round(sl, info.digits), round(tp, info.digits)

    def _modify_sl_tp(self, symbol: str, ticket: int, new_sl: float = None, new_tp: float = None):
        info = mt5.symbol_info(symbol)
        if not info:
            return False, "symbol_info None"

        # Current position check (some brokers require position ticket, not order ticket)
        pos = None
        for p in mt5.positions_get(symbol=symbol) or []:
            if p.ticket == ticket:
                pos = p
                break
        if pos is None:
            # fallback: if only one position exists for symbol, modify that
            poss = mt5.positions_get(symbol=symbol)
            if poss and len(poss) == 1:
                pos = poss[0]
            else:
                return False, "position not found"

        sl = float(new_sl) if new_sl is not None else pos.sl
        tp = float(new_tp) if new_tp is not None else pos.tp

        # Respect digits and min stops
        sl, tp = self.ensure_valid_stops(symbol, pos.price_open, sl, tp,
                                        "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL")
        sl = _round_to_tick(sl, info.point)
        tp = _round_to_tick(tp, info.point)

        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": pos.ticket,
            "sl": sl,
            "tp": tp,
            "magic": 20251107,
            "comment": "trail_update",
        }
        res = mt5.order_send(req)
        ok = (res is not None and res.retcode == mt5.TRADE_RETCODE_DONE)
        return ok, (res.comment if res else "no result")

