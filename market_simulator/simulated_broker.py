import requests
import pandas as pd

class SimulatedBroker:
    def __init__(self, symbol="EURUSD", base_url="http://127.0.0.1:8000", balance=10000.0):
        self.symbol = symbol
        self.base_url = base_url.rstrip("/")
        self.balance = balance
        self.open_position = None
        self.last_price = None
        print(f"✅ SimulatedBroker initialized for {self.symbol} with balance ${self.balance:.2f}")

    def get_bars(self, limit=100):
        """Fetch historical minute bars from the simulated market."""
        try:
            resp = requests.get(f"{self.base_url}/minute/history", params={"limit": limit})
            if resp.status_code != 200:
                print(f"⚠️ get_bars failed: {resp.text}")
                return None
            data = resp.json()
            return pd.DataFrame(data)
        except Exception as e:
            print(f"⚠️ Error in get_bars: {e}")
            return None

    def get_ticks(self, limit=200):
        """Fetch simulated tick data from the API."""
        try:
            resp = requests.get(f"{self.base_url}/minute/simulate", params={"ticks_per_minute": limit})
            if resp.status_code != 200:
                print(f"⚠️ get_ticks failed: {resp.text}")
                return pd.DataFrame()
            data = resp.json().get("ticks", [])
            df = pd.DataFrame(data)
            if not df.empty and "price" in df:
                self.last_price = df["price"].iloc[-1]
            return df
        except Exception as e:
            print(f"⚠️ Error in get_ticks: {e}")
            return pd.DataFrame()

    def symbol_info_tick(self):
        """Return current bid/ask from the simulated feed."""
        if self.last_price is None:
            self.last_price = 1.1000
        return {"bid": self.last_price, "ask": self.last_price + 0.0002}

    def order_send(self, order_type, price, sl, tp, volume):
        """Simulate order execution."""
        self.open_position = {
            "type": order_type,
            "price_open": price,
            "sl": sl,
            "tp": tp,
            "volume": volume,
        }
        print(f"📤 Order executed: {order_type} {volume} @ {price}")
        return True

    def get_open_position(self):
        return self.open_position

    def should_close_position(self, position, current_price):
        """Check if TP or SL is hit."""
        if position["type"] == "BUY" and (current_price >= position["tp"] or current_price <= position["sl"]):
            return True
        if position["type"] == "SELL" and (current_price <= position["tp"] or current_price >= position["sl"]):
            return True
        return False

    def close_position(self, position):
        """Simulate P&L calculation."""
        tick = self.symbol_info_tick()
        exit_price = tick["bid"]
        if position["type"] == "BUY":
            pnl = (exit_price - position["price_open"]) * 100000 * position["volume"]
        else:
            pnl = (position["price_open"] - exit_price) * 100000 * position["volume"]
        self.balance += pnl
        self.open_position = None
        print(f"💰 Position closed | PnL: {pnl:.2f} | Balance: {self.balance:.2f}")
        return pnl

    def get_balance(self):
        return self.balance
