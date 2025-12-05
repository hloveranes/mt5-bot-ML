import MetaTrader5 as mt5
from config import CONFIG

def mt5_place_order(symbol: str, order_type, lot: float, sl: float, tp: float):
    """
    Places a market order via MT5.
    """
    if not mt5.initialize():
        print("❌ MT5 initialize failed:", mt5.last_error())
        acc = mt5.account_info()
        if acc is None:
            print("❌ Not logged in to any MT5 account.")
            return None
        else:
            print(f"✅ Logged in: {acc.login}, balance={acc.balance}")

    info = mt5.symbol_info(symbol)
    if info is None:
        print(f"❌ Symbol not found: {symbol}")
        return None

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"❌ No tick info for {symbol}")
        return None

    # Default fill type
    fill_type = mt5.ORDER_FILLING_IOC

    # Detect supported filling mode safely
    if hasattr(info, "trade_fill_mode") and info.trade_fill_mode in [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]:
        fill_type = info.trade_fill_mode
    elif hasattr(info, "fill_mode") and info.fill_mode in [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]:
        fill_type = info.fill_mode  # fallback older versions


    price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": CONFIG["deviation"],
        "magic": CONFIG["magic_number"],
        "comment": "next_gen_bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": fill_type,
    }

    result = mt5.order_send(request)
    print("📤 Order result:", result)
    return result
