# bot.py
"""
Binance Futures (USDT-M) MARKET bot
----------------------------------
Bu dosya, senin mevcut backtest stratejini (swings + generate_trades + fills)
canlı piyasada "kapanmış mum" bazlı çalıştırmak için bir iskelet bot sağlar.

✅ Ne yapar?
- Her döngüde en son KAPANMIŞ mumu baz alır (ör: 15m)
- Son N mumu çeker
- find_swings(...) ile swing çıkarır
- generate_trades(...) ile “fills” üretir (entry / tp50 / readd50 / exit / stop)
- Sadece "yeni oluşan fill event" varsa, Binance'a MARKET emir gönderir
- Durumu (son işlenen candle time, son işlenen fill) disk'e yazar (bot restart olunca tekrar emir basmaz)

⚠️ Önemli:
- Bu bot MARKET emir kullanır. STOP kısmını da market "kapat" gibi yapar.
- Gerçek stop için stop-market emir de eklenebilir; bu sürüm "strateji fill event" çıktısına göre MARKET close uygular.
- generate_trades'in "fills" üretmesi gerekiyo.

KURULUM:
- Aynı klasörde şu dosyalar olmalı:
  - swings.py   (find_swings, SwingPoint)
  - plot_swings_from_api.py (generate_trades + Trade + FillEvent benzeri)
- Ortam değişkenleri:
  - BINANCE_API_KEY
  - BINANCE_API_SECRET
  - BINANCE_FUTURES_BASE_URL   (testnet için: https://testnet.binancefuture.com)
  - BOT_DRY_RUN=true/false

GÜVENLİK:
- BOT_DRY_RUN=false iken BASE_URL live ise, bot default olarak KENDİNİ KAPATIR.
- Live'a bilerek geçmek için ayrıca BOT_ALLOW_LIVE=true vermen gerekir.

KULLANIM:
python bot.py
"""

from __future__ import annotations

import os
import time
import json
import hmac
import hashlib
import urllib.parse
from typing import Dict, Any, List, Optional, Literal

import requests
import pandas as pd
import numpy as np

from dotenv import load_dotenv
load_dotenv()

from swings import find_swings
from plot_swings_from_api import generate_trades  # senin en güncel strateji (fills'li)


# =========================
# CONFIG
# =========================

SYMBOL: str = os.getenv("BOT_SYMBOL", "BTCUSDT")
INTERVAL: str = os.getenv("BOT_INTERVAL", "15m")  # Binance kline interval
LOOKBACK_LIMIT: int = int(os.getenv("BOT_LOOKBACK_LIMIT", "600"))  # kaç mum çekelim
LEFT_BARS: int = int(os.getenv("BOT_LEFT_BARS", "5"))
RIGHT_BARS: int = int(os.getenv("BOT_RIGHT_BARS", "1"))
MIN_DISTANCE: int = int(os.getenv("BOT_MIN_DISTANCE", "10"))

MAX_CHASE_PCT: float = float(os.getenv("BOT_MAX_CHASE_PCT", "0.03"))

# Risk / order sizing:
LEVERAGE: int = int(os.getenv("BOT_LEVERAGE", "10"))
MARGIN_PCT: float = float(os.getenv("BOT_MARGIN_PCT", "0.10"))

# Güvenlik:
DRY_RUN: bool = os.getenv("BOT_DRY_RUN", "true").lower() in ("1", "true", "yes", "y", "on")
ALLOW_LIVE: bool = os.getenv("BOT_ALLOW_LIVE", "false").lower() in ("1", "true", "yes", "y", "on")

STATE_FILE: str = os.getenv("BOT_STATE_FILE", "bot_state.json")

# Base URL ENV'den seçiliyor:
BASE_URL: str = os.getenv("BINANCE_FUTURES_BASE_URL", "https://fapi.binance.com").rstrip("/")


# =========================
# BINANCE SIGNED REST
# =========================

API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")


def _is_live_url(url: str) -> bool:
    u = url.rstrip("/")
    return u == "https://fapi.binance.com"


def _safety_checks_or_die():
    print("=== BOT START ===")
    print(f"SYMBOL={SYMBOL} INTERVAL={INTERVAL} LOOKBACK={LOOKBACK_LIMIT}")
    print(f"LEFT_BARS={LEFT_BARS} RIGHT_BARS={RIGHT_BARS} MIN_DISTANCE={MIN_DISTANCE}")
    print(f"LEVERAGE={LEVERAGE} MARGIN_PCT={MARGIN_PCT} MAX_CHASE_PCT={MAX_CHASE_PCT}")
    print(f"STATE_FILE={STATE_FILE}")
    print(f"DRY_RUN={DRY_RUN} ALLOW_LIVE={ALLOW_LIVE}")
    print(f"BASE_URL={BASE_URL}")

    # DRY_RUN=false iken key şart
    if not DRY_RUN and (not API_KEY or not API_SECRET):
        raise RuntimeError("BOT_DRY_RUN=false ama BINANCE_API_KEY / BINANCE_API_SECRET eksik.")

    # DRY_RUN=false iken live'a yanlışlıkla basmayı engelle
    if not DRY_RUN and _is_live_url(BASE_URL) and not ALLOW_LIVE:
        raise RuntimeError(
            "GÜVENLİK BLOĞU: BOT_DRY_RUN=false ve BASE_URL LIVE (https://fapi.binance.com).\n"
            "Live'a bilerek geçmek istiyorsan .env içine BOT_ALLOW_LIVE=true yaz.\n"
            "Testnet için: BINANCE_FUTURES_BASE_URL=https://testnet.binancefuture.com"
        )


def _sign_params(params: Dict[str, Any], secret: str) -> str:
    query = urllib.parse.urlencode(params, doseq=True)
    sig = hmac.new(secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
    return sig


def _headers() -> Dict[str, str]:
    return {"X-MBX-APIKEY": API_KEY}


def public_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    params = params or {}
    r = requests.get(BASE_URL + path, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def signed_request(method: str, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    if DRY_RUN:
        raise RuntimeError("signed_request DRY_RUN modunda çağrılmamalı (emir yok).")

    params = params or {}
    params["timestamp"] = int(time.time() * 1000)
    params["recvWindow"] = 5000
    params["signature"] = _sign_params(params, API_SECRET)

    url = BASE_URL + path
    method = method.upper()

    if method == "GET":
        r = requests.get(url, params=params, headers=_headers(), timeout=20)
    elif method == "POST":
        r = requests.post(url, params=params, headers=_headers(), timeout=20)
    elif method == "DELETE":
        r = requests.delete(url, params=params, headers=_headers(), timeout=20)
    else:
        raise ValueError(f"Unsupported HTTP method: {method}")

    r.raise_for_status()
    return r.json()


# =========================
# MARKET DATA
# =========================

def fetch_ohlc(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    data = public_get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})

    cols = [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "number_of_trades", "taker_buy_base_volume",
        "taker_buy_quote_volume", "ignore",
    ]
    df = pd.DataFrame(data, columns=cols)
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    df["Date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("Date", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]]
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    return df.sort_index()


def last_closed_bar_time(df: pd.DataFrame) -> pd.Timestamp:
    return df.index[-1]


# =========================
# STATE
# =========================

def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        return {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "last_processed_bar": None,       # ISO timestamp
            "last_processed_fill_key": None,  # string
        }
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "last_processed_bar": None,
            "last_processed_fill_key": None,
        }


def save_state(st: Dict[str, Any]) -> None:
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_FILE)


# =========================
# ACCOUNT / ORDERS
# =========================

def set_leverage(symbol: str, leverage: int) -> None:
    if DRY_RUN:
        print(f"[DRY_RUN] set_leverage({symbol}, {leverage})")
        return
    signed_request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})


def get_account_balance_usdt() -> float:
    """
    Futures account balance (USDT). Basit yaklaşım: totalWalletBalance.
    """
    if DRY_RUN:
        return float(os.getenv("BOT_DRYRUN_BALANCE", "1000"))
    acc = signed_request("GET", "/fapi/v2/account", {})
    return float(acc.get("totalWalletBalance", 0.0))


def get_position_amt(symbol: str) -> float:
    """
    positionAmt: + long, - short, 0 flat
    """
    if DRY_RUN:
        return 0.0
    pos = signed_request("GET", "/fapi/v2/positionRisk", {})
    for p in pos:
        if p.get("symbol") == symbol:
            return float(p.get("positionAmt", 0.0))
    return 0.0


def get_mark_price(symbol: str) -> float:
    px = public_get("/fapi/v1/premiumIndex", {"symbol": symbol})
    return float(px["markPrice"])


def round_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return float(np.floor(value / step) * step)


def get_symbol_filters(symbol: str) -> Dict[str, float]:
    info = public_get("/fapi/v1/exchangeInfo", {})
    for s in info.get("symbols", []):
        if s.get("symbol") == symbol:
            out: Dict[str, float] = {}
            for f in s.get("filters", []):
                if f.get("filterType") == "LOT_SIZE":
                    out["stepSize"] = float(f.get("stepSize", "0"))
                    out["minQty"] = float(f.get("minQty", "0"))
                if f.get("filterType") == "MARKET_LOT_SIZE":
                    out["marketStepSize"] = float(f.get("stepSize", "0"))
                    out["marketMinQty"] = float(f.get("minQty", "0"))
            return out
    return {}


def calc_market_qty(symbol: str, margin_pct: float, leverage: int) -> float:
    """
    notional = equity * margin_pct * leverage
    qty = notional / markPrice
    """
    equity = get_account_balance_usdt()
    notional = equity * margin_pct * float(leverage)
    price = get_mark_price(symbol)
    if price <= 0:
        return 0.0

    raw_qty = notional / price

    filters = get_symbol_filters(symbol)
    step = filters.get("marketStepSize") or filters.get("stepSize") or 0.0
    qty = round_step(raw_qty, step) if step else raw_qty

    min_qty = filters.get("marketMinQty") or filters.get("minQty") or 0.0
    if min_qty and qty < min_qty:
        return 0.0
    return float(qty)


def place_market_order(symbol: str, side: Literal["BUY", "SELL"], qty: float) -> Dict[str, Any]:
    if qty <= 0:
        raise ValueError("qty <= 0")

    if DRY_RUN:
        print(f"[DRY_RUN] MARKET {symbol} {side} qty={qty}")
        return {"dry_run": True, "symbol": symbol, "side": side, "origQty": qty}

    return signed_request("POST", "/fapi/v1/order", {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "quantity": qty,
    })


def close_position_market(symbol: str) -> None:
    amt = get_position_amt(symbol)
    if abs(amt) < 1e-12:
        return
    side = "SELL" if amt > 0 else "BUY"
    qty = abs(amt)
    place_market_order(symbol, side, qty)


# =========================
# STRATEGY -> LIVE ACTIONS
# =========================

def fill_key(ev: Any) -> str:
    i = int(getattr(ev, "index", -1))
    r = str(getattr(ev, "reason", ""))
    q = float(getattr(ev, "qty_delta", 0.0))
    p = float(getattr(ev, "price", 0.0))
    return f"{i}|{r}|{q:.6f}|{p:.2f}"


def decide_and_execute_from_latest(df: pd.DataFrame, st: Dict[str, Any]) -> None:
    highs = df["High"].values
    lows = df["Low"].values

    swings = find_swings(
        highs, lows,
        left_bars=LEFT_BARS,
        right_bars=RIGHT_BARS,
        min_distance=MIN_DISTANCE,
        alt_min_distance=None
    )
    print(f"[DBG] swings={len(swings)}")

    trades = generate_trades(
        df=df,
        swings=swings,
        stop_buffer_pct=0.0,
        right_bars=RIGHT_BARS,
        max_chase_pct=MAX_CHASE_PCT
    )
    print(f"[DBG] trades={len(trades)}")

    if not trades:
        return

    last_trade = trades[-1]
    direction = getattr(last_trade, "direction", None)
    fills = list(getattr(last_trade, "fills", None) or [])
    print(f"[DBG] last_trade.direction={direction} fills={len(fills)}")

    if not fills:
        return

    fills = sorted(fills, key=lambda e: int(getattr(e, "index", -1)))

    last_key = st.get("last_processed_fill_key")
    print(f"[DBG] last_processed_fill_key={last_key}")

    new_fills: List[Any] = []
    seen_last = (last_key is None)

    for ev in fills:
        k = fill_key(ev)
        if not seen_last:
            if k == last_key:
                seen_last = True
            continue
        else:
            if last_key is not None and k == last_key:
                continue
            new_fills.append(ev)

    print(f"[DBG] new_fills={len(new_fills)}")
    if not new_fills:
        return

    if direction not in ("long", "short"):
        return

    for ev in new_fills:
        reason = str(getattr(ev, "reason", ""))
        qty_delta = float(getattr(ev, "qty_delta", 0.0))

        print(f"[FILL] reason={reason} qty_delta={qty_delta} @ i={getattr(ev,'index',None)}")

        pos_amt = get_position_amt(SYMBOL)  # signed
        is_flat = (abs(pos_amt) < 1e-12)

        if reason == "entry":
            if not is_flat:
                print("  -> Skip entry (position already open).")
            else:
                qty = calc_market_qty(SYMBOL, MARGIN_PCT, LEVERAGE)
                if qty <= 0:
                    print("  -> Qty too small, skip.")
                else:
                    side = "BUY" if direction == "long" else "SELL"
                    resp = place_market_order(SYMBOL, side, qty)
                    print(f"  -> order_resp={resp}")

        elif reason in ("tp50_in_target", "readd50_in_target"):
            pos_amt = get_position_amt(SYMBOL)
            if abs(pos_amt) < 1e-12:
                print("  -> Skip (no position).")
            else:
                half_qty = abs(pos_amt) * 0.5
                if reason == "tp50_in_target":
                    side = "SELL" if pos_amt > 0 else "BUY"
                else:
                    side = "BUY" if pos_amt > 0 else "SELL"
                resp = place_market_order(SYMBOL, side, half_qty)
                print(f"  -> order_resp={resp}")

        elif reason in ("exit", "stop"):
            close_position_market(SYMBOL)

        else:
            print("  -> Unknown fill reason, ignore.")

        st["last_processed_fill_key"] = fill_key(ev)
        save_state(st)


# =========================
# LOOP
# =========================

def main():
    _safety_checks_or_die()

    # leverage set
    set_leverage(SYMBOL, LEVERAGE)

    st = load_state()

    while True:
        try:
            df = fetch_ohlc(SYMBOL, INTERVAL, LOOKBACK_LIMIT)
            if df.empty:
                time.sleep(5)
                continue

            t_last = last_closed_bar_time(df)
            t_last_iso = t_last.isoformat()

            if st.get("last_processed_bar") != t_last_iso:
                print(f"\n[NEW BAR] {t_last_iso} | Close={df['Close'].iloc[-1]}")
                st["last_processed_bar"] = t_last_iso
                save_state(st)

                decide_and_execute_from_latest(df, st)

            time.sleep(int(os.getenv("BOT_POLL_SECONDS", "20")))

        except requests.HTTPError as e:
            print("[HTTP ERROR]", e)
            time.sleep(5)
        except Exception as e:
            print("[ERROR]", repr(e))
            time.sleep(5)


if __name__ == "__main__":
    main()
