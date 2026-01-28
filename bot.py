# bot.py
from __future__ import annotations

import os
import time
import json
import hmac
import hashlib
import urllib.parse
from typing import Dict, Any, List, Optional, Literal, Tuple

import requests
import pandas as pd
import numpy as np

from dotenv import load_dotenv
load_dotenv()

from swings import find_swings
from plot_swings_from_api import generate_trades


# =========================
# CONFIG
# =========================

SYMBOL = os.getenv("BOT_SYMBOL", "BTCUSDT")
INTERVAL = os.getenv("BOT_INTERVAL", "15m")
LOOKBACK_LIMIT = int(os.getenv("BOT_LOOKBACK_LIMIT", "192"))

LEFT_BARS = int(os.getenv("BOT_LEFT_BARS", "5"))
RIGHT_BARS = int(os.getenv("BOT_RIGHT_BARS", "1"))
MIN_DISTANCE = int(os.getenv("BOT_MIN_DISTANCE", "10"))

# ✅ Yeni swing filtre parametreleri (son hal)
MIN_SAME_KIND_GAP = int(os.getenv("BOT_MIN_SAME_KIND_GAP", "7"))  # LH->LH ve LB->LB
MIN_OPPOSITE_GAP = int(os.getenv("BOT_MIN_OPPOSITE_GAP", "3"))    # LH<->LB arası 3
SWING_DEBUG = os.getenv("BOT_SWING_DEBUG", "false").lower() in ("1", "true", "yes", "y", "on")

MAX_CHASE_PCT = float(os.getenv("BOT_MAX_CHASE_PCT", "0.03"))

LEVERAGE = int(os.getenv("BOT_LEVERAGE", "10"))
MARGIN_PCT = float(os.getenv("BOT_MARGIN_PCT", "0.10"))

DRY_RUN = os.getenv("BOT_DRY_RUN", "true").lower() in ("1", "true", "yes", "y", "on")
ALLOW_LIVE = os.getenv("BOT_ALLOW_LIVE", "false").lower() in ("1", "true", "yes", "y", "on")

STATE_FILE = os.getenv("BOT_STATE_FILE", "bot_state.json")
BASE_URL = os.getenv("BINANCE_FUTURES_BASE_URL", "https://fapi.binance.com").rstrip("/")

API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

POLL_SECONDS = int(os.getenv("BOT_POLL_SECONDS", "20"))

HEARTBEAT_ENABLED = os.getenv("BOT_HEARTBEAT", "false").lower() in ("1", "true", "yes", "y", "on")
HEARTBEAT_EVERY_N = int(os.getenv("BOT_HEARTBEAT_EVERY", "15"))

DRYRUN_BALANCE = float(os.getenv("BOT_DRYRUN_BALANCE", "1000"))


# =========================
# TYPES
# =========================

TrendLabel = Literal["long", "short", "belirsiz", "trend_yok"]
WaitMode = Literal["wait_long", "wait_short", "accept_any"]


# =========================
# SAFETY / UTILS
# =========================

def _is_live_url(url: str) -> bool:
    return url.rstrip("/") == "https://fapi.binance.com"


def _safety_checks_or_die() -> None:
    print("=== BOT START ===")
    print(f"SYMBOL={SYMBOL} INTERVAL={INTERVAL} LOOKBACK={LOOKBACK_LIMIT}")
    print(f"LEFT_BARS={LEFT_BARS} RIGHT_BARS={RIGHT_BARS} MIN_DISTANCE={MIN_DISTANCE}")
    print(f"MIN_SAME_KIND_GAP={MIN_SAME_KIND_GAP} MIN_OPPOSITE_GAP={MIN_OPPOSITE_GAP} SWING_DEBUG={SWING_DEBUG}")
    print(f"LEVERAGE={LEVERAGE} MARGIN_PCT={MARGIN_PCT} MAX_CHASE_PCT={MAX_CHASE_PCT}")
    print(f"STATE_FILE={STATE_FILE}")
    print(f"DRY_RUN={DRY_RUN} ALLOW_LIVE={ALLOW_LIVE}")
    print(f"BASE_URL={BASE_URL}")

    if not DRY_RUN and (not API_KEY or not API_SECRET):
        raise RuntimeError("BOT_DRY_RUN=false ama BINANCE_API_KEY / BINANCE_API_SECRET eksik.")

    if not DRY_RUN and _is_live_url(BASE_URL) and not ALLOW_LIVE:
        raise RuntimeError(
            "GÜVENLİK BLOĞU: BOT_DRY_RUN=false ve BASE_URL LIVE (https://fapi.binance.com).\n"
            "Live'a bilerek geçmek istiyorsan .env içine BOT_ALLOW_LIVE=true yaz.\n"
            "Testnet için: BINANCE_FUTURES_BASE_URL=https://testnet.binancefuture.com"
        )

    # Binance limit guard (tek request max 1500)
    if LOOKBACK_LIMIT <= 0 or LOOKBACK_LIMIT > 1500:
        raise RuntimeError("BOT_LOOKBACK_LIMIT 1..1500 aralığında olmalı (Binance tek istekte max=1500).")


def _sign(params: Dict[str, Any]) -> str:
    q = urllib.parse.urlencode(params, doseq=True)
    return hmac.new(API_SECRET.encode("utf-8"), q.encode("utf-8"), hashlib.sha256).hexdigest()


def _headers() -> Dict[str, str]:
    return {"X-MBX-APIKEY": API_KEY}


def _public_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    r = requests.get(BASE_URL + path, params=params or {}, timeout=20)
    r.raise_for_status()
    return r.json()


# =========================
# STATE
# =========================

def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        return {
            "last_processed_bar": None,
            "last_processed_fill_key": None,

            "first_run": True,
            "startup_wait_mode": None,
            "startup_observed_trend": None,
            "startup_unlocked": False,

            "armed": False,
            "last_trade_direction": None,
        }

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            st = json.load(f)

        st.setdefault("last_processed_bar", None)
        st.setdefault("last_processed_fill_key", None)

        st.setdefault("first_run", True)
        st.setdefault("startup_wait_mode", None)
        st.setdefault("startup_observed_trend", None)
        st.setdefault("startup_unlocked", False)

        st.setdefault("armed", False)
        st.setdefault("last_trade_direction", None)
        return st
    except Exception:
        return {
            "last_processed_bar": None,
            "last_processed_fill_key": None,
            "first_run": True,
            "startup_wait_mode": None,
            "startup_observed_trend": None,
            "startup_unlocked": False,
            "armed": False,
            "last_trade_direction": None,
        }


def save_state(st: Dict[str, Any]) -> None:
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_FILE)


# =========================
# MARKET DATA
# =========================

def fetch_ohlc() -> pd.DataFrame:
    data = _public_get(
        "/fapi/v1/klines",
        {"symbol": SYMBOL, "interval": INTERVAL, "limit": LOOKBACK_LIMIT},
    )

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tb", "tq", "ignore"
    ])
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    df["Date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("Date", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]]
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    return df.sort_index()


def get_mark_price() -> Optional[float]:
    try:
        px = _public_get("/fapi/v1/premiumIndex", {"symbol": SYMBOL})
        return float(px["markPrice"])
    except Exception:
        return None


# =========================
# ACCOUNT / ORDERS
# =========================

def signed_request(method: str, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    if DRY_RUN:
        raise RuntimeError("signed_request in DRY_RUN")

    params = params or {}
    params["timestamp"] = int(time.time() * 1000)
    params["recvWindow"] = 5000
    params["signature"] = _sign(params)

    r = requests.request(method, BASE_URL + path, params=params, headers=_headers(), timeout=20)
    r.raise_for_status()
    return r.json()


def get_position_amt() -> float:
    if DRY_RUN:
        return 0.0
    pos = signed_request("GET", "/fapi/v2/positionRisk")
    for p in pos:
        if p.get("symbol") == SYMBOL:
            return float(p.get("positionAmt", 0.0))
    return 0.0


def get_account_balance_usdt() -> float:
    if DRY_RUN:
        return DRYRUN_BALANCE
    acc = signed_request("GET", "/fapi/v2/account")
    return float(acc.get("totalWalletBalance", 0.0))


def get_symbol_filters() -> Dict[str, float]:
    info = _public_get("/fapi/v1/exchangeInfo", {})
    for s in info.get("symbols", []):
        if s.get("symbol") == SYMBOL:
            out: Dict[str, float] = {}
            for f in s.get("filters", []):
                if f.get("filterType") == "MARKET_LOT_SIZE":
                    out["marketStepSize"] = float(f.get("stepSize", "0"))
                    out["marketMinQty"] = float(f.get("minQty", "0"))
                if f.get("filterType") == "LOT_SIZE":
                    out["stepSize"] = float(f.get("stepSize", "0"))
                    out["minQty"] = float(f.get("minQty", "0"))
            return out
    return {}


def round_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return float(np.floor(value / step) * step)


def calc_market_qty() -> float:
    equity = get_account_balance_usdt()
    mp = get_mark_price()
    price = mp if mp and mp > 0 else None
    if price is None:
        return 0.0

    notional = equity * MARGIN_PCT * float(LEVERAGE)
    raw_qty = notional / price

    filters = get_symbol_filters()
    step = filters.get("marketStepSize") or filters.get("stepSize") or 0.0
    qty = round_step(raw_qty, step) if step else raw_qty

    min_qty = filters.get("marketMinQty") or filters.get("minQty") or 0.0
    if min_qty and qty < min_qty:
        return 0.0
    return float(qty)


def market_order(side: Literal["BUY", "SELL"], qty: float) -> Any:
    if qty <= 0:
        raise ValueError("qty <= 0")

    if DRY_RUN:
        print(f"[DRY_RUN] MARKET {SYMBOL} {side} qty={qty}")
        return {"dry_run": True, "symbol": SYMBOL, "side": side, "origQty": qty}

    return signed_request("POST", "/fapi/v1/order", {
        "symbol": SYMBOL,
        "side": side,
        "type": "MARKET",
        "quantity": qty,
    })


def close_position() -> None:
    amt = get_position_amt()
    if abs(amt) < 1e-12:
        return
    side = "SELL" if amt > 0 else "BUY"
    market_order(side, abs(amt))


# =========================
# TREND HELPERS
# =========================

def fill_key(ev: Any) -> str:
    return f"{int(ev.index)}|{str(ev.reason)}|{float(ev.qty_delta):.6f}|{float(ev.price):.2f}"


def _compute_swings(df: pd.DataFrame):
    # Not: min_distance/alt_min_distance imzada duruyor (kullanmıyoruz)
    return find_swings(
        df["High"].values,
        df["Low"].values,
        left_bars=LEFT_BARS,
        right_bars=RIGHT_BARS,
        min_distance=MIN_DISTANCE,
        alt_min_distance=None,
        min_same_kind_gap=MIN_SAME_KIND_GAP,
        min_opposite_gap=MIN_OPPOSITE_GAP,
        debug=SWING_DEBUG,
    )


def detect_current_trend(df: pd.DataFrame) -> Tuple[TrendLabel, int, int]:
    swings = _compute_swings(df)

    trades = generate_trades(
        df=df,
        swings=swings,
        stop_buffer_pct=0.0,
        right_bars=RIGHT_BARS,
        max_chase_pct=MAX_CHASE_PCT
    )

    if not trades:
        return "trend_yok", len(swings), 0

    last_trade = trades[-1]
    d = getattr(last_trade, "direction", None)
    if d == "long":
        return "long", len(swings), len(trades)
    if d == "short":
        return "short", len(swings), len(trades)
    return "belirsiz", len(swings), len(trades)


def desired_wait_mode(observed_trend: TrendLabel) -> WaitMode:
    if observed_trend == "short":
        return "wait_long"
    if observed_trend == "long":
        return "wait_short"
    return "accept_any"


def format_trend_label(t: TrendLabel) -> str:
    if t == "trend_yok":
        return "trend yok"
    return t


def format_wait_mode(m: WaitMode) -> str:
    if m == "wait_long":
        return "LONG trend sinyali bekleniyor"
    if m == "wait_short":
        return "SHORT trend sinyali bekleniyor"
    return "LONG veya SHORT kabul (ilk gelen)"


def is_unlocked_by_trend(wait_mode: WaitMode, current_trend: TrendLabel) -> bool:
    if wait_mode == "wait_long":
        return current_trend == "long"
    if wait_mode == "wait_short":
        return current_trend == "short"
    return current_trend in ("long", "short")


# =========================
# CORE LOGIC
# =========================

def decide_and_execute(df: pd.DataFrame, st: Dict[str, Any]) -> None:
    swings = _compute_swings(df)

    trades = generate_trades(
        df=df,
        swings=swings,
        stop_buffer_pct=0.0,
        right_bars=RIGHT_BARS,
        max_chase_pct=MAX_CHASE_PCT
    )

    if not trades:
        current_trend: TrendLabel = "trend_yok"
        fills: List[Any] = []
    else:
        last_trade = trades[-1]
        d = getattr(last_trade, "direction", None)
        current_trend = d if d in ("long", "short") else "belirsiz"  # type: ignore
        fills = list(getattr(last_trade, "fills", None) or [])

    # İlk run state kaydı (safety)
    if st.get("first_run", False):
        st["startup_observed_trend"] = current_trend
        st["startup_wait_mode"] = desired_wait_mode(current_trend)
        st["startup_unlocked"] = False
        st["armed"] = False
        st["last_trade_direction"] = current_trend
        st["first_run"] = False
        save_state(st)
        return

    # Unlock kontrolü
    if not st.get("startup_unlocked", False):
        wm: WaitMode = st.get("startup_wait_mode") or "accept_any"
        if is_unlocked_by_trend(wm, current_trend):
            st["startup_unlocked"] = True
            st["armed"] = True
            print(f"[UNLOCK] {format_wait_mode(wm)} sağlandı -> artık entry serbest (armed=True).")
            save_state(st)
        else:
            st["last_trade_direction"] = current_trend
            save_state(st)
            return

    # Trend değiştiyse armed
    prev_dir = st.get("last_trade_direction")
    if prev_dir is not None and current_trend in ("long", "short") and prev_dir in ("long", "short"):
        if current_trend != prev_dir:
            st["armed"] = True
            print(f"[ARMED] Trend change: {prev_dir} -> {current_trend}")

    st["last_trade_direction"] = current_trend

    if not fills:
        save_state(st)
        return

    fills = sorted(fills, key=lambda x: int(getattr(x, "index", -1)))

    last_processed = st.get("last_processed_fill_key")
    new_fills: List[Any] = []

    seen = (last_processed is None)
    for f in fills:
        k = fill_key(f)
        if not seen:
            if k == last_processed:
                seen = True
            continue
        if last_processed is not None and k == last_processed:
            continue
        new_fills.append(f)

    if not new_fills:
        save_state(st)
        return

    for ev in new_fills:
        reason = str(getattr(ev, "reason", ""))
        qty_delta = float(getattr(ev, "qty_delta", 0.0))

        print(f"[FILL] reason={reason} qty_delta={qty_delta} @ i={getattr(ev, 'index', None)}")

        pos = get_position_amt()
        is_flat = abs(pos) < 1e-12

        if reason == "entry":
            if not st.get("armed", False):
                print("  -> entry blocked (armed=False; trend change bekleniyor)")
                continue
            if not is_flat:
                print("  -> skip entry (position already open)")
                continue
            if current_trend not in ("long", "short"):
                print("  -> skip entry (current trend belirsiz/trend yok)")
                continue

            qty = calc_market_qty()
            if qty <= 0:
                print("  -> qty too small, skip")
                continue

            side: Literal["BUY", "SELL"] = "BUY" if current_trend == "long" else "SELL"
            resp = market_order(side, qty)
            print(f"  -> order_resp={resp}")
            st["armed"] = False

        elif reason.startswith("tp50"):
            if is_flat:
                print("  -> skip tp50 (no position)")
                continue
            half = abs(pos) * 0.5
            side = "SELL" if pos > 0 else "BUY"
            resp = market_order(side, half)
            print(f"  -> order_resp={resp}")

        elif reason.startswith("readd"):
            if is_flat:
                print("  -> skip readd (no position)")
                continue
            half = abs(pos) * 0.5
            side = "BUY" if pos > 0 else "SELL"
            resp = market_order(side, half)
            print(f"  -> order_resp={resp}")

        elif reason in ("stop", "exit"):
            close_position()
            st["armed"] = False

        else:
            print("  -> unknown reason, ignore")

        st["last_processed_fill_key"] = fill_key(ev)
        save_state(st)


# =========================
# LOOP
# =========================

def main():
    _safety_checks_or_die()
    st = load_state()

    print(f"\n[INFO] Bot dinlemede: {SYMBOL} / {INTERVAL} | poll={POLL_SECONDS}s | source={BASE_URL}")

    # STARTUP bilgi
    try:
        df0 = fetch_ohlc()
        last_bar_iso = df0.index[-1].isoformat()
        close0 = float(df0["Close"].iloc[-1])
        mark0 = get_mark_price()

        trend0, swc0, trc0 = detect_current_trend(df0)
        wait_mode0 = desired_wait_mode(trend0)

        print(f"[STARTUP] son bar={last_bar_iso}")
        if mark0 is None:
            print(f"[STARTUP] close={close0:.4f}")
        else:
            print(f"[STARTUP] close={close0:.4f} | mark={mark0:.4f}")

        print(f"[STARTUP] swings={swc0} trades={trc0} | mevcut durum: {format_trend_label(trend0)}")
        print(f"[STARTUP] strateji: {format_wait_mode(wait_mode0)}")

        # İlk kez koşuyorsa state’e yaz
        if st.get("first_run", True):
            st["startup_observed_trend"] = trend0
            st["startup_wait_mode"] = wait_mode0
            st["startup_unlocked"] = False
            st["armed"] = False
            st["last_trade_direction"] = trend0
            st["first_run"] = False
            save_state(st)

        # spam azaltma: ilk bar kaydı
        if st.get("last_processed_bar") is None:
            st["last_processed_bar"] = last_bar_iso
            save_state(st)

        print("[STARTUP] Not: Bu aşamada mevcut trend yönünde entry yok; beklenen trend gelince unlock olur.\n")

    except Exception as e:
        print("[STARTUP] Okuma hatası:", repr(e))

    hb_counter = 0

    while True:
        try:
            df = fetch_ohlc()
            bar_time = df.index[-1].isoformat()

            if st.get("last_processed_bar") != bar_time:
                print(f"\n[NEW BAR] {bar_time} close={df['Close'].iloc[-1]}")
                st["last_processed_bar"] = bar_time
                save_state(st)
                decide_and_execute(df, st)
            else:
                if HEARTBEAT_ENABLED:
                    hb_counter += 1
                    if hb_counter >= HEARTBEAT_EVERY_N:
                        hb_counter = 0
                        print(f"[HB] waiting... last_bar={st.get('last_processed_bar')}")

            time.sleep(POLL_SECONDS)

        except requests.HTTPError as e:
            print("[HTTP ERROR]", e)
            time.sleep(5)
        except Exception as e:
            print("[ERROR]", repr(e))
            time.sleep(5)


if __name__ == "__main__":
    main()
