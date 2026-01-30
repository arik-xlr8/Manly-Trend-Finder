# bot.py
from __future__ import annotations

import os
import time
import json
import hmac
import hashlib
import urllib.parse
import threading
import queue
import re
from collections import deque
from typing import Dict, Any, List, Optional, Literal, Deque, Tuple

import requests
import pandas as pd
import numpy as np

from dotenv import load_dotenv
load_dotenv()

from websocket import WebSocketApp

from swings import find_swings
from plot_swings_from_api import generate_trades


# =========================
# CONFIG
# =========================

SYMBOL = os.getenv("BOT_SYMBOL", "BTCUSDT")
INTERVAL = os.getenv("BOT_INTERVAL", "15m")
LOOKBACK_LIMIT = int(os.getenv("BOT_LOOKBACK_LIMIT", "600"))

LEFT_BARS = int(os.getenv("BOT_LEFT_BARS", "5"))
RIGHT_BARS = int(os.getenv("BOT_RIGHT_BARS", "1"))
MIN_DISTANCE = int(os.getenv("BOT_MIN_DISTANCE", "10"))

MIN_SAME_KIND_GAP = int(os.getenv("BOT_MIN_SAME_KIND_GAP", "5"))
MIN_OPPOSITE_GAP = int(os.getenv("BOT_MIN_OPPOSITE_GAP", "2"))
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

CLIENT_ID_PREFIX = os.getenv("BOT_CLIENT_ID_PREFIX", "MANLYBOT")

# Stop tetikleme tipi: MARK_PRICE | CONTRACT_PRICE
WORKING_TYPE = os.getenv("BOT_WORKING_TYPE", "MARK_PRICE").upper().strip()

# pivot dışına buffer (örn 0.004 = %0.4)
STOP_BUFFER_PCT = float(os.getenv("BOT_STOP_BUFFER_PCT", "0.0"))

# Websocket URL (market stream)
WS_BASE_URL = os.getenv("BINANCE_WS_URL", "wss://fstream.binance.com/ws").rstrip("/")

# exchangeInfo cache (TTL)
EXCHANGEINFO_TTL_SEC = int(os.getenv("BOT_EXCHANGEINFO_TTL_SEC", "3600"))

# --- NEW: Per-band TP50/READD rules ---
# "0 ve 1 bandı" hariç => band_id 0 ve 1 engellenir
BANNED_BANDS = {0, 1}

# =========================
# TYPES
# =========================

TrendLabel = Literal["long", "short", "belirsiz", "trend_yok"]
Side = Literal["BUY", "SELL"]
Tunnel = Literal["YUKSELEN", "DUSEN", "BELIRSIZ", "YOK"]


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
    print(f"WS_BASE_URL={WS_BASE_URL}")
    print(f"WORKING_TYPE={WORKING_TYPE} STOP_BUFFER_PCT={STOP_BUFFER_PCT}")
    print(f"CLIENT_ID_PREFIX={CLIENT_ID_PREFIX}\n")

    if WORKING_TYPE not in ("MARK_PRICE", "CONTRACT_PRICE"):
        raise RuntimeError("BOT_WORKING_TYPE geçersiz. MARK_PRICE veya CONTRACT_PRICE olmalı.")

    if not DRY_RUN and (not API_KEY or not API_SECRET):
        raise RuntimeError("BOT_DRY_RUN=false ama BINANCE_API_KEY / BINANCE_API_SECRET eksik.")

    if not DRY_RUN and _is_live_url(BASE_URL) and not ALLOW_LIVE:
        raise RuntimeError(
            "GÜVENLİK BLOĞU: BOT_DRY_RUN=false ve BASE_URL LIVE (https://fapi.binance.com).\n"
            "Live'a bilerek geçmek istiyorsan .env içine BOT_ALLOW_LIVE=true yaz.\n"
            "Testnet için: BINANCE_FUTURES_BASE_URL=https://testnet.binancefuture.com"
        )


def _sign(params: Dict[str, Any]) -> str:
    q = urllib.parse.urlencode(params, doseq=True)
    return hmac.new(API_SECRET.encode("utf-8"), q.encode("utf-8"), hashlib.sha256).hexdigest()


def _headers() -> Dict[str, str]:
    return {"X-MBX-APIKEY": API_KEY}


def _public_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    r = requests.get(BASE_URL + path, params=params or {}, timeout=20)
    r.raise_for_status()
    return r.json()


def _almost_equal(a: Optional[float], b: Optional[float], eps: float) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= eps


# =========================
# STATE
# =========================

def _empty_band_map() -> Dict[str, Any]:
    # JSON friendly dict
    return {
        "tp50_done": {},    # band_id(str)->True
        "readd_done": {},   # band_id(str)->True
        "tp50_pivot_idx": {},  # band_id(str)->int (tp50 anında görülen pivot idx)
        "entry_pivot_idx": None,  # entry anındaki pivot idx (referans)
    }


def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        return {
            "last_closed_bar_time": None,
            "base_qty": 0.0,

            # stop algo
            "active_stop_client_id": None,   # clientAlgoId
            "last_stop_price": None,
            "stop_side": None,               # "BUY"/"SELL" (debug/sync)

            # legacy tp50 (kept for backward compatibility)
            "last_tp50_fill_bar": None,

            # first-entry gating
            "has_entered_once": False,

            # startup info (print once)
            "startup_printed": False,

            # ---- mode tracking ----
            "prev_pos_amt": 0.0,
            "uncertain_active": False,
            "last_entry_context": None,
            "uncertain_last_pivot_idx": None,
            "uncertain_last_pivot_kind": None,
            "uncertain_last_pivot_price": None,

            # ---- NEW: per-band action state ----
            "band_state": _empty_band_map(),
        }

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            st = json.load(f)

        st.setdefault("last_closed_bar_time", None)
        st.setdefault("base_qty", 0.0)
        st.setdefault("active_stop_client_id", None)
        st.setdefault("last_stop_price", None)
        st.setdefault("stop_side", None)
        st.setdefault("last_tp50_fill_bar", None)
        st.setdefault("has_entered_once", False)
        st.setdefault("startup_printed", False)

        st.setdefault("prev_pos_amt", 0.0)
        st.setdefault("uncertain_active", False)
        st.setdefault("last_entry_context", None)
        st.setdefault("uncertain_last_pivot_idx", None)
        st.setdefault("uncertain_last_pivot_kind", None)
        st.setdefault("uncertain_last_pivot_price", None)

        st.setdefault("band_state", _empty_band_map())
        # normalize missing keys
        st["band_state"].setdefault("tp50_done", {})
        st["band_state"].setdefault("readd_done", {})
        st["band_state"].setdefault("tp50_pivot_idx", {})
        st["band_state"].setdefault("entry_pivot_idx", None)

        return st
    except Exception:
        return {
            "last_closed_bar_time": None,
            "base_qty": 0.0,
            "active_stop_client_id": None,
            "last_stop_price": None,
            "stop_side": None,
            "last_tp50_fill_bar": None,
            "has_entered_once": False,
            "startup_printed": False,

            "prev_pos_amt": 0.0,
            "uncertain_active": False,
            "last_entry_context": None,
            "uncertain_last_pivot_idx": None,
            "uncertain_last_pivot_kind": None,
            "uncertain_last_pivot_price": None,

            "band_state": _empty_band_map(),
        }


def save_state(st: Dict[str, Any]) -> None:
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_FILE)


def _reset_band_state_for_new_position(st: Dict[str, Any], entry_pivot_idx: Optional[int]) -> None:
    st["band_state"] = _empty_band_map()
    st["band_state"]["entry_pivot_idx"] = int(entry_pivot_idx) if entry_pivot_idx is not None else None


# =========================
# ACCOUNT / ORDERS (SIGNED)
# =========================

def signed_request(method: str, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    if DRY_RUN:
        raise RuntimeError("signed_request in DRY_RUN")

    params = params or {}
    params["timestamp"] = int(time.time() * 1000)
    params["recvWindow"] = 5000
    params["signature"] = _sign(params)

    r = requests.request(method, BASE_URL + path, params=params, headers=_headers(), timeout=20)
    if r.status_code >= 400:
        try:
            print("[BINANCE-ERROR]", r.status_code, r.text)
        except Exception:
            pass
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
        return float(os.getenv("BOT_DRYRUN_BALANCE", "1000"))
    acc = signed_request("GET", "/fapi/v2/account")
    return float(acc.get("totalWalletBalance", 0.0))


def get_open_orders() -> List[Dict[str, Any]]:
    if DRY_RUN:
        return []
    return signed_request("GET", "/fapi/v1/openOrders", {"symbol": SYMBOL})


def get_open_algo_orders() -> List[Dict[str, Any]]:
    """
    GET /fapi/v1/openAlgoOrders
    """
    if DRY_RUN:
        return []
    return signed_request("GET", "/fapi/v1/openAlgoOrders", {"symbol": SYMBOL})


def cancel_order_by_client_id(client_id: str) -> None:
    if DRY_RUN:
        print(f"[DRY_RUN] CANCEL (normal) clientId={client_id}")
        return
    signed_request("DELETE", "/fapi/v1/order", {"symbol": SYMBOL, "origClientOrderId": client_id})


def cancel_algo_by_client_id(client_algo_id: str) -> None:
    """
    DELETE /fapi/v1/algoOrder  (clientAlgoId)
    """
    if DRY_RUN:
        print(f"[DRY_RUN] CANCEL (algo) clientAlgoId={client_algo_id}")
        return
    signed_request("DELETE", "/fapi/v1/algoOrder", {"symbol": SYMBOL, "clientAlgoId": client_algo_id})


def cancel_our_orders(prefix: str = CLIENT_ID_PREFIX) -> None:
    """
    Hem normal open orders hem algo open orders temizler.
    """
    try:
        oo = get_open_orders()
        for o in oo:
            cid = str(o.get("clientOrderId", ""))
            if cid.startswith(prefix):
                print(f"[CANCEL] normal open order: clientId={cid} type={o.get('type')} side={o.get('side')}")
                cancel_order_by_client_id(cid)
    except Exception as e:
        print("[WARN] cancel normal orders error:", repr(e))

    try:
        ao = get_open_algo_orders()
        for o in ao:
            cid = str(o.get("clientAlgoId", ""))
            if cid.startswith(prefix):
                print(f"[CANCEL] algo open order: clientAlgoId={cid} orderType={o.get('orderType')} side={o.get('side')}")
                cancel_algo_by_client_id(cid)
    except Exception as e:
        print("[WARN] cancel algo orders error:", repr(e))


# =========================
# EXCHANGEINFO FILTERS (CACHED)
# =========================

_symbol_filters_cache: Optional[Tuple[float, Dict[str, float]]] = None  # (ts, filters)


def get_symbol_filters() -> Dict[str, float]:
    global _symbol_filters_cache
    now = time.time()

    if _symbol_filters_cache is not None:
        ts, cached = _symbol_filters_cache
        if (now - ts) < EXCHANGEINFO_TTL_SEC and cached:
            return cached

    info = _public_get("/fapi/v1/exchangeInfo", {})
    for s in info.get("symbols", []):
        if s.get("symbol") == SYMBOL:
            out: Dict[str, float] = {}
            for f in s.get("filters", []):
                ft = f.get("filterType")
                if ft in ("MARKET_LOT_SIZE", "LOT_SIZE"):
                    out["stepSize"] = float(f.get("stepSize", "0"))
                    out["minQty"] = float(f.get("minQty", "0"))
                if ft == "PRICE_FILTER":
                    out["tickSize"] = float(f.get("tickSize", "0"))
                if ft in ("MIN_NOTIONAL", "NOTIONAL"):
                    out["minNotional"] = float(f.get("notional", f.get("minNotional", "0")))
            _symbol_filters_cache = (now, out)
            return out

    _symbol_filters_cache = (now, {})
    return {}


def round_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return float(np.floor(value / step) * step)


def round_tick_floor(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    return float(np.round(np.floor(price / tick) * tick, 12))


def round_tick_ceil(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    return float(np.round(np.ceil(price / tick) * tick, 12))


def get_mark_price() -> Optional[float]:
    try:
        px = _public_get("/fapi/v1/premiumIndex", {"symbol": SYMBOL})
        return float(px["markPrice"])
    except Exception:
        return None


def calc_market_qty() -> float:
    equity = get_account_balance_usdt()
    mp = get_mark_price()
    price = mp if mp and mp > 0 else None
    if price is None:
        return 0.0

    notional = equity * MARGIN_PCT * float(LEVERAGE)
    raw_qty = notional / price

    filters = get_symbol_filters()
    step = filters.get("stepSize", 0.0)
    qty = round_step(raw_qty, step) if step else raw_qty

    min_qty = filters.get("minQty", 0.0)
    if min_qty and qty < min_qty:
        return 0.0

    min_notional = filters.get("minNotional", 0.0)
    if min_notional and (qty * price) < min_notional:
        return 0.0

    return float(qty)


def market_order(side: Side, qty: float, reduce_only: bool = False, client_id: Optional[str] = None) -> Any:
    if qty <= 0:
        raise ValueError("qty <= 0")

    if DRY_RUN:
        print(f"[DRY_RUN] MARKET {SYMBOL} {side} qty={qty} reduceOnly={reduce_only} clientId={client_id}")
        return {"dry_run": True, "symbol": SYMBOL, "side": side, "origQty": qty, "reduceOnly": reduce_only, "clientOrderId": client_id}

    params: Dict[str, Any] = {
        "symbol": SYMBOL,
        "side": side,
        "type": "MARKET",
        "quantity": qty,
        "reduceOnly": "true" if reduce_only else "false",
        "newOrderRespType": "RESULT",
    }
    if client_id:
        params["newClientOrderId"] = client_id

    return signed_request("POST", "/fapi/v1/order", params)


def place_stop_market_closepos_algo(stop_side: Side, trigger_price: float, client_algo_id: str) -> Any:
    """
    STOP_MARKET closePosition -> Algo Order API
    POST /fapi/v1/algoOrder
      algoType=CONDITIONAL
      type=STOP_MARKET
      triggerPrice=<...>
      closePosition=true
      workingType=MARK_PRICE | CONTRACT_PRICE
      clientAlgoId=<...>
    """
    if DRY_RUN:
        print(f"[DRY_RUN] ALGO STOP_MARKET closePosition {SYMBOL} {stop_side} triggerPrice={trigger_price} clientAlgoId={client_algo_id} workingType={WORKING_TYPE}")
        return {"dry_run": True, "algoType": "CONDITIONAL", "type": "STOP_MARKET", "triggerPrice": trigger_price, "side": stop_side, "clientAlgoId": client_algo_id}

    params: Dict[str, Any] = {
        "algoType": "CONDITIONAL",
        "symbol": SYMBOL,
        "side": stop_side,
        "type": "STOP_MARKET",
        "triggerPrice": trigger_price,
        "closePosition": "true",
        "workingType": WORKING_TYPE,
        "clientAlgoId": client_algo_id,
        "newOrderRespType": "ACK",
    }
    return signed_request("POST", "/fapi/v1/algoOrder", params)


def _ensure_leverage():
    if DRY_RUN:
        return
    try:
        signed_request("POST", "/fapi/v1/leverage", {"symbol": SYMBOL, "leverage": LEVERAGE})
    except Exception as e:
        print("[WARN] leverage set error:", repr(e))


# =========================
# REST OHLC (startup print)
# =========================

def fetch_klines_limit(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    data = _public_get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": int(limit)})
    rows = []
    for r in data:
        rows.append(
            {
                "open_time": int(r[0]),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": float(r[5]),
            }
        )
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("Date").sort_index()
    df = df[["open", "high", "low", "close", "volume"]]
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    return df


# =========================
# STRATEGY / SWINGS
# =========================

def _compute_swings(df: pd.DataFrame):
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
        ignore_last_bar=True,
    )


def compute_trades(df_closed: pd.DataFrame):
    swings = _compute_swings(df_closed)
    trades = generate_trades(
        df=df_closed,
        swings=swings,
        stop_buffer_pct=STOP_BUFFER_PCT,
        right_bars=RIGHT_BARS,
        max_chase_pct=MAX_CHASE_PCT,
    )
    return swings, trades


def _swing_kind(s: Any) -> str:
    if isinstance(s, dict):
        return str(s.get("kind", s.get("type", s.get("label", ""))))
    if isinstance(s, (tuple, list)) and len(s) >= 2:
        return str(s[1])
    return str(getattr(s, "kind", getattr(s, "type", getattr(s, "label", ""))))


def _swing_price(s: Any) -> Optional[float]:
    if isinstance(s, dict):
        for k in ("price", "p", "value", "y"):
            if k in s:
                try:
                    return float(s[k])
                except Exception:
                    pass
        return None
    if isinstance(s, (tuple, list)):
        if len(s) >= 3:
            try:
                return float(s[2])
            except Exception:
                return None
        return None
    for k in ("price", "p", "value", "y"):
        if hasattr(s, k):
            try:
                return float(getattr(s, k))
            except Exception:
                pass
    return None


def _is_low_kind(kind_up: str) -> bool:
    return ("LB" in kind_up) or ("LL" in kind_up) or (kind_up in ("L", "LOW", "BOTTOM"))


def _is_high_kind(kind_up: str) -> bool:
    return ("LH" in kind_up) or ("HH" in kind_up) or (kind_up in ("H", "HIGH", "TOP"))


def last_two_confirmed_pivots(swings: List[Any], want_low: bool) -> Optional[Tuple[float, float]]:
    found: List[float] = []
    for s in reversed(swings):
        kind = _swing_kind(s).upper()
        price = _swing_price(s)
        if price is None:
            continue
        if want_low and _is_low_kind(kind):
            found.append(float(price))
        if (not want_low) and _is_high_kind(kind):
            found.append(float(price))
        if len(found) >= 2:
            break

    if len(found) < 2:
        return None

    newest = found[0]
    older = found[1]
    return (older, newest)


def structural_entry_signal(swings: List[Any]) -> TrendLabel:
    lows = last_two_confirmed_pivots(swings, want_low=True)
    highs = last_two_confirmed_pivots(swings, want_low=False)

    if lows is not None:
        l1, l2 = lows
        if l2 > l1:
            return "long"

    if highs is not None:
        h1, h2 = highs
        if h2 < h1:
            return "short"

    if lows is None and highs is None:
        return "trend_yok"
    return "belirsiz"


def detect_tunnel(swings: List[Any]) -> Tunnel:
    sig = structural_entry_signal(swings)
    if sig == "long":
        return "YUKSELEN"
    if sig == "short":
        return "DUSEN"
    if sig == "trend_yok":
        return "YOK"
    return "BELIRSIZ"


def entry_dir_from_tunnel(tunnel: Tunnel) -> Optional[TrendLabel]:
    if tunnel == "YUKSELEN":
        return "long"
    if tunnel == "DUSEN":
        return "short"
    return None


def last_confirmed_pivot_stop(swings: List[Any], trend: TrendLabel) -> Optional[float]:
    want_low = (trend == "long")
    for s in reversed(swings):
        kind = _swing_kind(s).upper()
        price = _swing_price(s)
        if price is None:
            continue
        if want_low and _is_low_kind(kind):
            return float(price)
        if (not want_low) and _is_high_kind(kind):
            return float(price)
    return None


def get_last_pivot_label(swings: List[Any]) -> Optional[Tuple[str, float]]:
    for s in reversed(swings):
        kind = _swing_kind(s).upper()
        price = _swing_price(s)
        if price is None:
            continue
        if _is_low_kind(kind):
            return ("BOTTOM", float(price))
        if _is_high_kind(kind):
            return ("TOP", float(price))
    return None


def get_last_pivot_info(swings: List[Any]) -> Optional[Tuple[str, float, int]]:
    """
    Son pivotu: ("BOTTOM"/"TOP", price, index)
    """
    for s in reversed(swings):
        kind_up = _swing_kind(s).upper()
        price = _swing_price(s)
        if price is None:
            continue

        idx = None
        if isinstance(s, dict) and "index" in s:
            try:
                idx = int(s["index"])
            except Exception:
                idx = None
        elif hasattr(s, "index"):
            try:
                idx = int(getattr(s, "index"))
            except Exception:
                idx = None

        if idx is None:
            continue

        if _is_low_kind(kind_up):
            return ("BOTTOM", float(price), idx)
        if _is_high_kind(kind_up):
            return ("TOP", float(price), idx)
    return None


# =========================
# FIB BAND / ACTION DETECTION (ROBUST)
# =========================

_REASON_RE = re.compile(
    r"(?P<kind>tp50|readd)\s*[:\-_]?\s*(?:band\s*[:=\-_]?\s*)?(?P<band>-?\d+)",
    re.IGNORECASE
)

def _extract_band_events_from_last_trade(last_trade: Any) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    fills = list(getattr(last_trade, "fills", None) or [])
    for f in fills:
        reason = str(getattr(f, "reason", "") or "")
        m = _REASON_RE.search(reason)
        if not m:
            continue
        kind = str(m.group("kind")).lower().strip()
        try:
            band = int(m.group("band"))
        except Exception:
            continue
        out.append((kind, band))
    return out


def _eligible_pivot_for_tp50(pos_dir: TrendLabel, last_pivot_kind: Optional[str]) -> bool:
    if last_pivot_kind is None:
        return False
    k = last_pivot_kind.upper()
    if pos_dir == "long":
        return k == "TOP"
    if pos_dir == "short":
        return k == "BOTTOM"
    return False


# ✅ NEW (MINIMAL PATCH): READD pivot uygunluğu (generate_trades ile aynı)
def _eligible_pivot_for_readd(pos_dir: TrendLabel, last_pivot_kind: Optional[str]) -> bool:
    if last_pivot_kind is None:
        return False
    k = last_pivot_kind.upper()
    # generate_trades:
    #   long: readd => LB => BOTTOM
    #   short: readd => LH => TOP
    if pos_dir == "long":
        return k == "BOTTOM"
    if pos_dir == "short":
        return k == "TOP"
    return False


def _band_key(band_id: int) -> str:
    return str(int(band_id))


# =========================
# STOP SYNC (ALGO)
# =========================

def _parse_algo_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


def _algo_trigger_price(o: Dict[str, Any]) -> Optional[float]:
    for k in ("triggerPrice", "stopPrice", "price"):
        if k in o and o[k] is not None:
            try:
                return float(o[k])
            except Exception:
                pass
    return None


def _algo_client_id(o: Dict[str, Any]) -> str:
    return str(o.get("clientAlgoId") or o.get("clientOrderId") or "")


def _algo_is_closepos_stop(o: Dict[str, Any]) -> bool:
    order_type = str(o.get("type") or o.get("orderType") or "").upper()
    algo_type = str(o.get("algoType") or "").upper()
    close_pos = _parse_algo_bool(o.get("closePosition"))
    return close_pos and (order_type == "STOP_MARKET") and (algo_type in ("CONDITIONAL", "") or True)


def _find_existing_closepos_stop(stop_side: Side) -> Optional[Dict[str, Any]]:
    if DRY_RUN:
        return None
    try:
        ao = get_open_algo_orders()
    except Exception:
        return None

    matches: List[Dict[str, Any]] = []
    for o in ao:
        side = str(o.get("side") or "").upper()
        if side != stop_side:
            continue
        if not _algo_is_closepos_stop(o):
            continue
        matches.append(o)

    if not matches:
        return None

    for o in matches:
        if _algo_client_id(o).startswith(CLIENT_ID_PREFIX):
            return o
    return matches[0]


def sync_state_with_exchange_stop(st: Dict[str, Any], pos_amt: float) -> None:
    if DRY_RUN:
        return

    if abs(pos_amt) < 1e-12:
        st["active_stop_client_id"] = None
        st["last_stop_price"] = None
        st["stop_side"] = None
        save_state(st)
        return

    stop_side: Side = "SELL" if pos_amt > 0 else "BUY"
    existing = _find_existing_closepos_stop(stop_side)
    if existing is None:
        st["active_stop_client_id"] = None
        st["last_stop_price"] = None
        st["stop_side"] = stop_side
        save_state(st)
        return

    cid = _algo_client_id(existing)
    px = _algo_trigger_price(existing)

    st["active_stop_client_id"] = cid if cid else None
    st["last_stop_price"] = float(px) if (px is not None) else None
    st["stop_side"] = stop_side
    save_state(st)

    px_txt = f"{st['last_stop_price']:.10f}" if st.get("last_stop_price") else "N/A"
    print(f"[STOP-SYNC] exchange has open closePosition STOP_MARKET side={stop_side} clientAlgoId={cid} triggerPrice={px_txt}")


def sync_stop_only(st: Dict[str, Any], pos_amt: float, stop_price: Optional[float]) -> None:
    if abs(pos_amt) < 1e-12:
        cancel_our_orders()
        st["active_stop_client_id"] = None
        st["last_stop_price"] = None
        st["stop_side"] = None
        save_state(st)
        return

    if stop_price is None or stop_price <= 0:
        return

    filters = get_symbol_filters()
    tick = float(filters.get("tickSize", 0.0))
    eps = tick * 2 if tick > 0 else 1e-10

    stop_side: Side = "SELL" if pos_amt > 0 else "BUY"
    st["stop_side"] = stop_side

    try:
        sync_state_with_exchange_stop(st, pos_amt)
    except Exception as e:
        print("[WARN] stop sync_state_with_exchange_stop error:", repr(e))

    if tick > 0:
        if stop_side == "SELL":
            stop_price = round_tick_floor(float(stop_price), tick)
        else:
            stop_price = round_tick_ceil(float(stop_price), tick)
    else:
        stop_price = float(stop_price)

    exch_existing = None
    try:
        exch_existing = _find_existing_closepos_stop(stop_side)
    except Exception:
        exch_existing = None

    if exch_existing is not None:
        exch_px = _algo_trigger_price(exch_existing)
        exch_cid = _algo_client_id(exch_existing)
        if exch_px is not None and _almost_equal(float(exch_px), float(stop_price), eps):
            st["active_stop_client_id"] = exch_cid or st.get("active_stop_client_id")
            st["last_stop_price"] = float(exch_px)
            save_state(st)
            return

    last_stop = st.get("last_stop_price")
    if last_stop is not None and _almost_equal(float(last_stop), float(stop_price), eps):
        return

    def _try_cancel(cid: Optional[str]) -> None:
        if not cid:
            return
        try:
            print(f"[STOP] cancel old algo stop clientAlgoId={cid}")
            cancel_algo_by_client_id(cid)
        except Exception as e:
            print("[WARN] cancel old stop error:", repr(e))

    _try_cancel(st.get("active_stop_client_id"))
    if exch_existing is not None:
        exch_cid = _algo_client_id(exch_existing)
        if exch_cid and exch_cid != st.get("active_stop_client_id"):
            if exch_cid.startswith(CLIENT_ID_PREFIX):
                _try_cancel(exch_cid)
            else:
                print(f"[STOP] WARN: exchange stop exists but not ours (clientAlgoId={exch_cid}). "
                      f"Will NOT replace it to avoid interfering.")
                st["active_stop_client_id"] = exch_cid
                st["last_stop_price"] = _algo_trigger_price(exch_existing)
                save_state(st)
                return

    new_cid = f"{CLIENT_ID_PREFIX}-STOP-{int(time.time())}"
    print(f"[STOP] place ALGO STOP_MARKET closePosition triggerPrice={stop_price} side={stop_side} clientAlgoId={new_cid} workingType={WORKING_TYPE}")

    try:
        resp = place_stop_market_closepos_algo(stop_side, float(stop_price), new_cid)
        st["active_stop_client_id"] = new_cid
        st["last_stop_price"] = float(stop_price)
        save_state(st)
        print(f"[STOP] resp={resp}")
        return
    except Exception as e:
        print("[STOP] place error:", repr(e))
        try:
            sync_state_with_exchange_stop(st, pos_amt)
        except Exception as e2:
            print("[WARN] stop resync after place error:", repr(e2))
        save_state(st)
        return


# =========================
# CORE LOGIC
# =========================

def _pos_label(pos: float) -> str:
    if abs(pos) < 1e-12:
        return "FLAT"
    return "LONG" if pos > 0 else "SHORT"


def compute_stop_level_from_pivot(swings: List[Any], direction: TrendLabel) -> Optional[float]:
    if direction not in ("long", "short"):
        return None
    pivot = last_confirmed_pivot_stop(swings, direction)
    if pivot is None:
        return None
    if direction == "long":
        return float(pivot) * (1.0 - STOP_BUFFER_PCT)
    return float(pivot) * (1.0 + STOP_BUFFER_PCT)


def compute_stop_level_from_uncertain_signal(st: Dict[str, Any], direction: TrendLabel) -> Optional[float]:
    px = st.get("uncertain_last_pivot_price")
    if px is None:
        return None
    px = float(px)
    if direction == "long":
        return px * (1.0 - STOP_BUFFER_PCT)
    if direction == "short":
        return px * (1.0 + STOP_BUFFER_PCT)
    return None


# =========================
# STARTUP: REST BUFFER + "TRENE ATLA"
# =========================

def _buf_seed_from_df(df: pd.DataFrame) -> Deque[Dict[str, Any]]:
    buf: Deque[Dict[str, Any]] = deque(maxlen=LOOKBACK_LIMIT)
    for dt, r in df.iterrows():
        buf.append({
            "open_time": int(dt.value // 10**6),  # ns -> ms
            "open": float(r["Open"]),
            "high": float(r["High"]),
            "low": float(r["Low"]),
            "close": float(r["Close"]),
            "volume": float(r["Volume"]),
            "is_closed": True,
        })
    return buf


def startup_try_jump_on_train(st: Dict[str, Any], df: pd.DataFrame, swings: List[Any]) -> None:
    if st.get("has_entered_once", False):
        return

    pos = get_position_amt()
    if abs(pos) >= 1e-12:
        return

    tunnel = detect_tunnel(swings)
    plan_dir = entry_dir_from_tunnel(tunnel)
    if plan_dir not in ("long", "short"):
        return

    qty = calc_market_qty()
    if qty <= 0:
        print("[STARTUP-ENTRY] SKIP (qty too small / minNotional/minQty)")
        return

    entry_piv = get_last_pivot_info(swings)
    entry_piv_idx = entry_piv[2] if entry_piv else None

    side: Side = "BUY" if plan_dir == "long" else "SELL"
    entry_cid = f"{CLIENT_ID_PREFIX}-ENTRY0-{int(time.time())}"
    print(f"[STARTUP-ENTRY] tunnel={tunnel} -> MARKET {plan_dir.upper()} qty={qty} clientId={entry_cid}")
    resp = market_order(side, qty, reduce_only=False, client_id=entry_cid)
    print(f"[STARTUP-ENTRY] resp={resp}")

    st["base_qty"] = float(qty)
    st["has_entered_once"] = True
    st["last_entry_context"] = "trend"
    _reset_band_state_for_new_position(st, entry_piv_idx)
    save_state(st)

    time.sleep(0.25)
    pos2 = get_position_amt()

    stop_dir2: TrendLabel = "long" if pos2 > 0 else "short"
    stop2 = compute_stop_level_from_pivot(swings, stop_dir2)
    print(f"[STARTUP-ENTRY] stop_from_pivot={stop2}")
    sync_stop_only(st, pos2, stop2)

    st["prev_pos_amt"] = float(pos2)
    save_state(st)


def print_startup_status_once(st: Dict[str, Any]) -> None:
    if st.get("startup_printed", False):
        return

    try:
        df = fetch_klines_limit(SYMBOL, INTERVAL, LOOKBACK_LIMIT)
    except Exception as e:
        print("[STARTUP-STATUS] REST kline çekilemedi:", repr(e))
        st["startup_printed"] = True
        save_state(st)
        return

    if len(df) < 10:
        st["startup_printed"] = True
        save_state(st)
        return

    swings = _compute_swings(df)
    tunnel = detect_tunnel(swings)

    plan_dir = entry_dir_from_tunnel(tunnel)

    last_bar_time = df.index[-1].isoformat()
    close_val = float(df["Close"].iloc[-1])

    last_pivot = get_last_pivot_label(swings)
    last_pivot_txt = f"{last_pivot[0]}@{last_pivot[1]:.8f}" if last_pivot else "N/A"

    pos = get_position_amt()
    pos_lbl = _pos_label(pos)

    try:
        sync_state_with_exchange_stop(st, pos)
    except Exception as e:
        print("[WARN] startup stop sync error:", repr(e))

    if tunnel == "DUSEN":
        tunnel_txt = "DÜŞEN (2 tepe düşüyor)"
    elif tunnel == "YUKSELEN":
        tunnel_txt = "YÜKSELEN (2 dip yükseliyor)"
    elif tunnel == "YOK":
        tunnel_txt = "YOK (yapısal sinyal yok)"
    else:
        tunnel_txt = "BELİRSİZ"

    if plan_dir == "long":
        plan_txt = "İlk giriş: LONG (trend yükselen, trene atla)"
        hypo = "LONG"
    elif plan_dir == "short":
        plan_txt = "İlk giriş: SHORT (trend düşen, trene atla)"
        hypo = "SHORT"
    else:
        plan_txt = "İlk giriş: BEKLE (tünel belirsiz/yok) [SADECE entry]"
        hypo = "YOK"

    print("[STARTUP-STATUS]")
    print(f"  Son bar: {last_bar_time} close={close_val:.10f}")
    print(f"  Tünel: {tunnel_txt}")
    print(f"  Plan:  {plan_txt}")
    print(f"  Hipotetik ilk giriş: {hypo}")
    print(f"  Pozisyon (teknik): {pos_lbl}")
    print(f"  Son pivot: {last_pivot_txt}")
    print("  İlk trade kuralı: 2 yükselen dip => LONG, 2 düşen tepe => SHORT (aksi halde bekle)")
    print(f"  Stop buffer: {STOP_BUFFER_PCT*100:.2f}% (pivot dışına)\n")

    st["startup_printed"] = True
    save_state(st)


def decide_and_execute(df_closed: pd.DataFrame, st: Dict[str, Any]) -> None:
    # --- her döngü başında pozisyonu tek sefer oku ---
    pos = get_position_amt()
    is_flat = abs(pos) < 1e-12
    pos_lbl = _pos_label(pos)

    prev_pos = float(st.get("prev_pos_amt", 0.0))
    was_in_pos = abs(prev_pos) >= 1e-12
    stop_or_exit_happened = (was_in_pos and is_flat)

    entry_signal: TrendLabel = "trend_yok"
    last_pivot_txt = "N/A"
    last_pivot_kind: Optional[str] = None
    last_pivot_idx: Optional[int] = None

    if len(df_closed) >= 10:
        swings, trades = compute_trades(df_closed)
        piv = get_last_pivot_info(swings)
        if piv:
            last_pivot_kind, last_pivot_price, last_pivot_idx = piv
            last_pivot_txt = f"{last_pivot_kind}@{last_pivot_price:.8f}"
        entry_signal = structural_entry_signal(swings)
    else:
        swings, trades = [], []

    print("[STATE]")
    print(f"  entry_signal={entry_signal}")
    print(f"  position={pos_lbl}")
    print(f"  last_pivot={last_pivot_txt}")

    if len(df_closed) < 10:
        st["prev_pos_amt"] = float(pos)
        save_state(st)
        return

    # =========================
    # A) STOP/EXIT SONRASI  ✅ FIX: aynı barda tekrar ENTRY yapma
    # =========================
    if stop_or_exit_happened:
        # Trend modunda stop olduysak => belirsiz moda geç
        if str(st.get("last_entry_context") or "") == "trend":
            st["uncertain_active"] = True

            # ✅ önemli: belirsiz moda geçince pivot "değişimini" beklemek yerine
            # ilk belirsiz barda karar verebilmek için last pivot idx'yi None yapıyoruz.
            st["uncertain_last_pivot_idx"] = None
            st["uncertain_last_pivot_kind"] = None
            st["uncertain_last_pivot_price"] = None
            save_state(st)

        # Stop oldu / flat oldu => emirleri temizle
        cancel_our_orders()
        st["active_stop_client_id"] = None
        st["last_stop_price"] = None
        st["stop_side"] = None

        # pozisyon kapandıysa per-band state sıfırla
        st["band_state"] = _empty_band_map()
        st["prev_pos_amt"] = float(pos)
        save_state(st)

        # ✅ kritik: stop'un gerçekleştiği bu bar içinde HİÇBİR şekilde yeniden entry yok.
        return

    # =========================
    # B) BELİRSİZ MOD
    # =========================
    if bool(st.get("uncertain_active", False)):
        # Yapısal sinyal netleşirse (long/short) belirsiz moddan çık
        if entry_signal in ("long", "short"):
            st["uncertain_active"] = False
            st["uncertain_last_pivot_idx"] = None
            st["uncertain_last_pivot_kind"] = None
            st["uncertain_last_pivot_price"] = None
            st["prev_pos_amt"] = float(pos)
            save_state(st)
            return

        piv = get_last_pivot_info(swings)
        if piv is None:
            st["prev_pos_amt"] = float(pos)
            save_state(st)
            return

        piv_kind, piv_price, piv_idx = piv
        last_idx = st.get("uncertain_last_pivot_idx")

        # ✅ stop sonrası last_idx=None olduğu için, ilk belirsiz barda pivot değişimi beklemeden karar verecek.
        if last_idx is None or int(last_idx) != int(piv_idx):
            desired: TrendLabel = "short" if piv_kind == "TOP" else "long"

            st["uncertain_last_pivot_idx"] = int(piv_idx)
            st["uncertain_last_pivot_kind"] = str(piv_kind)
            st["uncertain_last_pivot_price"] = float(piv_price)
            save_state(st)

            if is_flat:
                qty = calc_market_qty()
                if qty <= 0:
                    st["prev_pos_amt"] = float(pos)
                    save_state(st)
                    return

                side: Side = "BUY" if desired == "long" else "SELL"
                entry_cid = f"{CLIENT_ID_PREFIX}-ENTRY-{int(time.time())}"
                resp = market_order(side, qty, reduce_only=False, client_id=entry_cid)
                print(f"[ENTRY] ENTER {desired.upper()} resp={resp}")

                st["base_qty"] = float(qty)
                st["has_entered_once"] = True
                st["last_entry_context"] = "uncertain"
                _reset_band_state_for_new_position(st, piv_idx)
                save_state(st)

                time.sleep(0.25)
                pos2 = get_position_amt()

                stop_dir2: TrendLabel = "long" if pos2 > 0 else "short"
                stop2 = compute_stop_level_from_uncertain_signal(st, stop_dir2)
                sync_stop_only(st, pos2, stop2)

                st["prev_pos_amt"] = float(pos2)
                save_state(st)
                return

            current_dir: TrendLabel = "long" if pos > 0 else "short"
            if current_dir != desired:
                base_qty = float(st.get("base_qty", 0.0))
                qty = base_qty if base_qty > 0 else abs(pos)

                close_side: Side = "SELL" if pos > 0 else "BUY"
                cid_close = f"{CLIENT_ID_PREFIX}-FLIP-{int(time.time())}"
                market_order(close_side, float(qty), reduce_only=True, client_id=cid_close)

                time.sleep(0.15)

                open_side: Side = "BUY" if desired == "long" else "SELL"
                cid_open = f"{CLIENT_ID_PREFIX}-ENTRY-{int(time.time())}"
                resp = market_order(open_side, float(qty), reduce_only=False, client_id=cid_open)
                print(f"[ENTRY] ENTER {desired.upper()} resp={resp}")

                st["last_entry_context"] = "uncertain"
                _reset_band_state_for_new_position(st, piv_idx)
                save_state(st)

                time.sleep(0.25)
                pos3 = get_position_amt()

                stop_dir3: TrendLabel = "long" if pos3 > 0 else "short"
                stop3 = compute_stop_level_from_uncertain_signal(st, stop_dir3)
                sync_stop_only(st, pos3, stop3)

                st["prev_pos_amt"] = float(pos3)
                save_state(st)
                return

        if not is_flat:
            stop_dir_u: TrendLabel = "long" if pos > 0 else "short"
            stop_u = compute_stop_level_from_uncertain_signal(st, stop_dir_u)
            sync_stop_only(st, pos, stop_u)

        st["prev_pos_amt"] = float(pos)
        save_state(st)
        return

    # =========================
    # C) NORMAL MOD
    # =========================

    desired_first: Optional[TrendLabel] = None
    first_entry_wait = False

    if not st.get("has_entered_once", False):
        tunnel = detect_tunnel(swings)
        desired_first = entry_dir_from_tunnel(tunnel)
        if desired_first is None:
            first_entry_wait = True

    # 1) ENTRY
    if is_flat:
        if (not st.get("has_entered_once", False)) and first_entry_wait:
            st["prev_pos_amt"] = float(pos)
            save_state(st)
            return

        if entry_signal not in ("long", "short"):
            st["prev_pos_amt"] = float(pos)
            save_state(st)
            return

        if (not st.get("has_entered_once", False)) and desired_first in ("long", "short"):
            if entry_signal != desired_first:
                st["prev_pos_amt"] = float(pos)
                save_state(st)
                return

        qty = calc_market_qty()
        if qty <= 0:
            print("  -> ENTRY SKIP (qty too small / minNotional/minQty)")
            st["prev_pos_amt"] = float(pos)
            save_state(st)
            return

        entry_piv = get_last_pivot_info(swings)
        entry_piv_idx = entry_piv[2] if entry_piv else None

        side: Side = "BUY" if entry_signal == "long" else "SELL"
        entry_cid = f"{CLIENT_ID_PREFIX}-ENTRY-{int(time.time())}"
        resp = market_order(side, qty, reduce_only=False, client_id=entry_cid)
        print(f"[ENTRY] ENTER {entry_signal.upper()} resp={resp}")

        st["base_qty"] = float(qty)
        st["has_entered_once"] = True
        st["last_entry_context"] = "trend"
        _reset_band_state_for_new_position(st, entry_piv_idx)
        save_state(st)

        time.sleep(0.25)
        pos2 = get_position_amt()

        stop_dir2: TrendLabel = "long" if pos2 > 0 else "short"
        stop2 = compute_stop_level_from_pivot(swings, stop_dir2)
        sync_stop_only(st, pos2, stop2)

        st["prev_pos_amt"] = float(pos2)
        save_state(st)
        return

    # =========================
    # 2) TP50 + READD (PER BAND)
    # =========================
    pos_dir: TrendLabel = "long" if pos > 0 else "short"
    band_state = st.get("band_state") or _empty_band_map()
    tp50_done: Dict[str, Any] = band_state.get("tp50_done", {})
    readd_done: Dict[str, Any] = band_state.get("readd_done", {})
    tp50_pivot_idx_map: Dict[str, Any] = band_state.get("tp50_pivot_idx", {})
    entry_pivot_idx = band_state.get("entry_pivot_idx", None)

    last_trade = trades[-1] if trades else None
    events = _extract_band_events_from_last_trade(last_trade) if last_trade is not None else []

    did_any_action = False

    # ---- TP50 ----
    if last_trade is not None and events:
        for kind, band_id in events:
            if kind != "tp50":
                continue

            if band_id in BANNED_BANDS:
                continue

            bk = _band_key(band_id)
            if tp50_done.get(bk):
                continue

            if not _eligible_pivot_for_tp50(pos_dir, last_pivot_kind):
                continue
            if entry_pivot_idx is not None and last_pivot_idx is not None:
                try:
                    if int(last_pivot_idx) <= int(entry_pivot_idx):
                        continue
                except Exception:
                    pass

            base_qty = float(st.get("base_qty", 0.0))
            qty50 = base_qty * 0.5 if base_qty > 0 else abs(pos) * 0.5

            filters = get_symbol_filters()
            step = float(filters.get("stepSize", 0.0))
            if step > 0:
                qty50 = round_step(qty50, step)

            if qty50 <= 0:
                continue

            tp_side: Side = "SELL" if pos > 0 else "BUY"
            cid = f"{CLIENT_ID_PREFIX}-TP50-B{band_id}-{int(time.time())}"
            print(f"[TP50] band={band_id} pivot_ok={last_pivot_kind} -> MARKET reduceOnly qty={qty50} side={tp_side} clientId={cid}")

            try:
                resp = market_order(tp_side, float(qty50), reduce_only=True, client_id=cid)
                print(f"[TP50] resp={resp}")

                tp50_done[bk] = True
                tp50_pivot_idx_map[bk] = int(last_pivot_idx) if last_pivot_idx is not None else None
                band_state["tp50_done"] = tp50_done
                band_state["tp50_pivot_idx"] = tp50_pivot_idx_map
                st["band_state"] = band_state
                save_state(st)

                did_any_action = True
            except Exception as e:
                print("[TP50] market error:", repr(e))

    # ---- READD ----
    if abs(pos) >= 1e-12:
        # ✅ MINIMAL PATCH:
        # READD artık "opposite_signal" beklemiyor.
        # Sadece generate_trades'in ürettiği "readd band=k" event'lerini uygular.
        readd_bands_from_trade = [band_id for (k, band_id) in events if k == "readd" and band_id not in BANNED_BANDS]

        if readd_bands_from_trade:
            for band_id in readd_bands_from_trade:
                bk = _band_key(band_id)
                if not tp50_done.get(bk):
                    continue
                if readd_done.get(bk):
                    continue

                # Pivot türü: long => BOTTOM, short => TOP (generate_trades ile aynı)
                if not _eligible_pivot_for_readd(pos_dir, last_pivot_kind):
                    continue

                tp_piv_idx = tp50_pivot_idx_map.get(bk)
                if tp_piv_idx is not None and last_pivot_idx is not None:
                    try:
                        if int(last_pivot_idx) <= int(tp_piv_idx):
                            continue
                    except Exception:
                        pass

                base_qty = float(st.get("base_qty", 0.0))
                add_qty = base_qty * 0.5 if base_qty > 0 else abs(pos) * 0.5

                filters = get_symbol_filters()
                step = float(filters.get("stepSize", 0.0))
                if step > 0:
                    add_qty = round_step(add_qty, step)

                if add_qty <= 0:
                    continue

                add_side: Side = "BUY" if pos > 0 else "SELL"
                cid = f"{CLIENT_ID_PREFIX}-READD-B{band_id}-{int(time.time())}"
                print(f"[READD] band={band_id} pivot_ok={last_pivot_kind} -> MARKET qty={add_qty} side={add_side} clientId={cid}")

                try:
                    resp = market_order(add_side, float(add_qty), reduce_only=False, client_id=cid)
                    print(f"[READD] resp={resp}")

                    readd_done[bk] = True
                    band_state["readd_done"] = readd_done
                    st["band_state"] = band_state
                    save_state(st)

                    did_any_action = True
                except Exception as e:
                    print("[READD] market error:", repr(e))

    # =========================
    # 3) STOP sync (aksiyon sonrası pozisyonu refresh et)
    # =========================
    if did_any_action:
        time.sleep(0.20)
        pos = get_position_amt()
        is_flat = abs(pos) < 1e-12

        if is_flat:
            cancel_our_orders()
            st["active_stop_client_id"] = None
            st["last_stop_price"] = None
            st["stop_side"] = None
            st["prev_pos_amt"] = float(pos)
            save_state(st)
            return

    if not is_flat:
        stop_dir3: TrendLabel = "long" if pos > 0 else "short"
        stop3 = compute_stop_level_from_pivot(swings, stop_dir3)
        sync_stop_only(st, pos, stop3)

    st["prev_pos_amt"] = float(pos)
    save_state(st)


# =========================
# WEBSOCKET HELPERS
# =========================

def _interval_to_stream(interval: str) -> str:
    return interval


def _kline_msg_to_row(msg: Dict[str, Any]) -> Dict[str, Any]:
    k = msg["k"]
    return {
        "open_time": int(k["t"]),
        "open": float(k["o"]),
        "high": float(k["h"]),
        "low": float(k["l"]),
        "close": float(k["c"]),
        "volume": float(k["v"]),
        "is_closed": bool(k["x"]),
    }


def _df_from_buf(buf: Deque[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(list(buf))
    df["Date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("Date").sort_index()
    df = df[["open", "high", "low", "close", "volume"]]
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    return df


# =========================
# WEBSOCKET RUNNER (Queue + Worker)
# =========================

def run_ws_loop():
    st = load_state()

    # 1) Startup REST + status
    df_start: Optional[pd.DataFrame] = None
    swings_start: Optional[List[Any]] = None
    try:
        df_start = fetch_klines_limit(SYMBOL, INTERVAL, LOOKBACK_LIMIT)
        if len(df_start) >= 10:
            swings_start = _compute_swings(df_start)
    except Exception as e:
        print("[WARN] startup REST fetch error:", repr(e))
        df_start = None
        swings_start = None

    print_startup_status_once(st)

    # STARTUP: pozisyon + stop sync
    try:
        pos = get_position_amt()
        sync_state_with_exchange_stop(st, pos)
    except Exception as e:
        print("[WARN] initial stop sync error:", repr(e))

    # STARTUP: tünel long/short ise ANINDA market entry + stop (WS beklemeden)
    if df_start is not None and swings_start is not None:
        try:
            startup_try_jump_on_train(st, df_start, swings_start)
        except Exception as e:
            print("[WARN] startup jump-on-train error:", repr(e))

    buf_lock = threading.Lock()

    # WS buffer'ı REST ile seed et (boş başlama)
    if df_start is not None and len(df_start) > 0:
        buf: Deque[Dict[str, Any]] = _buf_seed_from_df(df_start)
    else:
        buf = deque(maxlen=LOOKBACK_LIMIT)

    work_q: "queue.Queue[Tuple[str, pd.DataFrame]]" = queue.Queue(maxsize=10)

    def worker():
        nonlocal st
        while True:
            closed_bar_time, df_closed = work_q.get()
            try:
                st["last_closed_bar_time"] = closed_bar_time
                save_state(st)
                decide_and_execute(df_closed, st)
            except Exception as e:
                print("[WORKER] error:", repr(e))
            finally:
                work_q.task_done()

    threading.Thread(target=worker, daemon=True).start()

    stream = f"{SYMBOL.lower()}@kline_{_interval_to_stream(INTERVAL)}"
    url = f"{WS_BASE_URL}/{stream}"

    print(f"[WS] connecting: {url}")

    def on_open(ws: WebSocketApp):
        print("[WS] opened")

    def on_error(ws: WebSocketApp, error):
        print("[WS] error:", repr(error))

    def on_close(ws: WebSocketApp, code, reason):
        print("[WS] closed:", code, reason)

    def on_message(ws: WebSocketApp, message: str):
        nonlocal st
        try:
            msg = json.loads(message)
        except Exception:
            return

        if "k" not in msg:
            return

        row = _kline_msg_to_row(msg)

        with buf_lock:
            if buf and buf[-1]["open_time"] == row["open_time"]:
                buf[-1] = row
            else:
                buf.append(row)

            if not row["is_closed"]:
                return

            df = _df_from_buf(buf)
            closed_bar_time = df.index[-1].isoformat()

            if st.get("last_closed_bar_time") == closed_bar_time:
                return

            close_val = float(df["Close"].iloc[-1])

        print(f"\n[NEW CLOSED BAR][WS] {closed_bar_time} close={close_val:.10f}")

        try:
            work_q.put_nowait((closed_bar_time, df.copy()))
        except queue.Full:
            print("[WS] work queue FULL -> dropping bar event (worker yetişemiyor)")

    while True:
        ws = WebSocketApp(
            url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        try:
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            print("[WS] run_forever exception:", repr(e))

        print("[WS] reconnecting in 3s...")
        time.sleep(3)


# =========================
# MAIN
# =========================

def main():
    _safety_checks_or_die()
    _ensure_leverage()
    run_ws_loop()


if __name__ == "__main__":
    main()
