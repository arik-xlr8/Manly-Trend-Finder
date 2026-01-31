# plot_swings_live.py
import os
import time
import threading
import queue
import requests
import sys

import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal, Tuple, Set

from dotenv import load_dotenv
load_dotenv()

from swings import find_swings, SwingPoint


# ======================
# 0) BEEP HELPERS
# ======================

def play_beep(kind: str):
    """
    kind: "LH" or "LB"
    Windows: winsound.Beep
    Others: terminal bell
    """
    try:
        if sys.platform.startswith("win"):
            import winsound
            if kind == "LH":
                winsound.Beep(1100, 180)
                winsound.Beep(1100, 180)
            else:  # "LB"
                winsound.Beep(500, 220)
        else:
            print("\a", end="", flush=True)
    except Exception:
        pass


# ======================
# 1) VERİ ÇEKME (REST)
# ======================

BASE_URL = os.getenv("BINANCE_FUTURES_BASE_URL", "https://fapi.binance.com").rstrip("/")
WS_BASE_URL = os.getenv("BINANCE_FUTURES_WS_BASE_URL", "wss://fstream.binance.com/ws").rstrip("/")


def fetch_ohlc_from_api(symbol: str = "BTCUSDT", interval: str = "15m", limit: int = 500) -> pd.DataFrame:
    url = f"{BASE_URL}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    cols = [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "number_of_trades", "taker_buy_base_volume",
        "taker_buy_quote_volume", "ignore",
    ]

    df = pd.DataFrame(data, columns=cols)
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    df[["open_time", "close_time"]] = df[["open_time", "close_time"]].astype(np.int64)

    # UTC index + sort
    df["Date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("Date", inplace=True)
    df = df.sort_index()

    # ✅ Sadece kapanmış mumlar (son mum kapanmadıysa at)
    now_ms = int(time.time() * 1000)
    if len(df) > 0:
        last_close_time = int(df["close_time"].iloc[-1])
        if now_ms < last_close_time:
            df = df.iloc[:-1]

    df = df[["open_time", "close_time", "open", "high", "low", "close", "volume"]]
    df.columns = ["OpenTime", "CloseTime", "Open", "High", "Low", "Close", "Volume"]
    return df


def _kline_to_row(k: Dict[str, Any]) -> Dict[str, Any]:
    """
    Binance WS kline payload'ından kapanmış mum satırına çevirir.
    """
    return {
        "OpenTime": int(k["t"]),
        "CloseTime": int(k["T"]),
        "Open": float(k["o"]),
        "High": float(k["h"]),
        "Low": float(k["l"]),
        "Close": float(k["c"]),
        "Volume": float(k["v"]),
    }


def upsert_closed_candle(df: pd.DataFrame, row: Dict[str, Any], max_len: int) -> pd.DataFrame:
    """
    - Kapanmış mumu DF'e ekler (aynı OpenTime ise overwrite)
    - Index: Date(open_time)
    - max_len üstünü kırpar
    """
    dt = pd.to_datetime(row["OpenTime"], unit="ms", utc=True)

    if dt in df.index:
        for c in ["OpenTime", "CloseTime", "Open", "High", "Low", "Close", "Volume"]:
            df.at[dt, c] = row[c]
    else:
        new = pd.DataFrame([row], index=[dt])
        df = pd.concat([df, new], axis=0)

    df = df.sort_index()

    if len(df) > max_len:
        df = df.iloc[-max_len:]

    return df


# ======================
# 2) FILLS + TRADE
# ======================

@dataclass
class FillEvent:
    index: int
    qty_delta: float
    price: float
    reason: str  # "entry", "tp50 band=<k>", "readd band=<k>", "exit", "stop"


def _safe_rnd(x: Optional[float], nd: int = 12) -> float:
    if x is None:
        return 0.0
    try:
        return float(np.round(float(x), nd))
    except Exception:
        return 0.0


def _make_trade_key(
    direction: str,
    entry_index: int,
    entry_price: float,
    fib0_price: Optional[float],
    fib1_price: Optional[float],
    fib0_index: Optional[int],
    fib1_index: Optional[int],
    fib_start_index: Optional[int],
) -> str:
    return (
        f"{direction}|"
        f"ei={int(entry_index)}|"
        f"ep={_safe_rnd(entry_price)}|"
        f"f0={_safe_rnd(fib0_price)}|"
        f"f1={_safe_rnd(fib1_price)}|"
        f"f0i={int(fib0_index) if fib0_index is not None else -1}|"
        f"f1i={int(fib1_index) if fib1_index is not None else -1}|"
        f"fsi={int(fib_start_index) if fib_start_index is not None else -1}"
    )


@dataclass
class Trade:
    direction: Literal["long", "short"]
    entry_index: int
    entry_price: float
    stop_level: float
    exit_index: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None

    fib_levels: Optional[List[Tuple[str, float]]] = None
    fib_start_index: Optional[int] = None

    fib0_price: Optional[float] = None
    fib1_price: Optional[float] = None
    fib0_index: Optional[int] = None
    fib1_index: Optional[int] = None

    base_k: Optional[int] = None
    band_actions: Dict[int, Set[str]] = field(default_factory=dict)
    fills: List[FillEvent] = field(default_factory=list)

    trade_key: str = ""

    def refresh_trade_key(self) -> None:
        self.trade_key = _make_trade_key(
            direction=self.direction,
            entry_index=self.entry_index,
            entry_price=self.entry_price,
            fib0_price=self.fib0_price,
            fib1_price=self.fib1_price,
            fib0_index=self.fib0_index,
            fib1_index=self.fib1_index,
            fib_start_index=self.fib_start_index,
        )


# ======================
# 3) FIB ORANLARI
# ======================

UPTREND_FIB_RATIOS = [0.0, 0.5, 1.0, 1.5, 2.5, 3.6, 4.5]
DOWNTREND_FIB_RATIOS = [0.0, 0.5, 1.0, 1.5, 2.5, 3.6, 4.5]


def compute_trend_based_fib_levels(fib0: float, impulse_range: float, ratios: List[float]) -> List[Tuple[str, float]]:
    if impulse_range == 0:
        return []
    return [(f"{r:g}", float(fib0 + impulse_range * r)) for r in ratios]


# ======================
# 4) FIB BAND HELPERS
# ======================

def _fib_u(price: float, fib0: float, fib1: float) -> Optional[float]:
    den = (fib1 - fib0)
    if den == 0:
        return None
    return (price - fib0) / den


BANNED_BANDS = {0, 1}


# ======================
# 5) STRATEJİ (SENİN KOD)
# ======================

def generate_trades(
    df: pd.DataFrame,
    swings: List[SwingPoint],
    stop_buffer_pct: float = 0.0,
    right_bars: int = 1,
    max_chase_pct: float = 0.03,
) -> List[Trade]:
    closes = df["Close"].values
    highs_arr = df["High"].values
    lows_arr = df["Low"].values
    n = len(df)

    trades: List[Trade] = []
    current_trade: Optional[Trade] = None

    last_lb: Optional[SwingPoint] = None
    last_lh: Optional[SwingPoint] = None

    mode: Literal["normal", "uncertain"] = "normal"
    last_checked_bar = -1

    long_last_high: Optional[SwingPoint] = None
    long_candidate_low: Optional[SwingPoint] = None
    short_last_low: Optional[SwingPoint] = None
    short_candidate_high: Optional[SwingPoint] = None

    last_signal: Optional[SwingPoint] = None

    def reset_trailing_state():
        nonlocal long_last_high, long_candidate_low, short_last_low, short_candidate_high
        long_last_high = None
        long_candidate_low = None
        short_last_low = None
        short_candidate_high = None

    def stop_hit_on_bar(tr: Trade, i: int) -> bool:
        return (lows_arr[i] <= tr.stop_level) if tr.direction == "long" else (highs_arr[i] >= tr.stop_level)

    def stop_exit_price(tr: Trade) -> float:
        return float(tr.stop_level)

    def pivot_stop_level(pivot_idx: int, kind: str) -> float:
        base = float(df["Low"].iloc[pivot_idx]) if kind == "LB" else float(df["High"].iloc[pivot_idx])
        if stop_buffer_pct and stop_buffer_pct != 0.0:
            if kind == "LB":
                return base * (1.0 - abs(stop_buffer_pct))
            else:
                return base * (1.0 + abs(stop_buffer_pct))
        return base

    def chase_too_much(direction: str, pivot_idx: int, signal_idx: int) -> bool:
        if pivot_idx < 0 or pivot_idx >= n or signal_idx < 0 or signal_idx >= n:
            return False
        pivot_close = float(closes[pivot_idx])
        entry_close = float(closes[signal_idx])
        if pivot_close <= 0:
            return False
        if direction == "long":
            return entry_close >= pivot_close * (1.0 + max_chase_pct)
        else:
            return entry_close <= pivot_close * (1.0 - max_chase_pct)

    def far_from_last_signal(pivot_idx: int, now_idx: int) -> bool:
        if pivot_idx < 0 or pivot_idx >= n or now_idx < 0 or now_idx >= n:
            return True
        pivot_close = float(closes[pivot_idx])
        now_close = float(closes[now_idx])
        if pivot_close <= 0:
            return True
        diff = abs(now_close - pivot_close) / pivot_close
        return diff > max_chase_pct

    def add_entry_fill(tr: Trade):
        tr.fills.append(FillEvent(
            index=tr.entry_index,
            qty_delta=(+1.0 if tr.direction == "long" else -1.0),
            price=float(closes[tr.entry_index]),
            reason="entry"
        ))

    def _current_pos(tr: Trade) -> float:
        return float(sum(f.qty_delta for f in tr.fills))

    def add_flatten_fill(tr: Trade, idx: int, reason: str):
        pos = _current_pos(tr)
        if abs(pos) < 1e-12:
            return
        tr.fills.append(FillEvent(
            index=idx,
            qty_delta=-pos,
            price=float(closes[idx]) if 0 <= idx < n else float(tr.entry_price),
            reason=reason
        ))

    def init_band(tr: Trade, fib0_price: float, impulse_range: float, fib0_index: int, fib1_index: int):
        if impulse_range == 0:
            return
        tr.fib0_price = float(fib0_price)
        tr.fib1_price = float(fib0_price + impulse_range)
        tr.fib0_index = int(fib0_index)
        tr.fib1_index = int(fib1_index)

        u = _fib_u(tr.entry_price, tr.fib0_price, tr.fib1_price)
        if u is None:
            return
        tr.base_k = int(np.floor(u))
        tr.band_actions = {}
        tr.refresh_trade_key()

    def handle_band_actions(sp: SwingPoint, signal_idx: int):
        nonlocal current_trade
        tr = current_trade
        if tr is None:
            return
        if tr.fib0_price is None or tr.fib1_price is None:
            return
        if signal_idx <= tr.entry_index:
            return

        pivot_u = _fib_u(float(sp.price), tr.fib0_price, tr.fib1_price)
        if pivot_u is None:
            return
        pivot_k = int(np.floor(pivot_u))

        if pivot_k in BANNED_BANDS:
            return

        if pivot_k not in tr.band_actions:
            tr.band_actions[pivot_k] = set()

        actions = tr.band_actions[pivot_k]
        px = float(closes[signal_idx])

        if tr.direction == "long":
            if sp.kind == "LH" and ("sold" not in actions):
                tr.fills.append(FillEvent(index=signal_idx, qty_delta=-0.5, price=px, reason=f"tp50 band={pivot_k}"))
                actions.add("sold")
                return
            if sp.kind == "LB" and ("sold" in actions) and ("readd" not in actions):
                tr.fills.append(FillEvent(index=signal_idx, qty_delta=+0.5, price=px, reason=f"readd band={pivot_k}"))
                actions.add("readd")
                return
        else:
            if sp.kind == "LB" and ("sold" not in actions):
                tr.fills.append(FillEvent(index=signal_idx, qty_delta=+0.5, price=px, reason=f"tp50 band={pivot_k}"))
                actions.add("sold")
                return
            if sp.kind == "LH" and ("sold" in actions) and ("readd" not in actions):
                tr.fills.append(FillEvent(index=signal_idx, qty_delta=-0.5, price=px, reason=f"readd band={pivot_k}"))
                actions.add("readd")
                return

    def _new_trade(direction: Literal["long", "short"], entry_idx: int, stop_level: float,
                   fib_levels=None, fib_start_index=None) -> Trade:
        tr = Trade(
            direction=direction,
            entry_index=int(entry_idx),
            entry_price=float(closes[entry_idx]),
            stop_level=float(stop_level),
            fib_levels=fib_levels,
            fib_start_index=fib_start_index
        )
        add_entry_fill(tr)
        tr.refresh_trade_key()
        return tr

    def open_uncertain_trade_by_signal(signal: SwingPoint, entry_idx: int):
        nonlocal current_trade
        if entry_idx < 0 or entry_idx >= n:
            return

        if signal.kind == "LH":
            direction: Literal["long", "short"] = "short"
            stop_level = pivot_stop_level(signal.index, "LH")
        else:
            direction = "long"
            stop_level = pivot_stop_level(signal.index, "LB")

        current_trade = _new_trade(direction, entry_idx, stop_level, fib_levels=None, fib_start_index=None)
        reset_trailing_state()

    def try_uncertain_entry_after_stop(now_idx: int):
        if last_signal is None:
            return
        if far_from_last_signal(last_signal.index, now_idx):
            return
        open_uncertain_trade_by_signal(last_signal, now_idx)

    for sp in swings:
        pivot_idx = sp.index
        signal_idx = pivot_idx + right_bars
        if signal_idx >= n:
            continue

        prev_lb = last_lb
        prev_lh = last_lh

        # A) STOP
        if current_trade is not None:
            start_i = max(current_trade.entry_index, last_checked_bar + 1)
            end_i = signal_idx
            for i in range(start_i, end_i + 1):
                if stop_hit_on_bar(current_trade, i):
                    current_trade.exit_index = i
                    current_trade.exit_price = stop_exit_price(current_trade)
                    current_trade.exit_reason = "stop_hit"
                    add_flatten_fill(current_trade, i, reason="stop")

                    trades.append(current_trade)
                    current_trade = None
                    reset_trailing_state()

                    mode = "uncertain"
                    try_uncertain_entry_after_stop(now_idx=i)

                    last_checked_bar = i
                    break

        last_checked_bar = max(last_checked_bar, signal_idx)

        # B) UNCERTAIN
        if mode == "uncertain":
            uptrend_confirm = (sp.kind == "LB" and prev_lb is not None and prev_lb.price < sp.price)
            downtrend_confirm = (sp.kind == "LH" and prev_lh is not None and prev_lh.price > sp.price)

            if uptrend_confirm or downtrend_confirm:
                mode = "normal"
            else:
                desired: Literal["long", "short"] = "long" if sp.kind == "LB" else "short"

                if not chase_too_much(desired, pivot_idx, signal_idx):
                    if current_trade is not None and current_trade.direction != desired:
                        exit_idx = signal_idx
                        if exit_idx > current_trade.entry_index:
                            current_trade.exit_index = exit_idx
                            current_trade.exit_price = float(closes[exit_idx])
                            current_trade.exit_reason = "uncertain_flip"
                            add_flatten_fill(current_trade, exit_idx, reason="exit")
                            trades.append(current_trade)
                        current_trade = None
                        reset_trailing_state()

                    if current_trade is None:
                        stop_level = pivot_stop_level(pivot_idx, "LB" if desired == "long" else "LH")
                        current_trade = _new_trade(desired, signal_idx, stop_level, fib_levels=None, fib_start_index=None)
                        reset_trailing_state()

                last_signal = sp
                if sp.kind == "LB":
                    last_lb = sp
                else:
                    last_lh = sp
                continue

        # C) NORMAL
        if current_trade is not None:
            handle_band_actions(sp, signal_idx)

        # trailing stop
        if current_trade is not None:
            if current_trade.direction == "long":
                if sp.kind == "LH":
                    if long_last_high is None:
                        long_last_high = sp
                        long_candidate_low = None
                    else:
                        if sp.price > long_last_high.price and long_candidate_low is not None:
                            if long_candidate_low.index > long_last_high.index:
                                new_stop = float(df["Low"].iloc[long_candidate_low.index])
                                if stop_buffer_pct and stop_buffer_pct != 0.0:
                                    new_stop = new_stop * (1.0 - abs(stop_buffer_pct))
                                if new_stop > current_trade.stop_level:
                                    current_trade.stop_level = new_stop
                                long_last_high = sp
                                long_candidate_low = None
                elif sp.kind == "LB":
                    if long_last_high is not None and sp.index > long_last_high.index:
                        long_candidate_low = sp
            else:
                if sp.kind == "LB":
                    if short_last_low is None:
                        short_last_low = sp
                        short_candidate_high = None
                    else:
                        if sp.price < short_last_low.price and short_candidate_high is not None:
                            if short_candidate_high.index > short_last_low.index:
                                new_stop = float(df["High"].iloc[short_candidate_high.index])
                                if stop_buffer_pct and stop_buffer_pct != 0.0:
                                    new_stop = new_stop * (1.0 + abs(stop_buffer_pct))
                                if new_stop < current_trade.stop_level:
                                    current_trade.stop_level = new_stop
                                short_last_low = sp
                                short_candidate_high = None
                elif sp.kind == "LH":
                    if short_last_low is not None and sp.index > short_last_low.index:
                        short_candidate_high = sp

        # flip exit
        if current_trade is not None:
            if current_trade.direction == "long":
                if sp.kind == "LH" and prev_lh is not None and prev_lh.price > sp.price:
                    exit_idx = signal_idx
                    if exit_idx > current_trade.entry_index:
                        current_trade.exit_index = exit_idx
                        current_trade.exit_price = float(closes[exit_idx])
                        current_trade.exit_reason = "long_exit_downtrend_LH"
                        add_flatten_fill(current_trade, exit_idx, reason="exit")
                        trades.append(current_trade)

                    current_trade = None
                    reset_trailing_state()

                    if not chase_too_much("short", pivot_idx, signal_idx):
                        entry_price = float(closes[signal_idx])

                        lh1 = prev_lh
                        lb_between = prev_lb
                        lh2 = sp

                        fib_levels = None
                        fib_start_index = None
                        impulse = 0.0

                        if lh1 is not None and lb_between is not None:
                            impulse = float(lb_between.price) - float(lh1.price)
                            fib0 = float(lh2.price)
                            fib_levels = compute_trend_based_fib_levels(fib0, impulse, DOWNTREND_FIB_RATIOS)
                            fib_start_index = pivot_idx

                        stop_level = pivot_stop_level(pivot_idx, "LH")
                        current_trade = Trade(
                            direction="short",
                            entry_index=signal_idx,
                            entry_price=entry_price,
                            stop_level=float(stop_level),
                            fib_levels=fib_levels,
                            fib_start_index=fib_start_index
                        )
                        add_entry_fill(current_trade)
                        current_trade.refresh_trade_key()

                        if lh1 is not None and lb_between is not None:
                            init_band(
                                current_trade,
                                fib0_price=float(lh2.price),
                                impulse_range=impulse,
                                fib0_index=pivot_idx,
                                fib1_index=lb_between.index,
                            )
                        reset_trailing_state()

            else:
                if sp.kind == "LB" and prev_lb is not None and prev_lb.price < sp.price:
                    exit_idx = signal_idx
                    if exit_idx > current_trade.entry_index:
                        current_trade.exit_index = exit_idx
                        current_trade.exit_price = float(closes[exit_idx])
                        current_trade.exit_reason = "short_exit_uptrend_LB"
                        add_flatten_fill(current_trade, exit_idx, reason="exit")
                        trades.append(current_trade)

                    current_trade = None
                    reset_trailing_state()

                    if not chase_too_much("long", pivot_idx, signal_idx):
                        entry_price = float(closes[signal_idx])

                        lb1 = prev_lb
                        lh_between = prev_lh
                        lb2 = sp

                        fib_levels = None
                        fib_start_index = None
                        impulse = 0.0

                        if lb1 is not None and lh_between is not None:
                            impulse = float(lh_between.price) - float(lb1.price)
                            fib0 = float(lb2.price)
                            fib_levels = compute_trend_based_fib_levels(fib0, impulse, UPTREND_FIB_RATIOS)
                            fib_start_index = pivot_idx

                        stop_level = pivot_stop_level(pivot_idx, "LB")
                        current_trade = Trade(
                            direction="long",
                            entry_index=signal_idx,
                            entry_price=entry_price,
                            stop_level=float(stop_level),
                            fib_levels=fib_levels,
                            fib_start_index=fib_start_index
                        )
                        add_entry_fill(current_trade)
                        current_trade.refresh_trade_key()

                        if lb1 is not None and lh_between is not None:
                            init_band(
                                current_trade,
                                fib0_price=float(lb2.price),
                                impulse_range=impulse,
                                fib0_index=pivot_idx,
                                fib1_index=lh_between.index,
                            )
                        reset_trailing_state()

        # entry
        if current_trade is None:
            if sp.kind == "LB" and prev_lb is not None and prev_lb.price < sp.price:
                if not chase_too_much("long", pivot_idx, signal_idx):
                    entry_price = float(closes[signal_idx])

                    lb1 = prev_lb
                    lh_between = prev_lh
                    lb2 = sp

                    fib_levels = None
                    fib_start_index = None
                    impulse = 0.0

                    if lb1 is not None and lh_between is not None:
                        impulse = float(lh_between.price) - float(lb1.price)
                        fib0 = float(lb2.price)
                        fib_levels = compute_trend_based_fib_levels(fib0, impulse, UPTREND_FIB_RATIOS)
                        fib_start_index = pivot_idx

                    stop_level = pivot_stop_level(pivot_idx, "LB")
                    current_trade = Trade(
                        direction="long",
                        entry_index=signal_idx,
                        entry_price=entry_price,
                        stop_level=float(stop_level),
                        fib_levels=fib_levels,
                        fib_start_index=fib_start_index
                    )
                    add_entry_fill(current_trade)
                    current_trade.refresh_trade_key()

                    if lb1 is not None and lh_between is not None:
                        init_band(
                            current_trade,
                            fib0_price=float(lb2.price),
                            impulse_range=impulse,
                            fib0_index=pivot_idx,
                            fib1_index=lh_between.index,
                        )
                    reset_trailing_state()

            elif sp.kind == "LH" and prev_lh is not None and prev_lh.price > sp.price:
                if not chase_too_much("short", pivot_idx, signal_idx):
                    entry_price = float(closes[signal_idx])

                    lh1 = prev_lh
                    lb_between = prev_lb
                    lh2 = sp

                    fib_levels = None
                    fib_start_index = None
                    impulse = 0.0

                    if lh1 is not None and lb_between is not None:
                        impulse = float(lb_between.price) - float(lh1.price)
                        fib0 = float(lh2.price)
                        fib_levels = compute_trend_based_fib_levels(fib0, impulse, DOWNTREND_FIB_RATIOS)
                        fib_start_index = pivot_idx

                    stop_level = pivot_stop_level(pivot_idx, "LH")
                    current_trade = Trade(
                        direction="short",
                        entry_index=signal_idx,
                        entry_price=entry_price,
                        stop_level=float(stop_level),
                        fib_levels=fib_levels,
                        fib_start_index=fib_start_index
                    )
                    add_entry_fill(current_trade)
                    current_trade.refresh_trade_key()

                    if lh1 is not None and lb_between is not None:
                        init_band(
                            current_trade,
                            fib0_price=float(lh2.price),
                            impulse_range=impulse,
                            fib0_index=pivot_idx,
                            fib1_index=lb_between.index,
                        )
                    reset_trailing_state()

        last_signal = sp
        if sp.kind == "LB":
            last_lb = sp
        else:
            last_lh = sp

    if current_trade is not None:
        trades.append(current_trade)

    return trades


# ======================
# 6) GRAFİK (CANLI REDRAW)
# ======================

def plot_with_trades(
    df: pd.DataFrame,
    swings: List[SwingPoint],
    trades: List[Trade],
    symbol: str,
    interval: str,
    fig: Optional[plt.Figure] = None,
):
    n = len(df)

    long_mask = np.zeros(n, dtype=bool)
    short_mask = np.zeros(n, dtype=bool)

    lh_series = np.full(n, np.nan)
    lb_series = np.full(n, np.nan)
    for sp in swings:
        if 0 <= sp.index < n:
            if sp.kind == "LH":
                lh_series[sp.index] = sp.price
            else:
                lb_series[sp.index] = sp.price

    for tr in trades:
        ei = tr.entry_index
        if ei < 0 or ei >= n:
            continue
        xi = (n - 1) if tr.exit_index is None else tr.exit_index
        if xi < 0 or xi >= n:
            continue
        if xi <= ei:
            continue

        if tr.direction == "long":
            long_mask[ei:xi + 1] = True
        else:
            short_mask[ei:xi + 1] = True

    lows = df["Low"].values
    highs = df["High"].values

    long_band_low = np.where(long_mask, lows, np.nan)
    long_band_high = np.where(long_mask, highs, np.nan)
    short_band_low = np.where(short_mask, lows, np.nan)
    short_band_high = np.where(short_mask, highs, np.nan)

    apds = []

    def has_data(arr: np.ndarray) -> bool:
        return np.isfinite(arr).any()

    if has_data(lh_series):
        apds.append(mpf.make_addplot(lh_series, type="scatter", markersize=50, marker="^", color="#ff9800"))
    if has_data(lb_series):
        apds.append(mpf.make_addplot(lb_series, type="scatter", markersize=50, marker="v", color="#03a9f4"))

    fb = []
    if long_mask.any():
        fb.append(dict(y1=long_band_low, y2=long_band_high, where=long_mask, alpha=0.15, color="#00e676"))
    if short_mask.any():
        fb.append(dict(y1=short_band_low, y2=short_band_high, where=short_mask, alpha=0.15, color="#ff5252"))

    alines_segments = []
    for tr in trades:
        if not tr.fib_levels or tr.fib_start_index is None:
            continue
        sidx = tr.fib_start_index
        if sidx < 0 or sidx >= n:
            continue

        x1 = df.index[sidx]
        x2 = df.index[n - 1] if tr.exit_index is None else df.index[tr.exit_index]
        for _, price in tr.fib_levels:
            alines_segments.append([(x1, price), (x2, price)])

    mc = mpf.make_marketcolors(up="#26a69a", down="#ef5350", edge="inherit", wick="inherit", volume="in")
    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        marketcolors=mc,
        gridstyle=":",
        facecolor="#0d1117",
        figcolor="#0b0f16",
        rc={"axes.labelcolor": "white", "xtick.color": "gray", "ytick.color": "gray"}
    )

    plot_kwargs = dict(
        type="candle",
        style=style,
        addplot=apds if apds else None,
        volume=True,
        figratio=(16, 9),
        figscale=1.2,
        title=f"{symbol} {interval} (LIVE)",
        ylabel="Fiyat",
        ylabel_lower="Hacim",
        tight_layout=True,
        datetime_format="%m-%d %H:%M",
    )

    if fb:
        plot_kwargs["fill_between"] = fb
    if alines_segments:
        plot_kwargs["alines"] = dict(alines=alines_segments, linewidths=0.7, alpha=0.45)

    # ✅ mplfinance fig reuse yok -> her seferinde yeni fig üret, eskisini kapat
    if fig is not None:
        try:
            plt.close(fig)
        except Exception:
            pass

    fig, _axes = mpf.plot(
        df[["Open", "High", "Low", "Close", "Volume"]],
        returnfig=True,
        **plot_kwargs
    )

    plt.pause(0.001)
    return fig


# ======================
# 7) WS DINLEYICI (KAPANAN MUM)
# ======================

def start_kline_ws(symbol: str, interval: str, out_queue: "queue.Queue[Dict[str, Any]]", stop_evt: threading.Event):
    """
    WebSocket'ten kline alır.
    Sadece kapanan mumu out_queue'ya atar.
    """
    import json
    from websocket import WebSocketApp

    stream = f"{symbol.lower()}@kline_{interval}"
    ws_url = f"{WS_BASE_URL}/{stream}"

    def on_message(ws, message: str):
        try:
            obj = json.loads(message)
            k = obj.get("k")
            if not k:
                return
            if k.get("x") is True:  # candle closed
                out_queue.put(k)
        except Exception:
            return

    def on_error(ws, err):
        print("[WS] error:", err)

    def on_close(ws, status_code, msg):
        print("[WS] closed:", status_code, msg)

    def on_open(ws):
        print("[WS] opened:", ws_url)

    ws = WebSocketApp(ws_url, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)

    while not stop_evt.is_set():
        try:
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            print("[WS] run_forever exception:", e)

        if stop_evt.is_set():
            break

        print("[WS] reconnecting in 2s...")
        time.sleep(2)


# ======================
# 8) MAIN (LIVE)
# ======================

def main():
    symbol = os.getenv("BOT_SYMBOL", "BTCUSDT")
    interval = os.getenv("BOT_INTERVAL", "15m")
    limit = int(os.getenv("BOT_LOOKBACK_LIMIT", "600"))

    left_bars = int(os.getenv("BOT_LEFT_BARS", "5"))
    right_bars = int(os.getenv("BOT_RIGHT_BARS", "1"))

    min_same_kind_gap = int(os.getenv("BOT_MIN_SAME_KIND_GAP", "5"))
    min_opposite_gap = int(os.getenv("BOT_MIN_OPPOSITE_GAP", "2"))

    swing_debug = os.getenv("BOT_SWING_DEBUG", "false").lower() in ("1", "true", "yes", "y", "on")
    max_chase_pct = float(os.getenv("BOT_MAX_CHASE_PCT", "0.03"))
    stop_buffer_pct = float(os.getenv("BOT_STOP_BUFFER_PCT", "0.0"))

    print(f"[LIVE] REST init: {symbol} {interval} limit={limit}")
    df = fetch_ohlc_from_api(symbol=symbol, interval=interval, limit=limit)

    # --- swing alarm state (confirmed pivots only) ---
    # Key: pivot candle timestamp (df.index[pivot_idx])
    # Val: (kind, rounded_price)
    last_confirmed: Dict[pd.Timestamp, Tuple[str, float]] = {}

    def confirmed_swings_only(df_: pd.DataFrame, swings_: List[SwingPoint], rb: int) -> List[SwingPoint]:
        """
        Pivot 'onaylı' sayılması için sağında rb mum daha kapanmış olmalı.
        ignore_last_bar=True olduğu için son bar zaten yok sayılıyor.
        Confirm sınırı:
          pivot_idx <= len(df)-rb-2
        """
        n_ = len(df_)
        max_pivot_idx = n_ - rb - 2
        if max_pivot_idx < 0:
            return []
        out_ = [sp for sp in swings_ if sp.index <= max_pivot_idx]
        return out_

    def build_confirmed_map(df_: pd.DataFrame, swings_: List[SwingPoint], rb: int) -> Dict[pd.Timestamp, Tuple[str, float]]:
        m: Dict[pd.Timestamp, Tuple[str, float]] = {}
        conf = confirmed_swings_only(df_, swings_, rb)
        for sp in conf:
            if 0 <= sp.index < len(df_):
                ts = df_.index[sp.index]
                m[ts] = (sp.kind, float(np.round(float(sp.price), 12)))
        return m

    def diff_and_beep(prev: Dict[pd.Timestamp, Tuple[str, float]], nowm: Dict[pd.Timestamp, Tuple[str, float]]):
        """
        Beep koşulları:
        - NEW confirmed pivot: ts yeni eklendiyse
        - REPAINT/UPDATE: ts var ama (kind/price) değiştiyse
        """
        # NEW
        for ts, (kind, price) in nowm.items():
            if ts not in prev:
                play_beep(kind)
                print(f"[SWING] CONFIRMED NEW {kind} t={ts.isoformat()} price={price}")
            else:
                old_kind, old_price = prev[ts]
                if old_kind != kind or abs(old_price - price) > 1e-12:
                    play_beep(kind)
                    print(
                        f"[SWING] CONFIRMED REPAINT t={ts.isoformat()} "
                        f"{old_kind}@{old_price} -> {kind}@{price}"
                    )

    # ilk hesap
    swings = find_swings(
        df["High"].values,
        df["Low"].values,
        left_bars=left_bars,
        right_bars=right_bars,
        min_distance=None,
        alt_min_distance=None,
        min_same_kind_gap=min_same_kind_gap,
        min_opposite_gap=min_opposite_gap,
        debug=swing_debug,
        ignore_last_bar=True,
    )
    trades = generate_trades(
        df,
        swings,
        stop_buffer_pct=stop_buffer_pct,
        right_bars=right_bars,
        max_chase_pct=max_chase_pct,
    )

    # initial confirmed snapshot (ilk açılışta beep yok)
    last_confirmed = build_confirmed_map(df, swings, right_bars)

    # canlı plot
    plt.ion()
    fig = None
    fig = plot_with_trades(df, swings, trades, symbol, interval, fig=fig)

    # ws thread + queue
    q: "queue.Queue[Dict[str, Any]]" = queue.Queue()
    stop_evt = threading.Event()
    t = threading.Thread(target=start_kline_ws, args=(symbol, interval, q, stop_evt), daemon=True)
    t.start()

    last_redraw_at = 0.0

    print("[LIVE] Listening closed candles via WS. Ctrl+C to stop.")
    try:
        while True:
            try:
                k = q.get(timeout=0.5)
            except queue.Empty:
                plt.pause(0.001)
                continue

            row = _kline_to_row(k)
            df = upsert_closed_candle(df, row, max_len=limit)

            # her kapanan mumda re-calc
            swings = find_swings(
                df["High"].values,
                df["Low"].values,
                left_bars=left_bars,
                right_bars=right_bars,
                min_distance=None,
                alt_min_distance=None,
                min_same_kind_gap=min_same_kind_gap,
                min_opposite_gap=min_opposite_gap,
                debug=swing_debug,
                ignore_last_bar=True,
            )

            # ✅ alarm: sadece confirmed NEW/REPAINT
            now_confirmed = build_confirmed_map(df, swings, right_bars)
            diff_and_beep(last_confirmed, now_confirmed)
            last_confirmed = now_confirmed

            trades = generate_trades(
                df,
                swings,
                stop_buffer_pct=stop_buffer_pct,
                right_bars=right_bars,
                max_chase_pct=max_chase_pct,
            )

            # redraw throttle
            now = time.time()
            if now - last_redraw_at >= 0.05:
                fig = plot_with_trades(df, swings, trades, symbol, interval, fig=fig)
                last_redraw_at = now

            last_close_dt = df.index[-1]
            last_close = float(df["Close"].iloc[-1])
            print(
                f"[LIVE] closed={last_close_dt.isoformat()} close={last_close:.8f} | "
                f"swings={len(swings)} trades={len(trades)}"
            )

    except KeyboardInterrupt:
        print("\n[LIVE] stopping...")
    finally:
        stop_evt.set()
        try:
            plt.ioff()
            plt.show(block=False)
            plt.pause(0.2)
        except Exception:
            pass


if __name__ == "__main__":
    main()
