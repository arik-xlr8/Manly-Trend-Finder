# plot_swings_from_api.py
import requests
import numpy as np
import pandas as pd
import mplfinance as mpf

from dataclasses import dataclass, field
from typing import Literal, Optional, List, Tuple

from swings import find_swings, SwingPoint


# ======================
# 1) VERİ ÇEKME
# ======================

def fetch_ohlc_from_api(symbol: str = "TRXUSDT", interval: str = "15m", limit: int = 500) -> pd.DataFrame:
    url = "https://fapi.binance.com/fapi/v1/klines"
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

    df["Date"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("Date", inplace=True)

    df = df[["open", "high", "low", "close", "volume"]]
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    return df


# ======================
# 2) FILLS + TRADE
# ======================

@dataclass
class FillEvent:
    index: int
    qty_delta: float     # + artır / - azalt ; 0 => flatten (kalanı kapat)
    price: float         # ham Close
    reason: str          # "entry", "tp50_in_target", "readd50_in_target", "exit", "stop"


@dataclass
class Trade:
    direction: Literal["long", "short"]
    entry_index: int
    entry_price: float
    stop_level: float
    exit_index: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None

    # Grafik için eski fib (senin ratio listelerin)
    fib_levels: Optional[List[Tuple[str, float]]] = None
    fib_start_index: Optional[int] = None

    # NEW: trade bazlı 0-1 anchor (swing’den)
    fib0_price: Optional[float] = None
    fib1_price: Optional[float] = None
    fib0_index: Optional[int] = None
    fib1_index: Optional[int] = None

    base_k: Optional[int] = None
    target_k: Optional[int] = None

    sold50_in_target: bool = False
    last_action_in_target: Optional[Literal["sold50", "readd50"]] = None

    fills: List[FillEvent] = field(default_factory=list)


# ======================
# 3) FIB ORANLARI (senin istediğin)
# ======================

UPTREND_FIB_RATIOS = [-2.618, -1.618, -1.0, -1.5, 0.0, 0.5, 1.5, 2.5, 3.6, 4.5]
DOWNTREND_FIB_RATIOS = [3.5, 2.5, 1.5, 1.0, 0.0, -0.5, -1.5, -2.5, -3.6, -4.5]


def get_pivot_anchor_price(df: pd.DataFrame, pivot_idx: int, pivot_kind: str) -> float:
    if pivot_kind == "LB":
        return float(df["Low"].iloc[pivot_idx])
    return float(df["High"].iloc[pivot_idx])


def compute_long_fib_from_ll3(anchor_ll3: float, target_lh2: float) -> List[Tuple[str, float]]:
    rng = target_lh2 - anchor_ll3
    if rng <= 0:
        return []
    return [(f"{r:g}", anchor_ll3 + rng * r) for r in UPTREND_FIB_RATIOS]


def compute_short_fib_from_hh3(anchor_hh3: float, target_lb2: float) -> List[Tuple[str, float]]:
    rng = anchor_hh3 - target_lb2
    if rng <= 0:
        return []
    return [(f"{r:g}", anchor_hh3 - rng * r) for r in DOWNTREND_FIB_RATIOS]


# ======================
# 4) NEW: FIB BAND HELPERS (0-1 üzerinden band)
# ======================

def _fib_u(price: float, fib0: float, fib1: float) -> Optional[float]:
    den = (fib1 - fib0)
    if den == 0:
        return None
    return (price - fib0) / den  # u=0 fib0, u=1 fib1, u=2 => 1-2 band


def _band_bounds_price(fib0: float, fib1: float, k: int) -> Tuple[float, float]:
    p1 = fib0 + (fib1 - fib0) * k
    p2 = fib0 + (fib1 - fib0) * (k + 1)
    return (min(p1, p2), max(p1, p2))


def _price_in_band(price: float, fib0: float, fib1: float, k: int) -> bool:
    lo, hi = _band_bounds_price(fib0, fib1, k)
    return lo <= price <= hi


# ======================
# 5) STRATEJİ LOJİĞİ
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

    # Trailing state
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
        return float(df["Low"].iloc[pivot_idx]) if kind == "LB" else float(df["High"].iloc[pivot_idx])

    # %3 no-chase (pivot close -> entry close)
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

    # === NEW: fib anchors + base/target init ===
    def init_fib_band_for_trade(tr: Trade, prev_lb: Optional[SwingPoint], prev_lh: Optional[SwingPoint]):
        if prev_lb is None or prev_lh is None:
            return

        if tr.direction == "long":
            if not (prev_lb.index < prev_lh.index < tr.entry_index):
                return
            tr.fib0_price, tr.fib1_price = float(prev_lb.price), float(prev_lh.price)
            tr.fib0_index, tr.fib1_index = int(prev_lb.index), int(prev_lh.index)
        else:
            if not (prev_lh.index < prev_lb.index < tr.entry_index):
                return
            tr.fib0_price, tr.fib1_price = float(prev_lh.price), float(prev_lb.price)
            tr.fib0_index, tr.fib1_index = int(prev_lh.index), int(prev_lb.index)

        u = _fib_u(tr.entry_price, tr.fib0_price, tr.fib1_price)
        if u is None:
            return

        tr.base_k = int(np.floor(u))
        tr.target_k = tr.base_k + 1
        tr.sold50_in_target = False
        tr.last_action_in_target = None

    # === NEW: band içinde %50 sat / %50 geri ekle ===
    def handle_fib_band_scaling(sp: SwingPoint, pivot_idx: int, signal_idx: int):
        nonlocal current_trade
        tr = current_trade
        if tr is None:
            return
        if tr.fib0_price is None or tr.fib1_price is None:
            return
        if tr.target_k is None:
            return
        if signal_idx <= tr.entry_index:
            return

        # swing'in fiyatı target band içinde mi?
        if not _price_in_band(float(sp.price), tr.fib0_price, tr.fib1_price, tr.target_k):
            return

        px = float(closes[signal_idx])

        if tr.direction == "long":
            if (not tr.sold50_in_target) and sp.kind == "LH":
                tr.fills.append(FillEvent(index=signal_idx, qty_delta=-0.5, price=px, reason="tp50_in_target"))
                tr.sold50_in_target = True
                tr.last_action_in_target = "sold50"
                return

            if tr.sold50_in_target and tr.last_action_in_target == "sold50" and sp.kind == "LB":
                tr.fills.append(FillEvent(index=signal_idx, qty_delta=+0.5, price=px, reason="readd50_in_target"))
                tr.last_action_in_target = "readd50"
                return

        else:  # short
            if (not tr.sold50_in_target) and sp.kind == "LB":
                tr.fills.append(FillEvent(index=signal_idx, qty_delta=+0.5, price=px, reason="tp50_in_target"))
                tr.sold50_in_target = True
                tr.last_action_in_target = "sold50"
                return

            if tr.sold50_in_target and tr.last_action_in_target == "sold50" and sp.kind == "LH":
                tr.fills.append(FillEvent(index=signal_idx, qty_delta=-0.5, price=px, reason="readd50_in_target"))
                tr.last_action_in_target = "readd50"
                return

    def add_entry_fill(tr: Trade):
        # full poz aç
        tr.fills.append(FillEvent(
            index=tr.entry_index,
            qty_delta=(+1.0 if tr.direction == "long" else -1.0),
            price=float(closes[tr.entry_index]),
            reason="entry"
        ))

    def add_flatten_fill(tr: Trade, idx: int, reason: str):
        # kalan qty'yi kapat (sim flatten yapacak)
        tr.fills.append(FillEvent(
            index=idx,
            qty_delta=0.0,
            price=float(closes[idx]) if 0 <= idx < n else float(tr.entry_price),
            reason=reason
        ))

    def open_uncertain_trade_by_signal(signal: SwingPoint, entry_idx: int):
        nonlocal current_trade
        if entry_idx < 0 or entry_idx >= n:
            return

        if signal.kind == "LH":
            direction: Literal["long", "short"] = "short"
            stop_level = float(df["High"].iloc[signal.index])
        else:
            direction = "long"
            stop_level = float(df["Low"].iloc[signal.index])

        current_trade = Trade(
            direction=direction,
            entry_index=entry_idx,
            entry_price=float(closes[entry_idx]),
            stop_level=stop_level,
            fib_levels=None,
            fib_start_index=None
        )
        add_entry_fill(current_trade)
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

        # ==================================================
        # A) Swing'ler arası STOP taraması (bar bar)
        # ==================================================
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

        # ==================================================
        # B) BELİRSİZ MOD (fib yok)
        # ==================================================
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
                        current_trade = Trade(
                            direction=desired,
                            entry_index=signal_idx,
                            entry_price=float(closes[signal_idx]),
                            stop_level=stop_level,
                            fib_levels=None,
                            fib_start_index=None
                        )
                        add_entry_fill(current_trade)
                        reset_trailing_state()

                last_signal = sp
                if sp.kind == "LB":
                    last_lb = sp
                else:
                    last_lh = sp
                continue

        # ==================================================
        # C) NORMAL MOD
        # ==================================================

        # NEW: band scaling (trade açıkken)
        if current_trade is not None:
            handle_fib_band_scaling(sp, pivot_idx, signal_idx)

        # Trailing stop (senin eski mantık)
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
                                if new_stop < current_trade.stop_level:
                                    current_trade.stop_level = new_stop
                                short_last_low = sp
                                short_candidate_high = None
                elif sp.kind == "LH":
                    if short_last_low is not None and sp.index > short_last_low.index:
                        short_candidate_high = sp

        # 1) poz açıkken ters sinyal -> kapat + flip
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

                    # flip short
                    if not chase_too_much("short", pivot_idx, signal_idx):
                        entry_price = float(closes[signal_idx])

                        fib_levels = None
                        fib_start_index = None
                        hh3_idx = pivot_idx
                        if 0 <= hh3_idx < n and prev_lb is not None:
                            anchor_hh3 = get_pivot_anchor_price(df, hh3_idx, "LH")
                            fib_levels = compute_short_fib_from_hh3(anchor_hh3=anchor_hh3, target_lb2=prev_lb.price)
                            fib_start_index = hh3_idx

                        stop_level = pivot_stop_level(pivot_idx, "LH")
                        current_trade = Trade(
                            direction="short",
                            entry_index=signal_idx,
                            entry_price=entry_price,
                            stop_level=stop_level,
                            fib_levels=fib_levels,
                            fib_start_index=fib_start_index
                        )
                        add_entry_fill(current_trade)
                        init_fib_band_for_trade(current_trade, prev_lb=prev_lb, prev_lh=prev_lh)
                        reset_trailing_state()

            else:  # short
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

                    # flip long
                    if not chase_too_much("long", pivot_idx, signal_idx):
                        entry_price = float(closes[signal_idx])

                        fib_levels = None
                        fib_start_index = None
                        ll3_idx = pivot_idx
                        if 0 <= ll3_idx < n and prev_lh is not None:
                            anchor_ll3 = get_pivot_anchor_price(df, ll3_idx, "LB")
                            fib_levels = compute_long_fib_from_ll3(anchor_ll3=anchor_ll3, target_lh2=prev_lh.price)
                            fib_start_index = ll3_idx

                        stop_level = pivot_stop_level(pivot_idx, "LB")
                        current_trade = Trade(
                            direction="long",
                            entry_index=signal_idx,
                            entry_price=entry_price,
                            stop_level=stop_level,
                            fib_levels=fib_levels,
                            fib_start_index=fib_start_index
                        )
                        add_entry_fill(current_trade)
                        init_fib_band_for_trade(current_trade, prev_lb=prev_lb, prev_lh=prev_lh)
                        reset_trailing_state()

        # 2) poz yokken entry
        if current_trade is None:
            # long entry: HL
            if sp.kind == "LB" and prev_lb is not None and prev_lb.price < sp.price:
                if not chase_too_much("long", pivot_idx, signal_idx):
                    entry_price = float(closes[signal_idx])

                    fib_levels = None
                    fib_start_index = None
                    ll3_idx = pivot_idx
                    if 0 <= ll3_idx < n and prev_lh is not None:
                        anchor_ll3 = get_pivot_anchor_price(df, ll3_idx, "LB")
                        fib_levels = compute_long_fib_from_ll3(anchor_ll3=anchor_ll3, target_lh2=prev_lh.price)
                        fib_start_index = ll3_idx

                    stop_level = pivot_stop_level(pivot_idx, "LB")
                    current_trade = Trade(
                        direction="long",
                        entry_index=signal_idx,
                        entry_price=entry_price,
                        stop_level=stop_level,
                        fib_levels=fib_levels,
                        fib_start_index=fib_start_index
                    )
                    add_entry_fill(current_trade)
                    init_fib_band_for_trade(current_trade, prev_lb=prev_lb, prev_lh=prev_lh)
                    reset_trailing_state()

            # short entry: LH lower-high
            elif sp.kind == "LH" and prev_lh is not None and prev_lh.price > sp.price:
                if not chase_too_much("short", pivot_idx, signal_idx):
                    entry_price = float(closes[signal_idx])

                    fib_levels = None
                    fib_start_index = None
                    hh3_idx = pivot_idx
                    if 0 <= hh3_idx < n and prev_lb is not None:
                        anchor_hh3 = get_pivot_anchor_price(df, hh3_idx, "LH")
                        fib_levels = compute_short_fib_from_hh3(anchor_hh3=anchor_hh3, target_lb2=prev_lb.price)
                        fib_start_index = hh3_idx

                    stop_level = pivot_stop_level(pivot_idx, "LH")
                    current_trade = Trade(
                        direction="short",
                        entry_index=signal_idx,
                        entry_price=entry_price,
                        stop_level=stop_level,
                        fib_levels=fib_levels,
                        fib_start_index=fib_start_index
                    )
                    add_entry_fill(current_trade)
                    init_fib_band_for_trade(current_trade, prev_lb=prev_lb, prev_lh=prev_lh)
                    reset_trailing_state()

        # ==================================================
        # D) Pivot update
        # ==================================================
        last_signal = sp
        if sp.kind == "LB":
            last_lb = sp
        else:
            last_lh = sp

    # sonda açık trade varsa ekle (exit yoksa backtest onu görmez)
    if current_trade is not None:
        trades.append(current_trade)

    return trades


# ======================
# 6) GRAFİK
# ======================

def plot_with_trades(df: pd.DataFrame, swings: List[SwingPoint], trades: List[Trade], symbol: str, interval: str):
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

        if tr.exit_index is None:
            xi = n - 1
        else:
            xi = tr.exit_index
            if xi < 0 or xi >= n:
                continue

        if xi <= ei:
            continue

        if tr.direction == "long":
            long_mask[ei:xi+1] = True
        else:
            short_mask[ei:xi+1] = True

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
        if tr.exit_index is None:
            x2 = df.index[n - 1]
        else:
            if 0 <= tr.exit_index < n:
                x2 = df.index[tr.exit_index]
            else:
                continue

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
        title=f"{symbol} {interval}",
        ylabel="Fiyat",
        ylabel_lower="Hacim",
        tight_layout=True,
        datetime_format="%m-%d %H:%M",
    )

    if fb:
        plot_kwargs["fill_between"] = fb
    if alines_segments:
        plot_kwargs["alines"] = dict(alines=alines_segments, linewidths=0.7, alpha=0.45)

    mpf.plot(df, **plot_kwargs)


# ======================
# 7) MAIN (debug plot)
# ======================

def main():
    symbol = "TRXUSDT"
    interval = "15m"

    df = fetch_ohlc_from_api(symbol=symbol, interval=interval, limit=200)

    highs = df["High"].values
    lows = df["Low"].values

    right_bars = 1

    swings = find_swings(
        highs,
        lows,
        left_bars=5,
        right_bars=right_bars,
        min_distance=10,
        alt_min_distance=None,
    )

    trades = generate_trades(df, swings, stop_buffer_pct=0.0, right_bars=right_bars, max_chase_pct=0.005)

    print("Trade sayısı:", len(trades))
    for t in trades[-10:]:
        print(
            t.direction,
            "entry:", t.entry_index,
            "exit:", t.exit_index,
            "stop:", round(t.stop_level, 6),
            "exit_reason:", t.exit_reason,
            "fills:", len(t.fills),
            "base:", t.base_k,
            "target:", t.target_k,
        )

    plot_with_trades(df, swings, trades, symbol, interval)


if __name__ == "__main__":
    main()
