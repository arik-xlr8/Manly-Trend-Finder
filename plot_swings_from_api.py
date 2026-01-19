import requests
import numpy as np
import pandas as pd
import mplfinance as mpf

from swings import find_swings, SwingPoint

from dataclasses import dataclass
from typing import Literal, Optional, List, Tuple


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
        "open_time","open","high","low","close","volume","close_time",
        "quote_asset_volume","number_of_trades","taker_buy_base_volume",
        "taker_buy_quote_volume","ignore",
    ]

    df = pd.DataFrame(data, columns=cols)
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

    df["Date"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("Date", inplace=True)

    df = df[["open", "high", "low", "close", "volume"]]
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    return df


# ======================
# 2) TRADE YAPISI
# ======================

@dataclass
class Trade:
    direction: Literal["long", "short"]
    entry_index: int
    entry_price: float
    stop_level: float
    exit_index: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None


# ======================
# 2.5) UP-TREND FIB (LB2 -> LH2 -> LB3)
# ======================

@dataclass
class UptrendFib:
    fib_id: int
    lb2_index: int
    lb2_price: float
    lh2_index: int
    lh2_price: float
    lb3_index: int
    lb3_price: float
    levels: List[Tuple[str, float]]


def compute_uptrend_fib_levels(lb2: float, lh2: float) -> List[Tuple[str, float]]:
    rng = lh2 - lb2
    if rng <= 0:
        return []

    retr = [0.236, 0.382, 0.5, 0.618, 0.786]
    ext  = [1.0, 1.272, 1.618]

    levels: List[Tuple[str, float]] = []
    levels.append(("0.000 (LH2)", lh2))
    for r in retr:
        levels.append((f"{r:.3f}", lh2 - rng * r))
    levels.append(("1.000 (LB2)", lb2))
    for e in ext:
        levels.append((f"ext {e:.3f}", lh2 + rng * (e - 1.0)))
    return levels


def find_uptrend_fibs(swings: List[SwingPoint]) -> List[UptrendFib]:
    fibs: List[UptrendFib] = []
    fib_id = 1

    last_lb: Optional[SwingPoint] = None
    last_lh: Optional[SwingPoint] = None

    for sp in swings:
        prev_lb = last_lb
        prev_lh = last_lh

        if sp.kind == "LB" and prev_lb is not None and prev_lb.price < sp.price:
            lb2 = prev_lb
            lb3 = sp

            if prev_lh is not None and (lb2.index < prev_lh.index < lb3.index):
                lh2 = prev_lh
                levels = compute_uptrend_fib_levels(lb2.price, lh2.price)
                if levels:
                    fibs.append(
                        UptrendFib(
                            fib_id=fib_id,
                            lb2_index=lb2.index, lb2_price=lb2.price,
                            lh2_index=lh2.index, lh2_price=lh2.price,
                            lb3_index=lb3.index, lb3_price=lb3.price,
                            levels=levels
                        )
                    )
                    fib_id += 1

        if sp.kind == "LB":
            last_lb = sp
        else:
            last_lh = sp

    return fibs


# ======================
# 2.6) DOWN-TREND FIB (LH2 -> LB2 -> LH3)
# ======================

@dataclass
class DowntrendFib:
    fib_id: int
    lh2_index: int
    lh2_price: float
    lb2_index: int
    lb2_price: float
    lh3_index: int
    lh3_price: float
    levels: List[Tuple[str, float]]


def compute_downtrend_fib_levels(lh2: float, lb2: float) -> List[Tuple[str, float]]:
    """
    Downtrend fib:
      Ana hareket: LH2 -> LB2 (range = lh2 - lb2)
      Retracement (yukarı dönüş): LB2 + range * r
      Extension (aşağı devam):    LB2 - range * (e - 1)
    """
    rng = lh2 - lb2
    if rng <= 0:
        return []

    retr = [0.236, 0.382, 0.5, 0.618, 0.786]
    ext  = [1.0, 1.272, 1.618]

    levels: List[Tuple[str, float]] = []
    levels.append(("0.000 (LB2)", lb2))
    for r in retr:
        levels.append((f"{r:.3f}", lb2 + rng * r))
    levels.append(("1.000 (LH2)", lh2))
    for e in ext:
        levels.append((f"ext {e:.3f}", lb2 - rng * (e - 1.0)))
    return levels


def find_downtrend_fibs(swings: List[SwingPoint]) -> List[DowntrendFib]:
    """
    Senin tarifin:
      Trend LH3'te başlar (Lower High anı).
      Fib seti = (LH2, LB2, LH3)
        - LH2: LH3'ten önceki tepe
        - LB2: LH2'den sonra gelen dip
        - LH3: LH (lower high) ile trend başlangıcı
    """
    fibs: List[DowntrendFib] = []
    fib_id = 1

    last_lh: Optional[SwingPoint] = None   # LH2 adayı
    last_lb: Optional[SwingPoint] = None   # LB2 adayı (son görülen LB)

    for sp in swings:
        prev_lh = last_lh
        prev_lb = last_lb

        # Lower High yakala: LH3 geldi ve LH2'ye göre daha düşük
        if sp.kind == "LH" and prev_lh is not None and prev_lh.price > sp.price:
            lh2 = prev_lh
            lh3 = sp

            # LB2 = LH2'den sonra gelen LB olmalı (LH2 < LB2 < LH3)
            if prev_lb is not None and (lh2.index < prev_lb.index < lh3.index):
                lb2 = prev_lb
                levels = compute_downtrend_fib_levels(lh2.price, lb2.price)
                if levels:
                    fibs.append(
                        DowntrendFib(
                            fib_id=fib_id,
                            lh2_index=lh2.index, lh2_price=lh2.price,
                            lb2_index=lb2.index, lb2_price=lb2.price,
                            lh3_index=lh3.index, lh3_price=lh3.price,
                            levels=levels
                        )
                    )
                    fib_id += 1

        # pivot cache güncelle
        if sp.kind == "LB":
            last_lb = sp
        else:
            last_lh = sp

    return fibs


# ======================
# 3) STRATEJİ LOJİĞİ (Aynen)
# ======================

def generate_trades(df: pd.DataFrame, swings: List[SwingPoint], stop_buffer_pct: float = 0.0, right_bars: int = 2) -> List[Trade]:
    closes = df["Close"].values
    n = len(df)

    trades: List[Trade] = []
    current_trade: Optional[Trade] = None
    last_lb: Optional[SwingPoint] = None
    last_lh: Optional[SwingPoint] = None

    for sp in swings:
        pivot_idx = sp.index
        signal_idx = pivot_idx + right_bars
        if signal_idx >= n:
            continue

        prev_lb = last_lb
        prev_lh = last_lh

        if current_trade is not None:
            if current_trade.direction == "long":
                if sp.kind == "LH" and prev_lh is not None and prev_lh.price > sp.price:
                    exit_idx = signal_idx
                    if exit_idx > current_trade.entry_index:
                        current_trade.exit_index = exit_idx
                        current_trade.exit_price = closes[exit_idx]
                        current_trade.exit_reason = "long_exit_downtrend_LH"
                        trades.append(current_trade)
                    current_trade = None
                    current_trade = Trade("short", signal_idx, closes[signal_idx], 0.0)

            elif current_trade.direction == "short":
                if sp.kind == "LB" and prev_lb is not None and prev_lb.price < sp.price:
                    exit_idx = signal_idx
                    if exit_idx > current_trade.entry_index:
                        current_trade.exit_index = exit_idx
                        current_trade.exit_price = closes[exit_idx]
                        current_trade.exit_reason = "short_exit_uptrend_LB"
                        trades.append(current_trade)
                    current_trade = None
                    current_trade = Trade("long", signal_idx, closes[signal_idx], 0.0)

        if current_trade is None:
            if sp.kind == "LB" and prev_lb is not None and prev_lb.price < sp.price:
                current_trade = Trade("long", signal_idx, closes[signal_idx], 0.0)
            elif sp.kind == "LH" and prev_lh is not None and prev_lh.price > sp.price:
                current_trade = Trade("short", signal_idx, closes[signal_idx], 0.0)

        if sp.kind == "LB":
            last_lb = sp
        else:
            last_lh = sp

    if current_trade is not None:
        trades.append(current_trade)

    return trades


# ======================
# 4) GRAFİKTE GÖSTERME (+ UP + DOWN FIB)
# ======================

def plot_with_trades_and_fibs(df: pd.DataFrame, swings: List[SwingPoint], trades: List[Trade],
                             up_fibs: List[UptrendFib], down_fibs: List[DowntrendFib],
                             symbol: str, interval: str):
    n = len(df)

    long_mask = np.zeros(n, dtype=bool)
    short_mask = np.zeros(n, dtype=bool)

    long_entry  = np.full(n, np.nan)
    long_exit   = np.full(n, np.nan)
    short_entry = np.full(n, np.nan)
    short_exit  = np.full(n, np.nan)

    for tr in trades:
        ei = tr.entry_index
        if ei < 0 or ei >= n:
            continue

        if tr.exit_index is None:
            has_exit = False
            xi = n - 1
        else:
            has_exit = True
            xi = tr.exit_index
            if xi < 0 or xi >= n:
                continue

        if xi <= ei:
            if tr.direction == "long":
                long_entry[ei] = df["Low"].iloc[ei] * 0.995
            else:
                short_entry[ei] = df["High"].iloc[ei] * 1.005
            continue

        if tr.direction == "long":
            long_mask[ei:xi+1] = True
            long_entry[ei] = df["Low"].iloc[ei] * 0.995
            if has_exit:
                long_exit[xi] = df["High"].iloc[xi] * 1.005
        else:
            short_mask[ei:xi+1] = True
            short_entry[ei] = df["High"].iloc[ei] * 1.005
            if has_exit:
                short_exit[xi] = df["Low"].iloc[xi] * 0.995

    lows  = df["Low"].values
    highs = df["High"].values
    long_band_low  = np.where(long_mask, lows,  np.nan)
    long_band_high = np.where(long_mask, highs, np.nan)
    short_band_low  = np.where(short_mask, lows,  np.nan)
    short_band_high = np.where(short_mask, highs, np.nan)

    # Pivot noktaları
    lh_series = np.full(n, np.nan)
    lb_series = np.full(n, np.nan)
    for sp in swings:
        if 0 <= sp.index < n:
            if sp.kind == "LH":
                lh_series[sp.index] = sp.price
            else:
                lb_series[sp.index] = sp.price

    # Up fib noktaları (LB2, LH2, LB3)
    up_lb2 = np.full(n, np.nan)
    up_lh2 = np.full(n, np.nan)
    up_lb3 = np.full(n, np.nan)
    for fb in up_fibs:
        if 0 <= fb.lb2_index < n: up_lb2[fb.lb2_index] = fb.lb2_price
        if 0 <= fb.lh2_index < n: up_lh2[fb.lh2_index] = fb.lh2_price
        if 0 <= fb.lb3_index < n: up_lb3[fb.lb3_index] = fb.lb3_price

    # Down fib noktaları (LH2, LB2, LH3)
    down_lh2 = np.full(n, np.nan)
    down_lb2 = np.full(n, np.nan)
    down_lh3 = np.full(n, np.nan)
    for fb in down_fibs:
        if 0 <= fb.lh2_index < n: down_lh2[fb.lh2_index] = fb.lh2_price
        if 0 <= fb.lb2_index < n: down_lb2[fb.lb2_index] = fb.lb2_price
        if 0 <= fb.lh3_index < n: down_lh3[fb.lh3_index] = fb.lh3_price

    apds = []
    def has_data(arr: np.ndarray) -> bool:
        return np.isfinite(arr).any()

    if has_data(lh_series):
        apds.append(mpf.make_addplot(lh_series, type="scatter", markersize=50, marker="^", color="#ff9800"))
    if has_data(lb_series):
        apds.append(mpf.make_addplot(lb_series, type="scatter", markersize=50, marker="v", color="#03a9f4"))

    if has_data(long_entry):
        apds.append(mpf.make_addplot(long_entry, type="scatter", markersize=70, marker="o", color="#00e676"))
    if has_data(long_exit):
        apds.append(mpf.make_addplot(long_exit, type="scatter", markersize=70, marker="o", color="#1de9b6"))
    if has_data(short_entry):
        apds.append(mpf.make_addplot(short_entry, type="scatter", markersize=70, marker="o", color="#ff5252"))
    if has_data(short_exit):
        apds.append(mpf.make_addplot(short_exit, type="scatter", markersize=70, marker="o", color="#ff8a80"))

    # Uptrend fib 3 noktası (büyük)
    if has_data(up_lb2):
        apds.append(mpf.make_addplot(up_lb2, type="scatter", markersize=140, marker="o", color="#ffd54f"))
    if has_data(up_lh2):
        apds.append(mpf.make_addplot(up_lh2, type="scatter", markersize=140, marker="o", color="#ffee58"))
    if has_data(up_lb3):
        apds.append(mpf.make_addplot(up_lb3, type="scatter", markersize=140, marker="o", color="#ffca28"))

    # Downtrend fib 3 noktası (büyük) - farklı tonlar
    if has_data(down_lh2):
        apds.append(mpf.make_addplot(down_lh2, type="scatter", markersize=140, marker="o", color="#ffb74d"))
    if has_data(down_lb2):
        apds.append(mpf.make_addplot(down_lb2, type="scatter", markersize=140, marker="o", color="#ff8a80"))
    if has_data(down_lh3):
        apds.append(mpf.make_addplot(down_lh3, type="scatter", markersize=140, marker="o", color="#ff7043"))

    # Trend tüneli
    fb_fill = []
    if long_mask.any():
        fb_fill.append(dict(y1=long_band_low, y2=long_band_high, where=long_mask, alpha=0.15, color="#00e676"))
    if short_mask.any():
        fb_fill.append(dict(y1=short_band_low, y2=short_band_high, where=short_mask, alpha=0.15, color="#ff5252"))

    # Fib çizgileri segmentleri:
    # Up: LB2 -> LB3
    # Down: LH2 -> LH3
    alines_segments = []

    for fb in up_fibs:
        if 0 <= fb.lb2_index < n and 0 <= fb.lb3_index < n:
            x1 = df.index[fb.lb2_index]
            x2 = df.index[fb.lb3_index]
            for _, price in fb.levels:
                alines_segments.append([(x1, price), (x2, price)])

    for fb in down_fibs:
        if 0 <= fb.lh2_index < n and 0 <= fb.lh3_index < n:
            x1 = df.index[fb.lh2_index]
            x2 = df.index[fb.lh3_index]
            for _, price in fb.levels:
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

    if fb_fill:
        plot_kwargs["fill_between"] = fb_fill
    if alines_segments:
        plot_kwargs["alines"] = dict(alines=alines_segments, linewidths=0.7, alpha=0.45)

    mpf.plot(df, **plot_kwargs)


# ======================
# 5) MAIN
# ======================

def main():
    symbol = "TRXUSDT"
    interval = "15m"

    df = fetch_ohlc_from_api(symbol=symbol, interval=interval, limit=200)

    highs = df["High"].values
    lows  = df["Low"].values

    right_bars = 1

    swings = find_swings(
        highs,
        lows,
        left_bars=5,
        right_bars=right_bars,
        min_distance=10,
        alt_min_distance=None,
    )

    trades = generate_trades(df, swings, stop_buffer_pct=0.0, right_bars=right_bars)

    up_fibs = find_uptrend_fibs(swings)
    down_fibs = find_downtrend_fibs(swings)

    print("Uptrend fib set:", len(up_fibs))
    print("Downtrend fib set:", len(down_fibs))

    plot_with_trades_and_fibs(df, swings, trades, up_fibs, down_fibs, symbol, interval)


if __name__ == "__main__":
    main()
