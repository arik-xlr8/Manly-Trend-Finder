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
# 2) TRADE YAPISI (+ FIB)
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

    # Fib trade açılırken 1 kere hesaplanır ve trade kapanana kadar SABİT kalır
    fib_levels: Optional[List[Tuple[str, float]]] = None
    fib_start_index: Optional[int] = None  # LL3/HH3 pivot bar index (çizgiler buradan başlar)


# ======================
# 3) FIB ORANLARI (SENİN İSTEDİĞİN)
# ======================

UPTREND_FIB_RATIOS = [-2.618, -1.618, -1.0, -1.5, 0.0, 0.5, 1.5, 2.5, 3.6, 4.5]
DOWNTREND_FIB_RATIOS = [3.5, 2.5, 1.5, 1.0, 0.0, -0.5, -1.5, -2.5, -3.6, -4.5]


def get_pivot_anchor_price(df: pd.DataFrame, pivot_idx: int, pivot_kind: str) -> float:
    """
    Yeni kural:
      - pivot dip (LB / LL3) -> anchor = o mumun LOW'u
      - pivot tepe (LH / HH3) -> anchor = o mumun HIGH'ı
    """
    if pivot_kind == "LB":
        return float(df["Low"].iloc[pivot_idx])
    return float(df["High"].iloc[pivot_idx])  # "LH"


def compute_long_fib_from_ll3(anchor_ll3: float, target_lh2: float) -> List[Tuple[str, float]]:
    """
    YÜKSELEN:
      0 = LL3 (LB3 pivot mumunun LOW'u)
      rng = LH2 - LL3
      level = LL3 + rng * ratio
    """
    rng = target_lh2 - anchor_ll3
    if rng <= 0:
        return []

    levels: List[Tuple[str, float]] = []
    for r in UPTREND_FIB_RATIOS:
        levels.append((f"{r:g}", anchor_ll3 + rng * r))
    return levels


def compute_short_fib_from_hh3(anchor_hh3: float, target_lb2: float) -> List[Tuple[str, float]]:
    """
    DÜŞEN:
      0 = HH3 (LH3 pivot mumunun HIGH'ı)
      rng = HH3 - LB2
      level = HH3 - rng * ratio
    """
    rng = anchor_hh3 - target_lb2
    if rng <= 0:
        return []

    levels: List[Tuple[str, float]] = []
    for r in DOWNTREND_FIB_RATIOS:
        levels.append((f"{r:g}", anchor_hh3 - rng * r))
    return levels


# ======================
# 4) STRATEJİ LOJİĞİ (HL / LH) + FIB TRADE'E KİLİTLİ
# ======================

def generate_trades(
    df: pd.DataFrame,
    swings: List[SwingPoint],
    stop_buffer_pct: float = 0.0,  # kullanılmıyor (imza dursun)
    right_bars: int = 2,
) -> List[Trade]:
    closes = df["Close"].values
    n = len(df)

    trades: List[Trade] = []
    current_trade: Optional[Trade] = None

    last_lb: Optional[SwingPoint] = None
    last_lh: Optional[SwingPoint] = None

    for sp in swings:
        pivot_idx = sp.index
        signal_idx = pivot_idx + right_bars  # trade giriş/çıkış barı

        if signal_idx >= n:
            continue

        prev_lb = last_lb
        prev_lh = last_lh

        # ======================
        # 1) POZ AÇIKKEN: SADECE TERS SİNYALLE KAPAT + FLIP
        # ======================
        if current_trade is not None:
            if current_trade.direction == "long":
                # LONG kapanma: gerçek downtrend (Lower High)
                if sp.kind == "LH" and prev_lh is not None and prev_lh.price > sp.price:
                    exit_idx = signal_idx
                    if exit_idx > current_trade.entry_index:
                        current_trade.exit_index = exit_idx
                        current_trade.exit_price = float(closes[exit_idx])
                        current_trade.exit_reason = "long_exit_downtrend_LH"
                        trades.append(current_trade)
                    current_trade = None

                    # Aynı anda SHORT aç: fib HH3 pivotundan
                    entry_price = float(closes[signal_idx])

                    fib_levels = None
                    fib_start_index = None

                    hh3_idx = pivot_idx  # bu pivot: LH3
                    if 0 <= hh3_idx < n and prev_lb is not None:
                        anchor_hh3 = get_pivot_anchor_price(df, hh3_idx, "LH")  # HIGH
                        fib_levels = compute_short_fib_from_hh3(anchor_hh3=anchor_hh3, target_lb2=prev_lb.price)
                        fib_start_index = hh3_idx

                    current_trade = Trade(
                        direction="short",
                        entry_index=signal_idx,
                        entry_price=entry_price,
                        stop_level=0.0,
                        fib_levels=fib_levels,
                        fib_start_index=fib_start_index
                    )

            else:  # short
                # SHORT kapanma: gerçek uptrend (Higher Low)
                if sp.kind == "LB" and prev_lb is not None and prev_lb.price < sp.price:
                    exit_idx = signal_idx
                    if exit_idx > current_trade.entry_index:
                        current_trade.exit_index = exit_idx
                        current_trade.exit_price = float(closes[exit_idx])
                        current_trade.exit_reason = "short_exit_uptrend_LB"
                        trades.append(current_trade)
                    current_trade = None

                    # Aynı anda LONG aç: fib LL3 pivotundan
                    entry_price = float(closes[signal_idx])

                    fib_levels = None
                    fib_start_index = None

                    ll3_idx = pivot_idx  # bu pivot: LB3
                    if 0 <= ll3_idx < n and prev_lh is not None:
                        anchor_ll3 = get_pivot_anchor_price(df, ll3_idx, "LB")  # LOW
                        fib_levels = compute_long_fib_from_ll3(anchor_ll3=anchor_ll3, target_lh2=prev_lh.price)
                        fib_start_index = ll3_idx

                    current_trade = Trade(
                        direction="long",
                        entry_index=signal_idx,
                        entry_price=entry_price,
                        stop_level=0.0,
                        fib_levels=fib_levels,
                        fib_start_index=fib_start_index
                    )

        # ======================
        # 2) POZ YOKKEN ENTRY
        # ======================
        if current_trade is None:
            # LONG entry: HL (Higher Low)
            if sp.kind == "LB" and prev_lb is not None and prev_lb.price < sp.price:
                entry_price = float(closes[signal_idx])

                fib_levels = None
                fib_start_index = None

                ll3_idx = pivot_idx  # LB3
                if 0 <= ll3_idx < n and prev_lh is not None:
                    anchor_ll3 = get_pivot_anchor_price(df, ll3_idx, "LB")  # LOW
                    fib_levels = compute_long_fib_from_ll3(anchor_ll3=anchor_ll3, target_lh2=prev_lh.price)
                    fib_start_index = ll3_idx

                current_trade = Trade(
                    direction="long",
                    entry_index=signal_idx,
                    entry_price=entry_price,
                    stop_level=0.0,
                    fib_levels=fib_levels,
                    fib_start_index=fib_start_index
                )

            # SHORT entry: LH (Lower High)
            elif sp.kind == "LH" and prev_lh is not None and prev_lh.price > sp.price:
                entry_price = float(closes[signal_idx])

                fib_levels = None
                fib_start_index = None

                hh3_idx = pivot_idx  # LH3
                if 0 <= hh3_idx < n and prev_lb is not None:
                    anchor_hh3 = get_pivot_anchor_price(df, hh3_idx, "LH")  # HIGH
                    fib_levels = compute_short_fib_from_hh3(anchor_hh3=anchor_hh3, target_lb2=prev_lb.price)
                    fib_start_index = hh3_idx

                current_trade = Trade(
                    direction="short",
                    entry_index=signal_idx,
                    entry_price=entry_price,
                    stop_level=0.0,
                    fib_levels=fib_levels,
                    fib_start_index=fib_start_index
                )

        # ======================
        # 3) SON PİVOTLARI GÜNCELLE
        # ======================
        if sp.kind == "LB":
            last_lb = sp
        else:
            last_lh = sp

    # Açık trade kalırsa ekle
    if current_trade is not None:
        trades.append(current_trade)

    return trades


# ======================
# 5) GRAFİK
# ======================

def plot_with_trades(df: pd.DataFrame, swings: List[SwingPoint], trades: List[Trade], symbol: str, interval: str):
    n = len(df)

    long_mask = np.zeros(n, dtype=bool)
    short_mask = np.zeros(n, dtype=bool)

    long_entry  = np.full(n, np.nan)
    long_exit   = np.full(n, np.nan)
    short_entry = np.full(n, np.nan)
    short_exit  = np.full(n, np.nan)

    # Pivot noktaları
    lh_series = np.full(n, np.nan)
    lb_series = np.full(n, np.nan)
    for sp in swings:
        if 0 <= sp.index < n:
            if sp.kind == "LH":
                lh_series[sp.index] = sp.price
            else:
                lb_series[sp.index] = sp.price

    # Trade tüneli + entry/exit
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

    lows = df["Low"].values
    highs = df["High"].values

    long_band_low  = np.where(long_mask, lows,  np.nan)
    long_band_high = np.where(long_mask, highs, np.nan)

    short_band_low  = np.where(short_mask, lows,  np.nan)
    short_band_high = np.where(short_mask, highs, np.nan)

    apds = []
    def has_data(arr: np.ndarray) -> bool:
        return np.isfinite(arr).any()

    # Pivotlar
    if has_data(lh_series):
        apds.append(mpf.make_addplot(lh_series, type="scatter", markersize=50, marker="^", color="#ff9800"))
    if has_data(lb_series):
        apds.append(mpf.make_addplot(lb_series, type="scatter", markersize=50, marker="v", color="#03a9f4"))

    # Trade entry/exit
    if has_data(long_entry):
        apds.append(mpf.make_addplot(long_entry, type="scatter", markersize=70, marker="o", color="#00e676"))
    if has_data(long_exit):
        apds.append(mpf.make_addplot(long_exit, type="scatter", markersize=70, marker="o", color="#1de9b6"))
    if has_data(short_entry):
        apds.append(mpf.make_addplot(short_entry, type="scatter", markersize=70, marker="o", color="#ff5252"))
    if has_data(short_exit):
        apds.append(mpf.make_addplot(short_exit, type="scatter", markersize=70, marker="o", color="#ff8a80"))

    # Trend tüneli
    fb = []
    if long_mask.any():
        fb.append(dict(y1=long_band_low, y2=long_band_high, where=long_mask, alpha=0.15, color="#00e676"))
    if short_mask.any():
        fb.append(dict(y1=short_band_low, y2=short_band_high, where=short_mask, alpha=0.15, color="#ff5252"))

    # Fib çizgileri: LL3/HH3 pivot barından başlar, trade kapanana kadar gider
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
# 6) MAIN
# ======================

def main():
    symbol = "TRXUSDT"
    interval = "15m"

    df = fetch_ohlc_from_api(symbol=symbol, interval=interval, limit=200)

    highs = df["High"].values
    lows  = df["Low"].values

    right_bars = 1  # find_swings ile aynı olmalı

    swings = find_swings(
        highs,
        lows,
        left_bars=5,
        right_bars=right_bars,
        min_distance=10,
        alt_min_distance=None,
    )

    trades = generate_trades(df, swings, stop_buffer_pct=0.0, right_bars=right_bars)

    print("Trade sayısı:", len(trades))
    for t in trades[-10:]:
        print(
            t.direction,
            "entry:", t.entry_index,
            "exit:", t.exit_index,
            "fib:", "var" if t.fib_levels else "yok",
            "fib_start:", t.fib_start_index
        )

    plot_with_trades(df, swings, trades, symbol, interval)


if __name__ == "__main__":
    main()
