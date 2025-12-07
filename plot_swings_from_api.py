import requests
import numpy as np
import pandas as pd
import mplfinance as mpf

from swings import find_swings, SwingPoint   # bizim pivot bulucu

from dataclasses import dataclass
from typing import Literal, Optional, List


# ======================
# 1) VERİ ÇEKME
# ======================

def fetch_ohlc_from_api(
    symbol: str = "AVAXUSDT",
    interval: str = "4h",
    limit: int = 500,
) -> pd.DataFrame:
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]

    df = pd.DataFrame(data, columns=cols)

    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)

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
# 3) STRATEJİ LOJİĞİ (HL / LH)
# ======================

def generate_trades(
    df: pd.DataFrame,
    swings: List[SwingPoint],
    stop_buffer_pct: float = 0.0,  # şimdilik kullanılmıyor ama imzayı bozmayalım
    right_bars: int = 2,           # pivot onayı için kaç bar bekleniyor (find_swings ile aynı olmalı)
) -> List[Trade]:
    """
    Strateji (senin tarif ettiğin sade versiyon):

    - Local Bottom (LB):
        Son LB (sp) ve ondan önceki LB (prev_lb) var.
        Eğer prev_lb.price < sp.price -> higher low -> yükselen trend.
        => Long aç.
        Long giriş: sp.index + right_bars (pivot onay mumunda).
        Long çıkış: Sonraki LH pivotunun onaylandığı mumda.

    - Local High (LH):
        Son LH (sp) ve ondan önceki LH (prev_lh) var.
        Eğer prev_lh.price > sp.price -> lower high -> düşen trend.
        => Short aç.
        Short giriş: sp.index + right_bars.
        Short çıkış: Sonraki LB pivotunun onaylandığı mumda.

    - Long kapandığında:
        Eğer bu LH aynı zamanda prev_lh'den düşükse -> aynı mumda short açılabilir.

    - Short kapandığında:
        Eğer bu LB aynı zamanda prev_lb'den yüksekse -> aynı mumda long açılabilir.
    """

    closes = df["Close"].values
    n = len(df)

    trades: List[Trade] = []

    current_trade: Optional[Trade] = None
    last_lb: Optional[SwingPoint] = None  # en son görülen LB
    last_lh: Optional[SwingPoint] = None  # en son görülen LH

    for sp in swings:
        pivot_idx = sp.index
        signal_idx = pivot_idx + right_bars   # pivotun onaylandığı bar

        if signal_idx >= n:
            # sağdan right_bars mum istiyoruz, data yetmiyorsa bu pivotu kullanma
            continue

        # Güncellemeden önceki pivot referanslarını snapshot al
        prev_lb = last_lb
        prev_lh = last_lh

        # ======================
        # 1) VAR OLAN TRADE'İ KAPATMA
        # ======================
        if current_trade is not None:
            if current_trade.direction == "long" and sp.kind == "LH":
                # Long'u bu LH onay mumunda kapat
                current_trade.exit_index = signal_idx
                current_trade.exit_price = closes[signal_idx]
                current_trade.exit_reason = "long_exit_LH"
                trades.append(current_trade)
                current_trade = None

                # Aynı LH pivotu, önceki LH'den düşükse -> düşen trend -> aynı mumda SHORT aç
                if prev_lh is not None and prev_lh.price > sp.price:
                    current_trade = Trade(
                        direction="short",
                        entry_index=signal_idx,
                        entry_price=closes[signal_idx],
                        stop_level=0.0,   # şu an stop kullanmıyoruz
                    )

            elif current_trade.direction == "short" and sp.kind == "LB":
                # Short'u bu LB onay mumunda kapat
                current_trade.exit_index = signal_idx
                current_trade.exit_price = closes[signal_idx]
                current_trade.exit_reason = "short_exit_LB"
                trades.append(current_trade)
                current_trade = None

                # Aynı LB pivotu, önceki LB'den yüksekse -> yükselen trend -> aynı mumda LONG aç
                if prev_lb is not None and prev_lb.price < sp.price:
                    current_trade = Trade(
                        direction="long",
                        entry_index=signal_idx,
                        entry_price=closes[signal_idx],
                        stop_level=0.0,
                    )

        # ======================
        # 2) HİÇ POZ YOKKEN YENİ ENTRY ARAMA
        # ======================
        if current_trade is None:
            if sp.kind == "LB":
                # Higher Low? (önceki dip daha aşağı, yenisi daha yukarı)
                if prev_lb is not None and prev_lb.price < sp.price:
                    # LONG aç
                    current_trade = Trade(
                        direction="long",
                        entry_index=signal_idx,
                        entry_price=closes[signal_idx],
                        stop_level=0.0,
                    )

            elif sp.kind == "LH":
                # Lower High? (önceki tepe daha yukarı, yenisi daha aşağı)
                if prev_lh is not None and prev_lh.price > sp.price:
                    # SHORT aç
                    current_trade = Trade(
                        direction="short",
                        entry_index=signal_idx,
                        entry_price=closes[signal_idx],
                        stop_level=0.0,
                    )

        # ======================
        # 3) SON PİVOTLARI GÜNCELLE
        # ======================
        if sp.kind == "LB":
            last_lb = sp
        else:  # "LH"
            last_lh = sp

    return trades


# ======================
# 4) GRAFİKTE GÖSTERME
# ======================

def plot_with_trades(df: pd.DataFrame, swings: List[SwingPoint], trades: List[Trade],
                     symbol: str, interval: str):
    n = len(df)

    # Trend tüneli için maskeler
    long_mask = np.zeros(n, dtype=bool)
    short_mask = np.zeros(n, dtype=bool)

    # Entry / Exit serileri (hepsi yuvarlak olacak)
    long_entry  = np.full(n, np.nan)
    long_exit   = np.full(n, np.nan)
    short_entry = np.full(n, np.nan)
    short_exit  = np.full(n, np.nan)

    for tr in trades:
        if tr.entry_index is None or tr.exit_index is None:
            continue

        ei = tr.entry_index
        xi = tr.exit_index

        if ei < 0 or ei >= n or xi < 0 or xi >= n:
            continue

        if tr.direction == "long":
            long_mask[ei: xi + 1] = True

            # Long girişi: mumun biraz altına yuvarlak
            long_entry[ei] = df["Low"].iloc[ei] * 0.995

            # Long çıkışı: mumun biraz üstüne yuvarlak
            long_exit[xi] = df["High"].iloc[xi] * 1.005

        else:  # short
            short_mask[ei: xi + 1] = True

            # Short girişi: mumun biraz üstüne yuvarlak
            short_entry[ei] = df["High"].iloc[ei] * 1.005

            # Short çıkışı: mumun biraz altına yuvarlak
            short_exit[xi] = df["Low"].iloc[xi] * 0.995

    # === Trend tüneli bandları (NumPy array) ===
    lows  = df["Low"].values
    highs = df["High"].values

    long_band_low  = np.where(long_mask, lows,  np.nan)
    long_band_high = np.where(long_mask, highs, np.nan)

    short_band_low  = np.where(short_mask, lows,  np.nan)
    short_band_high = np.where(short_mask, highs, np.nan)

    # Pivot noktaları (LH / LB)
    lh_series = np.full(n, np.nan)
    lb_series = np.full(n, np.nan)
    for sp in swings:
        if 0 <= sp.index < n:
            if sp.kind == "LH":
                lh_series[sp.index] = sp.price   # tepe: üçgen ^
            else:
                lb_series[sp.index] = sp.price   # dip: üçgen v

    apds = []

    def has_data(arr: np.ndarray) -> bool:
        return np.isfinite(arr).any()

    # PIVOTLAR: üçgen
    if has_data(lh_series):
        apds.append(
            mpf.make_addplot(
                lh_series,
                type="scatter",
                markersize=50,
                marker="^",
                color="#ff9800",  # turuncu tepe
            )
        )
    if has_data(lb_series):
        apds.append(
            mpf.make_addplot(
                lb_series,
                type="scatter",
                markersize=50,
                marker="v",
                color="#03a9f4",  # mavi dip
            )
        )

    # LONG / SHORT GİRİŞ & ÇIKIŞLAR: hepsi yuvarlak
    if has_data(long_entry):
        apds.append(
            mpf.make_addplot(
                long_entry,
                type="scatter",
                markersize=70,
                marker="o",
                color="#00e676",  # yeşil long entry
            )
        )
    if has_data(long_exit):
        apds.append(
            mpf.make_addplot(
                long_exit,
                type="scatter",
                markersize=70,
                marker="o",
                color="#1de9b6",  # açık yeşil long exit
            )
        )
    if has_data(short_entry):
        apds.append(
            mpf.make_addplot(
                short_entry,
                type="scatter",
                markersize=70,
                marker="o",
                color="#ff5252",  # kırmızı short entry
            )
        )
    if has_data(short_exit):
        apds.append(
            mpf.make_addplot(
                short_exit,
                type="scatter",
                markersize=70,
                marker="o",
                color="#ff8a80",  # açık kırmızı short exit
            )
        )

    # Market color & stil
    mc = mpf.make_marketcolors(
        up="#26a69a",
        down="#ef5350",
        edge="inherit",
        wick="inherit",
        volume="in"
    )

    style = mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        marketcolors=mc,
        gridstyle=":",
        facecolor="#0d1117",
        figcolor="#0b0f16",
        rc={
            "axes.labelcolor": "white",
            "xtick.color": "gray",
            "ytick.color": "gray",
        }
    )

    # Trend tüneli fill_between
    fb = []
    if long_mask.any():
        fb.append(
            dict(
                y1=long_band_low,
                y2=long_band_high,
                where=long_mask,
                alpha=0.15,
                color="#00e676"  # yeşil uptrend tüneli
            )
        )
    if short_mask.any():
        fb.append(
            dict(
                y1=short_band_low,
                y2=short_band_high,
                where=short_mask,
                alpha=0.15,
                color="#ff5252"  # kırmızı downtrend tüneli
            )
        )

    mpf.plot(
        df,
        type="candle",
        style=style,
        addplot=apds if apds else None,
        volume=True,
        figratio=(16, 9),
        figscale=1.2,
        title=f"",
        ylabel="Fiyat",
        ylabel_lower="Hacim",
        tight_layout=True,
        datetime_format="%m-%d %H:%M",
        fill_between=fb if fb else None,
    )


# ======================
# 5) MAIN
# ======================

def main():
    symbol = "AVAXUSDT"
    interval = "4h"

    df = fetch_ohlc_from_api(symbol=symbol, interval=interval, limit=100)

    highs = df["High"].values
    lows = df["Low"].values

    # Pivot parametreleri
    right_bars = 1  # pivot onayı için beklenen bar sayısı (find_swings ile aynı olmalı)

    swings = find_swings(
        highs,
        lows,
        left_bars=5,
        right_bars=right_bars,
        min_distance=10,
        alt_min_distance=None,
    )

    print("Pivot sayısı:", len(swings))

    # stop_buffer_pct şu an kullanılmıyor ama imza uyumlu dursun
    stop_buffer_pct = 0.0

    trades = generate_trades(
        df,
        swings,
        stop_buffer_pct=stop_buffer_pct,
        right_bars=right_bars,
    )

    print("Trade sayısı:", len(trades))
    for t in trades:
        print(t)

    plot_with_trades(df, swings, trades, symbol, interval)


if __name__ == "__main__":
    main()
