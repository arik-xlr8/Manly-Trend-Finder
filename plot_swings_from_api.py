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
    symbol: str = "BTCUSDT",
    interval: str = "15m",
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
    Trend mantığı:

    - Uptrend başlangıcı:
        İki ardışık LB var ve son LB daha yüksekse:
            prev_lb.price < sp.price  => Higher Low
        -> Bu pivotta (sp) uptrend başlar.

    - Downtrend başlangıcı:
        İki ardışık LH var ve son LH daha düşükse:
            prev_lh.price > sp.price  => Lower High
        -> Bu pivotta (sp) downtrend başlar.

    Kurallar:

    1) POZ YOKKEN:
        - Uptrend sinyali (HL) gelirse LONG aç.
        - Downtrend sinyali (LH) gelirse SHORT aç.

    2) POZ AÇIKKEN:
        - LONG açıkken sadece DOWNtrend sinyali (LH, prev_lh > sp.price) gelirse:
            -> LONG'u kapat ve aynı mumda SHORT aç.
        - SHORT açıkken sadece UPtrend sinyali (LB, prev_lb < sp.price) gelirse:
            -> SHORT'u kapat ve aynı mumda LONG aç.

    Yani karşıt pivot geldi diye poz kapanmıyor;
    sadece trend değiştirecek kadar anlamlıysa kapanıyor.
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
        # 1) VAR OLAN TRADE'İ KAPAT / TERSİNE ÇEVİR
        # ======================
        if current_trade is not None:
            if current_trade.direction == "long":
                # LONG sadece gerçek bir DOWNtrend sinyalinde kapanır:
                # -> LH ve önceki LH daha yüksek olmalı (Lower High)
                if (
                    sp.kind == "LH"
                    and prev_lh is not None
                    and prev_lh.price > sp.price
                ):
                    exit_idx = signal_idx

                    # En az 1 bar açık kaldıysa trade'i kaydedelim
                    if exit_idx > current_trade.entry_index:
                        current_trade.exit_index = exit_idx
                        current_trade.exit_price = closes[exit_idx]
                        current_trade.exit_reason = "long_exit_downtrend_LH"
                        trades.append(current_trade)
                    # entry == exit ise trade'i çöpe atıyoruz
                    current_trade = None

                    # Aynı anda, bu DOWNtrend sinyaliyle SHORT açıyoruz
                    current_trade = Trade(
                        direction="short",
                        entry_index=signal_idx,
                        entry_price=closes[signal_idx],
                        stop_level=0.0,
                    )

            elif current_trade.direction == "short":
                # SHORT sadece gerçek bir UPtrend sinyalinde kapanır:
                # -> LB ve önceki LB daha düşük olmalı (Higher Low)
                if (
                    sp.kind == "LB"
                    and prev_lb is not None
                    and prev_lb.price < sp.price
                ):
                    exit_idx = signal_idx

                    if exit_idx > current_trade.entry_index:
                        current_trade.exit_index = exit_idx
                        current_trade.exit_price = closes[exit_idx]
                        current_trade.exit_reason = "short_exit_uptrend_LB"
                        trades.append(current_trade)
                    current_trade = None

                    # Aynı anda, bu UPtrend sinyaliyle LONG açıyoruz
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
                # Higher Low? (önceki dip daha aşağı, yenisi daha yukarı) -> uptrend başlıyor
                if prev_lb is not None and prev_lb.price < sp.price:
                    current_trade = Trade(
                        direction="long",
                        entry_index=signal_idx,
                        entry_price=closes[signal_idx],
                        stop_level=0.0,
                    )

            elif sp.kind == "LH":
                # Lower High? (önceki tepe daha yukarı, yenisi daha aşağı) -> downtrend başlıyor
                if prev_lh is not None and prev_lh.price > sp.price:
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

    # ======================
    # 4) HÂLÂ AÇIK OLAN TRADE VARSA, EXIT'SİZ OLARAK EKLE
    # ======================
    if current_trade is not None:
        trades.append(current_trade)

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
        if tr.entry_index is None:
            continue

        ei = tr.entry_index

        if ei < 0 or ei >= n:
            continue

        # Çıkış var mı?
        if tr.exit_index is None:
            has_exit = False
            xi = n - 1  # açık trade ise tüneli grafiğin son mumuna kadar götür
        else:
            has_exit = True
            xi = tr.exit_index
            if xi < 0 or xi >= n:
                continue

        # Güvenlik: exit <= entry ise tünel boyama (zaten generate_trades bunları filtreliyor ama dursun)
        if xi <= ei:
            # sadece entry noktası çizilsin, tünel yok
            if tr.direction == "long":
                long_entry[ei] = df["Low"].iloc[ei] * 0.995
            else:
                short_entry[ei] = df["High"].iloc[ei] * 1.005
            continue

        # Tünel aralığı: ENTRY'den EXIT'e kadar (ikisinin de dahil)
        start = ei
        end = xi

        if start <= end:
            if tr.direction == "long":
                # Long trend tüneli
                long_mask[start: end + 1] = True

                # Long girişi: mumun biraz altına yuvarlak (entry barında)
                long_entry[ei] = df["Low"].iloc[ei] * 0.995

                # Long çıkışı varsa: mumun biraz üstüne yuvarlak (exit barında)
                if has_exit:
                    long_exit[xi] = df["High"].iloc[xi] * 1.005

            else:  # short
                # Short trend tüneli
                short_mask[start: end + 1] = True

                # Short girişi: mumun biraz üstüne yuvarlak (entry barında)
                short_entry[ei] = df["High"].iloc[ei] * 1.005

                # Short çıkışı varsa: mumun biraz altına yuvarlak (exit barında)
                if has_exit:
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

    plot_kwargs = dict(
        type="candle",
        style=style,
        addplot=apds if apds else None,
        volume=True,
        figratio=(16, 9),
        figscale=1.2,
        title="",
        ylabel="Fiyat",
        ylabel_lower="Hacim",
        tight_layout=True,
        datetime_format="%m-%d %H:%M",
    )

    if fb:  # boş değilse ekle
        plot_kwargs["fill_between"] = fb

    mpf.plot(df, **plot_kwargs)



# ======================
# 5) MAIN
# ======================

def main():
    symbol = "BTCUSDT"
    interval = "15m"

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
