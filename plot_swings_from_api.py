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
# 3) STRATEJİ LOJİĞİ
# ======================

def generate_trades(
    df: pd.DataFrame,
    swings: List[SwingPoint],
    stop_buffer_pct: float = 0.10,
    right_bars: int = 2,   # pivot onayı için kaç bar bekleniyor (find_swings ile aynı olmalı)
) -> List[Trade]:
    """
    Strateji:
    - Son pivot LB ise ve önceki LB'den yüksekse -> yükselen trend başlangıcı -> LONG aç
      (giriş: LB pivotunun onaylandığı mum = pivot_index + right_bars)
    - LONG exit:
        - Yeni bir LH pivotu onaylandığında (take profit)
        - Veya yeni bir LB pivotu, önceki dibin %X altına geldiyse (stop) ve bu da onaylandıysa

    - İşlem kapandıktan sonra:
        - Son pivot LH ise ve önceki LH daha yüksekse -> düşen trend
          -> bir SONRAKİ LB pivotu onaylandığında SHORT aç
    - SHORT exit:
        - Yeni bir LB pivotu onaylandığında (take profit)
        - Veya yeni bir LH pivotu, önceki tepenin %X üstüne geldiyse (stop) ve bu da onaylandıysa
    """
    closes = df["Close"].values
    n = len(df)

    trades: List[Trade] = []

    last_lb: Optional[SwingPoint] = None
    last_lh: Optional[SwingPoint] = None

    current_trade: Optional[Trade] = None
    pending_short_after_downtrend: Optional[float] = None  # stop referansı: önceki tepe fiyatı

    for sp in swings:
        pivot_idx = sp.index
        signal_idx = pivot_idx + right_bars  # pivotun onaylandığı bar

        # Veri dışına taştıysa bu pivotu kullanamayız
        if signal_idx >= n:
            continue

        # === 1) Aktif trade varsa önce EXIT kurallarını çalıştır ===
        if current_trade is not None:
            if current_trade.direction == "long":
                # 1.a) Yeni LH pivot onaylandı -> take profit
                if sp.kind == "LH" and current_trade.exit_index is None:
                    current_trade.exit_index = signal_idx
                    current_trade.exit_price = closes[signal_idx]
                    current_trade.exit_reason = "long_take_profit_LH"
                    trades.append(current_trade)
                    current_trade = None

                # 1.b) Yeni LB pivot onaylandı ve STOP seviyesinin altına indik
                elif sp.kind == "LB" and last_lb is not None and current_trade.exit_index is None:
                    if sp.price <= current_trade.stop_level:
                        current_trade.exit_index = signal_idx
                        current_trade.exit_price = closes[signal_idx]
                        current_trade.exit_reason = "long_stop_prev_low_break"
                        trades.append(current_trade)
                        current_trade = None

            elif current_trade.direction == "short":
                # 1.c) Yeni LB pivot onaylandı -> short take profit
                if sp.kind == "LB" and current_trade.exit_index is None:
                    current_trade.exit_index = signal_idx
                    current_trade.exit_price = closes[signal_idx]
                    current_trade.exit_reason = "short_take_profit_LB"
                    trades.append(current_trade)
                    current_trade = None

                # 1.d) Yeni LH pivot onaylandı ve STOP seviyesinin üstüne çıktık
                elif sp.kind == "LH" and last_lh is not None and current_trade.exit_index is None:
                    if sp.price >= current_trade.stop_level:
                        current_trade.exit_index = signal_idx
                        current_trade.exit_price = closes[signal_idx]
                        current_trade.exit_reason = "short_stop_prev_high_break"
                        trades.append(current_trade)
                        current_trade = None

        # === 2) Trade yoksa / yeni trade arıyorsak ENTRY kurallarını çalıştır ===
        if current_trade is None:
            if sp.kind == "LB":
                opened = False

                # 2.a) LB pivot onaylandıysa ve önceki LB'den yüksekse -> LONG
                if last_lb is not None and sp.price > last_lb.price:
                    entry_price = closes[signal_idx]
                    prev_low_for_stop = last_lb.price
                    stop_level = prev_low_for_stop * (1.0 - stop_buffer_pct)

                    current_trade = Trade(
                        direction="long",
                        entry_index=signal_idx,   # giriş pivot değil, ONAY barında
                        entry_price=entry_price,
                        stop_level=stop_level
                    )
                    pending_short_after_downtrend = None  # düşen trend planını sıfırla
                    opened = True

                # 2.b) Daha önce düşen trend tespit edildiyse ve long açmadıysak -> SHORT entry
                if (not opened) and pending_short_after_downtrend is not None:
                    entry_price = closes[signal_idx]
                    prev_high_for_stop = pending_short_after_downtrend
                    stop_level = prev_high_for_stop * (1.0 + stop_buffer_pct)

                    current_trade = Trade(
                        direction="short",
                        entry_index=signal_idx,
                        entry_price=entry_price,
                        stop_level=stop_level
                    )
                    pending_short_after_downtrend = None

            elif sp.kind == "LH":
                # 2.c) LH pivot onaylandı ve önceki LH daha yüksekse -> düşen trend tespiti
                if last_lh is not None and sp.price < last_lh.price:
                    # Bir sonraki LB onaylandığında SHORT açmak için hazır ol
                    pending_short_after_downtrend = last_lh.price  # stop referansı

        # === 3) Son pivot referanslarını güncelle ===
        if sp.kind == "LB":
            last_lb = sp
        else:
            last_lh = sp

    return trades


# ======================
# 4) GRAFİKTE GÖSTERME
# ======================

def plot_with_trades(df: pd.DataFrame, swings: List[SwingPoint], trades: List[Trade],
                     symbol: str, interval: str):
    n = len(df)

    # Trend tünelleri için maske
    long_mask = np.zeros(n, dtype=bool)
    short_mask = np.zeros(n, dtype=bool)

    # Entry / TP / Stop serileri
    long_entry = np.full(n, np.nan)
    long_tp    = np.full(n, np.nan)
    long_stop  = np.full(n, np.nan)

    short_entry = np.full(n, np.nan)
    short_tp    = np.full(n, np.nan)
    short_stop  = np.full(n, np.nan)

    for tr in trades:
        if tr.entry_index is None or tr.exit_index is None:
            continue

        ei = tr.entry_index
        xi = tr.exit_index

        if ei < 0 or ei >= n or xi < 0 or xi >= n:
            continue

        if tr.direction == "long":
            # Trend tüneli
            long_mask[ei: xi + 1] = True

            # Entry: yuvarlak, mumun biraz altı
            long_entry[ei] = df["Low"].iloc[ei] * 0.995

            # Exit: TP mi stop mu?
            if tr.exit_reason and "stop" in tr.exit_reason:
                # Stop: kare, mumun biraz altına
                long_stop[xi] = df["Low"].iloc[xi] * 0.995
            else:
                # TP: üçgen, mumun biraz üstü
                long_tp[xi] = df["High"].iloc[xi] * 1.005

        else:  # short
            short_mask[ei: xi + 1] = True

            # Entry: yuvarlak, mumun biraz üstü
            short_entry[ei] = df["High"].iloc[ei] * 1.005

            if tr.exit_reason and "stop" in tr.exit_reason:
                # Stop: kare, mumun biraz üstü
                short_stop[xi] = df["High"].iloc[xi] * 1.005
            else:
                # TP: üçgen, mumun biraz altı
                short_tp[xi] = df["Low"].iloc[xi] * 0.995

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
        # En az bir tane NaN olmayan (gerçek) değer var mı?
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

    # ENTRIES: yuvarlak
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

    # TAKE PROFITLER: üçgen
    if has_data(long_tp):
        apds.append(
            mpf.make_addplot(
                long_tp,
                type="scatter",
                markersize=80,
                marker="^",
                color="#e91e63",  # pembe long TP
            )
        )
    if has_data(short_tp):
        apds.append(
            mpf.make_addplot(
                short_tp,
                type="scatter",
                markersize=80,
                marker="v",
                color="#ff40ef",  # açık mavi short TP
            )
        )

    # STOPLAR: kare
    if has_data(long_stop):
        apds.append(
            mpf.make_addplot(
                long_stop,
                type="scatter",
                markersize=80,
                marker="s",          # KARE
                color="#ff1744",     # kırmızımsı long stop
            )
        )
    if has_data(short_stop):
        apds.append(
            mpf.make_addplot(
                short_stop,
                type="scatter",
                markersize=80,
                marker="s",          # KARE
                color="#d500f9",     # morumsu short stop
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

    # Trend tüneli fill_between → sadece gerçekten tünel varsa ekle
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
    symbol = "BTCUSDT"
    interval = "15m"

    df = fetch_ohlc_from_api(symbol=symbol, interval=interval, limit=500)

    highs = df["High"].values
    lows = df["Low"].values

    # Pivot parametreleri
    right_bars = 2  # pivot onayı için beklenen bar sayısı (find_swings ile aynı olmalı)

    swings = find_swings(
        highs,
        lows,
        left_bars=5,
        right_bars=right_bars,
        min_distance=8,
        alt_min_distance=None,
    )

    print("Pivot sayısı:", len(swings))

    # STOP buffer parametresi (örn. 0.10 = %10)
    # Stopları daha sık görmek için testte küçük tutabilirsin.
    stop_buffer_pct = 0.02

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
