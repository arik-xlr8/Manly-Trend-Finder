import requests
import numpy as np
import pandas as pd
import mplfinance as mpf

from swings import find_swings, SwingPoint   # artık HIGH/LOW kullanan versiyon


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


def main():
    symbol = "BTCUSDT"
    interval = "15m"

    df = fetch_ohlc_from_api(symbol=symbol, interval=interval, limit=400)

    print("Bar sayısı:", len(df))

    # === YENİ PIVOT ALGOSU ===
    highs = df["High"].values
    lows = df["Low"].values

    swings = find_swings(
        highs,
        lows,
        left_bars=5,
        right_bars=3,
        min_distance=14,   # LH-LH / LB-LB için güçlü seyreltme
        alt_min_distance=2 # LH'dan hemen sonra gelen LB için neredeyse serbest
    )

    print("Swing sayısı:", len(swings))

    # Grafik için seriler hazırlıyoruz
    lh_series = np.full(len(df), np.nan)
    lb_series = np.full(len(df), np.nan)

    for sp in swings:
        if 0 <= sp.index < len(df):
            if sp.kind == "LH":
                lh_series[sp.index] = sp.price
            else:
                lb_series[sp.index] = sp.price

    # === ADDPLOT listesi ===
    apds = []

    if not np.all(np.isnan(lh_series)):
        apds.append(
            mpf.make_addplot(
                lh_series,
                type="scatter",
                markersize=80,
                marker="^",
                color="#ff4d4d",  # kırmızı tepe
            )
        )

    if not np.all(np.isnan(lb_series)):
        apds.append(
            mpf.make_addplot(
                lb_series,
                type="scatter",
                markersize=80,
                marker="v",
                color="#00e676",  # yeşil dip
            )
        )

    # === ŞIK TEMA ===
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

    mpf.plot(
        df,
        type="candle",
        style=style,
        addplot=apds,
        volume=True,
        figratio=(16, 9),
        figscale=1.25,
        title=f"{symbol} {interval} - Pivot LH/LB",
        ylabel="Fiyat",
        ylabel_lower="Hacim",
        tight_layout=True,
        datetime_format="%m-%d %H:%M",
    )


if __name__ == "__main__":
    main()
