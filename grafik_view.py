import ccxt
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import mplfinance as mpf

# ==========================
#  EXCHANGE AYARI (PUBLIC)
#  - API KEY YOK, SADECE PUBLIC OHLCV
# ==========================

exchange = ccxt.binance({
    "options": {
        "defaultType": "future"   # İstersen 'spot' da yapabilirsin
    },
    "enableRateLimit": True
})

# ==========================
#  OHLC DATA FONKSİYONU
# ==========================

def fetch_candlestick_data(symbol, timeframe, limit=300):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df

# ==========================
#  PİVOT BULMA
# ==========================

def find_pivots(df, order=10):
    df = df.copy()
    df["min"] = df.iloc[argrelextrema(df["low"].values, np.less_equal, order=order)[0]]["low"]
    df["max"] = df.iloc[argrelextrema(df["high"].values, np.greater_equal, order=order)[0]]["high"]
    return df

# ==========================
#  ANA ÇALIŞMA
# ==========================

symbol = "RESOLV/USDT"   # İstediğin çifti buradan değiştirebilirsin
timeframe = "15m"

print(f"{symbol} verisi çekiliyor...")
df = fetch_candlestick_data(symbol, timeframe, 200)
df = find_pivots(df, order=10)

# Tam uzunlukta seri veriyoruz (NaN'ler sorun değil)
min_series = df["min"]
max_series = df["max"]

apds = []

if not min_series.isna().all():
    apds.append(
        mpf.make_addplot(
            min_series,
            type="scatter",
            marker="v",    # aşağı bakan üçgen
            markersize=60
        )
    )

if not max_series.isna().all():
    apds.append(
        mpf.make_addplot(
            max_series,
            type="scatter",
            marker="^",    # yukarı bakan üçgen
            markersize=60
        )
    )

mpf.plot(
    df,
    type="candle",
    volume=True,
    addplot=apds,
    title=f"{symbol} - {timeframe} Pivot Görünümü"
)
