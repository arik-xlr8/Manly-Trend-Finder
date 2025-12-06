import numpy as np
import pandas as pd
import mplfinance as mpf

from swings import find_swings  # az önce yazdığımız indikatör


# 1) VERİYİ YÜKLE
# Burada kendi dosya adını yaz: örn. 'btc_1h.csv' gibi
df = pd.read_csv("veri.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)

# Emin olalım ki bu kolonlar var:
# print(df.head())


# 2) LH / LB HESAPLA (Close fiyatı üzerinden)
prices = df["Close"].values
swings = find_swings(prices, window=3, min_distance=5)

# 3) LH ve LB için ayrı seriler oluşturalım (NaN + sadece swing noktaları dolu)
lh_series = np.full(len(df), np.nan)
lb_series = np.full(len(df), np.nan)

for sp in swings:
    if sp.kind == "LH":
        lh_series[sp.index] = sp.price
    else:  # "LB"
        lb_series[sp.index] = sp.price

# 4) mplfinance için addplot'lar
apds = [
    mpf.make_addplot(
        lh_series,
        type="scatter",
        markersize=80,
        marker="^",   # yukarı ok = LH
    ),
    mpf.make_addplot(
        lb_series,
        type="scatter",
        markersize=80,
        marker="v",   # aşağı ok = LB
    ),
]

# 5) CANDLESTICK GRAFİĞİ ÇİZDİR
mpf.plot(
    df,
    type="candle",
    style="yahoo",
    addplot=apds,
    volume=False,
    title="Mum Grafiği + LH / LB Swing Noktaları",
    ylabel="Fiyat",
)
