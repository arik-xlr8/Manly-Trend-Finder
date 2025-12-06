# clean_backtest.py
#
# Basit ve TEMİZ backtest motoru
# - Binance'ten OHLC verisi çeker (sadece public, API key gerekmez)
# - 15 dakikalık mumlarda dip/tepe (argrelextrema) bulur
# - Son dip/tepe yapısına göre trend belirler (Yukselen / Dusen / Belirsiz)
# - Trend yönünde LONG / SHORT açar, pivotlara göre stop ve kar al yapar
# - Sonunda toplam PnL ve basit özet verir

import ccxt
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt


# ==============================
#  AYARLAR
# ==============================

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "AVAX/USDT",
]

TIMEFRAME = "15m"
LIMIT = 350          # mum sayısı
INITIAL_BALANCE = 1000.0
LEVERAGE = 5         # kaldıraç
TRADE_NOTIONAL = 50  # her işlem için kullanılacak USDT (kaldıraçtan önce)
PIVOT_ORDER = 10     # argrelextrema penceresi
GC_PCT = 0.005       # ~%0.5 geri çekilme/ilerleme aralığı (stop/kar al için)


# ==============================
#  VERİ ÇEKME
# ==============================

def fetch_ohlc(symbol: str, timeframe: str, limit: int = 350) -> pd.DataFrame:
    """
    Binance'ten OHLCV çeker.
    Sadece public endpoint kullanıyor, API key gerekmiyor.
    """
    exchange = ccxt.binance({"enableRateLimit": True})
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


# ==============================
#  DİP / TEPE (PIVOT) HESABI
# ==============================

def find_pivots(df: pd.DataFrame, order: int):
    """
    argrelextrema ile local dip (min) ve tepe (max) index'leri bulur.
    """
    lows = df["low"].values
    highs = df["high"].values

    min_idx = argrelextrema(lows, np.less_equal, order=order)[0]
    max_idx = argrelextrema(highs, np.greater_equal, order=order)[0]

    min_idx = sorted(min_idx.tolist())
    max_idx = sorted(max_idx.tolist())
    return min_idx, max_idx


def filter_signals(df: pd.DataFrame, min_idx, max_idx, confirm_mum: int = 2):
    """
    Dip/tepe sinyallerini temizler:
    - Aynı tip arka arkaya gelirse:
        * diplerde daha DERİN olan kalır
        * tepelerde daha YÜKSEK olan kalır
    - Son pivotu 'confirm_mum' mum ile teyit eder (repaint azaltma)
    Geriye sadece index listesi döndürür.
    """
    min_idx = set(min_idx)
    max_idx = set(max_idx)

    pivots = []
    for idx in sorted(min_idx | max_idx):
        if idx in min_idx and idx in max_idx:
            # Hem dip hem tepe ise, mumun gövdesine göre karar ver
            low = df.iloc[idx]["low"]
            high = df.iloc[idx]["high"]
            close = df.iloc[idx]["close"]
            if abs(close - low) < abs(close - high):
                p_type = "min"
                price = low
            else:
                p_type = "max"
                price = high
        elif idx in min_idx:
            p_type = "min"
            price = df.iloc[idx]["low"]
        else:
            p_type = "max"
            price = df.iloc[idx]["high"]

        pivots.append({"idx": idx, "type": p_type, "price": float(price)})

    if not pivots:
        return []

    cleaned = []
    for p in pivots:
        if not cleaned:
            cleaned.append(p)
            continue

        last = cleaned[-1]
        if p["type"] == last["type"]:
            # iki dip arka arkaya → daha düşük olan kalsın
            if p["type"] == "min":
                if p["price"] < last["price"]:
                    cleaned[-1] = p
            # iki tepe arka arkaya → daha yüksek olan kalsın
            else:
                if p["price"] > last["price"]:
                    cleaned[-1] = p
        else:
            cleaned.append(p)

    # Son pivotu onayla (confirm_mum kadar sağa bak)
    if len(cleaned) >= 1 and confirm_mum > 0:
        last_pivot = cleaned[-1]
        idx = last_pivot["idx"]
        if idx + confirm_mum >= len(df):
            cleaned = cleaned[:-1]
        else:
            if last_pivot["type"] == "max":
                future_high = df["high"].iloc[idx + 1: idx + 1 + confirm_mum].max()
                if future_high > last_pivot["price"]:
                    cleaned = cleaned[:-1]
            else:
                future_low = df["low"].iloc[idx + 1: idx + 1 + confirm_mum].min()
                if future_low < last_pivot["price"]:
                    cleaned = cleaned[:-1]

    return [p["idx"] for p in cleaned]


# ==============================
#  TREND BELİRLEME
# ==============================

def determine_trend(df: pd.DataFrame, filtered_idx, min_idx, max_idx):
    """
    Son pivotlara göre trend belirler:
    - Son iki tepe aşağı iniyorsa → Dusen
    - Son iki dip yukarı çıkıyorsa → Yukselen
    - Yoksa → Belirsiz
    """
    trend = "Belirsiz"
    trend_points = []
    trend_start_idx = None

    min_set = set(min_idx)
    max_set = set(max_idx)

    filtered_max = [i for i in filtered_idx if i in max_set]
    filtered_min = [i for i in filtered_idx if i in min_set]

    # Düşen trend: son iki tepe aşağı
    if len(filtered_max) >= 2:
        last_max = filtered_max[-1]
        prev_max = filtered_max[-2]
        if df.iloc[last_max]["high"] < df.iloc[prev_max]["high"]:
            trend = "Dusen"
            trend_start_idx = prev_max
            trend_points.append((prev_max, last_max))

    # Yükselen trend: son iki dip yukarı (eğer düşen değilse)
    if trend != "Dusen" and len(filtered_min) >= 2:
        last_min = filtered_min[-1]
        prev_min = filtered_min[-2]
        if df.iloc[last_min]["low"] > df.iloc[prev_min]["low"]:
            trend = "Yukselen"
            trend_start_idx = prev_min
            trend_points.append((prev_min, last_min))

    return trend, trend_points, trend_start_idx


# ==============================
#  BASİT POZİSYON NESNESİ
# ==============================

class Position:
    def __init__(self, side: str, entry_price: float, qty: float):
        self.side = side          # "long" veya "short"
        self.entry = entry_price
        self.qty = qty
        self.stop = None
        self.partial_taken = False
        self.added = False

    def __repr__(self):
        return f"Position({self.side}, entry={self.entry:.6f}, qty={self.qty:.4f}, stop={self.stop})"


# ==============================
#  TEK COİN İÇİN BACKTEST
# ==============================

def backtest_single_symbol(symbol: str) -> dict:
    print(f"\n=== {symbol} için veri çekiliyor... ===")
    df = fetch_ohlc(symbol, TIMEFRAME, LIMIT)

    balance = INITIAL_BALANCE
    equity_curve = []
    position = None
    trades = []

    # PnL hesabında referans için son fiyat
    last_price = df["close"].iloc[0]

    for i in range(PIVOT_ORDER * 2, len(df)):
        # Şu ana kadarki veriyi kullanarak pivot & trend analizi
        window = df.iloc[: i + 1].copy()
        price = window["close"].iloc[-1]
        timestamp = window["timestamp"].iloc[-1]

        min_idx, max_idx = find_pivots(window, PIVOT_ORDER)
        filtered_idx = filter_signals(window, min_idx, max_idx)

        trend, trend_points, trend_start_idx = determine_trend(
            window, filtered_idx, min_idx, max_idx
        )

        # -------------------------
        #  POZİSYON VARSA STOP / KAR
        # -------------------------
        if position is not None:
            # Stop kontrolü
            if position.stop is not None:
                if position.side == "long" and price <= position.stop:
                    pnl = (price - position.entry) * position.qty
                    balance += pnl
                    trades.append(
                        {
                            "time": timestamp,
                            "symbol": symbol,
                            "side": "LONG_EXIT_STOP",
                            "price": price,
                            "pnl": pnl,
                        }
                    )
                    position = None

                elif position.side == "short" and price >= position.stop:
                    pnl = (position.entry - price) * position.qty
                    balance += pnl
                    trades.append(
                        {
                            "time": timestamp,
                            "symbol": symbol,
                            "side": "SHORT_EXIT_STOP",
                            "price": price,
                            "pnl": pnl,
                        }
                    )
                    position = None

            # Pozisyon hala açıksa basit kar al / ekleme mantığı
            if position is not None:
                gc = price * GC_PCT

                # LONG için: fiyat entry'den yeterince uzaklaştıysa ve geri çekilirse yarım kapat
                if position.side == "long":
                    if (
                        not position.partial_taken
                        and price > position.entry + 3 * gc
                        and price < window["high"].iloc[-2] - gc
                    ):
                        close_qty = position.qty / 2
                        pnl = (price - position.entry) * close_qty
                        balance += pnl
                        position.qty -= close_qty
                        position.partial_taken = True
                        trades.append(
                            {
                                "time": timestamp,
                                "symbol": symbol,
                                "side": "LONG_PARTIAL",
                                "price": price,
                                "pnl": pnl,
                            }
                        )

                    # Basit trailing stop: entry'nin biraz üstüne taşı
                    if position.stop is None:
                        position.stop = position.entry - 3 * gc
                    else:
                        # stop'u sadece yukarı çek
                        new_stop = price - 4 * gc
                        if new_stop > position.stop:
                            position.stop = new_stop

                # SHORT için benzer mantık
                elif position.side == "short":
                    if (
                        not position.partial_taken
                        and price < position.entry - 3 * gc
                        and price > window["low"].iloc[-2] + gc
                    ):
                        close_qty = position.qty / 2
                        pnl = (position.entry - price) * close_qty
                        balance += pnl
                        position.qty -= close_qty
                        position.partial_taken = True
                        trades.append(
                            {
                                "time": timestamp,
                                "symbol": symbol,
                                "side": "SHORT_PARTIAL",
                                "price": price,
                                "pnl": pnl,
                            }
                        )

                    # trailing stop
                    if position.stop is None:
                        position.stop = position.entry + 3 * gc
                    else:
                        new_stop = price + 4 * gc
                        if new_stop < position.stop:
                            position.stop = new_stop

        # -------------------------
        #  YENİ POZİSYON AÇMA KARARI
        # -------------------------
        # Pozisyon yoksa trend yönünde giriş dene
        if position is None:
            notional = TRADE_NOTIONAL * LEVERAGE
            qty = notional / price

            if trend == "Yukselen":
                # Yukselen trende geçtiysek LONG aç
                position = Position("long", price, qty)
                gc = price * GC_PCT
                position.stop = price - 3 * gc
                trades.append(
                    {
                        "time": timestamp,
                        "symbol": symbol,
                        "side": "LONG_ENTER",
                        "price": price,
                        "pnl": 0.0,
                    }
                )

            elif trend == "Dusen":
                # Dusen trende geçtiysek SHORT aç
                position = Position("short", price, qty)
                gc = price * GC_PCT
                position.stop = price + 3 * gc
                trades.append(
                    {
                        "time": timestamp,
                        "symbol": symbol,
                        "side": "SHORT_ENTER",
                        "price": price,
                        "pnl": 0.0,
                    }
                )

        # Equity curve (pozisyon varsa mark-to-market, yoksa balance)
        if position is None:
            equity_curve.append(balance)
        else:
            if position.side == "long":
                floating = (price - position.entry) * position.qty
            else:
                floating = (position.entry - price) * position.qty
            equity_curve.append(balance + floating)

        last_price = price

    # Backtest sonunda pozisyon açıksa kapat
    if position is not None:
        if position.side == "long":
            pnl = (last_price - position.entry) * position.qty
        else:
            pnl = (position.entry - last_price) * position.qty
        balance += pnl
        trades.append(
            {
                "time": df["timestamp"].iloc[-1],
                "symbol": symbol,
                "side": "FORCED_EXIT",
                "price": last_price,
                "pnl": pnl,
            }
        )

    result = {
        "symbol": symbol,
        "final_balance": balance,
        "pnl": balance - INITIAL_BALANCE,
        "trades": trades,
        "equity_curve": equity_curve,
        "timestamps": df["timestamp"].iloc[-len(equity_curve):].tolist(),
    }
    return result


# ==============================
#  ANA ÇALIŞMA
# ==============================

def main():
    all_results = []

    for sym in SYMBOLS:
        res = backtest_single_symbol(sym)
        all_results.append(res)

        print(f"\n>>> {sym} Sonuç:")
        print(f"   Başlangıç Bakiye : {INITIAL_BALANCE:.2f} USDT")
        print(f"   Bitiş Bakiye     : {res['final_balance']:.2f} USDT")
        print(f"   Toplam PnL       : {res['pnl']:.2f} USDT")
        print(f"   İşlem Sayısı     : {len(res['trades'])}")

    # Örnek: İlk symbol'ün equity curve'ünü çizelim
    if all_results:
        first = all_results[0]
        plt.figure(figsize=(10, 5))
        plt.plot(first["equity_curve"])
        plt.title(f"{first['symbol']} - Equity Curve")
        plt.xlabel("Bar")
        plt.ylabel("Equity (USDT)")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    main()
