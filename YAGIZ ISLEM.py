import ccxt
import pandas as pd
import numpy as np
import time
import sys
import math
from datetime import datetime
from scipy.signal import argrelextrema
from binance.client import Client

# ============================================================
#  GLOBAL AYARLAR
# ============================================================

DRY_RUN = False
USE_CCXT = True

leverage = 10
islemeGirecekPara = input("dolar: ")

symbols = [
    "DMC/USDT", "TOSHI/USDT", "MAV/USDT", "AIA/USDT", "XVG/USDT"
]

# ============================================================
#  GLOBAL STATE DICTIONARY’LER (HER COİN İÇİN)
# ============================================================

symbols_data = {}                     # Her coin için tüm durum

# FIB seviyeleri
fib_extension_prices = {}             # {sym: {'Yukselen': [...], 'Dusen': [...]} }
previous_fib_extension_prices = {}

# Sinyal fiyatları (pivotlar)
son_min_idx_price = {}                # son min sinyal fiyatı
son_max_idx_price = {}                # son max sinyal fiyatı
onceki_min_idx_price = {}             # önceki min sinyal fiyatı
onceki_max_idx_price = {}             # önceki max sinyal fiyatı
sabitlemis_min_sinyal = {}            # kilitlenmiş min
sabitlemis_max_sinyal = {}            # kilitlenmiş max

# FIB aralıkları
ust_deger = {}                        # anlık üst FIB seviyesi
alt_deger = {}                        # anlık alt FIB seviyesi
ust_deger1 = {}                       # trend içi birinci kademe seviye
alt_deger1 = {}
acilis = {}                           # ilk fib aralığı ayarı yapıldı mı? (0/1)

# Aralık değerleri
gcaralik = {}                         # kar aralığı (%0.5 vb)
stoparalik = {}                       # stop aralığı (%0.3 vb)

# Stop değerleri
long_stop_level = {}
short_stop_level = {}



def get_current_position(symbol: str):
    try:
        balance = exchange.fetch_balance()
        positions = (
            balance['info']['positions'] 
            if 'info' in balance and 'positions' in balance['info'] 
            else balance.get('positions', [])
        )

        clean_symbol = symbol.replace("/", "")
        current_positions = [
            p for p in positions 
            if float(p.get("positionAmt", 0)) != 0 
            and p.get("symbol") == clean_symbol
        ]

        if not current_positions:
            return None

        pos = current_positions[-1]
        amt = float(pos["positionAmt"])
        side = "LONG" if amt > 0 else "SHORT"
        entry = float(pos["entryPrice"])
        profit = float(pos.get("unrealizedProfit", 0))
        isolated_wallet = float(pos.get("isolatedWallet", 0))

        print(f"[POZİSYON] {symbol} {side} | entry={entry} | miktar={amt} | kar={profit} | cüzdan={isolated_wallet}")
        return {
            "symbol": symbol,
            "side": side,
            "amount": amt,
            "entry": entry,
            "profit": profit,
            "wallet": isolated_wallet
        }

    except Exception as e:
        print(f"[POZİSYON HATASI] {symbol} -> {e}")
        return None



# ============================================================
#  POZİSYON GÜVENLİ OKUMA (HATA VERMEYEN)
# ============================================================

def get_position_qty(symbol):
    """
    Binance futures üzerindeki GERÇEK pozisyon miktarını döndürür.
    Pozisyon yoksa 0.
    """
    pos = get_current_position(symbol)
    if pos:
        try:
            return abs(float(pos.get("amount", 0.0)))
        except Exception:
            return 0.0
    return 0.0


miktar_precision = {}   # sembol bazlı precision bilgisi

def get_market_lot_info(symbol: str):
    precision = None
    min_qty = None
    step = None
    try:
        m = exchange.market(symbol)  # CCXT'de markets preload edilmiş olmalı
        precision = (m.get('precision') or {}).get('amount')
        filters = (m.get('info') or {}).get('filters', [])
        for f in filters:
            if f.get('filterType') == 'LOT_SIZE':
                step = float(f.get('stepSize', '0'))
                min_qty = float(f.get('minQty', '0'))
                break
    except Exception:
        pass

    if precision is None:
        precision = (miktar_precision.get(symbol) if 'miktar_precision' in globals() else 8)

    return int(precision), (float(min_qty) if min_qty else 0.0), (float(step) if step else 0.0)

def floor_to_step(value: float, step: float) -> float:
    if not step or step <= 0:
        return value
    return math.floor(value / step) * step

def floor_to_precision(value: float, precision: int) -> float:
    factor = 10 ** int(precision)
    return math.floor(value * factor) / factor

def normalize_qty(symbol: str, raw_qty: float):
    raw = abs(float(raw_qty))
    precision, min_qty, step = get_market_lot_info(symbol)

    adj = raw
    if step and step > 0:
        adj = floor_to_step(adj, step)
    else:
        adj = floor_to_precision(adj, precision)

    if min_qty and adj < min_qty:
        print(f"[UYARI] {symbol} miktar {adj} minQty altinda ({min_qty}). 0'a düşürüldü.")
        adj = 0.0

    if adj > 0:
        adj = round(adj, 6)

    return adj, precision

    
# ============================================================
#  OHLC VERİ ÇEKME
# ============================================================

def fetch_candlestick_data(symbol, timeframe, limit=350):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


# ============================================================
#  SİNYAL FİLTRELEME (dip/tepe)
# ============================================================

def filter_signals(df, min_max_idx, min_idx, max_idx, confirm_mum=2):
    filtered_idx = []

    for i, current_idx in enumerate(min_max_idx):
        # --- MAX (TEPE) SİNYALİ ---
        if current_idx in max_idx:
            # ⿡ Onay mumlarıyla teyit et (mevcut kodun mantığı)
            if current_idx + confirm_mum < len(df):
                highs = [df.iloc[current_idx + j]['high'] for j in range(1, confirm_mum + 1)]
                if df.iloc[current_idx]['high'] < max(highs):
                    continue  # sonraki mumlar daha yüksekse sinyal zayıf

            # ⿢ Önceki max sinyaliyle karşılaştır (güçlü olanı koru)
            if filtered_idx:
                last_idx = filtered_idx[-1]
                if last_idx in max_idx and df.iloc[current_idx]['high'] <= df.iloc[last_idx]['high']:
                    continue  # mevcut daha zayıfsa atla
                elif last_idx in max_idx and df.iloc[current_idx]['high'] > df.iloc[last_idx]['high']:
                    filtered_idx[-1] = current_idx  # yeni tepe daha yüksekse eskiyi değiştir
                    continue

            # ⿣ 🔥 Yeni onay filtresi (geri çekilme kontrolü)
            if current_idx + 1 < len(df):
                current_close = df.iloc[current_idx]['close']
                next_close = df.iloc[current_idx + 1]['close']
                # fiyat %0.5'ten fazla düşmediyse sinyal onaylanmaz
                if (current_close - next_close) / current_close < 0.005:
                    continue

            filtered_idx.append(current_idx)

        # --- MIN (DİP) SİNYALİ ---
        elif current_idx in min_idx:
            # ⿡ Onay mumlarıyla teyit et (mevcut kodun mantığı)
            if current_idx + confirm_mum < len(df):
                lows = [df.iloc[current_idx + j]['low'] for j in range(1, confirm_mum + 1)]
                if df.iloc[current_idx]['low'] > min(lows):
                    continue  # sonraki mumlar daha düşükse sinyal zayıf

            # ⿢ Önceki min sinyaliyle karşılaştır (güçlü olanı koru)
            if filtered_idx:
                last_idx = filtered_idx[-1]
                if last_idx in min_idx and df.iloc[current_idx]['low'] >= df.iloc[last_idx]['low']:
                    continue  # mevcut daha zayıfsa atla
                elif last_idx in min_idx and df.iloc[current_idx]['low'] < df.iloc[last_idx]['low']:
                    filtered_idx[-1] = current_idx  # yeni dip daha derinse eskiyi değiştir
                    continue

            # ⿣ 🔥 Yeni onay filtresi (yükseliş kontrolü)
            if current_idx + 1 < len(df):
                current_close = df.iloc[current_idx]['close']
                next_close = df.iloc[current_idx + 1]['close']
                # fiyat %0.5'ten fazla yükselmediyse sinyal onaylanmaz
                if (next_close - current_close) / current_close < 0.005:
                    continue

            filtered_idx.append(current_idx)

    return filtered_idx



def balance_signals(filtered_idx, min_idx, max_idx):
    min_count = sum(1 for idx in filtered_idx if idx in min_idx)
    max_count = sum(1 for idx in filtered_idx if idx in max_idx)
    while min_count != max_count:
        if min_count > max_count:
            for idx in reversed(filtered_idx):
                if idx in min_idx:
                    filtered_idx.remove(idx)
                    min_count -= 1
                    break
        else:
            for idx in reversed(filtered_idx):
                if idx in max_idx:
                    filtered_idx.remove(idx)
                    max_count -= 1
                    break
    return filtered_idx


def determine_last_trend(filtered_idx, candlestick_data, min_idx, max_idx):
    trend = 'Belirsiz'
    trend_points = []
    trend_start_idx = None
    trend_start_price = None

    # filtered_idx içindeki max_idx ve min_idx sinyallerini belirle
    filtered_max_idx = [idx for idx in filtered_idx if idx in max_idx]
    filtered_min_idx = [idx for idx in filtered_idx if idx in min_idx]

    # -----------------------------
    # DÜŞEN TREND BELİRLEME
    # -----------------------------
    if len(filtered_max_idx) >= 2:
        last_max = filtered_max_idx[-1]
        second_last_max = filtered_max_idx[-2]

        # Eğer son max önceki max'tan daha düşükse => düşen trend
        if candlestick_data.iloc[last_max]['high'] < candlestick_data.iloc[second_last_max]['high']:
            trend = 'Dusen'
            # Trend başlangıcı en yüksek max’tan sonra gelen düşük max olmalı
            trend_start_idx = last_max
            trend_points.append((second_last_max, last_max))

    # -----------------------------
    # DÜŞEN TREND GEÇMİŞE GENİŞLETME (DÜZELTİLDİ)
    # -----------------------------
    if trend == 'Dusen' and trend_start_idx is not None:
        try:
            current_idx = filtered_max_idx.index(trend_start_idx)
        except ValueError:
            current_idx = -1

        for i in range(current_idx - 1, -1, -1):
            previous_max = filtered_max_idx[i]
            current_max = filtered_max_idx[i - 1]
            # Düşen trendde tepe fiyatları azalmalı
            if candlestick_data.iloc[current_max]['high'] < candlestick_data.iloc[previous_max]['high']:
                trend_start_idx = current_max
                trend_points.insert(0, (previous_max, current_max))
            else:
                break


    # -----------------------------
    # YÜKSELEN TREND BELİRLEME
    # -----------------------------
    if len(filtered_min_idx) >= 2 and trend != 'Dusen':
        last_min = filtered_min_idx[-1]
        second_last_min = filtered_min_idx[-2]

        if candlestick_data.iloc[last_min]['low'] > candlestick_data.iloc[second_last_min]['low']:
            trend = 'Yukselen'
            trend_start_idx = last_min
            trend_points.append((second_last_min, last_min))

    # -----------------------------
    # YÜKSELEN TREND GEÇMİŞE GENİŞLETME
    # -----------------------------
    if trend == 'Yukselen' and trend_start_idx is not None:
        try:
            current_idx = filtered_min_idx.index(trend_start_idx)
        except ValueError:
            current_idx = -1

        for i in range(current_idx - 1, -1, -1):
            previous_min = filtered_min_idx[i]
            current_min = filtered_min_idx[i + 1]
            if candlestick_data.iloc[current_min]['low'] > candlestick_data.iloc[previous_min]['low']:
                trend_start_idx = current_min
                trend_points.insert(0, (previous_min, current_min))
            else:
                break

    
    # -----------------------------
    # TREND BOZULMA KONTROLLERİ
    # -----------------------------
    if trend == 'Yukselen':
        if len(filtered_max_idx) >= 1 and len(filtered_min_idx) >= 2:
            last_max_idx = filtered_max_idx[-1]
            previous_mins = [idx for idx in filtered_min_idx if idx < last_max_idx]
            if previous_mins:
                last_min_before_max_idx = previous_mins[-1]
                min_low = candlestick_data.iloc[last_min_before_max_idx]['low']
                if candlestick_data.iloc[-1]['close'] < min_low:
                    trend = 'Belirsiz'
                    trend_points = []
                    trend_start_idx = None

    if trend == 'Dusen':
        if len(filtered_min_idx) >= 1 and len(filtered_max_idx) >= 2:
            last_min_idx = filtered_min_idx[-1]
            previous_maxs = [idx for idx in filtered_max_idx if idx < last_min_idx]
            if previous_maxs:
                last_max_before_min_idx = previous_maxs[-1]
                max_high = candlestick_data.iloc[last_max_before_min_idx]['high']
                if candlestick_data.iloc[-1]['close'] > max_high:
                    trend = 'Belirsiz'
                    trend_points = []
                    trend_start_idx = None

    # -----------------------------
    # TREND BAŞLANGIÇ FİYATI (EKLENDİ)
    # -----------------------------
    if trend_start_idx is not None:
        if trend == 'Yukselen':
            trend_start_price = candlestick_data.iloc[trend_start_idx]['low']
        elif trend == 'Dusen':
            trend_start_price = candlestick_data.iloc[trend_start_idx]['high']

    return trend, trend_points, trend_start_idx, trend_start_price


# ============================================================
#  FIBONACCI EXTENSION
# ============================================================

# Fibonacci Extension seviyelerini hesapla
def calculate_fib_extension(start_price, prev_min_price, prev_max_price, trend_direction):
    if trend_direction == 'Yukselen':
        # Yukarı trend için Fibonacci Extension hesaplama
        fib_levels = [-2.618, -1.618, -1, -1.5, 0, 0.5, 1.5, 2.5, 3.6, 4.5, 5.5, 6.5, 7.5, 8.5, 9.8, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5]
        price_range = abs(prev_max_price - prev_min_price)
        fib_prices = [start_price + level * price_range for level in fib_levels]
    elif trend_direction == 'Dusen':
        # Aşağı trend için Fibonacci Extension hesaplama
        fib_levels = [ 3.5, 2.5,  1.5, 1, 0, -0.5, -1.5, -2.5, -3.6, -4.5, -5.5, -6.5, -7.5, -8.5, -9.8, -10.5, -11.5, -12.5, -13.5, -14.5, -15.5, -16.5]
        price_range = abs(prev_max_price - prev_min_price)
        fib_prices = [start_price + level * price_range for level in fib_levels]
    else:
        return []

    return fib_prices


# ============================================================
#  API / EXCHANGE BAŞLANGICI
# ============================================================

api_key = "XNYtJ2hI8KMIH1RujaDKfNvRN90IaASV0lxGDtpBThsL8fWc97d67IxaF3xCzcWc"
api_secret = "TgExmWB33xsF2mg3T08RcaFp1500ZUp0K4ZdEY9g2nOIsQkRt3JYwFb32SQt2Y3z"

# Binance resmi client
client = Client(api_key, api_secret)

# CCXT Binance Futures
exchange = ccxt.binance({
    "apiKey": api_key,
    "secret": api_secret,
    "options": {
        "defaultType": "future"
    },
    "enableRateLimit": True
})


# ============================================================
#  YARDIMCI: GÜVENLİ FLOAT
# ============================================================

def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


# ============================================================
#  STOP LOSS GÜNCELLEYİCİ
# ============================================================

def update_stop_loss(side, fiyat, son_min, son_max, onceki_min, onceki_max, mevcut_stop, gc):
    """
    side: 'long' / 'short'
    fiyat: anlık fiyat
    son_min / son_max: son pivotlar
    onceki_min / onceki_max: önceki pivotlar
    mevcut_stop: daha önce set edilen stop
    gc: gcaralik[sym] (yüzde değil, oran: 0.005 gibi)
    """
    stop_tetik = False
    gc = safe_float(gc, 0.0)
    mevcut = safe_float(mevcut_stop, 0.0)

    if side == "long":
        # Long için stop her zaman AŞAĞIDA, sadece yukarı çekilir
        aday_stop = 0.0

        # Öncelik: önceki min > 0 ise
        if onceki_min > 0:
            aday_stop = onceki_min
        elif son_min > 0:
            aday_stop = son_min

        # Fiyat önceki max'ı kırdıysa stop'u son min'e çek
        if onceki_max > 0 and son_min > 0 and fiyat > onceki_max:
            aday_stop = son_min

        if aday_stop > 0:
            yeni_stop = max(mevcut, aday_stop)
        else:
            yeni_stop = mevcut

        # Stop tetikleme: fiyat stop'un gc altına düşerse
        if yeni_stop > 0 and fiyat <= yeni_stop * (1 - gc):
            stop_tetik = True

        return yeni_stop, stop_tetik

    elif side == "short":
        # Short için stop her zaman YUKARIDA, sadece aşağı çekilir
        aday_stop = 0.0

        if onceki_max > 0:
            aday_stop = onceki_max
        elif son_max > 0:
            aday_stop = son_max

        # Fiyat önceki min altına kırdıysa stop'u son max'a çek
        if onceki_min > 0 and son_max > 0 and fiyat < onceki_min:
            aday_stop = son_max

        if aday_stop > 0:
            if mevcut == 0:
                yeni_stop = aday_stop
            else:
                yeni_stop = min(mevcut, aday_stop)
        else:
            yeni_stop = mevcut

        # Stop tetikleme: fiyat stop'un gc üstüne çıkarsa
        if yeni_stop > 0 and fiyat >= yeni_stop * (1 + gc):
            stop_tetik = True

        return yeni_stop, stop_tetik

    return mevcut_stop, False


# ============================================================
#  EMİR GÖNDERME (GERÇEK MOD )
# ============================================================

def place_order(side: str, symbol: str, qty: float, reduce_only: bool = False, dry_run: bool = DRY_RUN):
    """
    side: 'BUY' veya 'SELL'
    symbol: 'BTC/USDT' formatı
    qty: pozitif miktar
    reduce_only: True ise sadece pozisyon kapatır
    dry_run: True -> sadece yazdır, emir gönderme
    """
    try:
        qty_adj, _ = normalize_qty(symbol, qty)
        if qty_adj <= 0:
            print(f"[HATA] Geçersiz miktar: {qty} -> normalize: {qty_adj}")
            return None

        params = {"reduceOnly": reduce_only} if reduce_only else {}

        if dry_run:
            print(f"[DRY-RUN] {side} {symbol} miktar={qty_adj} reduceOnly={reduce_only}")
            return {"status": "dry-run", "side": side, "symbol": symbol, "qty": qty_adj}

        # Gerçek emir
        if side.upper() == "BUY":
            order = exchange.create_market_buy_order(symbol, qty_adj, params)
        else:
            order = exchange.create_market_sell_order(symbol, qty_adj, params)

        print(f"[EMİR] {side} {symbol} qty={qty_adj} gönderildi (reduceOnly={reduce_only})")
        return order

    except Exception as e:
        print(f"[EMİR HATASI] {side} {symbol} qty={qty} -> {e}")
        return None

# ============================================================
#  KULLANICI DEĞERLERİNİ FLOAT'A ÇEVİR
# ============================================================

try:
    islemeGirecekPara = float(islemeGirecekPara)
except Exception:
    islemeGirecekPara = float(str(islemeGirecekPara).replace(",", "."))

try:
    leverage = float(leverage)
except Exception:
    leverage = float(str(leverage).replace(",", "."))


# ============================================================
#  ANA DÖNGÜ (TEMİZ / OPTİMİZE SÜRÜM)
# ============================================================

while True:
    try:
        for sym in symbols:

            # ===============================
            #   HER COIN İÇİN STATE BAŞLANGICI
            # ===============================
            if sym not in symbols_data:
                symbols_data[sym] = {
                    "longPoz": False,
                    "shortPoz": False,
                    "Satildi": 0,
                    "Eklendi": 0,
                    "ust2": 0.0,
                    "alt2": 0.0,
                    "gc": 0.0,
                    "stop_gc": 0.0,
                    "aralik_set": 0
                }

            data = symbols_data[sym]

            # ----------------------------
            #  OHLC DATA AL
            # ----------------------------
            df = fetch_candlestick_data(sym, "15m", 350)
            fiyat = float(df["close"].iloc[-1])

            # ----------------------------
            #   MİN / MAX PİVOT HESABI
            # ----------------------------
            n = 10
            df["min"] = df.iloc[argrelextrema(df["low"].values,
                                               np.less_equal,
                                               order=n)[0]]["low"]
            df["max"] = df.iloc[argrelextrema(df["high"].values,
                                               np.greater_equal,
                                               order=n)[0]]["high"]

            min_idx = df[df["min"].notnull()].index.tolist()
            max_idx = df[df["max"].notnull()].index.tolist()
            mm_idx = sorted(min_idx + max_idx)

            filtered_idx = filter_signals(df, mm_idx, min_idx, max_idx)
            balanced_idx = balance_signals(filtered_idx, min_idx, max_idx)

            # =====================================
            #   TREND BELİRLE
            # =====================================
            trend, trend_pts, trend_start_idx, trend_start_price = determine_last_trend(
                balanced_idx, df, min_idx, max_idx
            )

            print(f"\n{sym} → Trend: {trend}")

            # =====================================================
            #   FIB INNER PIVOT KONTROL (Trend var ama pivot yoksa)
            # =====================================================
            if trend in ("Yukselen", "Dusen") and trend_start_idx is not None:

                prev_min_idx = max([i for i in balanced_idx if i in min_idx and i < trend_start_idx], default=None)
                prev_max_idx = max([i for i in balanced_idx if i in max_idx and i < trend_start_idx], default=None)

                if prev_min_idx is None or prev_max_idx is None:
                    print(f"[FIB UYARI] {sym}: Trend var ama FIB için pivot eksik → Belirsiz yapıldı")
                    trend = "Belirsiz"
                    trend_start_idx = None
                    fib_extension_prices[sym] = {"Yukselen": [], "Dusen": []}

            # =====================================
            #   FIBONACCI EXTENSION HESABI
            # =====================================
            fib_extension_prices.setdefault(sym, {"Yukselen": [], "Dusen": []})
            previous_fib_extension_prices.setdefault(sym, {"Yukselen": [], "Dusen": []})

            fib_levels = []

            if trend in ("Yukselen", "Dusen") and trend_start_idx is not None:

                prev_min_idx = max([i for i in balanced_idx if i in min_idx and i < trend_start_idx], default=None)
                prev_max_idx = max([i for i in balanced_idx if i in max_idx and i < trend_start_idx], default=None)

                if prev_min_idx is not None and prev_max_idx is not None:

                    if trend == "Yukselen":
                        start = df.iloc[trend_start_idx]["low"]
                    else:
                        start = df.iloc[trend_start_idx]["high"]

                    prev_min = df.iloc[prev_min_idx]["low"]
                    prev_max = df.iloc[prev_max_idx]["high"]

                    fib_levels = calculate_fib_extension(start, prev_min, prev_max, trend)
                    fib_levels = sorted(fib_levels)

                    fib_extension_prices[sym][trend] = fib_levels
                    previous_fib_extension_prices[sym][trend] = list(fib_levels)

                    print(f"{sym} FIB:", [round(x, 13) for x in fib_levels[:13]], "...")

                else:
                    print(f"[FIB] {sym}: Pivot eksik, hesaplanmadı.")

            else:
                print(f"[FIB] {sym}: Trend belirsiz, hesap yapılmadı.")

            # =====================================
            #   FIB ARALIĞI BUL (UST/ALT DEG)
            # =====================================
            ust_deger[sym] = 0
            alt_deger[sym] = 0

            if fib_levels:
                for i in range(len(fib_levels) - 1):
                    if fib_levels[i] <= fiyat < fib_levels[i + 1]:
                        alt_deger[sym] = fib_levels[i]
                        ust_deger[sym] = fib_levels[i + 1]
                        break

            print(f"Alt={alt_deger[sym]:.6f}  Üst={ust_deger[sym]:.6f}")

            # =====================================
            #   GC ARALIK SET (İLK defa)
            # =====================================
            if data["aralik_set"] == 0:
                data["gc"] = fiyat * 0.3     # %0.5
                data["stop_gc"] = fiyat * 0.3
                data["aralik_set"] = 1
                print(f"GC set edildi → {data['gc']:.6f}")

            gc = data["gc"]


            # ================================
            #   SON MİN / MAX SİNYAL FİYATLARI
            # ================================
            if balanced_idx:
                son_sinyal_idx = balanced_idx[-1]

                # Yeni MIN sinyal
                if son_sinyal_idx in min_idx:
                    onceki_min_idx_price[sym] = son_min_idx_price.get(sym, 0.0)
                    son_min_idx_price[sym] = df.iloc[son_sinyal_idx]["low"]
                    sabitlemis_min_sinyal[sym] = son_min_idx_price[sym]

                    # Son max sıfırlanır
                    son_max_idx_price[sym] = 0.0
                    sabitlemis_max_sinyal[sym] = 0.0

                # Yeni MAX sinyal
                if son_sinyal_idx in max_idx:
                    onceki_max_idx_price[sym] = son_max_idx_price.get(sym, 0.0)
                    son_max_idx_price[sym] = df.iloc[son_sinyal_idx]["high"]
                    sabitlemis_max_sinyal[sym] = son_max_idx_price[sym]

                    # Son min sıfırlanır
                    son_min_idx_price[sym] = 0.0
                    sabitlemis_min_sinyal[sym] = 0.0

                # --- Sinyal geçersizleşme kontrolü ---
                anlik_fiyat = df["close"].iloc[-1]

                # Min sinyal kırılırsa
                if son_sinyal_idx in min_idx:
                    min_fiyat = df.iloc[son_sinyal_idx]["low"]
                    if anlik_fiyat < min_fiyat:
                        if son_sinyal_idx in balanced_idx:
                            balanced_idx.remove(son_sinyal_idx)
                        if son_sinyal_idx in min_idx:
                            min_idx.remove(son_sinyal_idx)
                        # Gerideki son MAX'i bul
                        for geri_idx in reversed(balanced_idx):
                            if geri_idx in max_idx:
                                son_max_idx_price[sym] = df.iloc[geri_idx]["high"]
                                sabitlemis_max_sinyal[sym] = son_max_idx_price[sym]
                                son_min_idx_price[sym] = 0.0
                                sabitlemis_min_sinyal[sym] = 0.0
                                break

                # Max sinyal kırılırsa
                if son_sinyal_idx in max_idx:
                    max_fiyat = df.iloc[son_sinyal_idx]["high"]
                    if anlik_fiyat > max_fiyat:
                        if son_sinyal_idx in balanced_idx:
                            balanced_idx.remove(son_sinyal_idx)
                        if son_sinyal_idx in max_idx:
                            max_idx.remove(son_sinyal_idx)
                        # Gerideki son MIN'i bul
                        for geri_idx in reversed(balanced_idx):
                            if geri_idx in min_idx:
                                son_min_idx_price[sym] = df.iloc[geri_idx]["low"]
                                sabitlemis_min_sinyal[sym] = son_min_idx_price[sym]
                                son_max_idx_price[sym] = 0.0
                                sabitlemis_max_sinyal[sym] = 0.0
                                break


            def longEnter(symbol, miktar):
                print(f"[LONG ENTER] {symbol}")
                return place_order("BUY", symbol, miktar, reduce_only=False)

            def longExit(symbol, miktar):
                print(f"[LONG EXIT] {symbol}")
                return place_order("SELL", symbol, miktar, reduce_only=True)

            def longEnterEkleme(symbol, miktar):
                print(f"[LONG EKLEME] {symbol}")
                return place_order("BUY", symbol, miktar, reduce_only=False)

            def longExitKarSatisi(symbol, miktar):
                print(f"[LONG KAR SATIŞI] {symbol}")
                return place_order("SELL", symbol, miktar, reduce_only=True)

            def shortEnter(symbol, miktar):
                print(f"[SHORT ENTER] {symbol}")
                return place_order("SELL", symbol, miktar, reduce_only=False)

            def shortExit(symbol, miktar):
                print(f"[SHORT EXIT] {symbol}")
                return place_order("BUY", symbol, miktar, reduce_only=True)

            def shortEnterEkleme(symbol, miktar):
                print(f"[SHORT EKLEME] {symbol}")
                return place_order("SELL", symbol, miktar, reduce_only=False)

            def shortExitKarSatisi(symbol, miktar):
                print(f"[SHORT KAR SATIŞI] {symbol}")
                return place_order("BUY", symbol, miktar, reduce_only=True)



            # ================================
            #   STOP LOSS GÜNCELLEME
            # ================================
            fiyat = float(df["close"].iloc[-1])

            # LONG stop
            if data["longPoz"]:
                yeni_stop, stop_tetik = update_stop_loss(
                    "long",
                    fiyat,
                    son_min_idx_price.get(sym, 0.0),
                    son_max_idx_price.get(sym, 0.0),
                    onceki_min_idx_price.get(sym, 0.0),
                    onceki_max_idx_price.get(sym, 0.0),
                    long_stop_level.get(sym, 0.0),
                    gc,
                )
                long_stop_level[sym] = yeni_stop

                if stop_tetik:
                    poz_miktar = get_position_qty(sym)
                    if poz_miktar > 0:
                        longExit(sym, poz_miktar)
                    data["longPoz"] = False
                    data["Satildi"] = 0
                    data["Eklendi"] = 0
                    data["ust2"] = 0.0
                    data["alt2"] = 0.0
                    print(f"🚨 {sym} LONG STOP tetiklendi, pozisyon kapatıldı.")

            # SHORT stop
            if data["shortPoz"]:
                yeni_stop, stop_tetik = update_stop_loss(
                    "short",
                    fiyat,
                    son_min_idx_price.get(sym, 0.0),
                    son_max_idx_price.get(sym, 0.0),
                    onceki_min_idx_price.get(sym, 0.0),
                    onceki_max_idx_price.get(sym, 0.0),
                    short_stop_level.get(sym, 0.0),
                    gc,
                )
                short_stop_level[sym] = yeni_stop

                if stop_tetik:
                    poz_miktar = get_position_qty(sym)
                    if poz_miktar > 0:
                        shortExit(sym, poz_miktar)
                    data["shortPoz"] = False
                    data["Satildi"] = 0
                    data["Eklendi"] = 0
                    data["ust2"] = 0.0
                    data["alt2"] = 0.0
                    print(f"🚨 {sym} SHORT STOP tetiklendi, pozisyon kapatıldı.")

            # ================================
            #   DEBUG ÇIKTILARI
            # ================================
            print("=============================")
            print(f"\033[38;5;214mSon max sinyal: {son_max_idx_price.get(sym, 0):.6f}\033[0m")
            print(f"\033[38;5;214mSon min sinyal: {son_min_idx_price.get(sym, 0):.6f}\033[0m")
            print(f"\033[38;5;208mÖnceki max sinyal: {onceki_max_idx_price.get(sym, 0):.6f}\033[0m")
            print(f"\033[38;5;208mÖnceki min sinyal: {onceki_min_idx_price.get(sym, 0):.6f}\033[0m")
            print("=============================")
            print(f"\033[92must_deger: {ust_deger.get(sym, 0):.6f}\033[0m")
            print(f"\033[91malt_deger: {alt_deger.get(sym, 0):.6f}\033[0m")
            print("=============================")
            print(f"\033[92mUST2 (kilit): {data.get('ust2', 0.0):.6f}\033[0m")
            print(f"\033[91mALT2 (kilit): {data.get('alt2', 0.0):.6f}\033[0m")
            print("=============================")

            # ================================
            #   GİRİŞ / ÇIKIŞ STRATEJİSİ
            # ================================
            poz_miktar = get_position_qty(sym)

            # --------------------------------
            #   YÜKSELEN TREND (LONG STRATEJİ)
            # --------------------------------
            if trend == "Yukselen":

                # Açık SHORT varsa kapat → LONG'a geç
                if data["shortPoz"]:
                    if poz_miktar > 0:
                        shortExit(sym, poz_miktar)
                    data["shortPoz"] = False
                    data["Satildi"] = 0
                    data["Eklendi"] = 0
                    data["ust2"] = 0.0
                    data["alt2"] = 0.0

                # LONG yoksa aç
                if not data["longPoz"]:
                    miktar = (islemeGirecekPara * leverage) / fiyat
                    longEnter(sym, miktar)
                    data["longPoz"] = True
                    data["Satildi"] = 0
                    data["Eklendi"] = 0
                    # İlk FIB bandını long için referans al
                    if alt_deger.get(sym, 0) > 0 and ust_deger.get(sym, 0) > 0:
                        alt_deger1[sym] = alt_deger[sym]
                        ust_deger1[sym] = ust_deger[sym]
                        print(f"[LONG] İlk band: {alt_deger1[sym]:.6f} - {ust_deger1[sym]:.6f}")

                # Fiyat ust_deger1 üzerine çıkınca UST2 KİLİTLENİR
                if data["longPoz"] and ust_deger1.get(sym, 0) > 0:
                    if fiyat >= ust_deger1[sym]:
                        data["ust2"] = ust_deger1[sym]
                        data["Satildi"] = 0
                        data["Eklendi"] = 0
                        print(f"[UST2 KİLİTLENDİ] {sym}: {data['ust2']:.6f}")

                # KAR SATIŞI (LONG)
                if data["longPoz"] and data["ust2"] > 0 and data["Satildi"] == 0:
                    # fiyat, kilitli seviyeden gc kadar geri çekildiyse
                    if fiyat <= data["ust2"] - gc:
                        poz_miktar = get_position_qty(sym)
                        if poz_miktar > 0:
                            satilacak = round(poz_miktar / 2, 6)
                            longExitKarSatisi(sym, satilacak)
                            data["Satildi"] = 1
                            print(f"[LONG KAR SATIŞI] {sym}: {satilacak} miktar")

                # EKLEME (LONG)
                if data["longPoz"] and data["Satildi"] == 1 and data["Eklendi"] == 0:
                    if son_min_idx_price.get(sym, 0) > 0:
                        if fiyat >= son_min_idx_price.get(sym, 0) + gc:
                            poz_miktar = get_position_qty(sym)
                            if poz_miktar > 0:
                                longEnterEkleme(sym, poz_miktar)
                                data["Eklendi"] = 1
                                print(f"[LONG EKLEME] {sym}: {poz_miktar} miktar")

            # --------------------------------
            #   DÜŞEN TREND (SHORT STRATEJİ)
            # --------------------------------
            elif trend == "Dusen":

                # Açık LONG varsa kapat → SHORT'a geç
                if data["longPoz"]:
                    if poz_miktar > 0:
                        longExit(sym, poz_miktar)
                    data["longPoz"] = False
                    data["Satildi"] = 0
                    data["Eklendi"] = 0
                    data["ust2"] = 0.0
                    data["alt2"] = 0.0

                # SHORT yoksa aç
                if not data["shortPoz"]:
                    miktar = (islemeGirecekPara * leverage) / fiyat
                    shortEnter(sym, miktar)
                    data["shortPoz"] = True
                    data["Satildi"] = 0
                    data["Eklendi"] = 0
                    if alt_deger.get(sym, 0) > 0 and ust_deger.get(sym, 0) > 0:
                        alt_deger1[sym] = alt_deger[sym]
                        ust_deger1[sym] = ust_deger[sym]
                        print(f"[SHORT] İlk band: {alt_deger1[sym]:.6f} - {ust_deger1[sym]:.6f}")

                # Fiyat alt_deger1 altına inince ALT2 KİLİTLENİR
                if data["shortPoz"] and alt_deger1.get(sym, 0) > 0:
                    if fiyat <= alt_deger1[sym]:
                        data["alt2"] = alt_deger1[sym]
                        data["Satildi"] = 0
                        data["Eklendi"] = 0
                        print(f"[ALT2 KİLİTLENDİ] {sym}: {data['alt2']:.6f}")

                # KAR SATIŞI (SHORT)
                if data["shortPoz"] and data["alt2"] > 0 and data["Satildi"] == 0:
                    # fiyat, kilitli seviyeden gc kadar yukarı tepki verirse
                    if fiyat >= data["alt2"] + gc:
                        poz_miktar = get_position_qty(sym)
                        if poz_miktar > 0:
                            satilacak = round(poz_miktar / 2, 6)
                            shortExitKarSatisi(sym, satilacak)
                            data["Satildi"] = 1
                            print(f"[SHORT KAR SATIŞI] {sym}: {satilacak} miktar")

                # EKLEME (SHORT)
                if data["shortPoz"] and data["Satildi"] == 1 and data["Eklendi"] == 0:
                    if son_max_idx_price.get(sym, 0) > 0:
                        if fiyat <= son_max_idx_price.get(sym, 0) - gc:
                            poz_miktar = get_position_qty(sym)
                            if poz_miktar > 0:
                                shortEnterEkleme(sym, poz_miktar)
                                data["Eklendi"] = 1
                                print(f"[SHORT EKLEME] {sym}: {poz_miktar} miktar")

            # --------------------------------
            #   BELİRSİZ TREND
            # --------------------------------
            else:
                print(f"⚪ {sym} Belirsiz trend – yeni yön aranıyor...")

                # LONG açma denemesi
                if not data["longPoz"] and son_min_idx_price.get(sym, 0) > 0:
                    if fiyat >= son_min_idx_price.get(sym, 0) + gc:
                        # önce SHORT varsa kapat
                        if data["shortPoz"]:
                            poz_miktar = get_position_qty(sym)
                            if poz_miktar > 0:
                                shortExit(sym, poz_miktar)
                            data["shortPoz"] = False

                        miktar = (islemeGirecekPara * leverage) / fiyat
                        print(f"📈 {sym} BELİRSİZ TREND – LONG açılıyor @ {fiyat:.6f}")
                        longEnter(sym, miktar)
                        data["longPoz"] = True
                        data["Satildi"] = 0
                        data["Eklendi"] = 0

                # SHORT açma denemesi
                if not data["shortPoz"] and son_max_idx_price.get(sym, 0) > 0:
                    if fiyat <= son_max_idx_price.get(sym, 0) - gc:
                        # önce LONG varsa kapat
                        if data["longPoz"]:
                            poz_miktar = get_position_qty(sym)
                            if poz_miktar > 0:
                                longExit(sym, poz_miktar)
                            data["longPoz"] = False

                        miktar = (islemeGirecekPara * leverage) / fiyat
                        print(f"📉 {sym} BELİRSİZ TREND – SHORT açılıyor @ {fiyat:.6f}")
                        shortEnter(sym, miktar)
                        data["shortPoz"] = True
                        data["Satildi"] = 0
                        data["Eklendi"] = 0

            ## ================================
            ##   DASHBOARD & DURUM ÇIKTISI
            ## ================================
            #mevcut_kasa = simulated_balance.get(sym, 1000.0)
            #son_pnl = 0.0
            #pos = open_positions.get(sym)
            #if pos:
            #    entry = pos.get("entry_price", fiyat)
            #    direction = pos.get("direction", "")
            #    if entry > 0:
            #        if direction == "LONG":
            #            son_pnl = ((fiyat - entry) / entry) * 100
            #        elif direction == "SHORT":
            #            son_pnl = ((entry - fiyat) / entry) * 100

            aktif_pozisyon = "ARAMA"
            if data["longPoz"]:
                aktif_pozisyon = "LONG"
            elif data["shortPoz"]:
                aktif_pozisyon = "SHORT"

            #print_status(sym, fiyat, mevcut_kasa, pozisyon=aktif_pozisyon, pnl=son_pnl, trend=trend)

            # Pozisyon durumu yaz
            if data["shortPoz"]:
                print("\033[91mSHORT POZİSYONDA BEKLİYOR\033[0m")
                print(f"\033[91mFiyat: {fiyat:.6f}\033[0m")
            if data["longPoz"]:
                print("\033[92mLONG POZİSYONDA BEKLİYOR\033[0m")
                print(f"\033[92mFiyat: {fiyat:.6f}\033[0m")

            if not data["longPoz"] and not data["shortPoz"]:
                print(f"🔍 {sym} Pozisyon yok, yeni fırsat aranıyor...")

            print("===========================================================")
            time.sleep(1)
            

    except ccxt.BaseError as error:
        print("[ERROR - CCXT] ", error)
        time.sleep(1)
        continue
    


