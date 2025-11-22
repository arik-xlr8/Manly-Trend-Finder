import ccxt
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.signal import argrelextrema
import time 

# Binance API anahtarları
api_key = 'XNYtJ2hI8KMIH1RujaDKfNvRN90IaASV0lxGDtpBThsL8fWc97d67IxaF3xCzcWc'
api_secret = 'TgExmWB33xsF2mg3T08RcaFp1500ZUp0K4ZdEY9g2nOIsQkRt3JYwFb32SQt2Y3z'

exchange = ccxt.binance({'apiKey': api_key, 'secret': api_secret})


def fetch_candlestick_data(symbol, timeframe, limit):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not ohlcv:
            print(f"No data received for {symbol}")
            return pd.DataFrame()
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        #df = df.iloc[:-1]  # Son mumu çıkar (canlı olan)
        return df
    except Exception as e:
        print(f"Error fetching candlestick data for {symbol}: {e}")
        return pd.DataFrame()

def filter_signals(df, min_max_idx, min_idx, max_idx, confirm_mum=2):
    """
    df            : OHLC dataframe (timestamp, open, high, low, close ...)
    min_max_idx   : min_idx + max_idx birleştirilip sıralanmış hali
    min_idx       : argrelextrema'dan gelen dip indeksleri
    max_idx       : argrelextrema'dan gelen tepe indeksleri

    Amaç:
        - Sinyalleri tamamen yeniden organize etmek.
        - Aynı tipten ardışık sinyaller varsa, sadece en uç (en derin dip / en yüksek tepe) kalsın.
        - Sonuçta min-max-min-max şeklinde temiz bir pivot dizisi üretmek.
    """
    # 1) Bütün pivotları (min / max) tek bir listeye topla ve zaman sırasına göre sırala
    pivots = []
    for idx in sorted(set(min_max_idx)):
        if idx in min_idx and idx in max_idx:
            # Hem min hem max olması çok nadir; mumu inceleyip karar verebiliriz.
            low = df.iloc[idx]["low"]
            high = df.iloc[idx]["high"]
            # Mum gövdesi nereye daha yakınsa onu pivot tipi seç
            if abs(df.iloc[idx]["close"] - low) < abs(df.iloc[idx]["close"] - high):
                p_type = "min"
                price = low
            else:
                p_type = "max"
                price = high
        elif idx in min_idx:
            p_type = "min"
            price = df.iloc[idx]["low"]
        elif idx in max_idx:
            p_type = "max"
            price = df.iloc[idx]["high"]
        else:
            continue

        pivots.append({"idx": idx, "type": p_type, "price": float(price)})

    if not pivots:
        return []

    # 2) Aynı tipten ardışık pivotları temizle:
    #    - İki dip arka arkaya gelirse: daha DERİN olan kalsın
    #    - İki tepe arka arkaya gelirse: daha YÜKSEK olan kalsın
    cleaned = []
    for p in pivots:
        if not cleaned:
            cleaned.append(p)
            continue

        last = cleaned[-1]
        if p["type"] == last["type"]:
            # İkisi de dip ise -> daha düşük fiyatlı olan dip tutulur
            if p["type"] == "min":
                if p["price"] < last["price"]:
                    cleaned[-1] = p
            # İkisi de tepe ise -> daha yüksek fiyatlı olan tepe tutulur
            else:  # "max"
                if p["price"] > last["price"]:
                    cleaned[-1] = p
        else:
            cleaned.append(p)

    # 3) İsteğe bağlı: uçtaki (en sağdaki) pivot hala "tam oturmamış" olabilir.
    #    Bunun için confirm_mum kullanarak sadece SON pivotu kontrol edeceğiz.
    #    Bu, repaint riskini azaltır ama sinyal sayısını çok azaltmaz.
    if len(cleaned) >= 1 and confirm_mum is not None and confirm_mum > 0:
        last_pivot = cleaned[-1]
        idx = last_pivot["idx"]
        # Sağda yeterli mum yoksa -> bu pivotu geçici say ve çıkar
        if idx + confirm_mum >= len(df):
            cleaned = cleaned[:-1]
        else:
            if last_pivot["type"] == "max":
                future_high = df["high"].iloc[idx + 1 : idx + 1 + confirm_mum].max()
                # Eğer ileride daha yüksek bir high oluşmuşsa bu tepeyi iptal et
                if future_high > last_pivot["price"]:
                    cleaned = cleaned[:-1]
            else:  # "min"
                future_low = df["low"].iloc[idx + 1 : idx + 1 + confirm_mum].min()
                # Eğer ileride daha düşük bir low oluşmuşsa bu dipi iptal et
                if future_low < last_pivot["price"]:
                    cleaned = cleaned[:-1]

    # Artık min-max-min-max şeklinde temiz bir dizi var.
    # Fonksiyon eski API'sini korumak için sadece indeks listesini döndürüyor.
    filtered_idx = [p["idx"] for p in cleaned]
    return filtered_idx


def balance_signals(filtered_idx, min_idx, max_idx):
    """
    Eski versiyon min ve max sayısını zorla eşitliyordu
    (sonlardan sinyal silerek). Bu, özellikle son trend yapısını
    bozuyordu. Artık bu fonksiyon sadece mevcut listeyi aynen döndürüyor.

    Trend analizi için min / max sayısının eşit olmasına gerek yok;
    önemli olan doğru sırayla (min-max-min / max-min-max) dizilmiş olmaları.
    Bu sıralamayı zaten filter_signals içinde sağladığımız için
    burada ek bir müdahaleye ihtiyaç yok.
    """
    return list(filtered_idx)


def determine_last_trend(filtered_idx, candlestick_data, min_idx, max_idx):
    trend = 'Belirsiz'
    trend_points = []
    trend_start_idx = None

    # --- Sinyal listelerini oluştur ---
    filtered_max_idx = [idx for idx in filtered_idx if idx in max_idx]
    filtered_min_idx = [idx for idx in filtered_idx if idx in min_idx]

    # --- Düşen trend kontrolü (Son iki tepe) ---
    if len(filtered_max_idx) >= 2:
        last_max = filtered_max_idx[-1]
        second_last_max = filtered_max_idx[-2]
        if candlestick_data.iloc[last_max]['high'] < candlestick_data.iloc[second_last_max]['high']:
            trend = 'Dusen'
            trend_start_idx = second_last_max
            trend_points.append((second_last_max, last_max))

    # --- Yükselen trend kontrolü (Son iki dip) ---
    if len(filtered_min_idx) >= 2 and trend != 'Dusen':
        last_min = filtered_min_idx[-1]
        second_last_min = filtered_min_idx[-2]
        if candlestick_data.iloc[last_min]['low'] > candlestick_data.iloc[second_last_min]['low']:
            trend = 'Yukselen'
            trend_start_idx = second_last_min
            trend_points.append((second_last_min, last_min))

    # --- Yükselen trend genişletme ---
    if trend == 'Yukselen' and trend_start_idx is not None:
        try:
            current_idx = filtered_min_idx.index(trend_start_idx)
        except ValueError:
            current_idx = -1

        for i in range(current_idx - 1, -1, -1):
            prev_min = filtered_min_idx[i]
            curr_min = filtered_min_idx[i + 1]
            if candlestick_data.iloc[curr_min]['low'] > candlestick_data.iloc[prev_min]['low']:
                trend_points.insert(0, (prev_min, curr_min))
                trend_start_idx = prev_min
            else:
                break

    # --- Düşen trend genişletme ---
    if trend == 'Dusen' and trend_start_idx is not None:
        try:
            current_idx = filtered_max_idx.index(trend_start_idx)
        except ValueError:
            current_idx = -1

        for i in range(current_idx - 1, -1, -1):
            prev_max = filtered_max_idx[i - 1]
            curr_max = filtered_max_idx[i]
            if candlestick_data.iloc[curr_max]['high'] < candlestick_data.iloc[prev_max]['high']:
                trend_points.insert(0, (prev_max, curr_max))
                trend_start_idx = prev_max
            else:
                break

    # --- Trend bozulma kontrolü ---
    if trend == 'Yukselen':
        if len(filtered_max_idx) >= 1 and len(filtered_min_idx) >= 2:
            last_max_idx = filtered_max_idx[-1]
            last_min_before_max = max([i for i in filtered_min_idx if i < last_max_idx], default=None)
            if last_min_before_max is not None:
                min_low = candlestick_data.iloc[last_min_before_max]['low']
                if candlestick_data.iloc[-1]['close'] < min_low:
                    trend = 'Belirsiz'
                    trend_points = []
                    trend_start_idx = None

    if trend == 'Dusen':
        if len(filtered_min_idx) >= 1 and len(filtered_max_idx) >= 2:
            last_min_idx = filtered_min_idx[-1]
            last_max_before_min = max([i for i in filtered_max_idx if i < last_min_idx], default=None)
            if last_max_before_min is not None:
                max_high = candlestick_data.iloc[last_max_before_min]['high']
                if candlestick_data.iloc[-1]['close'] > max_high:
                    trend = 'Belirsiz'
                    trend_points = []
                    trend_start_idx = None

    return trend, trend_points, trend_start_idx





# ================================================================
#  TREND PIVOTLARINI FIB İÇİN OTOMATİK SEÇEN FONKSİYON
# ================================================================
def trend_pivots_for_fib(trend, trend_points, df, min_idx, max_idx):
    if trend == "Belirsiz" or len(trend_points) == 0:
        return None, None, None

    # -----------------------------------------------
    # DÜŞEN trend için: High1 → High2 → Low1
    # -----------------------------------------------
    if trend == "Dusen":
        high1_idx = trend_points[0][0]   # En yüksek tepe
        high2_idx = trend_points[0][1]   # Lower-high (trend start)

        # High2’den sonra gelen ilk dip = Low1
        low_candidates = [i for i in min_idx if i > high2_idx]
        if not low_candidates:
            return None, None, None

        low1_idx = low_candidates[0]

        high1 = df.iloc[high1_idx]["high"]
        high2 = df.iloc[high2_idx]["high"]
        low1  = df.iloc[low1_idx]["low"]

        # prev_max, start_price, prev_min
        return high1, high2, low1

    # -----------------------------------------------
    # YÜKSELEN trend için: Low1 → Low2 → High1
    # -----------------------------------------------
    if trend == "Yukselen":
        low1_idx = trend_points[0][0]
        low2_idx = trend_points[0][1]

        high_candidates = [i for i in max_idx if i > low2_idx]
        if not high_candidates:
            return None, None, None

        high1_idx = high_candidates[0]

        low1  = df.iloc[low1_idx]["low"]
        low2  = df.iloc[low2_idx]["low"]
        high1 = df.iloc[high1_idx]["high"]

        # prev_min, start_price, prev_max
        return low1, low2, high1

    return None, None, None



def calculate_fib_extension(start_price, prev_min_price, prev_max_price, trend_direction):
    if trend_direction == 'Yukselen':
        fib_levels = [-1, 0, 0.5, 1.5]
        price_range = abs(prev_max_price - prev_min_price)
        fib_prices = [start_price + level * price_range for level in fib_levels]
    elif trend_direction == 'Dusen':
        fib_levels = [1, 0, -0.5, -1.5]
        price_range = abs(prev_max_price - prev_min_price)
        fib_prices = [start_price + level * price_range for level in fib_levels]
    else:
        return []

    return fib_prices


def plot_signals(candlestick_data, filtered_idx, min_idx, max_idx, trend_direction, trend_points, trend_start_idx):
    fig = go.Figure(data=[go.Candlestick(x=candlestick_data['timestamp'],
                open=candlestick_data['open'],
                high=candlestick_data['high'],
                low=candlestick_data['low'],
                close=candlestick_data['close'])])

    for idx in filtered_idx:
        signal_type = 'Minima' if idx in min_idx else 'Maxima'
        color = 'blue' if signal_type == 'Minima' else 'red'
        price = candlestick_data.iloc[idx]['low'] if signal_type == 'Minima' else candlestick_data.iloc[idx]['high']
        marker_symbol = 'triangle-down' if signal_type == 'Minima' else 'triangle-up'
        fig.add_trace(go.Scatter(x=[candlestick_data.iloc[idx]['timestamp']], 
                                 y=[price],
                                 mode='markers',
                                 marker=dict(color=color, symbol=marker_symbol, size=10),
                                 name=f'{signal_type} #{idx}'))

    

    if trend_direction != 'Belirsiz':
        for point in trend_points:
            start_idx, end_idx = point
            start_price = candlestick_data.iloc[start_idx]['low'] if start_idx in min_idx else candlestick_data.iloc[start_idx]['high']
            end_price = candlestick_data.iloc[end_idx]['low'] if end_idx in min_idx else candlestick_data.iloc[end_idx]['high']
            fig.add_trace(go.Scatter(x=[candlestick_data.iloc[start_idx]['timestamp'], candlestick_data.iloc[end_idx]['timestamp']], 
                                     y=[start_price, end_price],
                                     mode='lines',
                                     line=dict(color='green' if trend_direction == 'Yukselen' else 'red', width=2),
                                     name=f'Trend {trend_direction}'))

        if trend_start_idx is not None:
            start_price = candlestick_data.iloc[trend_start_idx]['low'] if trend_start_idx in min_idx else candlestick_data.iloc[trend_start_idx]['high']
            fig.add_trace(go.Scatter(x=[candlestick_data.iloc[trend_start_idx]['timestamp']], 
                                     y=[start_price],
                                     mode='markers+text',
                                     marker=dict(color='purple', symbol='star', size=15),
                                     text=[f'Trend Baslangici ({candlestick_data.iloc[trend_start_idx]["timestamp"]})'],
                                     textposition="bottom right"))


            
            for fib_price in fib_extension_prices:
                fig.add_trace(go.Scatter(x=[candlestick_data.iloc[trend_start_idx]['timestamp'], candlestick_data.iloc[-1]['timestamp']], 
                             y=[fib_price, fib_price],
                             mode='lines',
                             line=dict(color='orange', dash='dash'),
                             text=[f'Fib Extension: {fib_price:.4f}'],
                             textposition="bottom right"))

    fig.update_layout(title='Candlestick Chart with Signals',
                      xaxis_title='Timestamp',
                      yaxis_title='Price',
                      template='plotly_dark')

    
    fig.update_layout(title=f'{symbol} Sinyalleri ve Trend Yonu ({trend_direction} trend)', xaxis_title='Tarih', yaxis_title='Fiyat')
    fig.show()
    #, "RESOLV/USDT", "PENGU/USDT", "AIN/USDT", "STX/USDT"
symbols = [
    "ETH/USDT",
    "BTC/USDT",
    "AVAX/USDT",
    "OP/USDT",
]

timeframe = '4h'
limit = 350

while True:
    for symbol in symbols:
        try:
            candlestick_data = fetch_candlestick_data(symbol, timeframe, limit)

            n = 10  # argrelextrema i�in pencere boyutu

            candlestick_data['min'] = candlestick_data.iloc[argrelextrema(candlestick_data['low'].values, np.less_equal, order=n)[0]]['low']
            candlestick_data['max'] = candlestick_data.iloc[argrelextrema(candlestick_data['high'].values, np.greater_equal, order=n)[0]]['high']

            min_idx = candlestick_data[candlestick_data['min'].notnull()].index.tolist()
            max_idx = candlestick_data[candlestick_data['max'].notnull()].index.tolist()

            min_max_idx = sorted(list(min_idx) + list(max_idx))
            filtered_idx = filter_signals(candlestick_data, min_max_idx, min_idx, max_idx)
            balanced_idx = balance_signals(filtered_idx, min_idx, max_idx)

            trend_direction, trend_points, trend_start_idx = determine_last_trend(filtered_idx, candlestick_data, min_idx, max_idx)

            plot_signals(candlestick_data, balanced_idx, min_idx, max_idx, trend_direction, trend_points, trend_start_idx)

            # Trend yönü ve başlangıç indeksi güncellenir
            if trend_direction != 'Belirsiz':
                prev_trend = trend_direction
                prev_trend_start_idx = trend_start_idx

            # Fibonacci Extension seviyelerini ekle
            if trend_direction == 'Yukselen':
                # Yükselen trend için
                min_candidates = [i for i in filtered_idx if i in min_idx]
                max_candidates = [i for i in filtered_idx if i in max_idx]
    
                if len(min_candidates) >= 2 and len(max_candidates) >= 2:
                    prev_min_idx = min_candidates[-2]
                    prev_max_idx = max_candidates[-2]
                    fib_extension_prices = calculate_fib_extension(
                        candlestick_data.iloc[trend_start_idx]['low'], 
                        candlestick_data.iloc[prev_min_idx]['high'],
                        candlestick_data.iloc[prev_max_idx]['low'],
                        'Yukselen'
                    )
                else:
                    continue  # Ya da pass / break

            if trend_direction == 'Dusen':
                # Düşen trend için
                min_candidates = [i for i in filtered_idx if i in min_idx]
                max_candidates = [i for i in filtered_idx if i in max_idx]
    
                if len(min_candidates) >= 2 and len(max_candidates) >= 2:
                    prev_min_idx = min_candidates[-2]
                    prev_max_idx = max_candidates[-2]
                    fib_extension_prices = calculate_fib_extension(
                        candlestick_data.iloc[trend_start_idx]['high'], 
                        candlestick_data.iloc[prev_min_idx]['low'],
                        candlestick_data.iloc[prev_max_idx]['high'],
                        'Dusen'
                    )
                else:
                    continue  # Ya da pass / break

                    # Hedef fiyatların %10'dan fazla olduğu ve anlık fiyatın trend başlangıç fiyatına %3'ten daha az farkı olan coini seç
                    for fib_price in fib_extension_prices:
                        fib_price_diff_percent = ((fib_price - start_price) / start_price) * 100

                        if abs(fib_price_diff_percent) > 5.0:
                            price_diff_from_start_percent = abs(price_diff_percent)
                            
                            if price_diff_from_start_percent < 6.0:
                                print(f"\nSymbol: {symbol}")
                                print("Trend Yonu:", trend_direction)
                                print("Sinyallerin Tarih, Saat ve Fiyat Degerleri:")
                                for idx in balanced_idx:
                                    signal_type = 'Minima' if idx in min_idx else 'Maxima'
                                    price = candlestick_data.iloc[idx]['low'] if signal_type == 'Minima' else candlestick_data.iloc[idx]['high']
                                    print(f" [ {idx} ] Tarih ve Saat: {candlestick_data.iloc[idx]['timestamp']}, Sinyal Tipi: {signal_type}, Fiyat Degeri: {price}")

                                print(f"\nTrend Baslangic Noktasi ({trend_direction}):")
                                print(f"Tarih ve Saat: {candlestick_data.iloc[trend_start_idx]['timestamp']}, Fiyat Degeri: {start_price}")
                               
                                print("\nFibonacci Extension Seviyeleri:")
                                for i, price in enumerate(fib_extension_prices):
                                    print(f"Seviye {i+1}: {price:.4f}")

                                plot_signals(candlestick_data, balanced_idx, min_idx, max_idx, trend_direction, trend_points, trend_start_idx, symbol)
                                time.sleep(300)  # 1 saniye bekle

                        ## Tüm coinler bittiğinde 5 dakika (300 saniye) bekle
                        #print("\nTüm coinler tamamlandı. 5 dakika bekleniyor...")
                        

        except Exception as e:
            print(f"{symbol} için hata oluştu: {e}")
            time.sleep(10)
if __name__ == "__main__":
    main()


