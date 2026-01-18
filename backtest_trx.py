# import time
# import requests
# import numpy as np
# import pandas as pd
# from dataclasses import dataclass
# from typing import Literal, Optional, List, Tuple

# # ✅ Pivot fonksiyonlarını senin dosyadan çekiyoruz:
# # plot_swings_from_api.py içinde find_swings ve SwingPoint olmalı
# from plot_swings_from_api import find_swings, SwingPoint  # type: ignore


# BINANCE_FAPI = "https://fapi.binance.com"


# # ======================
# # Helpers
# # ======================

# def _to_utc(ts: pd.Timestamp) -> pd.Timestamp:
#     if ts.tzinfo is None:
#         return ts.tz_localize("UTC")
#     return ts.tz_convert("UTC")

# def _ms(dt: pd.Timestamp) -> int:
#     dt = _to_utc(dt)
#     return int(dt.value // 10**6)


# # ======================
# # Binance Futures: OHLC + Funding
# # ======================

# def fetch_klines_chunk(symbol: str, interval: str, start_ms: int, limit: int = 1500) -> pd.DataFrame:
#     url = f"{BINANCE_FAPI}/fapi/v1/klines"
#     params = {"symbol": symbol, "interval": interval, "startTime": start_ms, "limit": limit}
#     r = requests.get(url, params=params, timeout=20)
#     r.raise_for_status()
#     data = r.json()
#     if not data:
#         return pd.DataFrame()

#     cols = [
#         "open_time", "open", "high", "low", "close", "volume",
#         "close_time", "qav", "num_trades", "tbbv", "tbqv", "ignore"
#     ]
#     df = pd.DataFrame(data, columns=cols)
#     df["Date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
#     df.set_index("Date", inplace=True)

#     for c in ["open", "high", "low", "close", "volume"]:
#         df[c] = df[c].astype(float)

#     df = df[["open", "high", "low", "close", "volume"]]
#     df.columns = ["Open", "High", "Low", "Close", "Volume"]
#     return df


# def fetch_ohlc_years(symbol: str, interval: str, years: int = 3, end_utc: Optional[pd.Timestamp] = None) -> pd.DataFrame:
#     """
#     Binance Futures Klines ile belirtilen yıl kadar geçmiş OHLC çeker.
#     Chunk'lı çektiği için 3 yıl gibi uzun aralıklarda da çalışır.
#     """
#     if end_utc is None:
#         # pandas sürümlerinde utcnow tz-aware gelebiliyor -> localize yapmıyoruz
#         end_utc = pd.Timestamp.utcnow()
#     end_utc = _to_utc(end_utc)

#     start_utc = end_utc - pd.Timedelta(days=365 * years)
#     start_ms = _ms(start_utc)

#     out = []
#     while True:
#         chunk = fetch_klines_chunk(symbol, interval, start_ms, limit=1500)
#         if chunk.empty:
#             break
#         out.append(chunk)

#         last_open_ms = int(chunk.index[-1].value // 10**6)
#         start_ms = last_open_ms + 1

#         # rate limit'e takılmamak için küçük uyku
#         time.sleep(0.05)

#         if chunk.index[-1] >= end_utc:
#             break

#     if not out:
#         raise RuntimeError("OHLC çekilemedi. Sembol/interval doğru mu?")

#     df = pd.concat(out).sort_index()
#     df = df[~df.index.duplicated(keep="first")]
#     df = df[(df.index >= start_utc) & (df.index <= end_utc)]
#     return df


# def fetch_funding_rates(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
#     """
#     Binance Futures funding rate history.
#     (Not: Bu endpoint bazı sembollerde eski datayı sınırlayabilir; yine de mümkün olanı çeker.)
#     """
#     url = f"{BINANCE_FAPI}/fapi/v1/fundingRate"

#     rows = []
#     cursor = start_ms
#     while True:
#         params = {"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000}
#         r = requests.get(url, params=params, timeout=20)
#         r.raise_for_status()
#         data = r.json()
#         if not data:
#             break

#         rows.extend(data)
#         last_time = int(data[-1]["fundingTime"])
#         cursor = last_time + 1

#         time.sleep(0.05)

#         if last_time >= end_ms or len(data) < 1000:
#             break

#     if not rows:
#         return pd.DataFrame(columns=["fundingTime", "fundingRate"])

#     fr = pd.DataFrame(rows)
#     fr["fundingTime"] = pd.to_datetime(fr["fundingTime"].astype(np.int64), unit="ms", utc=True)
#     fr["fundingRate"] = fr["fundingRate"].astype(float)
#     fr = fr[["fundingTime", "fundingRate"]].sort_values("fundingTime")
#     fr = fr.drop_duplicates(subset=["fundingTime"], keep="first")
#     return fr


# # ======================
# # Strategy Trades (+ STOP in swings logic)
# # ======================

# @dataclass
# class Trade:
#     direction: Literal["long", "short"]
#     entry_index: int
#     entry_price: float
#     stop_level: float
#     exit_index: Optional[int] = None
#     exit_price: Optional[float] = None
#     exit_reason: Optional[str] = None
#     entry_time: Optional[pd.Timestamp] = None
#     exit_time: Optional[pd.Timestamp] = None


# def _first_stop_hit_index(
#     df: pd.DataFrame,
#     direction: str,
#     start_idx: int,
#     end_idx: int,
#     stop_level: float
# ) -> Optional[int]:
#     """
#     (start_idx, end_idx] aralığında stop'a ilk değilen bar indexini döndürür.
#     LONG : Low  <= stop_level
#     SHORT: High >= stop_level
#     """
#     if end_idx <= start_idx:
#         return None

#     window = df.iloc[start_idx + 1: end_idx + 1]
#     if window.empty:
#         return None

#     if direction == "long":
#         hits = window["Low"].values <= stop_level
#     else:
#         hits = window["High"].values >= stop_level

#     if not hits.any():
#         return None

#     first_pos = int(np.argmax(hits))  # ilk True
#     return (start_idx + 1) + first_pos


# def generate_trades(
#     df: pd.DataFrame,
#     swings: List[SwingPoint],
#     right_bars: int = 1,
#     enable_flip: bool = True,

#     # ✅ stop kontrolleri swings mantığında
#     enable_stop: bool = True,
#     stop_loss_pct: float = 0.015,  # %1.5
# ) -> List[Trade]:
#     """
#     Entry:
#       - LB (Higher Low): prev_lb.price < curr_lb.price -> LONG
#       - LH (Lower High): prev_lh.price > curr_lh.price -> SHORT

#     Exit (greed killer):
#       - LONG açıkken LH onaylanınca kapat
#       - SHORT açıkken LB onaylanınca kapat

#     STOP LOSS:
#       - LONG: entry'den sonra Low <= entry*(1-0.015) -> STOP
#       - SHORT: entry'den sonra High >= entry*(1+0.015) -> STOP
#       Stop pivot beklemez; her pivotta bir sonraki “signal_idx”e kadar arayı tarar.
#     """
#     closes = df["Close"].values
#     n = len(df)

#     trades: List[Trade] = []
#     current: Optional[Trade] = None

#     last_lb: Optional[SwingPoint] = None
#     last_lh: Optional[SwingPoint] = None

#     for sp in swings:
#         pivot_idx = sp.index
#         signal_idx = pivot_idx + right_bars
#         if signal_idx >= n:
#             continue

#         prev_lb = last_lb
#         prev_lh = last_lh

#         # ✅ 0) STOP LOSS kontrolü: pivotu işlemeden önce arada stop tetiklendi mi?
#         if enable_stop and current is not None:
#             stop_idx = _first_stop_hit_index(
#                 df=df,
#                 direction=current.direction,
#                 start_idx=current.entry_index,
#                 end_idx=signal_idx,
#                 stop_level=current.stop_level
#             )
#             if stop_idx is not None and stop_idx > current.entry_index:
#                 current.exit_index = stop_idx
#                 current.exit_price = current.stop_level  # stop fill price kabul ediyoruz
#                 current.exit_reason = "STOP_LOSS_1_5"
#                 trades.append(current)
#                 current = None
#                 # Stop olduysa bu pivotta yeni entry aramaya devam edeceğiz.

#         # 1) Poz açıkken greed-killer exit + opsiyonel flip
#         if current is not None:
#             if current.direction == "long" and sp.kind == "LH":
#                 if signal_idx > current.entry_index:
#                     current.exit_index = signal_idx
#                     current.exit_price = closes[signal_idx]
#                     current.exit_reason = "long_exit_peak_LH"
#                     trades.append(current)
#                 current = None

#                 if enable_flip and (prev_lh is not None) and (prev_lh.price > sp.price):
#                     ep = closes[signal_idx]
#                     current = Trade(
#                         direction="short",
#                         entry_index=signal_idx,
#                         entry_price=ep,
#                         stop_level=ep * (1.0 + stop_loss_pct),
#                     )

#             elif current.direction == "short" and sp.kind == "LB":
#                 if signal_idx > current.entry_index:
#                     current.exit_index = signal_idx
#                     current.exit_price = closes[signal_idx]
#                     current.exit_reason = "short_exit_dip_LB"
#                     trades.append(current)
#                 current = None

#                 if enable_flip and (prev_lb is not None) and (prev_lb.price < sp.price):
#                     ep = closes[signal_idx]
#                     current = Trade(
#                         direction="long",
#                         entry_index=signal_idx,
#                         entry_price=ep,
#                         stop_level=ep * (1.0 - stop_loss_pct),
#                     )

#         # 2) Poz yokken entry
#         if current is None:
#             if sp.kind == "LB" and prev_lb is not None and prev_lb.price < sp.price:
#                 ep = closes[signal_idx]
#                 current = Trade(
#                     direction="long",
#                     entry_index=signal_idx,
#                     entry_price=ep,
#                     stop_level=ep * (1.0 - stop_loss_pct),
#                 )

#             elif sp.kind == "LH" and prev_lh is not None and prev_lh.price > sp.price:
#                 ep = closes[signal_idx]
#                 current = Trade(
#                     direction="short",
#                     entry_index=signal_idx,
#                     entry_price=ep,
#                     stop_level=ep * (1.0 + stop_loss_pct),
#                 )

#         # pivotları güncelle
#         if sp.kind == "LB":
#             last_lb = sp
#         else:
#             last_lh = sp

#     # açık trade kalırsa (exit yok) ekle
#     if current is not None:
#         trades.append(current)

#     return trades


# # ======================
# # Realistic Backtest Engine
# # ======================

# def calc_trade_pnl(direction: str, entry: float, exit: float, notional: float) -> float:
#     if entry <= 0:
#         return 0.0
#     if direction == "long":
#         return (exit - entry) / entry * notional
#     return (entry - exit) / entry * notional


# def apply_slippage(price: float, direction: str, side: str, slippage_pct: float) -> float:
#     """
#     Market order slippage (aleyhe):
#     LONG entry -> pahalı, LONG exit -> ucuz
#     SHORT entry -> ucuz (sell kötü), SHORT exit -> pahalı (buy kötü)
#     """
#     if slippage_pct <= 0:
#         return price

#     s = slippage_pct
#     if direction == "long":
#         return price * (1.0 + s) if side == "entry" else price * (1.0 - s)
#     else:
#         return price * (1.0 - s) if side == "entry" else price * (1.0 + s)


# def estimate_liquidation_price(entry_price: float, direction: str, leverage: float, maint_margin_rate: float) -> float:
#     """
#     Yaklaşık liquidation eşiği (kaba):
#       LONG  liq ~ entry*(1 - 1/L + mmr)
#       SHORT liq ~ entry*(1 + 1/L - mmr)
#     """
#     if leverage <= 1:
#         return 0.0 if direction == "long" else float("inf")

#     inv = 1.0 / leverage
#     if direction == "long":
#         return entry_price * (1.0 - inv + maint_margin_rate)
#     return entry_price * (1.0 + inv - maint_margin_rate)


# def check_liquidation(df: pd.DataFrame, direction: str, entry_idx: int, exit_idx: int, liq_price: float) -> bool:
#     if exit_idx <= entry_idx:
#         return False
#     window = df.iloc[entry_idx: exit_idx + 1]
#     if direction == "long":
#         return float(window["Low"].min()) <= liq_price
#     return float(window["High"].max()) >= liq_price


# def run_backtest_realistic(
#     df: pd.DataFrame,
#     trades: List[Trade],
#     funding_df: pd.DataFrame,
#     start_balance: float = 1000.0,
#     leverage: float = 10.0,

#     # risk & limits
#     risk_per_trade: float = 0.02,
#     max_notional_usdt: float = 50_000,
#     min_balance: float = 1.0,

#     # fees
#     taker_fee_rate: float = 0.0005,
#     maker_fee_rate: float = 0.0002,
#     assume_market_orders: bool = True,

#     # slippage
#     slippage_pct: float = 0.0003,

#     # liquidation
#     maint_margin_rate: float = 0.005,
# ) -> Tuple[pd.DataFrame, float]:
#     """
#     Çıktı:
#       - details DataFrame (trade bazlı)
#       - end_balance
#     """
#     fee_rate = taker_fee_rate if assume_market_orders else maker_fee_rate

#     f_times = funding_df["fundingTime"].values if not funding_df.empty else np.array([])
#     f_rates = funding_df["fundingRate"].values if not funding_df.empty else np.array([])

#     balance = float(start_balance)
#     idx_to_time = df.index.to_list()
#     rows = []

#     for i, tr in enumerate(trades, start=1):
#         if balance <= min_balance:
#             break

#         if tr.exit_index is None or tr.exit_price is None:
#             continue

#         entry_idx = tr.entry_index
#         exit_idx = tr.exit_index
#         if exit_idx <= entry_idx:
#             continue

#         entry_time = idx_to_time[entry_idx]
#         exit_time = idx_to_time[exit_idx]

#         # risk cap: her trade margin = balance * risk_per_trade
#         margin = balance * risk_per_trade
#         if margin <= 0:
#             continue

#         notional = min(margin * leverage, max_notional_usdt)

#         # slippage (aleyhe)
#         entry_price = apply_slippage(tr.entry_price, tr.direction, "entry", slippage_pct)
#         exit_price  = apply_slippage(tr.exit_price,  tr.direction, "exit",  slippage_pct)

#         # fees (entry+exit)
#         fees = notional * fee_rate * 2.0

#         # funding (entry, exit]
#         funding = 0.0
#         if len(f_times) > 0:
#             et = np.datetime64(entry_time.to_datetime64())
#             xt = np.datetime64(exit_time.to_datetime64())
#             mask = (f_times > et) & (f_times <= xt)
#             sel = f_rates[mask]
#             if sel.size > 0:
#                 raw = float(np.sum(notional * sel))
#                 funding = -raw if tr.direction == "long" else +raw

#         # liquidation check
#         liq_price = estimate_liquidation_price(entry_price, tr.direction, leverage, maint_margin_rate)
#         liquidated = check_liquidation(df, tr.direction, entry_idx, exit_idx, liq_price)

#         balance_before = balance

#         if liquidated:
#             # basit liquidation: margin gider
#             liq_loss = margin
#             net_trade = -liq_loss - fees + funding
#             balance = balance + net_trade

#             rows.append({
#                 "trade_no": i,
#                 "direction": tr.direction,
#                 "entry_time": entry_time,
#                 "exit_time": exit_time,
#                 "entry_price": entry_price,
#                 "exit_price": exit_price,
#                 "notional": notional,
#                 "pnl": 0.0,
#                 "fees": fees,
#                 "funding": funding,
#                 "liquidated": True,
#                 "liq_price_est": liq_price,
#                 "liq_loss": liq_loss,
#                 "net_trade": net_trade,
#                 "balance_before": balance_before,
#                 "balance_after": balance,
#                 "exit_reason": "LIQUIDATION"
#             })
#             continue

#         pnl = calc_trade_pnl(tr.direction, entry_price, exit_price, notional)
#         net_trade = pnl - fees + funding
#         balance = balance + net_trade

#         rows.append({
#             "trade_no": i,
#             "direction": tr.direction,
#             "entry_time": entry_time,
#             "exit_time": exit_time,
#             "entry_price": entry_price,
#             "exit_price": exit_price,
#             "notional": notional,
#             "pnl": pnl,
#             "fees": fees,
#             "funding": funding,
#             "liquidated": False,
#             "liq_price_est": liq_price,
#             "liq_loss": 0.0,
#             "net_trade": net_trade,
#             "balance_before": balance_before,
#             "balance_after": balance,
#             "exit_reason": tr.exit_reason or "pivot_exit"
#         })

#     details = pd.DataFrame(rows)
#     return details, balance


# def print_summary(title: str, start_balance: float, details: pd.DataFrame, end_balance: float) -> None:
#     net = end_balance - start_balance
#     profit_pct = (end_balance / start_balance - 1.0) * 100.0 if start_balance > 0 else 0.0

#     print(f"\n==================== {title} ====================")
#     print(f"Start balance:   ${start_balance:,.2f}")
#     print(f"End balance:     ${end_balance:,.2f}")
#     print(f"NET PnL:         ${net:,.2f}")
#     print(f"Profit %:        {profit_pct:.2f}%")

#     if details.empty:
#         print("No realized trades.")
#         return

#     wins = int((details["net_trade"] > 0).sum())
#     losses = int((details["net_trade"] <= 0).sum())

#     total_fees = float(details["fees"].sum())
#     total_funding = float(details["funding"].sum())
#     total_liq_loss = float(details["liq_loss"].sum())
#     gross_pnl = float(details["pnl"].sum())

#     stop_hits = int((details["exit_reason"].astype(str) == "STOP_LOSS_1_5").sum())
#     stop_net = float(details.loc[details["exit_reason"].astype(str) == "STOP_LOSS_1_5", "net_trade"].sum()) if stop_hits > 0 else 0.0

#     print(f"Trades (realized): {len(details)}")
#     print(f"W/L:             {wins}/{losses}")
#     print(f"Gross PnL:       ${gross_pnl:,.2f}")
#     print(f"Total Fees:      ${total_fees:,.2f}")
#     print(f"Total Funding:   ${total_funding:,.2f}")
#     print(f"Total Liq Loss:  ${total_liq_loss:,.2f}")
#     print(f"Stop hits:       {stop_hits}")
#     print(f"Stop net PnL:    ${stop_net:,.2f}")


# # ======================
# # MAIN
# # ======================

# def main():
#     symbol = "TRXUSDT"
#     interval = "15m"
#     right_bars = 1

#     # ✅ kaç yıl test etmek istiyorsun?
#     years = 5

#     start_balance = 1000.0
#     leverage = 10.0

#     # risk realism knobs
#     risk_per_trade = 0.02
#     max_notional_usdt = 50_000
#     slippage_pct = 0.0003
#     maint_margin_rate = 0.005

#     # Binance fee (tier’ına göre güncelleyebilirsin)
#     taker_fee = 0.0005
#     maker_fee = 0.0002
#     assume_market = True

#     stop_loss_pct = 0.015
#     enable_flip = True

#     print(f"1) OHLC çekiliyor... (years={years})")
#     df = fetch_ohlc_years(symbol, interval, years=years)
#     print(f"OHLC bars: {len(df)} | Range: {df.index[0]} -> {df.index[-1]}")

#     highs = df["High"].values
#     lows = df["Low"].values

#     print("2) Pivotlar bulunuyor...")
#     swings = find_swings(
#         highs,
#         lows,
#         left_bars=5,
#         right_bars=right_bars,
#         min_distance=10,
#         alt_min_distance=None,
#     )
#     print("Pivot sayısı:", len(swings))

#     start_ms = int(df.index[0].value // 10**6)
#     end_ms = int(df.index[-1].value // 10**6)

#     print("3) Funding rate history çekiliyor...")
#     funding_df = fetch_funding_rates(symbol, start_ms, end_ms)
#     print("Funding kayıt:", len(funding_df))

#     # --------------------------
#     # A) STOP KAPALI
#     # --------------------------
#     print("\n4A) Trade’ler üretiliyor (STOP KAPALI)...")
#     trades_no_stop = generate_trades(
#         df=df,
#         swings=swings,
#         right_bars=right_bars,
#         enable_flip=enable_flip,
#         enable_stop=False,          # ✅ stop kapalı
#         stop_loss_pct=stop_loss_pct
#     )

#     print("5A) Backtest (STOP KAPALI)...")
#     details_no_stop, end_no_stop = run_backtest_realistic(
#         df=df,
#         trades=trades_no_stop,
#         funding_df=funding_df,
#         start_balance=start_balance,
#         leverage=leverage,
#         risk_per_trade=risk_per_trade,
#         max_notional_usdt=max_notional_usdt,
#         taker_fee_rate=taker_fee,
#         maker_fee_rate=maker_fee,
#         assume_market_orders=assume_market,
#         slippage_pct=slippage_pct,
#         maint_margin_rate=maint_margin_rate,
#     )
#     print_summary("RESULTS (NO STOP)", start_balance, details_no_stop, end_no_stop)
#     if not details_no_stop.empty:
#         details_no_stop.to_csv("trx_backtest_details_no_stop.csv", index=False)

#     # --------------------------
#     # B) STOP AÇIK %1.5
#     # --------------------------
#     print("\n4B) Trade’ler üretiliyor (STOP AÇIK %1.5)...")
#     trades_with_stop = generate_trades(
#         df=df,
#         swings=swings,
#         right_bars=right_bars,
#         enable_flip=enable_flip,
#         enable_stop=True,           # ✅ stop açık
#         stop_loss_pct=stop_loss_pct
#     )

#     print("5B) Backtest (STOP AÇIK %1.5)...")
#     details_with_stop, end_with_stop = run_backtest_realistic(
#         df=df,
#         trades=trades_with_stop,
#         funding_df=funding_df,
#         start_balance=start_balance,
#         leverage=leverage,
#         risk_per_trade=risk_per_trade,
#         max_notional_usdt=max_notional_usdt,
#         taker_fee_rate=taker_fee,
#         maker_fee_rate=maker_fee,
#         assume_market_orders=assume_market,
#         slippage_pct=slippage_pct,
#         maint_margin_rate=maint_margin_rate,
#     )
#     print_summary("RESULTS (WITH STOP 1.5%)", start_balance, details_with_stop, end_with_stop)
#     if not details_with_stop.empty:
#         details_with_stop.to_csv("trx_backtest_details_with_stop.csv", index=False)

#     # --------------------------
#     # Karşılaştırma
#     # --------------------------
#     profit_no_stop = (end_no_stop / start_balance - 1.0) * 100.0
#     profit_with_stop = (end_with_stop / start_balance - 1.0) * 100.0

#     print("\n==================== STOP COMPARISON ====================")
#     print(f"End (no stop):   ${end_no_stop:,.2f}  | Profit%: {profit_no_stop:.2f}%")
#     print(f"End (with stop): ${end_with_stop:,.2f}  | Profit%: {profit_with_stop:.2f}%")
#     print(f"Delta Profit%:   {(profit_with_stop - profit_no_stop):.2f}%")
#     if not details_with_stop.empty:
#         print(f"Stop hits:       {int((details_with_stop['exit_reason'].astype(str) == 'STOP_LOSS_1_5').sum())}")
#     else:
#         print("Stop hits:       0")

#     print("\nCSV çıktıları:")
#     print(" - trx_backtest_details_no_stop.csv")
#     print(" - trx_backtest_details_with_stop.csv")


# if __name__ == "__main__":
#     main()



# import time
# import math
# import requests
# import numpy as np
# import pandas as pd
# from dataclasses import dataclass
# from typing import Literal, Optional, List, Tuple

# # plot_swings_from_api.py içinde find_swings ve SwingPoint olmalı
# from plot_swings_from_api import find_swings, SwingPoint  # type: ignore

# BINANCE_FAPI = "https://fapi.binance.com"


# # ======================
# # Helpers
# # ======================

# def _to_utc(ts: pd.Timestamp) -> pd.Timestamp:
#     if ts.tzinfo is None:
#         return ts.tz_localize("UTC")
#     return ts.tz_convert("UTC")


# def _ms(dt: pd.Timestamp) -> int:
#     dt = _to_utc(dt)
#     return int(dt.value // 10**6)


# # ======================
# # Binance Futures: OHLC + Funding
# # ======================

# def fetch_klines_chunk(symbol: str, interval: str, start_ms: int, limit: int = 1500) -> pd.DataFrame:
#     url = f"{BINANCE_FAPI}/fapi/v1/klines"
#     params = {"symbol": symbol, "interval": interval, "startTime": start_ms, "limit": limit}
#     r = requests.get(url, params=params, timeout=20)
#     r.raise_for_status()
#     data = r.json()
#     if not data:
#         return pd.DataFrame()

#     cols = [
#         "open_time", "open", "high", "low", "close", "volume",
#         "close_time", "qav", "num_trades", "tbbv", "tbqv", "ignore"
#     ]
#     df = pd.DataFrame(data, columns=cols)
#     df["Date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
#     df.set_index("Date", inplace=True)

#     for c in ["open", "high", "low", "close", "volume"]:
#         df[c] = df[c].astype(float)

#     df = df[["open", "high", "low", "close", "volume"]]
#     df.columns = ["Open", "High", "Low", "Close", "Volume"]
#     return df


# def fetch_ohlc_years(symbol: str, interval: str, years: int = 3, end_utc: Optional[pd.Timestamp] = None) -> pd.DataFrame:
#     if end_utc is None:
#         end_utc = pd.Timestamp.utcnow()
#     end_utc = _to_utc(end_utc)

#     start_utc = end_utc - pd.Timedelta(days=365 * years)
#     start_ms = _ms(start_utc)

#     out = []
#     while True:
#         chunk = fetch_klines_chunk(symbol, interval, start_ms, limit=1500)
#         if chunk.empty:
#             break
#         out.append(chunk)

#         last_open_ms = int(chunk.index[-1].value // 10**6)
#         start_ms = last_open_ms + 1

#         time.sleep(0.05)

#         if chunk.index[-1] >= end_utc:
#             break

#     if not out:
#         raise RuntimeError("OHLC çekilemedi. Sembol/interval doğru mu?")

#     df = pd.concat(out).sort_index()
#     df = df[~df.index.duplicated(keep="first")]
#     df = df[(df.index >= start_utc) & (df.index <= end_utc)]
#     return df


# def fetch_funding_rates(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
#     url = f"{BINANCE_FAPI}/fapi/v1/fundingRate"

#     rows = []
#     cursor = start_ms
#     while True:
#         params = {"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000}
#         r = requests.get(url, params=params, timeout=20)
#         r.raise_for_status()
#         data = r.json()
#         if not data:
#             break

#         rows.extend(data)
#         last_time = int(data[-1]["fundingTime"])
#         cursor = last_time + 1

#         time.sleep(0.05)

#         if last_time >= end_ms or len(data) < 1000:
#             break

#     if not rows:
#         return pd.DataFrame(columns=["fundingTime", "fundingRate"])

#     fr = pd.DataFrame(rows)
#     fr["fundingTime"] = pd.to_datetime(fr["fundingTime"].astype(np.int64), unit="ms", utc=True)
#     fr["fundingRate"] = fr["fundingRate"].astype(float)
#     fr = fr[["fundingTime", "fundingRate"]].sort_values("fundingTime")
#     fr = fr.drop_duplicates(subset=["fundingTime"], keep="first")
#     return fr


# # ======================
# # Strategy Trades (+ STOP in swings logic)
# # ======================

# @dataclass
# class Trade:
#     direction: Literal["long", "short"]
#     entry_index: int
#     entry_price: float
#     stop_level: float
#     exit_index: Optional[int] = None
#     exit_price: Optional[float] = None
#     exit_reason: Optional[str] = None


# def _first_stop_hit_index(
#     df: pd.DataFrame,
#     direction: str,
#     start_idx: int,
#     end_idx: int,
#     stop_level: float
# ) -> Optional[int]:
#     """
#     (start_idx, end_idx] aralığında stop'a ilk değilen bar indexini döndürür.
#     LONG : Low  <= stop_level
#     SHORT: High >= stop_level
#     """
#     if end_idx <= start_idx:
#         return None

#     window = df.iloc[start_idx + 1: end_idx + 1]
#     if window.empty:
#         return None

#     if direction == "long":
#         hits = window["Low"].values <= stop_level
#     else:
#         hits = window["High"].values >= stop_level

#     if not hits.any():
#         return None

#     first_pos = int(np.argmax(hits))
#     return (start_idx + 1) + first_pos


# def generate_trades(
#     df: pd.DataFrame,
#     swings: List[SwingPoint],
#     right_bars: int = 1,
#     enable_flip: bool = True,
#     enable_stop: bool = True,
#     stop_loss_pct: float = 0.015,
# ) -> List[Trade]:
#     closes = df["Close"].values
#     n = len(df)

#     trades: List[Trade] = []
#     current: Optional[Trade] = None

#     last_lb: Optional[SwingPoint] = None
#     last_lh: Optional[SwingPoint] = None

#     for sp in swings:
#         pivot_idx = sp.index
#         signal_idx = pivot_idx + right_bars
#         if signal_idx >= n:
#             continue

#         prev_lb = last_lb
#         prev_lh = last_lh

#         # 0) STOP kontrol (bar-içi worst-case: stop önce gelir)
#         if enable_stop and current is not None:
#             stop_idx = _first_stop_hit_index(
#                 df=df,
#                 direction=current.direction,
#                 start_idx=current.entry_index,
#                 end_idx=signal_idx,
#                 stop_level=current.stop_level
#             )
#             if stop_idx is not None and stop_idx > current.entry_index:
#                 current.exit_index = stop_idx
#                 current.exit_price = current.stop_level
#                 current.exit_reason = "STOP_LOSS"
#                 trades.append(current)
#                 current = None

#         # 1) Poz açıkken pivot exit + opsiyonel flip
#         if current is not None:
#             if current.direction == "long" and sp.kind == "LH":
#                 if signal_idx > current.entry_index:
#                     current.exit_index = signal_idx
#                     current.exit_price = closes[signal_idx]
#                     current.exit_reason = "long_exit_LH"
#                     trades.append(current)
#                 current = None

#                 if enable_flip and (prev_lh is not None) and (prev_lh.price > sp.price):
#                     ep = closes[signal_idx]
#                     current = Trade(
#                         direction="short",
#                         entry_index=signal_idx,
#                         entry_price=ep,
#                         stop_level=ep * (1.0 + stop_loss_pct),
#                     )

#             elif current.direction == "short" and sp.kind == "LB":
#                 if signal_idx > current.entry_index:
#                     current.exit_index = signal_idx
#                     current.exit_price = closes[signal_idx]
#                     current.exit_reason = "short_exit_LB"
#                     trades.append(current)
#                 current = None

#                 if enable_flip and (prev_lb is not None) and (prev_lb.price < sp.price):
#                     ep = closes[signal_idx]
#                     current = Trade(
#                         direction="long",
#                         entry_index=signal_idx,
#                         entry_price=ep,
#                         stop_level=ep * (1.0 - stop_loss_pct),
#                     )

#         # 2) Poz yokken entry
#         if current is None:
#             if sp.kind == "LB" and prev_lb is not None and prev_lb.price < sp.price:
#                 ep = closes[signal_idx]
#                 current = Trade(
#                     direction="long",
#                     entry_index=signal_idx,
#                     entry_price=ep,
#                     stop_level=ep * (1.0 - stop_loss_pct),
#                 )

#             elif sp.kind == "LH" and prev_lh is not None and prev_lh.price > sp.price:
#                 ep = closes[signal_idx]
#                 current = Trade(
#                     direction="short",
#                     entry_index=signal_idx,
#                     entry_price=ep,
#                     stop_level=ep * (1.0 + stop_loss_pct),
#                 )

#         if sp.kind == "LB":
#             last_lb = sp
#         else:
#             last_lh = sp

#     # Açık trade kalırsa ekle (exit yok) -> backtest'te force-close edeceğiz
#     if current is not None:
#         trades.append(current)

#     return trades


# # ======================
# # Worst-case Execution Model
# # ======================

# @dataclass
# class WorstCaseExecConfig:
#     # fees
#     taker_fee_rate: float = 0.0007   # normalden daha kötü
#     maker_fee_rate: float = 0.0003
#     assume_market_orders: bool = True

#     # slippage base
#     base_slippage_pct: float = 0.0008  # 0.08% base (kötü)
#     vol_slippage_k: float = 0.35       # mum volatilitesine çarpan
#     impact_k: float = 0.20             # market impact çarpanı (kötü)
#     max_slippage_pct: float = 0.01     # slippage üst sınır %1

#     # stop extra penalty
#     stop_extra_slippage_pct: float = 0.0015  # stop fill ekstra kötüleşsin

#     # funding worst-case
#     funding_always_pay_abs: bool = True

#     # liquidation harsher
#     maint_margin_rate: float = 0.012  # daha yüksek MMR (daha kolay liq)

#     # end-of-test behavior
#     force_close_last_trade: bool = True


# def _adv_usdt(df: pd.DataFrame, idx: int, lookback_bars: int = 96) -> float:
#     """
#     Basit ADV(USDT) yaklaşımı: sum(Volume * Close) lookback.
#     96 bar ~ 1 gün (15m).
#     """
#     a = max(0, idx - lookback_bars)
#     w = df.iloc[a: idx + 1]
#     if w.empty:
#         return 0.0
#     adv = float(np.sum(w["Volume"].values * w["Close"].values))
#     return max(adv, 1.0)


# def _bar_volatility(df: pd.DataFrame, idx: int) -> float:
#     """
#     Basit bar volatilitesi: (High-Low)/Close
#     """
#     c = float(df["Close"].iloc[idx])
#     if c <= 0:
#         return 0.0
#     h = float(df["High"].iloc[idx])
#     l = float(df["Low"].iloc[idx])
#     return max(0.0, (h - l) / c)


# def _impact_pct(notional: float, adv_usdt: float, impact_k: float) -> float:
#     """
#     impact ~ k * sqrt(notional/adv)
#     """
#     ratio = notional / max(adv_usdt, 1.0)
#     return impact_k * math.sqrt(max(ratio, 0.0))


# def _clamp(x: float, lo: float, hi: float) -> float:
#     return max(lo, min(hi, x))


# def _worst_fill_price(
#     df: pd.DataFrame,
#     idx: int,
#     direction: str,
#     side: Literal["entry", "exit"],
#     base_price: float,
#     slippage_pct: float,
#     impact_pct: float
# ) -> float:
#     """
#     Worst-case fill: OHLC içinde "aleyhe" fiyat seç + slippage + impact uygula.

#     - LONG entry: olası en pahalı (max(Open, Close)) * (1 + slip+impact)
#     - LONG exit (sell): olası en ucuz (min(Open, Close)) * (1 - slip-impact)

#     - SHORT entry (sell): olası en ucuz (min(Open, Close)) * (1 - slip-impact)
#     - SHORT exit (buy): olası en pahalı (max(Open, Close)) * (1 + slip+impact)

#     base_price'ı doğrudan kullanmak yerine, bar içi muhafazakar seçime gidiyoruz.
#     """
#     o = float(df["Open"].iloc[idx])
#     c = float(df["Close"].iloc[idx])

#     slip = slippage_pct + impact_pct
#     slip = _clamp(slip, 0.0, 0.05)  # güvenlik: %5 üstünü burada kesiyoruz

#     if direction == "long":
#         if side == "entry":
#             p = max(o, c, base_price)
#             return p * (1.0 + slip)
#         else:
#             p = min(o, c, base_price)
#             return p * (1.0 - slip)
#     else:
#         if side == "entry":
#             p = min(o, c, base_price)
#             return p * (1.0 - slip)
#         else:
#             p = max(o, c, base_price)
#             return p * (1.0 + slip)


# def calc_trade_pnl(direction: str, entry: float, exit: float, notional: float) -> float:
#     if entry <= 0:
#         return 0.0
#     if direction == "long":
#         return (exit - entry) / entry * notional
#     return (entry - exit) / entry * notional


# def estimate_liquidation_price(entry_price: float, direction: str, leverage: float, maint_margin_rate: float) -> float:
#     """
#     Kaba liq:
#       LONG  liq ~ entry*(1 - 1/L + mmr)
#       SHORT liq ~ entry*(1 + 1/L - mmr)
#     """
#     if leverage <= 1:
#         return 0.0 if direction == "long" else float("inf")

#     inv = 1.0 / leverage
#     if direction == "long":
#         return entry_price * (1.0 - inv + maint_margin_rate)
#     return entry_price * (1.0 + inv - maint_margin_rate)


# def check_liquidation(df: pd.DataFrame, direction: str, entry_idx: int, exit_idx: int, liq_price: float) -> bool:
#     if exit_idx <= entry_idx:
#         return False
#     window = df.iloc[entry_idx: exit_idx + 1]
#     if direction == "long":
#         return float(window["Low"].min()) <= liq_price
#     return float(window["High"].max()) >= liq_price


# def run_backtest_worst_case(
#     df: pd.DataFrame,
#     trades: List[Trade],
#     funding_df: pd.DataFrame,
#     exec_cfg: WorstCaseExecConfig,
#     start_balance: float = 1000.0,
#     leverage: float = 10.0,

#     # risk & limits
#     risk_per_trade: float = 0.02,
#     max_notional_usdt: float = 50_000,
#     min_balance: float = 1.0,
# ) -> Tuple[pd.DataFrame, float]:
#     fee_rate = exec_cfg.taker_fee_rate if exec_cfg.assume_market_orders else exec_cfg.maker_fee_rate

#     f_times = funding_df["fundingTime"].values if not funding_df.empty else np.array([])
#     f_rates = funding_df["fundingRate"].values if not funding_df.empty else np.array([])

#     balance = float(start_balance)
#     idx_to_time = df.index.to_list()
#     rows = []

#     last_bar_idx = len(df) - 1

#     for i, tr in enumerate(trades, start=1):
#         if balance <= min_balance:
#             break

#         # force-close last open trade at end of dataset (worst-case)
#         if (tr.exit_index is None or tr.exit_price is None) and exec_cfg.force_close_last_trade:
#             tr.exit_index = last_bar_idx
#             tr.exit_price = float(df["Close"].iloc[last_bar_idx])
#             tr.exit_reason = tr.exit_reason or "FORCE_CLOSE_END"

#         if tr.exit_index is None or tr.exit_price is None:
#             continue

#         entry_idx = tr.entry_index
#         exit_idx = tr.exit_index
#         if exit_idx <= entry_idx:
#             continue

#         entry_time = idx_to_time[entry_idx]
#         exit_time = idx_to_time[exit_idx]

#         margin = balance * risk_per_trade
#         if margin <= 0:
#             continue

#         notional = min(margin * leverage, max_notional_usdt)

#         # dynamic slippage components
#         adv_e = _adv_usdt(df, entry_idx, lookback_bars=96)
#         adv_x = _adv_usdt(df, exit_idx, lookback_bars=96)
#         vol_e = _bar_volatility(df, entry_idx)
#         vol_x = _bar_volatility(df, exit_idx)

#         impact_e = _impact_pct(notional, adv_e, exec_cfg.impact_k)
#         impact_x = _impact_pct(notional, adv_x, exec_cfg.impact_k)

#         slip_e = exec_cfg.base_slippage_pct + exec_cfg.vol_slippage_k * vol_e
#         slip_x = exec_cfg.base_slippage_pct + exec_cfg.vol_slippage_k * vol_x

#         slip_e = _clamp(slip_e, 0.0, exec_cfg.max_slippage_pct)
#         slip_x = _clamp(slip_x, 0.0, exec_cfg.max_slippage_pct)

#         # worst-case entry/exit fills
#         entry_price = _worst_fill_price(
#             df=df, idx=entry_idx, direction=tr.direction, side="entry",
#             base_price=tr.entry_price, slippage_pct=slip_e, impact_pct=impact_e
#         )

#         # stop exit ise ekstra stop slippage cezası uygula
#         is_stop = (str(tr.exit_reason).startswith("STOP_LOSS") if tr.exit_reason else False)
#         extra_stop = exec_cfg.stop_extra_slippage_pct if is_stop else 0.0

#         exit_price = _worst_fill_price(
#             df=df, idx=exit_idx, direction=tr.direction, side="exit",
#             base_price=tr.exit_price, slippage_pct=(slip_x + extra_stop), impact_pct=impact_x
#         )

#         # fees
#         fees = notional * fee_rate * 2.0

#         # funding (entry, exit]
#         funding = 0.0
#         if len(f_times) > 0:
#             et = np.datetime64(entry_time.to_datetime64())
#             xt = np.datetime64(exit_time.to_datetime64())
#             mask = (f_times > et) & (f_times <= xt)
#             sel = f_rates[mask]
#             if sel.size > 0:
#                 raw = float(np.sum(notional * sel))
#                 if exec_cfg.funding_always_pay_abs:
#                     funding = -abs(raw)  # WORST CASE: her zaman öde
#                 else:
#                     # daha normal yaklaşım (istersen kapat):
#                     # pozitif funding: longs pay, shorts receive; negatif tersi
#                     # burada sign zaten raw içinde var
#                     funding = -raw if tr.direction == "long" else +raw

#         # liquidation
#         liq_price = estimate_liquidation_price(entry_price, tr.direction, leverage, exec_cfg.maint_margin_rate)
#         liquidated = check_liquidation(df, tr.direction, entry_idx, exit_idx, liq_price)

#         balance_before = balance

#         if liquidated:
#             liq_loss = margin
#             net_trade = -liq_loss - fees + funding
#             balance = balance + net_trade

#             rows.append({
#                 "trade_no": i,
#                 "direction": tr.direction,
#                 "entry_time": entry_time,
#                 "exit_time": exit_time,
#                 "entry_price": entry_price,
#                 "exit_price": exit_price,
#                 "notional": notional,
#                 "pnl": 0.0,
#                 "fees": fees,
#                 "funding": funding,
#                 "liquidated": True,
#                 "liq_price_est": liq_price,
#                 "liq_loss": liq_loss,
#                 "net_trade": net_trade,
#                 "balance_before": balance_before,
#                 "balance_after": balance,
#                 "exit_reason": "LIQUIDATION"
#             })
#             continue

#         pnl = calc_trade_pnl(tr.direction, entry_price, exit_price, notional)
#         net_trade = pnl - fees + funding
#         balance = balance + net_trade

#         rows.append({
#             "trade_no": i,
#             "direction": tr.direction,
#             "entry_time": entry_time,
#             "exit_time": exit_time,
#             "entry_price": entry_price,
#             "exit_price": exit_price,
#             "notional": notional,
#             "pnl": pnl,
#             "fees": fees,
#             "funding": funding,
#             "liquidated": False,
#             "liq_price_est": liq_price,
#             "liq_loss": 0.0,
#             "net_trade": net_trade,
#             "balance_before": balance_before,
#             "balance_after": balance,
#             "exit_reason": tr.exit_reason or "pivot_exit"
#         })

#     details = pd.DataFrame(rows)
#     return details, balance


# def print_summary(title: str, start_balance: float, details: pd.DataFrame, end_balance: float) -> None:
#     net = end_balance - start_balance
#     profit_pct = (end_balance / start_balance - 1.0) * 100.0 if start_balance > 0 else 0.0

#     print(f"\n==================== {title} ====================")
#     print(f"Start balance:   ${start_balance:,.2f}")
#     print(f"End balance:     ${end_balance:,.2f}")
#     print(f"NET PnL:         ${net:,.2f}")
#     print(f"Profit %:        {profit_pct:.2f}%")

#     if details.empty:
#         print("No realized trades.")
#         return

#     wins = int((details["net_trade"] > 0).sum())
#     losses = int((details["net_trade"] <= 0).sum())

#     total_fees = float(details["fees"].sum())
#     total_funding = float(details["funding"].sum())
#     total_liq_loss = float(details["liq_loss"].sum())
#     gross_pnl = float(details["pnl"].sum())

#     stop_hits = int(details["exit_reason"].astype(str).str.startswith("STOP_LOSS").sum())
#     stop_net = float(details.loc[details["exit_reason"].astype(str).str.startswith("STOP_LOSS"), "net_trade"].sum()) if stop_hits > 0 else 0.0

#     print(f"Trades (realized): {len(details)}")
#     print(f"W/L:             {wins}/{losses}")
#     print(f"Gross PnL:       ${gross_pnl:,.2f}")
#     print(f"Total Fees:      ${total_fees:,.2f}")
#     print(f"Total Funding:   ${total_funding:,.2f}")
#     print(f"Total Liq Loss:  ${total_liq_loss:,.2f}")
#     print(f"Stop hits:       {stop_hits}")
#     print(f"Stop net PnL:    ${stop_net:,.2f}")


# # ======================
# # MAIN
# # ======================

# def main():
#     symbol = "TRXUSDT"
#     interval = "15m"
#     right_bars = 1
#     years = 5

#     start_balance = 1000.0
#     leverage = 10.0

#     # risk
#     risk_per_trade = 0.02
#     max_notional_usdt = 50_000

#     stop_loss_pct = 0.015
#     enable_flip = True

#     # WORST CASE config
#     exec_cfg = WorstCaseExecConfig(
#         taker_fee_rate=0.0007,
#         maker_fee_rate=0.0003,
#         assume_market_orders=True,
#         base_slippage_pct=0.0008,
#         vol_slippage_k=0.35,
#         impact_k=0.20,
#         max_slippage_pct=0.01,
#         stop_extra_slippage_pct=0.0015,
#         funding_always_pay_abs=True,
#         maint_margin_rate=0.012,
#         force_close_last_trade=True
#     )

#     print(f"1) OHLC çekiliyor... (years={years})")
#     df = fetch_ohlc_years(symbol, interval, years=years)
#     print(f"OHLC bars: {len(df)} | Range: {df.index[0]} -> {df.index[-1]}")

#     highs = df["High"].values
#     lows = df["Low"].values

#     print("2) Pivotlar bulunuyor...")
#     swings = find_swings(
#         highs,
#         lows,
#         left_bars=5,
#         right_bars=right_bars,
#         min_distance=10,
#         alt_min_distance=None,
#     )
#     print("Pivot sayısı:", len(swings))

#     start_ms = int(df.index[0].value // 10**6)
#     end_ms = int(df.index[-1].value // 10**6)

#     print("3) Funding rate history çekiliyor...")
#     funding_df = fetch_funding_rates(symbol, start_ms, end_ms)
#     print("Funding kayıt:", len(funding_df))

#     # --------------------------
#     # A) STOP KAPALI (worst-case execution)
#     # --------------------------
#     print("\n4A) Trade’ler üretiliyor (STOP KAPALI)...")
#     trades_no_stop = generate_trades(
#         df=df,
#         swings=swings,
#         right_bars=right_bars,
#         enable_flip=enable_flip,
#         enable_stop=False,
#         stop_loss_pct=stop_loss_pct
#     )

#     print("5A) Backtest (STOP KAPALI) [WORST-CASE EXEC]...")
#     details_no_stop, end_no_stop = run_backtest_worst_case(
#         df=df,
#         trades=trades_no_stop,
#         funding_df=funding_df,
#         exec_cfg=exec_cfg,
#         start_balance=start_balance,
#         leverage=leverage,
#         risk_per_trade=risk_per_trade,
#         max_notional_usdt=max_notional_usdt,
#     )
#     print_summary("RESULTS (NO STOP) [WORST-CASE]", start_balance, details_no_stop, end_no_stop)
#     if not details_no_stop.empty:
#         details_no_stop.to_csv("trx_backtest_details_no_stop_worst.csv", index=False)

#     # --------------------------
#     # B) STOP AÇIK %1.5 (worst-case execution)
#     # --------------------------
#     print("\n4B) Trade’ler üretiliyor (STOP AÇIK %1.5)...")
#     trades_with_stop = generate_trades(
#         df=df,
#         swings=swings,
#         right_bars=right_bars,
#         enable_flip=enable_flip,
#         enable_stop=True,
#         stop_loss_pct=stop_loss_pct
#     )

#     print("5B) Backtest (STOP AÇIK %1.5) [WORST-CASE EXEC]...")
#     details_with_stop, end_with_stop = run_backtest_worst_case(
#         df=df,
#         trades=trades_with_stop,
#         funding_df=funding_df,
#         exec_cfg=exec_cfg,
#         start_balance=start_balance,
#         leverage=leverage,
#         risk_per_trade=risk_per_trade,
#         max_notional_usdt=max_notional_usdt,
#     )
#     print_summary("RESULTS (WITH STOP 1.5%) [WORST-CASE]", start_balance, details_with_stop, end_with_stop)
#     if not details_with_stop.empty:
#         details_with_stop.to_csv("trx_backtest_details_with_stop_worst.csv", index=False)

#     # --------------------------
#     # Compare
#     # --------------------------
#     profit_no_stop = (end_no_stop / start_balance - 1.0) * 100.0
#     profit_with_stop = (end_with_stop / start_balance - 1.0) * 100.0

#     print("\n==================== STOP COMPARISON [WORST-CASE] ====================")
#     print(f"End (no stop):   ${end_no_stop:,.2f}  | Profit%: {profit_no_stop:.2f}%")
#     print(f"End (with stop): ${end_with_stop:,.2f}  | Profit%: {profit_with_stop:.2f}%")
#     print(f"Delta Profit%:   {(profit_with_stop - profit_no_stop):.2f}%")
#     if not details_with_stop.empty:
#         print(f"Stop hits:       {int(details_with_stop['exit_reason'].astype(str).str.startswith('STOP_LOSS').sum())}")
#     else:
#         print("Stop hits:       0")

#     print("\nCSV çıktıları:")
#     print(" - trx_backtest_details_no_stop_worst.csv")
#     print(" - trx_backtest_details_with_stop_worst.csv")


# if __name__ == "__main__":
#     main()


import time
import math
import requests
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal, Optional, List, Tuple

# plot_swings_from_api.py içinde find_swings ve SwingPoint olmalı
from plot_swings_from_api import find_swings, SwingPoint  # type: ignore

BINANCE_FAPI = "https://fapi.binance.com"


# ======================
# Helpers
# ======================

def _to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _ms(dt: pd.Timestamp) -> int:
    dt = _to_utc(dt)
    return int(dt.value // 10**6)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ======================
# Binance Futures: OHLC + Funding
# ======================

def fetch_klines_chunk(symbol: str, interval: str, start_ms: int, limit: int = 1500) -> pd.DataFrame:
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "startTime": start_ms, "limit": limit}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not data:
        return pd.DataFrame()

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "num_trades", "tbbv", "tbqv", "ignore"
    ]
    df = pd.DataFrame(data, columns=cols)
    df["Date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("Date", inplace=True)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    df = df[["open", "high", "low", "close", "volume"]]
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    return df


def fetch_ohlc_years(symbol: str, interval: str, years: int = 3, end_utc: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    if end_utc is None:
        end_utc = pd.Timestamp.utcnow()
    end_utc = _to_utc(end_utc)

    start_utc = end_utc - pd.Timedelta(days=365 * years)
    start_ms = _ms(start_utc)

    out = []
    while True:
        chunk = fetch_klines_chunk(symbol, interval, start_ms, limit=1500)
        if chunk.empty:
            break
        out.append(chunk)

        last_open_ms = int(chunk.index[-1].value // 10**6)
        start_ms = last_open_ms + 1

        time.sleep(0.05)

        if chunk.index[-1] >= end_utc:
            break

    if not out:
        raise RuntimeError("OHLC çekilemedi. Sembol/interval doğru mu?")

    df = pd.concat(out).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df[(df.index >= start_utc) & (df.index <= end_utc)]
    return df


def fetch_funding_rates(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    url = f"{BINANCE_FAPI}/fapi/v1/fundingRate"

    rows = []
    cursor = start_ms
    while True:
        params = {"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": 1000}
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not data:
            break

        rows.extend(data)
        last_time = int(data[-1]["fundingTime"])
        cursor = last_time + 1

        time.sleep(0.05)

        if last_time >= end_ms or len(data) < 1000:
            break

    if not rows:
        return pd.DataFrame(columns=["fundingTime", "fundingRate"])

    fr = pd.DataFrame(rows)
    fr["fundingTime"] = pd.to_datetime(fr["fundingTime"].astype(np.int64), unit="ms", utc=True)
    fr["fundingRate"] = fr["fundingRate"].astype(float)
    fr = fr[["fundingTime", "fundingRate"]].sort_values("fundingTime")
    fr = fr.drop_duplicates(subset=["fundingTime"], keep="first")
    return fr


# ======================
# Strategy Trades (+ STOP in swings logic)
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


def _first_stop_hit_index(
    df: pd.DataFrame,
    direction: str,
    start_idx: int,
    end_idx: int,
    stop_level: float
) -> Optional[int]:
    """
    (start_idx, end_idx] aralığında stop'a ilk değilen bar indexini döndürür.
    LONG : Low  <= stop_level
    SHORT: High >= stop_level
    """
    if end_idx <= start_idx:
        return None

    window = df.iloc[start_idx + 1: end_idx + 1]
    if window.empty:
        return None

    if direction == "long":
        hits = window["Low"].values <= stop_level
    else:
        hits = window["High"].values >= stop_level

    if not hits.any():
        return None

    first_pos = int(np.argmax(hits))
    return (start_idx + 1) + first_pos


def generate_trades(
    df: pd.DataFrame,
    swings: List[SwingPoint],
    right_bars: int = 1,
    enable_flip: bool = True,
    enable_stop: bool = True,
    stop_loss_pct: float = 0.015,
) -> List[Trade]:
    closes = df["Close"].values
    n = len(df)

    trades: List[Trade] = []
    current: Optional[Trade] = None

    last_lb: Optional[SwingPoint] = None
    last_lh: Optional[SwingPoint] = None

    for sp in swings:
        pivot_idx = sp.index
        signal_idx = pivot_idx + right_bars
        if signal_idx >= n:
            continue

        prev_lb = last_lb
        prev_lh = last_lh

        # 0) STOP kontrol (bar-içi: stop önce gelir)
        if enable_stop and current is not None:
            stop_idx = _first_stop_hit_index(
                df=df,
                direction=current.direction,
                start_idx=current.entry_index,
                end_idx=signal_idx,
                stop_level=current.stop_level
            )
            if stop_idx is not None and stop_idx > current.entry_index:
                current.exit_index = stop_idx
                current.exit_price = current.stop_level
                current.exit_reason = "STOP_LOSS"
                trades.append(current)
                current = None

        # 1) Poz açıkken pivot exit + opsiyonel flip
        if current is not None:
            if current.direction == "long" and sp.kind == "LH":
                if signal_idx > current.entry_index:
                    current.exit_index = signal_idx
                    current.exit_price = closes[signal_idx]
                    current.exit_reason = "long_exit_LH"
                    trades.append(current)
                current = None

                if enable_flip and (prev_lh is not None) and (prev_lh.price > sp.price):
                    ep = closes[signal_idx]
                    current = Trade(
                        direction="short",
                        entry_index=signal_idx,
                        entry_price=ep,
                        stop_level=ep * (1.0 + stop_loss_pct),
                    )

            elif current.direction == "short" and sp.kind == "LB":
                if signal_idx > current.entry_index:
                    current.exit_index = signal_idx
                    current.exit_price = closes[signal_idx]
                    current.exit_reason = "short_exit_LB"
                    trades.append(current)
                current = None

                if enable_flip and (prev_lb is not None) and (prev_lb.price < sp.price):
                    ep = closes[signal_idx]
                    current = Trade(
                        direction="long",
                        entry_index=signal_idx,
                        entry_price=ep,
                        stop_level=ep * (1.0 - stop_loss_pct),
                    )

        # 2) Poz yokken entry
        if current is None:
            if sp.kind == "LB" and prev_lb is not None and prev_lb.price < sp.price:
                ep = closes[signal_idx]
                current = Trade(
                    direction="long",
                    entry_index=signal_idx,
                    entry_price=ep,
                    stop_level=ep * (1.0 - stop_loss_pct),
                )

            elif sp.kind == "LH" and prev_lh is not None and prev_lh.price > sp.price:
                ep = closes[signal_idx]
                current = Trade(
                    direction="short",
                    entry_index=signal_idx,
                    entry_price=ep,
                    stop_level=ep * (1.0 + stop_loss_pct),
                )

        if sp.kind == "LB":
            last_lb = sp
        else:
            last_lh = sp

    # açık trade kalırsa ekle (exit yok) -> backtest'te force-close edeceğiz
    if current is not None:
        trades.append(current)

    return trades


# ======================
# Realistic Execution Model (Balance-aware slippage + spread)
# ======================

@dataclass
class RealisticExecConfig:
    # Fees (Binance Futures USDT-M tipik taker/maker; tier'a göre değişir)
    taker_fee_rate: float = 0.0005
    maker_fee_rate: float = 0.0002
    assume_market_orders: bool = True

    # Base slippage (normal koşul)
    base_slippage_pct: float = 0.00025   # 0.025%
    vol_slippage_k: float = 0.25         # bar vol ile artış
    impact_k: float = 0.08               # sqrt(notional/adv) çarpanı
    max_slippage_pct: float = 0.006      # 0.6% tavan

    # Balance büyüdükçe slippage şişsin:
    # slip *= (1 + balance_slip_k * log10(balance/start_balance))
    balance_slip_k: float = 0.18

    # Spread modeli: spread_pct ~ spread_base + spread_vol_k * bar_vol
    spread_base_pct: float = 0.00010     # 0.01%
    spread_vol_k: float = 0.20

    # Stop exit'lerinde ekstra kötü fill (stoplar genelde kayar)
    stop_extra_slippage_pct: float = 0.00035

    # Funding: gerçekçi sign (long/short)
    funding_worstcase: bool = False

    # Liquidation hesabı (bakiyeyi 0'lamayacağız; ayrı blow-up kuralı ekleyeceğiz)
    maint_margin_rate: float = 0.0075

    # Dataset sonu: açık trade'i kapat
    force_close_last_trade: bool = True


def _adv_usdt(df: pd.DataFrame, idx: int, lookback_bars: int = 96) -> float:
    w = df.iloc[max(0, idx - lookback_bars): idx + 1]
    if w.empty:
        return 1.0
    adv = float(np.sum(w["Volume"].values * w["Close"].values))
    return max(adv, 1.0)


def _bar_volatility(df: pd.DataFrame, idx: int) -> float:
    c = float(df["Close"].iloc[idx])
    if c <= 0:
        return 0.0
    h = float(df["High"].iloc[idx])
    l = float(df["Low"].iloc[idx])
    return max(0.0, (h - l) / c)


def _impact_pct(notional: float, adv_usdt: float, impact_k: float) -> float:
    ratio = notional / max(adv_usdt, 1.0)
    return impact_k * math.sqrt(max(ratio, 0.0))


def _balance_multiplier(balance: float, start_balance: float, k: float) -> float:
    """
    balance büyüdükçe 1 -> 1 + k*log10(balance/start)
    Örn: start=1k, balance=100k => log10(100)=2 => 1+2k
    """
    if balance <= 0 or start_balance <= 0:
        return 1.0
    x = balance / start_balance
    if x <= 1:
        return 1.0
    return 1.0 + k * math.log10(x)


def _spread_pct(df: pd.DataFrame, idx: int, cfg: RealisticExecConfig) -> float:
    v = _bar_volatility(df, idx)
    s = cfg.spread_base_pct + cfg.spread_vol_k * v
    return _clamp(s, 0.0, 0.01)  # %1 tavan güvenlik


def _fill_price_with_costs(
    df: pd.DataFrame,
    idx: int,
    direction: str,
    side: Literal["entry", "exit"],
    base_price: float,
    slip_pct: float,
    spr_pct: float,
    extra_slip_pct: float = 0.0,
) -> float:
    """
    Market order gibi düşün:
      - Spread: entry'de aleyhe +spr/2, exit'te aleyhe +spr/2 (yönüne göre)
      - Slippage: aleyhe
    """
    o = float(df["Open"].iloc[idx])
    c = float(df["Close"].iloc[idx])

    slip = _clamp(slip_pct + extra_slip_pct, 0.0, 0.05)
    spr = _clamp(spr_pct, 0.0, 0.05)
    half_spr = spr * 0.5

    # bar içi daha muhafazakar base seçelim (average-case ama temkinli)
    if direction == "long":
        if side == "entry":
            p = max(o, c, base_price)
            # buy: spread + slip yukarı iter
            return p * (1.0 + half_spr + slip)
        else:
            p = min(o, c, base_price)
            # sell: spread + slip aşağı iter
            return p * (1.0 - half_spr - slip)
    else:
        if side == "entry":
            p = min(o, c, base_price)
            # sell: spread + slip aşağı iter
            return p * (1.0 - half_spr - slip)
        else:
            p = max(o, c, base_price)
            # buy: spread + slip yukarı iter
            return p * (1.0 + half_spr + slip)


# ======================
# PnL / Funding / Liquidation helpers
# ======================

def calc_trade_pnl(direction: str, entry: float, exit: float, notional: float) -> float:
    if entry <= 0:
        return 0.0
    if direction == "long":
        return (exit - entry) / entry * notional
    return (entry - exit) / entry * notional


def estimate_liquidation_price(entry_price: float, direction: str, leverage: float, maint_margin_rate: float) -> float:
    """
    Kaba liq:
      LONG  liq ~ entry*(1 - 1/L + mmr)
      SHORT liq ~ entry*(1 + 1/L - mmr)
    """
    if leverage <= 1:
        return 0.0 if direction == "long" else float("inf")

    inv = 1.0 / leverage
    if direction == "long":
        return entry_price * (1.0 - inv + maint_margin_rate)
    return entry_price * (1.0 + inv - maint_margin_rate)


def check_liquidation(df: pd.DataFrame, direction: str, entry_idx: int, exit_idx: int, liq_price: float) -> bool:
    if exit_idx <= entry_idx:
        return False
    window = df.iloc[entry_idx: exit_idx + 1]
    if direction == "long":
        return float(window["Low"].min()) <= liq_price
    return float(window["High"].max()) >= liq_price


def check_blowup_10pct_adverse(df: pd.DataFrame, direction: str, entry_idx: int, exit_idx: int, entry_price: float, adverse_pct: float) -> bool:
    """
    Trade açıkken fiyat entry'ye karşı adverse_pct kadar giderse "hesap sıfırlandı" kabul et.
    LONG: Low <= entry*(1-adverse)
    SHORT: High >= entry*(1+adverse)
    """
    if exit_idx <= entry_idx:
        return False
    thr = entry_price * (1.0 - adverse_pct) if direction == "long" else entry_price * (1.0 + adverse_pct)
    window = df.iloc[entry_idx: exit_idx + 1]
    if direction == "long":
        return float(window["Low"].min()) <= thr
    return float(window["High"].max()) >= thr


# ======================
# Backtest (Realistic, balance-aware, with blow-up rules)
# ======================

def run_backtest_realistic_balance_aware(
    df: pd.DataFrame,
    trades: List[Trade],
    funding_df: pd.DataFrame,
    exec_cfg: RealisticExecConfig,
    start_balance: float = 1000.0,
    leverage: float = 10.0,

    # risk & limits
    risk_per_trade: float = 0.02,
    max_notional_usdt: float = 50_000,
    min_balance: float = 1e-9,

    # blow-up rules
    blowup_on_10_consecutive_stops: bool = True,
    max_stop_streak: int = 10,
    blowup_on_10pct_adverse_move: bool = True,
    adverse_move_pct: float = 0.10,
    apply_adverse_rule_only_when_stop_disabled: bool = True,
    stop_enabled_in_this_run: bool = True,
) -> Tuple[pd.DataFrame, float]:

    fee_rate = exec_cfg.taker_fee_rate if exec_cfg.assume_market_orders else exec_cfg.maker_fee_rate
    f_times = funding_df["fundingTime"].values if not funding_df.empty else np.array([])
    f_rates = funding_df["fundingRate"].values if not funding_df.empty else np.array([])

    balance = float(start_balance)
    idx_to_time = df.index.to_list()
    rows = []
    last_bar_idx = len(df) - 1

    stop_streak = 0
    blew_up = False
    blowup_reason = None

    for i, tr in enumerate(trades, start=1):
        if balance <= min_balance or blew_up:
            break

        # force close open trade at end
        if (tr.exit_index is None or tr.exit_price is None) and exec_cfg.force_close_last_trade:
            tr.exit_index = last_bar_idx
            tr.exit_price = float(df["Close"].iloc[last_bar_idx])
            tr.exit_reason = tr.exit_reason or "FORCE_CLOSE_END"

        if tr.exit_index is None or tr.exit_price is None:
            continue

        entry_idx = tr.entry_index
        exit_idx = tr.exit_index
        if exit_idx <= entry_idx:
            continue

        entry_time = idx_to_time[entry_idx]
        exit_time = idx_to_time[exit_idx]

        # margin sizing
        margin = balance * risk_per_trade
        if margin <= 0:
            continue
        notional = min(margin * leverage, max_notional_usdt)

        # slippage components
        adv_e = _adv_usdt(df, entry_idx, lookback_bars=96)
        adv_x = _adv_usdt(df, exit_idx, lookback_bars=96)
        vol_e = _bar_volatility(df, entry_idx)
        vol_x = _bar_volatility(df, exit_idx)

        impact_e = _impact_pct(notional, adv_e, exec_cfg.impact_k)
        impact_x = _impact_pct(notional, adv_x, exec_cfg.impact_k)

        slip_e = exec_cfg.base_slippage_pct + exec_cfg.vol_slippage_k * vol_e + impact_e
        slip_x = exec_cfg.base_slippage_pct + exec_cfg.vol_slippage_k * vol_x + impact_x

        # BALANCE multiplier (as balance grows -> worse fills)
        mult = _balance_multiplier(balance, start_balance, exec_cfg.balance_slip_k)
        slip_e *= mult
        slip_x *= mult

        slip_e = _clamp(slip_e, 0.0, exec_cfg.max_slippage_pct)
        slip_x = _clamp(slip_x, 0.0, exec_cfg.max_slippage_pct)

        # spread
        spr_e = _spread_pct(df, entry_idx, exec_cfg)
        spr_x = _spread_pct(df, exit_idx, exec_cfg)

        # stop extra penalty
        is_stop = (str(tr.exit_reason).startswith("STOP_LOSS") if tr.exit_reason else False)
        extra_stop = exec_cfg.stop_extra_slippage_pct if is_stop else 0.0

        # fills
        entry_price = _fill_price_with_costs(
            df=df, idx=entry_idx, direction=tr.direction, side="entry",
            base_price=tr.entry_price, slip_pct=slip_e, spr_pct=spr_e, extra_slip_pct=0.0
        )
        exit_price = _fill_price_with_costs(
            df=df, idx=exit_idx, direction=tr.direction, side="exit",
            base_price=tr.exit_price, slip_pct=slip_x, spr_pct=spr_x, extra_slip_pct=extra_stop
        )

        # blow-up on adverse move (%10)
        if blowup_on_10pct_adverse_move:
            apply_rule = True
            if apply_adverse_rule_only_when_stop_disabled and stop_enabled_in_this_run:
                apply_rule = False
            if apply_rule:
                if check_blowup_10pct_adverse(
                    df=df,
                    direction=tr.direction,
                    entry_idx=entry_idx,
                    exit_idx=exit_idx,
                    entry_price=entry_price,
                    adverse_pct=adverse_move_pct
                ):
                    # account blown up => balance 0, stop trading
                    blew_up = True
                    blowup_reason = f"ADVERSE_MOVE_{int(adverse_move_pct*100)}pct"
                    balance_before = balance
                    balance = 0.0
                    rows.append({
                        "trade_no": i,
                        "direction": tr.direction,
                        "entry_time": entry_time,
                        "exit_time": exit_time,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "notional": notional,
                        "pnl": 0.0,
                        "fees": 0.0,
                        "funding": 0.0,
                        "liquidated": False,
                        "liq_price_est": np.nan,
                        "liq_loss": 0.0,
                        "net_trade": -balance_before,   # komple sıfırlandı varsay
                        "balance_before": balance_before,
                        "balance_after": balance,
                        "exit_reason": blowup_reason,
                        "stop_streak": stop_streak
                    })
                    break

        # fees
        fees = notional * fee_rate * 2.0

        # funding
        funding = 0.0
        if len(f_times) > 0:
            et = np.datetime64(entry_time.to_datetime64())
            xt = np.datetime64(exit_time.to_datetime64())
            mask = (f_times > et) & (f_times <= xt)
            sel = f_rates[mask]
            if sel.size > 0:
                raw = float(np.sum(notional * sel))
                if exec_cfg.funding_worstcase:
                    funding = -abs(raw)
                else:
                    funding = -raw if tr.direction == "long" else +raw

        # liquidation check (isolated-margin gibi değil; sadece "margin gider" modeli)
        liq_price = estimate_liquidation_price(entry_price, tr.direction, leverage, exec_cfg.maint_margin_rate)
        liquidated = check_liquidation(df, tr.direction, entry_idx, exit_idx, liq_price)

        balance_before = balance

        if liquidated:
            liq_loss = margin
            net_trade = -liq_loss - fees + funding
            balance = balance + net_trade

            # stop streak update: liquidation stop sayılmaz
            stop_streak = 0

            rows.append({
                "trade_no": i,
                "direction": tr.direction,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "notional": notional,
                "pnl": 0.0,
                "fees": fees,
                "funding": funding,
                "liquidated": True,
                "liq_price_est": liq_price,
                "liq_loss": liq_loss,
                "net_trade": net_trade,
                "balance_before": balance_before,
                "balance_after": balance,
                "exit_reason": "LIQUIDATION",
                "stop_streak": stop_streak
            })

            if balance <= min_balance:
                balance = 0.0
                break
            continue

        pnl = calc_trade_pnl(tr.direction, entry_price, exit_price, notional)
        net_trade = pnl - fees + funding
        balance = balance + net_trade

        # stop streak rule (10 stop in a row => blow up to zero)
        if is_stop:
            stop_streak += 1
        else:
            stop_streak = 0

        rows.append({
            "trade_no": i,
            "direction": tr.direction,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "notional": notional,
            "pnl": pnl,
            "fees": fees,
            "funding": funding,
            "liquidated": False,
            "liq_price_est": liq_price,
            "liq_loss": 0.0,
            "net_trade": net_trade,
            "balance_before": balance_before,
            "balance_after": balance,
            "exit_reason": tr.exit_reason or "pivot_exit",
            "stop_streak": stop_streak
        })

        if blowup_on_10_consecutive_stops and stop_streak >= max_stop_streak:
            blew_up = True
            blowup_reason = f"STOP_STREAK_{max_stop_streak}"
            balance_before2 = balance
            balance = 0.0
            rows.append({
                "trade_no": i,
                "direction": tr.direction,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "notional": 0.0,
                "pnl": 0.0,
                "fees": 0.0,
                "funding": 0.0,
                "liquidated": False,
                "liq_price_est": np.nan,
                "liq_loss": 0.0,
                "net_trade": -balance_before2,
                "balance_before": balance_before2,
                "balance_after": balance,
                "exit_reason": blowup_reason,
                "stop_streak": stop_streak
            })
            break

        if balance <= min_balance:
            balance = 0.0
            break

    details = pd.DataFrame(rows)
    return details, float(balance)


def print_summary(title: str, start_balance: float, details: pd.DataFrame, end_balance: float) -> None:
    net = end_balance - start_balance
    profit_pct = (end_balance / start_balance - 1.0) * 100.0 if start_balance > 0 else 0.0

    print(f"\n==================== {title} ====================")
    print(f"Start balance:   ${start_balance:,.2f}")
    print(f"End balance:     ${end_balance:,.2f}")
    print(f"NET PnL:         ${net:,.2f}")
    print(f"Profit %:        {profit_pct:.2f}%")

    if details.empty:
        print("No realized trades.")
        return

    wins = int((details["net_trade"] > 0).sum())
    losses = int((details["net_trade"] <= 0).sum())

    total_fees = float(details["fees"].sum())
    total_funding = float(details["funding"].sum())
    total_liq_loss = float(details["liq_loss"].sum())
    gross_pnl = float(details["pnl"].sum())

    stop_hits = int(details["exit_reason"].astype(str).str.startswith("STOP_LOSS").sum())
    stop_net = float(details.loc[details["exit_reason"].astype(str).str.startswith("STOP_LOSS"), "net_trade"].sum()) if stop_hits > 0 else 0.0

    blew_up = int(details["exit_reason"].astype(str).str.startswith("ADVERSE_MOVE").sum()) > 0 or \
              int(details["exit_reason"].astype(str).str.startswith("STOP_STREAK").sum()) > 0

    print(f"Trades (realized): {len(details)}")
    print(f"W/L:             {wins}/{losses}")
    print(f"Gross PnL:       ${gross_pnl:,.2f}")
    print(f"Total Fees:      ${total_fees:,.2f}")
    print(f"Total Funding:   ${total_funding:,.2f}")
    print(f"Total Liq Loss:  ${total_liq_loss:,.2f}")
    print(f"Stop hits:       {stop_hits}")
    print(f"Stop net PnL:    ${stop_net:,.2f}")
    if blew_up:
        last_reason = str(details.iloc[-1]["exit_reason"])
        print(f"!!! BLOW-UP TRIGGERED: {last_reason}")


# ======================
# MAIN
# ======================

def main():
    symbol = "TRXUSDT"
    interval = "15m"
    right_bars = 1
    years = 5

    start_balance = 1000.0
    leverage = 10.0

    # risk
    risk_per_trade = 0.02
    max_notional_usdt = 50_000

    stop_loss_pct = 0.015
    enable_flip = True

    # Realistic exec config (balance-aware)
    exec_cfg = RealisticExecConfig(
        taker_fee_rate=0.0005,
        maker_fee_rate=0.0002,
        assume_market_orders=True,

        base_slippage_pct=0.00025,
        vol_slippage_k=0.25,
        impact_k=0.08,
        max_slippage_pct=0.006,

        balance_slip_k=0.18,

        spread_base_pct=0.00010,
        spread_vol_k=0.20,

        stop_extra_slippage_pct=0.00035,

        funding_worstcase=False,
        maint_margin_rate=0.0075,
        force_close_last_trade=True
    )

    print(f"1) OHLC çekiliyor... (years={years})")
    df = fetch_ohlc_years(symbol, interval, years=years)
    print(f"OHLC bars: {len(df)} | Range: {df.index[0]} -> {df.index[-1]}")

    highs = df["High"].values
    lows = df["Low"].values

    print("2) Pivotlar bulunuyor...")
    swings = find_swings(
        highs,
        lows,
        left_bars=5,
        right_bars=right_bars,
        min_distance=10,
        alt_min_distance=None,
    )
    print("Pivot sayısı:", len(swings))

    start_ms = int(df.index[0].value // 10**6)
    end_ms = int(df.index[-1].value // 10**6)

    print("3) Funding rate history çekiliyor...")
    funding_df = fetch_funding_rates(symbol, start_ms, end_ms)
    print("Funding kayıt:", len(funding_df))

    # --------------------------
    # A) STOP KAPALI
    # --------------------------
    print("\n4A) Trade’ler üretiliyor (STOP KAPALI)...")
    trades_no_stop = generate_trades(
        df=df,
        swings=swings,
        right_bars=right_bars,
        enable_flip=enable_flip,
        enable_stop=False,
        stop_loss_pct=stop_loss_pct
    )

    print("5A) Backtest (STOP KAPALI) [REALISTIC BALANCE-AWARE]...")
    details_no_stop, end_no_stop = run_backtest_realistic_balance_aware(
        df=df,
        trades=trades_no_stop,
        funding_df=funding_df,
        exec_cfg=exec_cfg,
        start_balance=start_balance,
        leverage=leverage,
        risk_per_trade=risk_per_trade,
        max_notional_usdt=max_notional_usdt,

        blowup_on_10_consecutive_stops=True,
        max_stop_streak=10,
        blowup_on_10pct_adverse_move=True,
        adverse_move_pct=0.10,
        apply_adverse_rule_only_when_stop_disabled=True,
        stop_enabled_in_this_run=False,
    )
    print_summary("RESULTS (NO STOP) [REALISTIC]", start_balance, details_no_stop, end_no_stop)
    if not details_no_stop.empty:
        details_no_stop.to_csv("trx_backtest_details_no_stop_realistic.csv", index=False)

    # --------------------------
    # B) STOP AÇIK %1.5
    # --------------------------
    print("\n4B) Trade’ler üretiliyor (STOP AÇIK %1.5)...")
    trades_with_stop = generate_trades(
        df=df,
        swings=swings,
        right_bars=right_bars,
        enable_flip=enable_flip,
        enable_stop=True,
        stop_loss_pct=stop_loss_pct
    )

    print("5B) Backtest (STOP AÇIK %1.5) [REALISTIC BALANCE-AWARE]...")
    details_with_stop, end_with_stop = run_backtest_realistic_balance_aware(
        df=df,
        trades=trades_with_stop,
        funding_df=funding_df,
        exec_cfg=exec_cfg,
        start_balance=start_balance,
        leverage=leverage,
        risk_per_trade=risk_per_trade,
        max_notional_usdt=max_notional_usdt,

        blowup_on_10_consecutive_stops=True,
        max_stop_streak=10,
        blowup_on_10pct_adverse_move=True,
        adverse_move_pct=0.10,
        # burada adverse kuralını istersen stop açıkken de uygulayabilirsin:
        apply_adverse_rule_only_when_stop_disabled=True,
        stop_enabled_in_this_run=True,
    )
    print_summary("RESULTS (WITH STOP 1.5%) [REALISTIC]", start_balance, details_with_stop, end_with_stop)
    if not details_with_stop.empty:
        details_with_stop.to_csv("trx_backtest_details_with_stop_realistic.csv", index=False)

    # --------------------------
    # Compare
    # --------------------------
    profit_no_stop = (end_no_stop / start_balance - 1.0) * 100.0
    profit_with_stop = (end_with_stop / start_balance - 1.0) * 100.0

    print("\n==================== STOP COMPARISON [REALISTIC] ====================")
    print(f"End (no stop):   ${end_no_stop:,.2f}  | Profit%: {profit_no_stop:.2f}%")
    print(f"End (with stop): ${end_with_stop:,.2f}  | Profit%: {profit_with_stop:.2f}%")
    print(f"Delta Profit%:   {(profit_with_stop - profit_no_stop):.2f}%")
    if not details_with_stop.empty:
        print(f"Stop hits:       {int(details_with_stop['exit_reason'].astype(str).str.startswith('STOP_LOSS').sum())}")
    else:
        print("Stop hits:       0")

    print("\nCSV çıktıları:")
    print(" - trx_backtest_details_no_stop_realistic.csv")
    print(" - trx_backtest_details_with_stop_realistic.csv")


if __name__ == "__main__":
    main()

