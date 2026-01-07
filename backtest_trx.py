import time
import requests
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal, Optional, List, Tuple

# ✅ Pivot fonksiyonlarını senin dosyadan çekiyoruz:
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
    """
    Binance Futures Klines ile belirtilen yıl kadar geçmiş OHLC çeker.
    Chunk'lı çektiği için 3 yıl gibi uzun aralıklarda da çalışır.
    """
    if end_utc is None:
        # pandas sürümlerinde utcnow tz-aware gelebiliyor -> localize yapmıyoruz
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

        # rate limit'e takılmamak için küçük uyku
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
    """
    Binance Futures funding rate history.
    (Not: Bu endpoint bazı sembollerde eski datayı sınırlayabilir; yine de mümkün olanı çeker.)
    """
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
    entry_time: Optional[pd.Timestamp] = None
    exit_time: Optional[pd.Timestamp] = None


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

    first_pos = int(np.argmax(hits))  # ilk True
    return (start_idx + 1) + first_pos


def generate_trades(
    df: pd.DataFrame,
    swings: List[SwingPoint],
    right_bars: int = 1,
    enable_flip: bool = True,

    # ✅ stop kontrolleri swings mantığında
    enable_stop: bool = True,
    stop_loss_pct: float = 0.015,  # %1.5
) -> List[Trade]:
    """
    Entry:
      - LB (Higher Low): prev_lb.price < curr_lb.price -> LONG
      - LH (Lower High): prev_lh.price > curr_lh.price -> SHORT

    Exit (greed killer):
      - LONG açıkken LH onaylanınca kapat
      - SHORT açıkken LB onaylanınca kapat

    STOP LOSS:
      - LONG: entry'den sonra Low <= entry*(1-0.015) -> STOP
      - SHORT: entry'den sonra High >= entry*(1+0.015) -> STOP
      Stop pivot beklemez; her pivotta bir sonraki “signal_idx”e kadar arayı tarar.
    """
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

        # ✅ 0) STOP LOSS kontrolü: pivotu işlemeden önce arada stop tetiklendi mi?
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
                current.exit_price = current.stop_level  # stop fill price kabul ediyoruz
                current.exit_reason = "STOP_LOSS_1_5"
                trades.append(current)
                current = None
                # Stop olduysa bu pivotta yeni entry aramaya devam edeceğiz.

        # 1) Poz açıkken greed-killer exit + opsiyonel flip
        if current is not None:
            if current.direction == "long" and sp.kind == "LH":
                if signal_idx > current.entry_index:
                    current.exit_index = signal_idx
                    current.exit_price = closes[signal_idx]
                    current.exit_reason = "long_exit_peak_LH"
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
                    current.exit_reason = "short_exit_dip_LB"
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

        # pivotları güncelle
        if sp.kind == "LB":
            last_lb = sp
        else:
            last_lh = sp

    # açık trade kalırsa (exit yok) ekle
    if current is not None:
        trades.append(current)

    return trades


# ======================
# Realistic Backtest Engine
# ======================

def calc_trade_pnl(direction: str, entry: float, exit: float, notional: float) -> float:
    if entry <= 0:
        return 0.0
    if direction == "long":
        return (exit - entry) / entry * notional
    return (entry - exit) / entry * notional


def apply_slippage(price: float, direction: str, side: str, slippage_pct: float) -> float:
    """
    Market order slippage (aleyhe):
    LONG entry -> pahalı, LONG exit -> ucuz
    SHORT entry -> ucuz (sell kötü), SHORT exit -> pahalı (buy kötü)
    """
    if slippage_pct <= 0:
        return price

    s = slippage_pct
    if direction == "long":
        return price * (1.0 + s) if side == "entry" else price * (1.0 - s)
    else:
        return price * (1.0 - s) if side == "entry" else price * (1.0 + s)


def estimate_liquidation_price(entry_price: float, direction: str, leverage: float, maint_margin_rate: float) -> float:
    """
    Yaklaşık liquidation eşiği (kaba):
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


def run_backtest_realistic(
    df: pd.DataFrame,
    trades: List[Trade],
    funding_df: pd.DataFrame,
    start_balance: float = 1000.0,
    leverage: float = 10.0,

    # risk & limits
    risk_per_trade: float = 0.02,
    max_notional_usdt: float = 50_000,
    min_balance: float = 1.0,

    # fees
    taker_fee_rate: float = 0.0005,
    maker_fee_rate: float = 0.0002,
    assume_market_orders: bool = True,

    # slippage
    slippage_pct: float = 0.0003,

    # liquidation
    maint_margin_rate: float = 0.005,
) -> Tuple[pd.DataFrame, float]:
    """
    Çıktı:
      - details DataFrame (trade bazlı)
      - end_balance
    """
    fee_rate = taker_fee_rate if assume_market_orders else maker_fee_rate

    f_times = funding_df["fundingTime"].values if not funding_df.empty else np.array([])
    f_rates = funding_df["fundingRate"].values if not funding_df.empty else np.array([])

    balance = float(start_balance)
    idx_to_time = df.index.to_list()
    rows = []

    for i, tr in enumerate(trades, start=1):
        if balance <= min_balance:
            break

        if tr.exit_index is None or tr.exit_price is None:
            continue

        entry_idx = tr.entry_index
        exit_idx = tr.exit_index
        if exit_idx <= entry_idx:
            continue

        entry_time = idx_to_time[entry_idx]
        exit_time = idx_to_time[exit_idx]

        # risk cap: her trade margin = balance * risk_per_trade
        margin = balance * risk_per_trade
        if margin <= 0:
            continue

        notional = min(margin * leverage, max_notional_usdt)

        # slippage (aleyhe)
        entry_price = apply_slippage(tr.entry_price, tr.direction, "entry", slippage_pct)
        exit_price  = apply_slippage(tr.exit_price,  tr.direction, "exit",  slippage_pct)

        # fees (entry+exit)
        fees = notional * fee_rate * 2.0

        # funding (entry, exit]
        funding = 0.0
        if len(f_times) > 0:
            et = np.datetime64(entry_time.to_datetime64())
            xt = np.datetime64(exit_time.to_datetime64())
            mask = (f_times > et) & (f_times <= xt)
            sel = f_rates[mask]
            if sel.size > 0:
                raw = float(np.sum(notional * sel))
                funding = -raw if tr.direction == "long" else +raw

        # liquidation check
        liq_price = estimate_liquidation_price(entry_price, tr.direction, leverage, maint_margin_rate)
        liquidated = check_liquidation(df, tr.direction, entry_idx, exit_idx, liq_price)

        balance_before = balance

        if liquidated:
            # basit liquidation: margin gider
            liq_loss = margin
            net_trade = -liq_loss - fees + funding
            balance = balance + net_trade

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
                "exit_reason": "LIQUIDATION"
            })
            continue

        pnl = calc_trade_pnl(tr.direction, entry_price, exit_price, notional)
        net_trade = pnl - fees + funding
        balance = balance + net_trade

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
            "exit_reason": tr.exit_reason or "pivot_exit"
        })

    details = pd.DataFrame(rows)
    return details, balance


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

    stop_hits = int((details["exit_reason"].astype(str) == "STOP_LOSS_1_5").sum())
    stop_net = float(details.loc[details["exit_reason"].astype(str) == "STOP_LOSS_1_5", "net_trade"].sum()) if stop_hits > 0 else 0.0

    print(f"Trades (realized): {len(details)}")
    print(f"W/L:             {wins}/{losses}")
    print(f"Gross PnL:       ${gross_pnl:,.2f}")
    print(f"Total Fees:      ${total_fees:,.2f}")
    print(f"Total Funding:   ${total_funding:,.2f}")
    print(f"Total Liq Loss:  ${total_liq_loss:,.2f}")
    print(f"Stop hits:       {stop_hits}")
    print(f"Stop net PnL:    ${stop_net:,.2f}")


# ======================
# MAIN
# ======================

def main():
    symbol = "TRXUSDT"
    interval = "15m"
    right_bars = 1

    # ✅ kaç yıl test etmek istiyorsun?
    years = 5

    start_balance = 1000.0
    leverage = 10.0

    # risk realism knobs
    risk_per_trade = 0.02
    max_notional_usdt = 50_000
    slippage_pct = 0.0003
    maint_margin_rate = 0.005

    # Binance fee (tier’ına göre güncelleyebilirsin)
    taker_fee = 0.0005
    maker_fee = 0.0002
    assume_market = True

    stop_loss_pct = 0.015
    enable_flip = True

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
        enable_stop=False,          # ✅ stop kapalı
        stop_loss_pct=stop_loss_pct
    )

    print("5A) Backtest (STOP KAPALI)...")
    details_no_stop, end_no_stop = run_backtest_realistic(
        df=df,
        trades=trades_no_stop,
        funding_df=funding_df,
        start_balance=start_balance,
        leverage=leverage,
        risk_per_trade=risk_per_trade,
        max_notional_usdt=max_notional_usdt,
        taker_fee_rate=taker_fee,
        maker_fee_rate=maker_fee,
        assume_market_orders=assume_market,
        slippage_pct=slippage_pct,
        maint_margin_rate=maint_margin_rate,
    )
    print_summary("RESULTS (NO STOP)", start_balance, details_no_stop, end_no_stop)
    if not details_no_stop.empty:
        details_no_stop.to_csv("trx_backtest_details_no_stop.csv", index=False)

    # --------------------------
    # B) STOP AÇIK %1.5
    # --------------------------
    print("\n4B) Trade’ler üretiliyor (STOP AÇIK %1.5)...")
    trades_with_stop = generate_trades(
        df=df,
        swings=swings,
        right_bars=right_bars,
        enable_flip=enable_flip,
        enable_stop=True,           # ✅ stop açık
        stop_loss_pct=stop_loss_pct
    )

    print("5B) Backtest (STOP AÇIK %1.5)...")
    details_with_stop, end_with_stop = run_backtest_realistic(
        df=df,
        trades=trades_with_stop,
        funding_df=funding_df,
        start_balance=start_balance,
        leverage=leverage,
        risk_per_trade=risk_per_trade,
        max_notional_usdt=max_notional_usdt,
        taker_fee_rate=taker_fee,
        maker_fee_rate=maker_fee,
        assume_market_orders=assume_market,
        slippage_pct=slippage_pct,
        maint_margin_rate=maint_margin_rate,
    )
    print_summary("RESULTS (WITH STOP 1.5%)", start_balance, details_with_stop, end_with_stop)
    if not details_with_stop.empty:
        details_with_stop.to_csv("trx_backtest_details_with_stop.csv", index=False)

    # --------------------------
    # Karşılaştırma
    # --------------------------
    profit_no_stop = (end_no_stop / start_balance - 1.0) * 100.0
    profit_with_stop = (end_with_stop / start_balance - 1.0) * 100.0

    print("\n==================== STOP COMPARISON ====================")
    print(f"End (no stop):   ${end_no_stop:,.2f}  | Profit%: {profit_no_stop:.2f}%")
    print(f"End (with stop): ${end_with_stop:,.2f}  | Profit%: {profit_with_stop:.2f}%")
    print(f"Delta Profit%:   {(profit_with_stop - profit_no_stop):.2f}%")
    if not details_with_stop.empty:
        print(f"Stop hits:       {int((details_with_stop['exit_reason'].astype(str) == 'STOP_LOSS_1_5').sum())}")
    else:
        print("Stop hits:       0")

    print("\nCSV çıktıları:")
    print(" - trx_backtest_details_no_stop.csv")
    print(" - trx_backtest_details_with_stop.csv")


if __name__ == "__main__":
    main()
