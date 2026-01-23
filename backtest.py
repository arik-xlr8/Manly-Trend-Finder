# backtest.py
import requests
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta, timezone

# =============================
# 0) STRATEJİYİ IMPORT ET
# =============================
from plot_swings_from_api import generate_trades  # senin en güncel stratejin
from swings import find_swings

# ======================
# 1) BINANCE API HELPERS
# ======================
BASE = "https://fapi.binance.com"

def _get(url, params):
    r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    return r.json()

def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    out = []
    limit = 1500
    url = f"{BASE}/fapi/v1/klines"

    t = start_ms
    while True:
        data = _get(url, {"symbol": symbol, "interval": interval, "startTime": t, "endTime": end_ms, "limit": limit})
        if not data:
            break
        out.extend(data)
        last_open = int(data[-1][0])
        t = last_open + 1
        if len(data) < limit:
            break
        if last_open >= end_ms:
            break

    if not out:
        return pd.DataFrame()

    cols = [
        "open_time","open","high","low","close","volume","close_time",
        "quote_asset_volume","number_of_trades","taker_buy_base_volume",
        "taker_buy_quote_volume","ignore",
    ]
    df = pd.DataFrame(out, columns=cols)
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
    df["Date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("Date", inplace=True)
    df = df[["open","high","low","close","volume"]]
    df.columns = ["Open","High","Low","Close","Volume"]
    return df.sort_index()

def fetch_funding_rates(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    url = f"{BASE}/fapi/v1/fundingRate"
    limit = 1000
    all_rows = []
    t = start_ms
    while True:
        rows = _get(url, {"symbol": symbol, "startTime": t, "endTime": end_ms, "limit": limit})
        if not rows:
            break
        all_rows.extend(rows)
        last_time = int(rows[-1]["fundingTime"])
        t = last_time + 1
        if len(rows) < limit:
            break

    if not all_rows:
        return pd.DataFrame(columns=["fundingTime", "fundingRate"])

    fr = pd.DataFrame(all_rows)
    fr["fundingTime"] = pd.to_datetime(fr["fundingTime"], unit="ms", utc=True)
    fr["fundingRate"] = fr["fundingRate"].astype(float)
    return fr[["fundingTime","fundingRate"]].sort_values("fundingTime").reset_index(drop=True)

# ======================
# 2) BACKTEST RESULT
# ======================
@dataclass
class FillResult:
    end_balance: float
    total_fees: float
    total_funding: float
    trades_closed: int
    wins: int
    losses: int
    stop_hits: int
    liquidations: int
    exit_reason_counts: Dict[str, int]
    equity_curve: List[Tuple[pd.Timestamp, float]]

def apply_fee(notional: float, fee_rate: float) -> float:
    return notional * fee_rate

def calc_pnl(direction: str, entry: float, exit: float, notional: float) -> float:
    if entry <= 0:
        return 0.0
    if direction == "long":
        return (exit - entry) / entry * notional
    else:
        return (entry - exit) / entry * notional

# ======================
# 3) SLIPPAGE HELPERS
# ======================
def apply_entry_slippage(direction: str, price: float, slip: float) -> float:
    # entry her zaman kötüleşir
    if direction == "long":
        return price * (1.0 + slip)
    else:
        return price * (1.0 - slip)

def apply_exit_slippage(direction: str, price: float, slip: float) -> float:
    # exit her zaman kötüleşir
    if direction == "long":
        return price * (1.0 - slip)
    else:
        return price * (1.0 + slip)

# ======================
# 4) REALISTIC SIM (MONTH INSIDE COMPOUND)
# ======================
def simulate_balance_realistic(
    df: pd.DataFrame,
    trades: List,
    funding: pd.DataFrame,
    leverage: float,
    start_balance: float,
    taker_fee_rate: float,
    *,
    margin_pct: float = 0.10,
    maintenance_margin_rate: float = 0.005,
    entry_slip_pct: float = 0.0002,
    exit_slip_pct: float = 0.0002,
    stop_slip_pct: float = 0.0005,
    enable_liquidation_check: bool = True,   # <-- ister kapat
) -> FillResult:
    balance = float(start_balance)
    total_fees = 0.0
    total_funding = 0.0
    wins = 0
    losses = 0
    closed = 0
    stop_hits = 0
    liquidations = 0
    exit_reason_counts: Dict[str, int] = {}
    equity_curve: List[Tuple[pd.Timestamp, float]] = []

    if len(df):
        equity_curve.append((df.index[0], balance))

    if len(funding):
        f_times = funding["fundingTime"]  # tz-aware Timestamp
        f_rates = funding["fundingRate"].values
    else:
        f_times = None
        f_rates = np.array([])

    for tr in trades:
        if getattr(tr, "exit_index", None) is None:
            continue

        ei = int(tr.entry_index)
        xi = int(tr.exit_index)
        if ei < 0 or xi <= ei or xi >= len(df):
            continue

        if balance <= 0:
            break

        entry_time = df.index[ei]
        exit_time  = df.index[xi]

        raw_entry = float(tr.entry_price)
        raw_exit  = float(tr.exit_price) if tr.exit_price is not None else float(df["Close"].iloc[xi])

        reason = getattr(tr, "exit_reason", None) or "unknown_exit"

        # --- Slippage prices ---
        entry_price = apply_entry_slippage(tr.direction, raw_entry, entry_slip_pct)
        if "stop" in reason.lower():
            exit_price = apply_exit_slippage(tr.direction, raw_exit, stop_slip_pct)
        else:
            exit_price = apply_exit_slippage(tr.direction, raw_exit, exit_slip_pct)

        # --- Position sizing ---
        margin = balance * margin_pct
        if margin <= 0:
            break
        notional = margin * leverage

        # --- Fees (entry+exit) ---
        fees = apply_fee(notional, taker_fee_rate) + apply_fee(notional, taker_fee_rate)
        balance -= fees
        total_fees += fees

        # --- Funding ---
        if f_times is not None and len(f_rates):
            mask = (f_times > entry_time) & (f_times <= exit_time)
            if mask.any():
                rates = f_rates[mask.values]
                sign = -1.0 if tr.direction == "long" else 1.0
                funding_pnl = float(np.sum(notional * rates * sign))
                balance += funding_pnl
                total_funding += funding_pnl

        # --- PnL ---
        pnl = calc_pnl(tr.direction, entry_price, exit_price, notional)
        balance += pnl

        # --- Stats ---
        exit_reason_counts[reason] = exit_reason_counts.get(reason, 0) + 1
        if "stop" in reason.lower():
            stop_hits += 1

        # --- Liquidation check (optional, basit) ---
        if enable_liquidation_check:
            required_maint = notional * maintenance_margin_rate
            if balance <= required_maint:
                liquidations += 1
                balance = 0.0
                equity_curve.append((exit_time, balance))
                closed += 1
                losses += 1
                break

        closed += 1
        if pnl >= 0:
            wins += 1
        else:
            losses += 1

        equity_curve.append((exit_time, balance))

    return FillResult(
        end_balance=balance,
        total_fees=total_fees,
        total_funding=total_funding,
        trades_closed=closed,
        wins=wins,
        losses=losses,
        stop_hits=stop_hits,
        liquidations=liquidations,
        exit_reason_counts=exit_reason_counts,
        equity_curve=equity_curve,
    )

# ======================
# 5) MONTH SLICING (RESET MONTHLY TO $1000)
# ======================
def iter_month_ranges(start_dt: datetime, end_dt: datetime):
    current_start = start_dt
    while current_start < end_dt:
        y = current_start.year
        m = current_start.month
        if m == 12:
            next_month = datetime(y + 1, 1, 1, tzinfo=timezone.utc)
        else:
            next_month = datetime(y, m + 1, 1, tzinfo=timezone.utc)
        current_end = min(next_month, end_dt)
        yield current_start, current_end
        current_start = current_end

def month_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m")

# ======================
# 6) RUN ONE SYMBOL (MONTHLY RESET) FOR ONE CASE
# ======================
def run_symbol_monthly_reset_case(
    symbol: str,
    interval: str,
    start_dt: datetime,
    end_dt: datetime,
    *,
    monthly_start_balance: float,
    leverage: float,
    taker_fee_rate: float,
    margin_pct: float,
    maintenance_margin_rate: float,
    max_chase_pct: float,
    entry_slip_pct: float,
    exit_slip_pct: float,
    stop_slip_pct: float,
    enable_liquidation_check: bool,
    left_bars: int = 5,
    right_bars: int = 1,
    min_distance: int = 10,
):
    month_reports = []

    total_net = 0.0
    total_fees = 0.0
    total_funding = 0.0
    total_closed = 0
    total_wins = 0
    total_losses = 0
    total_stop_hits = 0
    total_liqs = 0
    exit_reasons: Dict[str, int] = {}

    for m_start, m_end in iter_month_ranges(start_dt, end_dt):
        start_ms = int(m_start.timestamp() * 1000)
        end_ms   = int(m_end.timestamp() * 1000)

        df = fetch_klines(symbol, interval, start_ms, end_ms)
        if df.empty:
            month_reports.append({
                "month": month_key(m_start),
                "start": monthly_start_balance,
                "end": monthly_start_balance,
                "pnl": 0.0,
                "ret_pct": 0.0,
                "note": "no_data"
            })
            continue

        funding = fetch_funding_rates(symbol, start_ms, end_ms)

        highs = df["High"].values
        lows  = df["Low"].values

        swings = find_swings(
            highs, lows,
            left_bars=left_bars,
            right_bars=right_bars,
            min_distance=min_distance,
            alt_min_distance=None
        )

        trades = generate_trades(
            df=df,
            swings=swings,
            stop_buffer_pct=0.0,
            right_bars=right_bars,
            max_chase_pct=max_chase_pct
        )

        res = simulate_balance_realistic(
            df=df,
            trades=trades,
            funding=funding,
            leverage=leverage,
            start_balance=monthly_start_balance,
            taker_fee_rate=taker_fee_rate,
            margin_pct=margin_pct,
            maintenance_margin_rate=maintenance_margin_rate,
            entry_slip_pct=entry_slip_pct,
            exit_slip_pct=exit_slip_pct,
            stop_slip_pct=stop_slip_pct,
            enable_liquidation_check=enable_liquidation_check,
        )

        month_end = res.end_balance
        pnl = month_end - monthly_start_balance
        ret = (month_end / monthly_start_balance - 1.0) * 100.0 if monthly_start_balance > 0 else 0.0

        total_net += pnl
        total_fees += res.total_fees
        total_funding += res.total_funding
        total_closed += res.trades_closed
        total_wins += res.wins
        total_losses += res.losses
        total_stop_hits += res.stop_hits
        total_liqs += res.liquidations
        for k, v in res.exit_reason_counts.items():
            exit_reasons[k] = exit_reasons.get(k, 0) + v

        month_reports.append({
            "month": month_key(m_start),
            "start": monthly_start_balance,
            "end": month_end,
            "pnl": pnl,
            "ret_pct": ret,
            "trades": res.trades_closed,
            "wins": res.wins,
            "losses": res.losses,
            "stop_hits": res.stop_hits,
            "liq": res.liquidations,
            "fees": res.total_fees,
            "funding": res.total_funding,
        })

    return {
        "symbol": symbol,
        "months": month_reports,
        "total_net": total_net,
        "total_fees": total_fees,
        "total_funding": total_funding,
        "total_closed": total_closed,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "total_stop_hits": total_stop_hits,
        "total_liqs": total_liqs,
        "exit_reasons": exit_reasons,
    }

# ======================
# 7) CASE CONFIG
# ======================
@dataclass
class CaseConfig:
    name: str
    taker_fee_rate: float
    entry_slip: float
    exit_slip: float
    stop_slip: float
    margin_pct: float
    maintenance_margin_rate: float
    enable_liquidation_check: bool

# ======================
# 8) MAIN
# ======================
def main():
    SYMBOLS = ["BTCUSDT", "ETHUSDT", "TRXUSDT", "SOLUSDT", "TURBOUSDT"]
    INTERVAL = "15m"

    # Her ay sıfırdan: 1000$
    MONTHLY_START_BALANCE = 1000.0
    LEVERAGE = 10.0

    # senin %3 kuralın
    MAX_CHASE_PCT = 0.03

    # --- 3 CASE ---
    # AVG = senin mevcut ayarlar
    # BEST = daha iyi fill + daha düşük fee
    # WORST = daha kötü fill + daha yüksek fee + stop daha kötü
    CASES: List[CaseConfig] = [
        CaseConfig(
            name="BEST",
            taker_fee_rate=0.0002,     # 0.02%
            entry_slip=0.00005,        # 0.005%
            exit_slip=0.00005,
            stop_slip=0.00015,
            margin_pct=0.10,
            maintenance_margin_rate=0.005,
            enable_liquidation_check=False,  # stop var diyorsun -> kapalı
        ),
        CaseConfig(
            name="AVG",
            taker_fee_rate=0.0004,     # 0.04%
            entry_slip=0.0002,         # 0.02%
            exit_slip=0.0002,
            stop_slip=0.0005,          # 0.05%
            margin_pct=0.10,
            maintenance_margin_rate=0.005,
            enable_liquidation_check=False,  # stop var diyorsun -> kapalı
        ),
        CaseConfig(
            name="WORST",
            taker_fee_rate=0.0006,     # 0.06%
            entry_slip=0.0008,         # 0.08%
            exit_slip=0.0008,
            stop_slip=0.0015,          # 0.15%
            margin_pct=0.10,
            maintenance_margin_rate=0.005,
            enable_liquidation_check=False,  # stop var diyorsun -> kapalı
        ),
    ]

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=90)

    overall_by_case: Dict[str, List[Dict]] = {c.name: [] for c in CASES}

    for sym in SYMBOLS:
        print(f"\n================ {sym} ================")
        for case in CASES:
            print(f"\n--- CASE: {case.name} ---")
            rep = run_symbol_monthly_reset_case(
                sym,
                INTERVAL,
                start_dt,
                end_dt,
                monthly_start_balance=MONTHLY_START_BALANCE,
                leverage=LEVERAGE,
                taker_fee_rate=case.taker_fee_rate,
                margin_pct=case.margin_pct,
                maintenance_margin_rate=case.maintenance_margin_rate,
                max_chase_pct=MAX_CHASE_PCT,
                entry_slip_pct=case.entry_slip,
                exit_slip_pct=case.exit_slip,
                stop_slip_pct=case.stop_slip,
                enable_liquidation_check=case.enable_liquidation_check,
                left_bars=5,
                right_bars=1,
                min_distance=10,
            )
            overall_by_case[case.name].append(rep)

            # aylık çıktı
            for row in rep["months"]:
                if row.get("note") == "no_data":
                    print(f"{row['month']}: no_data")
                    continue
                print(
                    f"{row['month']}: start=${row['start']:.2f} end=${row['end']:.2f} "
                    f"pnl=${row['pnl']:.2f} ret={row['ret_pct']:.2f}% "
                    f"| trades={row['trades']} W/L={row['wins']}/{row['losses']} "
                    f"| stop={row['stop_hits']} "
                    f"| fees=${row['fees']:.2f} funding=${row['funding']:.2f}"
                )

            print("\n-- Symbol Totals (monthly reset, NO carry) --")
            print(f"Total net PnL: ${rep['total_net']:.2f}")
            print(f"Total fees:    ${rep['total_fees']:.2f}")
            print(f"Total funding: ${rep['total_funding']:.2f}")
            print(f"Trades closed: {rep['total_closed']} | W/L: {rep['total_wins']}/{rep['total_losses']}")
            print(f"Stop hits:     {rep['total_stop_hits']}")

            if rep["exit_reasons"]:
                top = sorted(rep["exit_reasons"].items(), key=lambda x: x[1], reverse=True)[:5]
                print("-- Exit reasons (top 5) --")
                for k, v in top:
                    print(f"{k}: {v}")

    # OVERALL SUMMARY
    print("\n==================== OVERALL SUMMARY (MONTHLY RESET, 3 CASES) ====================")
    for case in CASES:
        reps = overall_by_case[case.name]
        total_net = sum(r["total_net"] for r in reps)
        total_fees = sum(r["total_fees"] for r in reps)
        total_funding = sum(r["total_funding"] for r in reps)
        total_closed = sum(r["total_closed"] for r in reps)
        total_wins = sum(r["total_wins"] for r in reps)
        total_losses = sum(r["total_losses"] for r in reps)
        total_stop_hits = sum(r["total_stop_hits"] for r in reps)

        print(
            f"{case.name:5} | TotalNet=${total_net:,.2f} | Closed={total_closed} | "
            f"StopHits={total_stop_hits} | Fees=${total_fees:,.2f} | Funding=${total_funding:,.2f} | "
            f"W/L={total_wins}/{total_losses}"
        )

if __name__ == "__main__":
    main()
