# backtest.py
import requests
import pandas as pd
import numpy as np

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
from datetime import datetime, timedelta, timezone

from plot_swings_from_api import generate_trades  # en güncel strateji (fills'li)
from swings import find_swings

# ======================
# 1) BINANCE API HELPERS
# ======================
BASE = "https://fapi.binance.com"


def _get(url: str, params: Dict[str, Any]):
    r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    return r.json()


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Binance Futures klines: [open_time, open, high, low, close, volume, ...]
    """
    out = []
    limit = 1500
    url = f"{BASE}/fapi/v1/klines"

    t = start_ms
    while True:
        data = _get(
            url,
            {"symbol": symbol, "interval": interval, "startTime": t, "endTime": end_ms, "limit": limit},
        )
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
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "number_of_trades", "taker_buy_base_volume",
        "taker_buy_quote_volume", "ignore",
    ]
    df = pd.DataFrame(out, columns=cols)
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    df["Date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("Date", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]]
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    return df.sort_index()


def fetch_funding_rates(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Binance Futures fundingRate endpoint.
    fundingTime, fundingRate
    """
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
    return fr[["fundingTime", "fundingRate"]].sort_values("fundingTime").reset_index(drop=True)


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


def apply_stop_slippage_from_position(qty_signed: float, price: float, slip: float) -> float:
    """
    Stop'ta:
      - long pozisyon stop => SELL => fiyat kötüleşir => daha aşağı
      - short pozisyon stop => BUY  => fiyat kötüleşir => daha yukarı
    """
    if qty_signed > 0:  # long -> sell
        return price * (1.0 - slip)
    else:              # short -> buy
        return price * (1.0 + slip)


# ======================
# 4) REALISTIC SIM (fills destekli)
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
    enable_liquidation_check: bool = False,
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
        f_times = funding["fundingTime"]
        f_rates = funding["fundingRate"].values
    else:
        f_times = None
        f_rates = np.array([])

    def funding_pnl_between(
        t0: pd.Timestamp,
        t1: pd.Timestamp,
        notional: float,
        qty_abs: float,
        is_long: bool
    ) -> float:
        if f_times is None or not len(f_rates) or qty_abs <= 0:
            return 0.0

        mask = (f_times > t0) & (f_times <= t1)
        if not mask.any():
            return 0.0

        rates = f_rates[mask.values]
        # varsayım: long pays(-), short receives(+)
        sign = -1.0 if is_long else 1.0
        return float(np.sum(notional * qty_abs * rates * sign))

    for tr in trades:
        xi_obj = getattr(tr, "exit_index", None)
        if xi_obj is None:
            continue

        ei = int(tr.entry_index)
        xi = int(tr.exit_index)

        if ei < 0 or xi <= ei or xi >= len(df):
            continue
        if balance <= 0:
            break

        bal0 = balance

        # her trade için kullanılacak margin/notional
        margin = balance * margin_pct
        if margin <= 0:
            break
        notional_base = margin * leverage  # qty=1.0 notional

        reason = (getattr(tr, "exit_reason", None) or "unknown_exit")
        exit_reason_counts[reason] = exit_reason_counts.get(reason, 0) + 1
        if "stop" in reason.lower():
            stop_hits += 1

        fills = getattr(tr, "fills", None)

        # ------------------------------------------------------
        # A) Fallback: fills yoksa tek entry/exit sim
        # ------------------------------------------------------
        if not fills:
            entry_time = df.index[ei]
            exit_time = df.index[xi]

            raw_entry = float(tr.entry_price)
            raw_exit = float(tr.exit_price) if tr.exit_price is not None else float(df["Close"].iloc[xi])

            entry_price = apply_entry_slippage(tr.direction, raw_entry, entry_slip_pct)

            if "stop" in reason.lower():
                exit_price = apply_exit_slippage(tr.direction, raw_exit, stop_slip_pct)
            else:
                exit_price = apply_exit_slippage(tr.direction, raw_exit, exit_slip_pct)

            fees = apply_fee(notional_base, taker_fee_rate) + apply_fee(notional_base, taker_fee_rate)
            balance -= fees
            total_fees += fees

            fp = funding_pnl_between(entry_time, exit_time, notional_base, 1.0, is_long=(tr.direction == "long"))
            if fp != 0.0:
                balance += fp
                total_funding += fp

            if tr.direction == "long":
                pnl = (exit_price - entry_price) / entry_price * notional_base
            else:
                pnl = (entry_price - exit_price) / entry_price * notional_base

            balance += pnl
            closed += 1
            if pnl > 0:
                wins += 1
            else:
                losses += 1

            equity_curve.append((exit_time, balance))
            continue

        # ------------------------------------------------------
        # B) fills-based execution
        # ------------------------------------------------------
        events = sorted(fills, key=lambda e: int(e.index))

        qty = 0.0          # signed fraction (+ long, - short)
        avg = None         # avg entry price
        trade_pnl = 0.0    # trade bazında pnl - fee + funding

        last_time = df.index[ei]

        def apply_funding_to(t_next: pd.Timestamp):
            nonlocal balance, total_funding, trade_pnl, last_time, qty
            if qty == 0.0:
                last_time = t_next
                return
            fp = funding_pnl_between(last_time, t_next, notional_base, abs(qty), is_long=(qty > 0))
            if fp != 0.0:
                balance += fp
                total_funding += fp
                trade_pnl += fp
            last_time = t_next

        for ev in events:
            idx = int(ev.index)
            if idx < 0 or idx >= len(df):
                continue

            t_ev = df.index[idx]
            apply_funding_to(t_ev)

            raw_px = float(ev.price)

            # flatten event mi? (exit/stop veya qty_delta=0 => flatten)
            is_flatten = (getattr(ev, "reason", "") in ("exit", "stop")) or (float(ev.qty_delta) == 0.0)
            if is_flatten:
                delta = -qty
            else:
                delta = float(ev.qty_delta)

            if abs(delta) < 1e-12:
                continue

            # slippage seçimi
            if getattr(ev, "reason", "") == "stop":
                # stop'ta pozisyon yönüne göre kötüleşen fiyat
                px = apply_stop_slippage_from_position(qty if qty != 0 else delta, raw_px, stop_slip_pct)
            else:
                # aç/artar => entry slip, azalt/kapat => exit slip
                if (qty == 0.0) or (np.sign(delta) == np.sign(qty)):
                    px = apply_entry_slippage("long" if delta > 0 else "short", raw_px, entry_slip_pct)
                else:
                    px = apply_exit_slippage("long" if qty > 0 else "short", raw_px, exit_slip_pct)

            # fee: parça notional üzerinden
            fee = apply_fee(abs(delta) * notional_base, taker_fee_rate)
            balance -= fee
            total_fees += fee
            trade_pnl -= fee

            # open/add (aynı yönde)
            if qty == 0.0 or np.sign(delta) == np.sign(qty):
                new_qty = qty + delta

                if abs(new_qty) < 1e-12:
                    qty = 0.0
                    avg = None
                else:
                    if qty == 0.0:
                        avg = px
                    else:
                        w_old = abs(qty)
                        w_new = abs(delta)
                        avg = (avg * w_old + px * w_new) / (w_old + w_new)
                    qty = new_qty
                continue

            # reduce/close (ters delta)
            if avg is None or avg <= 0:
                avg = px

            portion = abs(delta)  # kapanan fraction
            if qty > 0:  # long reduce => sell
                pnl = (px - avg) / avg * (portion * notional_base)
            else:        # short reduce => buy
                pnl = (avg - px) / avg * (portion * notional_base)

            balance += pnl
            trade_pnl += pnl

            qty = qty + delta
            if abs(qty) < 1e-12:
                qty = 0.0
                avg = None

        # trade bitti
        closed += 1
        if trade_pnl > 0:
            wins += 1
        else:
            losses += 1

        # liquidation check (kapalıysa etkisiz)
        if enable_liquidation_check:
            required_maint = notional_base * maintenance_margin_rate
            if balance <= required_maint:
                liquidations += 1
                balance = 0.0

        equity_curve.append((df.index[xi], balance))
        if balance <= 0:
            break

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
# 5) FIB STATS (Readable)
# ======================
@dataclass
class FibMonthStats:
    trade_count: int = 0
    trades_with_fills: int = 0
    fib_anchor_ready: int = 0
    fib_anchor_missing: int = 0
    partial_tp50: int = 0
    readd_50: int = 0
    exit_events: int = 0
    stop_events: int = 0

    @property
    def size_actions(self) -> int:
        return self.partial_tp50 + self.readd_50


def _get_trade_fib_anchors(trade: Any) -> Tuple[Optional[float], Optional[float]]:
    fib0 = getattr(trade, "fib0_price", None)
    fib1 = getattr(trade, "fib1_price", None)
    if fib0 is None:
        fib0 = getattr(trade, "fib0", None)
    if fib1 is None:
        fib1 = getattr(trade, "fib1", None)
    return fib0, fib1


def compute_fib_month_stats(trades: List) -> FibMonthStats:
    st = FibMonthStats(trade_count=len(trades))

    for t in trades:
        fills = getattr(t, "fills", None) or []
        if fills:
            st.trades_with_fills += 1

        fib0, fib1 = _get_trade_fib_anchors(t)
        if fib0 is not None and fib1 is not None:
            st.fib_anchor_ready += 1
        else:
            st.fib_anchor_missing += 1

        for ev in fills:
            r = getattr(ev, "reason", None)

            # ✅ STRATEJİDEKİ İSİMLERLE UYUMLU
            if r == "tp50_band":
                st.partial_tp50 += 1
            elif r == "readd50_band":
                st.readd_50 += 1
            elif r == "exit":
                st.exit_events += 1
            elif r == "stop":
                st.stop_events += 1

    return st


def merge_fib_stats(a: FibMonthStats, b: FibMonthStats) -> FibMonthStats:
    return FibMonthStats(
        trade_count=a.trade_count + b.trade_count,
        trades_with_fills=a.trades_with_fills + b.trades_with_fills,
        fib_anchor_ready=a.fib_anchor_ready + b.fib_anchor_ready,
        fib_anchor_missing=a.fib_anchor_missing + b.fib_anchor_missing,
        partial_tp50=a.partial_tp50 + b.partial_tp50,
        readd_50=a.readd_50 + b.readd_50,
        exit_events=a.exit_events + b.exit_events,
        stop_events=a.stop_events + b.stop_events,
    )


def format_fib_inline(st: FibMonthStats) -> str:
    return f"FIB[anchor={st.fib_anchor_ready}/{st.trade_count} | tp50={st.partial_tp50} | readd={st.readd_50} | actions={st.size_actions}]"


# ======================
# 6) MONTH SLICING
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
# 7) RUN ONE SYMBOL (MONTHLY RESET)
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
    show_fib_each_month: bool = True,
    show_fib_symbol_summary: bool = True,
) -> Dict[str, Any]:

    month_reports: List[Dict[str, Any]] = []

    total_net = 0.0
    total_fees = 0.0
    total_funding = 0.0
    total_closed = 0
    total_wins = 0
    total_losses = 0
    total_stop_hits = 0
    total_liqs = 0
    exit_reasons: Dict[str, int] = {}

    fib_total = FibMonthStats()

    for m_start, m_end in iter_month_ranges(start_dt, end_dt):
        start_ms = int(m_start.timestamp() * 1000)
        end_ms = int(m_end.timestamp() * 1000)

        df = fetch_klines(symbol, interval, start_ms, end_ms)
        if df.empty:
            month_reports.append({
                "month": month_key(m_start),
                "start": monthly_start_balance,
                "end": monthly_start_balance,
                "pnl": 0.0,
                "ret_pct": 0.0,
                "note": "no_data",
                "fib": None,
            })
            continue

        funding = fetch_funding_rates(symbol, start_ms, end_ms)

        highs = df["High"].values
        lows = df["Low"].values

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

        fib_st = compute_fib_month_stats(trades)
        fib_total = merge_fib_stats(fib_total, fib_st)

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
            "fib": fib_st,
        })

    if show_fib_symbol_summary:
        print(
            f"[FIB-SUMMARY] {symbol} | "
            f"anchors_ready={fib_total.fib_anchor_ready}/{fib_total.trade_count} "
            f"({(fib_total.fib_anchor_ready / fib_total.trade_count * 100.0) if fib_total.trade_count else 0.0:.1f}%) | "
            f"tp50={fib_total.partial_tp50} | readd50={fib_total.readd_50} | actions={fib_total.size_actions}"
        )

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
        "fib_total": fib_total,
        "show_fib_each_month": show_fib_each_month,
    }


# ======================
# 8) CASE CONFIG
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
# 9) MAIN
# ======================
def main():
    SYMBOLS = ["BTCUSDT", "ETHUSDT", "TRXUSDT", "SOLUSDT", "TURBOUSDT"]
    INTERVAL = "15m"

    MONTHLY_START_BALANCE = 1000.0
    LEVERAGE = 10.0

    MAX_CHASE_PCT = 0.03

    CASES: List[CaseConfig] = [
        CaseConfig(
            name="BEST",
            taker_fee_rate=0.0002,
            entry_slip=0.00005,
            exit_slip=0.00005,
            stop_slip=0.00015,
            margin_pct=0.10,
            maintenance_margin_rate=0.005,
            enable_liquidation_check=False,
        ),
        CaseConfig(
            name="AVG",
            taker_fee_rate=0.0004,
            entry_slip=0.0002,
            exit_slip=0.0002,
            stop_slip=0.0005,
            margin_pct=0.10,
            maintenance_margin_rate=0.005,
            enable_liquidation_check=False,
        ),
        CaseConfig(
            name="WORST",
            taker_fee_rate=0.0006,
            entry_slip=0.0008,
            exit_slip=0.0008,
            stop_slip=0.0015,
            margin_pct=0.10,
            maintenance_margin_rate=0.005,
            enable_liquidation_check=False,
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
                show_fib_each_month=True,
                show_fib_symbol_summary=True,
            )
            overall_by_case[case.name].append(rep)

            for row in rep["months"]:
                if row.get("note") == "no_data":
                    print(f"{row['month']}: no_data")
                    continue

                fib_st: Optional[FibMonthStats] = row.get("fib")
                fib_part = ""
                if rep.get("show_fib_each_month") and fib_st is not None:
                    fib_part = " | " + format_fib_inline(fib_st)

                print(
                    f"{row['month']}: start=${row['start']:.2f} end=${row['end']:.2f} "
                    f"pnl=${row['pnl']:.2f} ret={row['ret_pct']:.2f}% "
                    f"| trades={row['trades']} W/L={row['wins']}/{row['losses']} "
                    f"| stop={row['stop_hits']} "
                    f"| fees=${row['fees']:.2f} funding=${row['funding']:.2f}"
                    f"{fib_part}"
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
