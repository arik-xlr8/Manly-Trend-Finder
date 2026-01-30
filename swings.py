# swings.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Sequence, Optional

import numpy as np


@dataclass
class SwingPoint:
    index: int
    price: float
    kind: Literal["LH", "LB"]  # Local High / Local Bottom


def find_swings(
    highs: Sequence[float],
    lows: Sequence[float],
    left_bars: int = 3,
    right_bars: int = 1,

    # artık kullanılmıyor, imza için duruyor
    min_distance: Optional[int] = None,
    alt_min_distance: Optional[int] = None,

    # kurallar
    min_same_kind_gap: int = 10,  # LH->LH ve LB->LB arası min mum (repaint modunda "kabul" değil, "replace penceresi" gibi düşün)
    min_opposite_gap: int = 4,    # LH<->LB arası min fark

    debug: bool = False,

    # repaint fix (canlıda kapanmamış bar)
    ignore_last_bar: bool = True,
) -> List[SwingPoint]:
    """
    ✅ REPAINT (ZIGZAG) mantığı:

    1) left/right penceresiyle "confirmed" pivot adayları çıkarılır.
    2) Aynı index'te hem LH hem LB varsa tekine düşür (alternation/prominence).
    3) Pivotları soldan sağa yürürken:
       - Eğer yeni pivot, son pivotla AYNI TİP ise (LH->LH / LB->LB):
            -> daha ekstrem ise (LH daha yüksek / LB daha düşük) SON PIVOTU REPLACE eder
            -> değilse drop eder
          (Bu, "repaint" davranışıdır: son tepe/dip daha iyi bir ekstremle güncellenir.)
       - Eğer yeni pivot TERS TİP ise (LH<->LB):
            -> gap < min_opposite_gap ise gürültü sayıp drop eder
            -> değilse ekler
    4) min_same_kind_gap:
       - repaint modunda aynı tipin "eklenmesi" zaten yok; sadece replace var.
       - yine de istersen, çok yakın aynı-tip replace'leri engellemek için "replace penceresi" gibi uygulanır:
            -> aynı tip geldiğinde gap < min_same_kind_gap ise replace'e izin ver
            -> gap >= min_same_kind_gap ise (zaten aynı tip üst üste geldiği için) yine replace mantığı çalışır.
         (Yani bu parametre repaintte “accept” değil, “ne kadar hızlı güncelleyeyim” gibi davranır.)
    """

    h_arr = np.asarray(highs, dtype=float)
    l_arr = np.asarray(lows, dtype=float)
    n = len(h_arr)

    if n == 0 or n != len(l_arr):
        return []

    # Son barı yok say (kapanmamış olabilir)
    effective_n = n - 1 if ignore_last_bar else n
    if effective_n <= 0:
        return []

    # Pivot aday aralığı: i in [left_bars, effective_n - right_bars)
    last_i_exclusive = effective_n - right_bars
    if last_i_exclusive <= left_bars:
        return []

    # -----------------------------
    # 1) HAM (CONFIRMED) PIVOTLAR
    # -----------------------------
    raw: List[SwingPoint] = []
    last_kind: Optional[str] = None  # aynı mumda çift pivot olursa alternation için

    for i in range(left_bars, last_i_exclusive):
        high = h_arr[i]
        low = l_arr[i]

        left_h = h_arr[i - left_bars: i]
        left_l = l_arr[i - left_bars: i]

        right_h = h_arr[i + 1: i + 1 + right_bars]
        right_l = l_arr[i + 1: i + 1 + right_bars]

        is_pivot_high = np.all(high >= left_h) and np.all(high >= right_h)
        is_pivot_low = np.all(low <= left_l) and np.all(low <= right_l)

        if not is_pivot_high and not is_pivot_low:
            continue

        # aynı mumda hem LH hem LB olmasın
        if is_pivot_high and is_pivot_low:
            if last_kind == "LH":
                chosen_kind = "LB"
            elif last_kind == "LB":
                chosen_kind = "LH"
            else:
                # prominence (mevcut mantık)
                if len(left_h) > 0 or len(right_h) > 0:
                    neigh_h_max = float(np.max(np.concatenate([left_h, right_h])))
                else:
                    neigh_h_max = float(np.max(h_arr[max(0, i - 1):min(effective_n, i + 2)]))
                high_prom = float(high - neigh_h_max)

                if len(left_l) > 0 or len(right_l) > 0:
                    neigh_l_min = float(np.min(np.concatenate([left_l, right_l])))
                else:
                    neigh_l_min = float(np.min(l_arr[max(0, i - 1):min(effective_n, i + 2)]))
                low_prom = float(neigh_l_min - low)

                chosen_kind = "LH" if high_prom >= low_prom else "LB"

            if chosen_kind == "LH":
                raw.append(SwingPoint(index=i, price=float(high), kind="LH"))
                last_kind = "LH"
            else:
                raw.append(SwingPoint(index=i, price=float(low), kind="LB"))
                last_kind = "LB"
            continue

        if is_pivot_high:
            raw.append(SwingPoint(index=i, price=float(high), kind="LH"))
            last_kind = "LH"
        elif is_pivot_low:
            raw.append(SwingPoint(index=i, price=float(low), kind="LB"))
            last_kind = "LB"

    raw.sort(key=lambda sp: sp.index)
    if not raw:
        return []

    # -----------------------------
    # 2) REPAINT ZIGZAG FİLTRE
    # -----------------------------
    out: List[SwingPoint] = []

    drop_opp = 0
    drop_same = 0
    repl_same = 0

    def is_more_extreme(new_sp: SwingPoint, old_sp: SwingPoint) -> bool:
        if new_sp.kind != old_sp.kind:
            return False
        if new_sp.kind == "LH":
            return new_sp.price > old_sp.price
        else:
            return new_sp.price < old_sp.price

    for sp in raw:
        if not out:
            out.append(sp)
            continue

        prev = out[-1]
        gap = sp.index - prev.index

        # Aynı tip geldiyse: REPAINT = daha ekstremse REPLACE
        if sp.kind == prev.kind:
            # Çok yakınsa da (min_same_kind_gap), repaint mantığında bu zaten replace penceresi gibi iş görüyor
            if is_more_extreme(sp, prev):
                out[-1] = sp
                repl_same += 1
                if debug:
                    print(f"[REPLACE samekind] {sp.kind} prev@{prev.index}:{prev.price} -> new@{sp.index}:{sp.price}")
            else:
                drop_same += 1
                if debug:
                    print(f"[DROP samekind] {sp.kind} new@{sp.index}:{sp.price} (prev@{prev.index}:{prev.price})")
            continue

        # Ters tip geldiyse: min_opposite_gap şartı
        if gap < min_opposite_gap:
            drop_opp += 1
            if debug:
                print(f"[DROP opp_gap] {sp.kind} i={sp.index} (prev={prev.kind}@{prev.index}) need>={min_opposite_gap}")
            continue

        out.append(sp)

    if debug:
        print(
            f"[SWINGS-REPAINT] raw={len(raw)} out={len(out)} | "
            f"replace_same={repl_same} drop_same={drop_same} drop_opp={drop_opp} "
            f"(ignore_last_bar={ignore_last_bar}, right_bars={right_bars}, "
            f"min_opposite_gap={min_opposite_gap}, min_same_kind_gap={min_same_kind_gap})"
        )

    return out
