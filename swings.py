# swings.py
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
    right_bars: int = 3,
    min_distance: Optional[int] = None,        # artık kullanılmıyor, imza için duruyor
    alt_min_distance: Optional[int] = None,    # artık kullanılmıyor, imza için duruyor
) -> List[SwingPoint]:
    """
    Kural:
    - Bir bar (mum) en fazla 1 swing üretir (LH veya LB).
    - Ham pivotlar çıkarılır.
    - Aynı index'te hem LH hem LB çıkarsa tekini seçeriz:
        * önceki pivot LH ise LB seç (alternation)
        * önceki pivot LB ise LH seç
        * önceki yoksa "prominence" ile seç
    - Ardından ardışık aynı tip pivotlar "en uç" pivot ile birleştirilir.
    """

    h_arr = np.asarray(highs, dtype=float)
    l_arr = np.asarray(lows, dtype=float)
    n = len(h_arr)

    if n == 0 or n != len(l_arr):
        return []

    raw: List[SwingPoint] = []

    last_kind: Optional[str] = None  # aynı mumda çift pivot olursa alternation için

    for i in range(left_bars, n - right_bars):
        high = h_arr[i]
        low = l_arr[i]

        left_h = h_arr[i - left_bars: i]
        right_h = h_arr[i + 1: i + 1 + right_bars]

        left_l = l_arr[i - left_bars: i]
        right_l = l_arr[i + 1: i + 1 + right_bars]

        is_pivot_high = np.all(high >= left_h) and np.all(high >= right_h)
        is_pivot_low = np.all(low <= left_l) and np.all(low <= right_l)

        if not is_pivot_high and not is_pivot_low:
            continue

        # --- aynı mumda hem LH hem LB olmasın ---
        if is_pivot_high and is_pivot_low:
            # 1) alternation'a göre seç
            if last_kind == "LH":
                chosen_kind = "LB"
            elif last_kind == "LB":
                chosen_kind = "LH"
            else:
                # 2) prominence ile seç (hangisi çevreye göre daha "uç")
                # yüksek için: high - max(neighbors_high)
                neigh_h_max = float(np.max(np.concatenate([left_h, right_h]))) if (len(left_h) and len(right_h)) else float(np.max(h_arr[max(0, i-1):min(n, i+2)]))
                high_prom = float(high - neigh_h_max)

                # düşük için: min(neighbors_low) - low
                neigh_l_min = float(np.min(np.concatenate([left_l, right_l]))) if (len(left_l) and len(right_l)) else float(np.min(l_arr[max(0, i-1):min(n, i+2)]))
                low_prom = float(neigh_l_min - low)

                # prominence büyük olanı seç
                chosen_kind = "LH" if high_prom >= low_prom else "LB"

            if chosen_kind == "LH":
                raw.append(SwingPoint(index=i, price=float(high), kind="LH"))
                last_kind = "LH"
            else:
                raw.append(SwingPoint(index=i, price=float(low), kind="LB"))
                last_kind = "LB"

            continue

        # normal tek pivot
        if is_pivot_high:
            raw.append(SwingPoint(index=i, price=float(high), kind="LH"))
            last_kind = "LH"
        elif is_pivot_low:
            raw.append(SwingPoint(index=i, price=float(low), kind="LB"))
            last_kind = "LB"

    raw.sort(key=lambda sp: sp.index)
    if not raw:
        return []

    # --- ardışık aynı tip pivotları "en uç" pivot ile birleştir ---
    grouped: List[SwingPoint] = []

    current_kind = raw[0].kind
    current_group: List[SwingPoint] = [raw[0]]

    for sp in raw[1:]:
        if sp.kind == current_kind:
            current_group.append(sp)
        else:
            if current_kind == "LH":
                best = max(current_group, key=lambda s: s.price)
            else:
                best = min(current_group, key=lambda s: s.price)

            grouped.append(best)
            current_kind = sp.kind
            current_group = [sp]

    if current_group:
        if current_kind == "LH":
            best = max(current_group, key=lambda s: s.price)
        else:
            best = min(current_group, key=lambda s: s.price)
        grouped.append(best)

    return grouped
