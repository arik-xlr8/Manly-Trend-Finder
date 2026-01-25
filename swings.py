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
    min_distance: Optional[int] = None,
    alt_min_distance: Optional[int] = None,
) -> List[SwingPoint]:
    """
    1) Ham LH/LB pivotları bul.
    2) Ardışık aynı tip pivotları birleştir:
       - LH grubunda en yüksek olan kalsın
       - LB grubunda en düşük olan kalsın
    """

    h_arr = np.asarray(highs, dtype=float)
    l_arr = np.asarray(lows, dtype=float)
    n = len(h_arr)

    if n == 0 or n != len(l_arr):
        return []

    swings: List[SwingPoint] = []

    # 1) Ham pivotlar
    for i in range(left_bars, n - right_bars):
        high = h_arr[i]
        low = l_arr[i]

        left_h = h_arr[i - left_bars: i]
        right_h = h_arr[i + 1: i + 1 + right_bars]

        left_l = l_arr[i - left_bars: i]
        right_l = l_arr[i + 1: i + 1 + right_bars]

        is_pivot_high = np.all(high >= left_h) and np.all(high >= right_h)
        is_pivot_low = np.all(low <= left_l) and np.all(low <= right_l)

        if is_pivot_high:
            swings.append(SwingPoint(index=i, price=float(high), kind="LH"))
        if is_pivot_low:
            swings.append(SwingPoint(index=i, price=float(low), kind="LB"))

    swings.sort(key=lambda sp: sp.index)
    if not swings:
        return []

    # 2) Aynı tip ardışık pivotları birleştir
    grouped: List[SwingPoint] = []

    current_kind = swings[0].kind
    current_group: List[SwingPoint] = [swings[0]]

    for sp in swings[1:]:
        if sp.kind == current_kind:
            current_group.append(sp)
        else:
            best = max(current_group, key=lambda s: s.price) if current_kind == "LH" else min(current_group, key=lambda s: s.price)
            grouped.append(best)
            current_kind = sp.kind
            current_group = [sp]

    if current_group:
        best = max(current_group, key=lambda s: s.price) if current_kind == "LH" else min(current_group, key=lambda s: s.price)
        grouped.append(best)

    return grouped
