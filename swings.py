from dataclasses import dataclass
from typing import List, Literal, Sequence, Optional

import numpy as np


@dataclass
class SwingPoint:
    index: int          # hangi mum
    price: float        # o mumun (iğnenin) fiyatı
    kind: Literal["LH", "LB"]  # Local High / Local Bottom (pivot high / low)


def find_swings(
    highs: Sequence[float],
    lows: Sequence[float],
    left_bars: int = 3,
    right_bars: int = 3,
    min_distance: Optional[int] = None,        # aynı tip (LH-LH / LB-LB) için
    alt_min_distance: Optional[int] = None,    # farklı tip (LH-LB / LB-LH) için
) -> List[SwingPoint]:
    """
    highs, lows     : mumların HIGH ve LOW değerleri
    left_bars       : pivot saymak için solda bakılacak bar sayısı
    right_bars      : pivot saymak için sağda bakılacak bar sayısı

    min_distance    : AYNI tip pivotlar (LH-LH veya LB-LB) arasında
                      olması gereken minimum bar mesafesi.
                      Bu mesafeden daha yakınlarsa daha 'uç' pivot kalır.

    alt_min_distance: FARKLI tip pivotlar (LH-LB, LB-LH) arasında
                      uygulanacak, genellikle ÇOK DAHA KÜÇÜK mesafe.
                      None ise, farklı tip pivotlar için seyreltme YAPILMAZ.
    """
    h_arr = np.asarray(highs, dtype=float)
    l_arr = np.asarray(lows, dtype=float)
    n = len(h_arr)

    if n == 0 or n != len(l_arr):
        return []

    swings: List[SwingPoint] = []

    # --- 1) Ham pivotları bul (left/right bar kuralına göre) ---
    for i in range(left_bars, n - right_bars):
        high = h_arr[i]
        low = l_arr[i]

        left_h = h_arr[i - left_bars : i]
        right_h = h_arr[i + 1 : i + 1 + right_bars]

        left_l = l_arr[i - left_bars : i]
        right_l = l_arr[i + 1 : i + 1 + right_bars]

        is_pivot_high = np.all(high >= left_h) and np.all(high >= right_h)
        is_pivot_low  = np.all(low  <= left_l) and np.all(low  <= right_l)

        if is_pivot_high:
            swings.append(SwingPoint(index=i, price=high, kind="LH"))

        if is_pivot_low:
            swings.append(SwingPoint(index=i, price=low, kind="LB"))

    swings.sort(key=lambda sp: sp.index)

    # Eğer hiç seyreltme parametresi yoksa, ham pivotları dön
    if (min_distance is None or min_distance <= 0) and \
       (alt_min_distance is None or alt_min_distance <= 0):
        return swings

    # --- 2) Seyreltme filtresi ---
    filtered: List[SwingPoint] = []
    for sp in swings:
        if not filtered:
            filtered.append(sp)
            continue

        last = filtered[-1]
        bar_dist = sp.index - last.index

        if sp.kind == last.kind:
            # AYNI TİP pivotlar (LH-LH veya LB-LB)
            if min_distance is not None and min_distance > 0 and bar_dist < min_distance:
                # daha 'uç' olan pivot kalsın
                if sp.kind == "LH":
                    if sp.price > last.price:
                        filtered[-1] = sp
                else:  # LB
                    if sp.price < last.price:
                        filtered[-1] = sp
                # yenisini eklemiyoruz
                continue
            else:
                # yeterince uzakta, ikisi de mantıklı
                filtered.append(sp)
        else:
            # FARKLI TİP pivotlar (LH-LB veya LB-LH)
            if alt_min_distance is not None and alt_min_distance > 0:
                if bar_dist < alt_min_distance:
                    # çok çok dibine yapışmışsa, istersek hafif filtre uygularız
                    # burada sade davranalım: last'i koruyup yeniyi at
                    continue
                else:
                    filtered.append(sp)
            else:
                # alt_min_distance yoksa → farklı tip pivotlarda seyreltme YOK
                filtered.append(sp)

    return filtered
