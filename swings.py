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
    min_distance: Optional[int] = None,        # artık kullanılmıyor, imza için duruyor
    alt_min_distance: Optional[int] = None,    # artık kullanılmıyor, imza için duruyor
) -> List[SwingPoint]:
    """
    highs, lows     : mumların HIGH ve LOW değerleri
    left_bars       : pivot saymak için solda bakılacak bar sayısı
    right_bars      : pivot saymak için sağda bakılacak bar sayısı

    Yeni kural:
    - Önce klasik şekilde bütün LH / LB pivotlarını çıkar.
    - Sonra:
        * Eğer art arda birden fazla LH varsa,
          bu aralıktaki en yüksek fiyatlı LH'yi seç, diğer LH'leri sil.
        * Eğer art arda birden fazla LB varsa,
          bu aralıktaki en düşük fiyatlı LB'yi seç, diğer LB'leri sil.
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

    # Index'e göre sırala (zaman)
    swings.sort(key=lambda sp: sp.index)

    if not swings:
        return []

    # --- 2) Aynı tip ardışık pivot gruplarını "en uç" pivot ile birleştir ---
    grouped: List[SwingPoint] = []

    current_kind = swings[0].kind
    current_group: List[SwingPoint] = [swings[0]]

    for sp in swings[1:]:
        if sp.kind == current_kind:
            # Aynı tip -> aynı gruba ekle
            current_group.append(sp)
        else:
            # Farklı tipe geçtik: mevcut grubu kapat
            if current_kind == "LH":
                # Grup içindeki en yüksek tepe
                best = max(current_group, key=lambda s: s.price)
            else:  # "LB"
                # Grup içindeki en düşük dip
                best = min(current_group, key=lambda s: s.price)

            grouped.append(best)

            # Yeni grup başlat
            current_kind = sp.kind
            current_group = [sp]

    # Son grubu da flush et
    if current_group:
        if current_kind == "LH":
            best = max(current_group, key=lambda s: s.price)
        else:
            best = min(current_group, key=lambda s: s.price)
        grouped.append(best)

    return grouped
