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

    # ✅ SENİN İSTEDİĞİN KURALLAR
    min_same_kind_gap: int = 20,  # LH->LH ve LB->LB arası min mum
    min_opposite_gap: int = 5,    # LH<->LB arası min fark

    debug: bool = False,

    # ✅ repaint fix
    ignore_last_bar: bool = True,
) -> List[SwingPoint]:
    """
    Swing üretim kuralları:
    1) Pivot high/low adayları çıkarılır (left/right penceresi ile).
    2) Aynı index'te hem LH hem LB varsa tekine düşür (alternation/prominence).
    3) Gap filtreleri uygulanır:
        - opposite gap: min_opposite_gap
        - same-kind gap: min_same_kind_gap
    4) ✅ KİLİT: Bir pivot seçildiyse ASLA override/replace edilmez.
       (Daha iyi tepe/dip gelse bile eski pivot kıpırdamaz; yeni aday drop edilir.)
    5) ÇIKTI doğal olarak "1 tepe 1 dip" alternasyonuna gider; aynı tip gelirse drop edilir.

    Repaint notu:
    - right_bars > 0 ise pivot "confirmed" için sağda right_bars kadar kapanmış bar gerekir.
    - ignore_last_bar=True: canlıda kapanmamış son bar sağ pencereye girip pivotları oynatmasın.
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

    # Pivot aday aralığı:
    # i in [left_bars, effective_n - right_bars)
    last_i_exclusive = effective_n - right_bars
    if last_i_exclusive <= left_bars:
        return []

    # -----------------------------
    # 1) HAM PIVOTLAR
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
                # prominence hesabı (mevcut mantık)
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
    # 2) GAP + ALTERNATION (✅ NO REPLACE, SADECE DROP)
    # -----------------------------
    out: List[SwingPoint] = []

    drop_opp = 0
    drop_same = 0
    drop_alt_samekind = 0

    for sp in raw:
        if not out:
            out.append(sp)
            continue

        prev = out[-1]
        gap = sp.index - prev.index

        # Same-kind (LH->LH veya LB->LB): asla replace yok, sadece şart sağlanıyorsa bile
        # "1 tepe 1 dip" istediğin için aynı tip gelince direkt drop.
        if sp.kind == prev.kind:
            # aynı tip gelmiş; sen "override istemiyorum" dediğin için replace yok.
            # ayrıca min_same_kind_gap kuralı da var ama zaten aynı tip gelince drop ediyoruz.
            drop_alt_samekind += 1
            if debug:
                print(f"[DROP alt_samekind] {sp.kind} i={sp.index} (prev={prev.kind}@{prev.index})")
            continue

        # Opposite gap (LB<->LH)
        if gap < min_opposite_gap:
            drop_opp += 1
            if debug:
                print(f"[DROP opp_gap] {sp.kind} i={sp.index} (prev={prev.kind}@{prev.index}) need>={min_opposite_gap}")
            continue

        # Ek güvenlik: “iki pivot arası” dediğin için burada aynı zamanda
        # önceki pivot ile yeni pivot farklı olsa bile genel minimumu sağlıyor zaten.

        out.append(sp)

    # -----------------------------
    # 3) (Opsiyonel) Aynı tip arası 20 mum kuralı:
    # Zaten alternation ile aynı tip yan yana gelmiyor.
    # Ama teorik olarak "LB ... LH ... LB" içinde LB-LB mesafesini de kontrol etmek istersen:
    # burada ikinci bir pass ile aynı tip son görülen indexe bakıp drop edebiliriz.
    # Sen bunu net istediğin için ekledim.
    # -----------------------------
    final: List[SwingPoint] = []
    last_seen_same: dict[str, int] = {"LH": -10**18, "LB": -10**18}

    for sp in out:
        last_i = last_seen_same.get(sp.kind, -10**18)
        if (sp.index - last_i) < min_same_kind_gap:
            drop_same += 1
            if debug:
                print(f"[DROP same_kind_gap] {sp.kind} i={sp.index} last_same={last_i} need>={min_same_kind_gap}")
            continue
        final.append(sp)
        last_seen_same[sp.kind] = sp.index

    if debug:
        print(
            f"[SWINGS] raw={len(raw)} out={len(out)} final={len(final)} | "
            f"drop_opp={drop_opp} drop_alt_samekind={drop_alt_samekind} drop_same_kind_gap={drop_same} "
            f"(ignore_last_bar={ignore_last_bar}, right_bars={right_bars}, "
            f"min_opposite_gap={min_opposite_gap}, min_same_kind_gap={min_same_kind_gap})"
        )

    return final
