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

    # ✅ yeni kurallar
    min_same_kind_gap: int = 7,   # LH->LH arası ve LB->LB arası en az kaç mum
    min_opposite_gap: int = 3,    # LH<->LB arası en az fark (en az 1 mum arada => 2)
    debug: bool = False,
) -> List[SwingPoint]:
    """
    Swing üretim kuralları:
    1) Pivot high/low adayları çıkarılır (left/right penceresi ile).
    2) Aynı index'te hem LH hem LB varsa tekine düşür (alternation/prominence).
    3) Ardışık aynı tip pivotlar en uç olanla birleştirilir (LH -> max, LB -> min).
    4) Gap filtreleri uygulanır (same-kind ve opposite).
    5) ✅ EN ÖNEMLİ: ÇIKTI MUTLAKA ALTERNATE OLUR (LH,LB,LH,LB,...)
       - Eğer aynı tip üst üste geliyorsa drop'lamak yerine "merge/replace" yapılır:
         LH->LH: daha yüksek olan kalsın
         LB->LB: daha düşük olan kalsın
    """

    h_arr = np.asarray(highs, dtype=float)
    l_arr = np.asarray(lows, dtype=float)
    n = len(h_arr)

    if n == 0 or n != len(l_arr):
        return []

    # -----------------------------
    # 1) HAM PIVOTLAR
    # -----------------------------
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

        # aynı mumda hem LH hem LB olmasın
        if is_pivot_high and is_pivot_low:
            if last_kind == "LH":
                chosen_kind = "LB"
            elif last_kind == "LB":
                chosen_kind = "LH"
            else:
                neigh_h_max = float(np.max(np.concatenate([left_h, right_h]))) if (len(left_h) and len(right_h)) else float(np.max(h_arr[max(0, i-1):min(n, i+2)]))
                high_prom = float(high - neigh_h_max)

                neigh_l_min = float(np.min(np.concatenate([left_l, right_l]))) if (len(left_l) and len(right_l)) else float(np.min(l_arr[max(0, i-1):min(n, i+2)]))
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
    # 2) ARDIŞIK AYNI TİP PIVOTLARI GRUPLA (EN UÇ OLANI SEÇ)
    # -----------------------------
    grouped: List[SwingPoint] = []
    current_kind = raw[0].kind
    current_group: List[SwingPoint] = [raw[0]]

    for sp in raw[1:]:
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

    # -----------------------------
    # 3) GAP FİLTRESİ (DROP DEĞİL, MÜMKÜNSE REPLACE)
    # -----------------------------
    filtered: List[SwingPoint] = []

    drop_opp = 0
    drop_same = 0
    replace_same = 0
    replace_opp = 0

    for sp in grouped:
        if not filtered:
            filtered.append(sp)
            continue

        prev = filtered[-1]

        # Opposite gap: prev.kind != sp.kind olmalı ve arada yeterli mum olmalı
        if prev.kind != sp.kind:
            if (sp.index - prev.index) < min_opposite_gap:
                # Çok yakın: burda mantık şu:
                # - İki farklı tip çok yakındaysa, genelde daha "anlamlı" olanı tutmak isteriz.
                # - Basit ve stabil: daha yeni olanı (sp) tutmak yerine, öncekiyle aynı tipi
                #   zorlamamak için genelde prev'ı koruyup sp'yi düşürürüz.
                drop_opp += 1
                if debug:
                    print(f"[DROP opp_gap] {sp.kind} i={sp.index} (prev={prev.kind}@{prev.index}) need>={min_opposite_gap}")
                continue

            filtered.append(sp)
            continue

        # Same kind gap: prev.kind == sp.kind (normalde istemiyoruz zaten)
        # Ama daha bu aşamada bile olabiliyor (çünkü gap/opp drop vs.)
        # Burada drop yerine "en uç" olanla replace yapacağız.
        if (sp.index - prev.index) < min_same_kind_gap:
            # Çok yakınsa: en uç olan kalsın
            if sp.kind == "LH":
                if sp.price >= prev.price:
                    filtered[-1] = sp
                    replace_same += 1
                    if debug:
                        print(f"[REPLACE same_LH close] {prev.price}@{prev.index} -> {sp.price}@{sp.index}")
                else:
                    drop_same += 1
                    if debug:
                        print(f"[DROP same_LH close] {sp.price}@{sp.index} (keep {prev.price}@{prev.index})")
            else:  # LB
                if sp.price <= prev.price:
                    filtered[-1] = sp
                    replace_same += 1
                    if debug:
                        print(f"[REPLACE same_LB close] {prev.price}@{prev.index} -> {sp.price}@{sp.index}")
                else:
                    drop_same += 1
                    if debug:
                        print(f"[DROP same_LB close] {sp.price}@{sp.index} (keep {prev.price}@{prev.index})")
            continue

        # Enough gap ama aynı tip: yine de alternation için replace/merge
        if sp.kind == "LH":
            if sp.price >= prev.price:
                filtered[-1] = sp
                replace_same += 1
                if debug:
                    print(f"[REPLACE same_LH] {prev.price}@{prev.index} -> {sp.price}@{sp.index}")
            else:
                drop_same += 1
                if debug:
                    print(f"[DROP same_LH] {sp.price}@{sp.index} (keep {prev.price}@{prev.index})")
        else:
            if sp.price <= prev.price:
                filtered[-1] = sp
                replace_same += 1
                if debug:
                    print(f"[REPLACE same_LB] {prev.price}@{prev.index} -> {sp.price}@{sp.index}")
            else:
                drop_same += 1
                if debug:
                    print(f"[DROP same_LB] {sp.price}@{sp.index} (keep {prev.price}@{prev.index})")

    # -----------------------------
    # 4) ✅ ALTERNATION ENFORCE (LH,LB,LH,LB...)
    #    - aynı tip yakalanırsa: en uç olan kalsın (replace)
    # -----------------------------
    alt: List[SwingPoint] = []
    for sp in filtered:
        if not alt:
            alt.append(sp)
            continue

        last = alt[-1]
        if sp.kind != last.kind:
            alt.append(sp)
            continue

        # aynı tip geldiyse: en uç olanla replace
        if sp.kind == "LH":
            if sp.price >= last.price:
                alt[-1] = sp
                replace_opp += 1
                if debug:
                    print(f"[ALT REPLACE LH] {last.price}@{last.index} -> {sp.price}@{sp.index}")
            else:
                drop_same += 1
                if debug:
                    print(f"[ALT DROP LH] {sp.price}@{sp.index} (keep {last.price}@{last.index})")
        else:  # LB
            if sp.price <= last.price:
                alt[-1] = sp
                replace_opp += 1
                if debug:
                    print(f"[ALT REPLACE LB] {last.price}@{last.index} -> {sp.price}@{sp.index}")
            else:
                drop_same += 1
                if debug:
                    print(f"[ALT DROP LB] {sp.price}@{sp.index} (keep {last.price}@{last.index})")

    if debug:
        print(
            f"[SWINGS] raw={len(raw)} grouped={len(grouped)} filtered={len(filtered)} alt={len(alt)} | "
            f"drop_same={drop_same} drop_opp={drop_opp} replace_same={replace_same} replace_alt={replace_opp}"
        )

    return alt
