from __future__ import annotations

from typing import Dict, Sequence


def allocate_round_robin_heights(total_h: int, panel_order: Sequence[str], *, min_panel_h: int) -> Dict[str, int]:
    total_h = max(0, int(total_h))
    order = list(panel_order)
    if not order:
        return {}

    if total_h >= min_panel_h * len(order):
        heights = {k: int(min_panel_h) for k in order}
        extra = total_h - (min_panel_h * len(order))
        idx = 0
        while extra > 0:
            key = order[idx]
            heights[key] += 1
            extra -= 1
            idx = (idx + 1) % len(order)
        return heights

    base = total_h // len(order)
    rem = total_h % len(order)
    heights = {k: base for k in order}
    for i in range(rem):
        heights[order[i]] += 1
    return heights