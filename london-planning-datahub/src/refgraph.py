"""Link applications into scheme 'families' using planning reference numbers.

Condition discharges, reserved matters, S73 variations and NMAs nearly always cite
the parent permission's reference in their description ("pursuant to condition 3 of
planning permission 21/2055/F"). Matching those citations against every reference
number in the same planning authority gives a child -> parent edge list.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

TOKEN_RE = re.compile(r"[A-Z0-9][A-Z0-9/\-\.]*[A-Z0-9]")
SUFFIX_RE = re.compile(r"/[A-Z][A-Z0-9]*$")


def norm_ref(ref: str) -> str:
    return re.sub(r"\s+", "", (ref or "").upper())


def base_ref(ref: str) -> str:
    """Reference with its trailing type suffix removed (PA/21/00716/NC -> PA/21/00716)."""
    return SUFFIX_RE.sub("", ref)


def _useful(ref: str) -> bool:
    return len(ref) >= 6 and sum(ch.isdigit() for ch in ref) >= 4 and ("/" in ref or re.search(r"\d[A-Z]{2,}$", ref))


def build_edges(df: pd.DataFrame, max_km: float = 3.0) -> pd.DataFrame:
    """Return child_idx -> parent_idx edges (row positions in df) within each LPA."""
    refs = df.lpa_app_no.map(lambda r: norm_ref(r) if isinstance(r, str) else "").to_numpy()
    lpas = df.lpa_name.to_numpy()
    lat, lon = df.lat.to_numpy(), df.lon.to_numpy()
    exact: dict[tuple[str, str], list[int]] = {}
    loose: dict[tuple[str, str], list[int]] = {}
    for i, (lpa, r) in enumerate(zip(lpas, refs)):
        if not _useful(r):
            continue
        exact.setdefault((lpa, r), []).append(i)
        b = base_ref(r)
        if b != r and _useful(b):
            loose.setdefault((lpa, b), []).append(i)
    edges = []
    for i, (lpa, desc) in enumerate(zip(lpas, df.description.to_numpy())):
        if not isinstance(desc, str) or not desc:
            continue
        own = refs[i]
        seen = set()
        for tok in TOKEN_RE.findall(desc.upper()):
            tok = tok.rstrip(".-/")
            if tok == own or tok in seen or not _useful(tok):
                continue
            seen.add(tok)
            targets = exact.get((lpa, tok)) or loose.get((lpa, tok)) or loose.get((lpa, base_ref(tok)))
            if not targets or len(targets) > 3:
                continue
            for j in targets:
                if j != i and _near(lat[i], lon[i], lat[j], lon[j], max_km):
                    edges.append((i, j))
    return pd.DataFrame(edges, columns=["child", "parent"]).drop_duplicates()


def _near(la1, lo1, la2, lo2, max_km) -> bool:
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in (la1, lo1, la2, lo2)):
        return True
    dy = (la1 - la2) * 111.0
    dx = (lo1 - lo2) * 111.0 * math.cos(math.radians(51.5))
    return dx * dx + dy * dy <= max_km * max_km


def components(n: int, edges: pd.DataFrame) -> np.ndarray:
    """Union-find connected components; returns a component id per row."""
    parent = np.arange(n)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b in zip(edges.child.to_numpy(), edges.parent.to_numpy()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(n)])
