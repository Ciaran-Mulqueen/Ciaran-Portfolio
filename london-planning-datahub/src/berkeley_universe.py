"""Identify the Berkeley Group application universe in the PLD extract.

1. Seeds: applications whose lead developer is a Berkeley brand (Berkeley Homes divisions,
   St George, St James, St Edward, St William, St Joseph). Only populated from ~2021.
2. Curated seeds: the head permissions of known Berkeley schemes (found by site name and
   checked by hand) so that schemes whose activity pre-dates the developer field are included.
3. Expansion: walk the reference-number graph (child cites parent permission) out from the
   seeds, a few hops deep and within a radius of the seed, to collect each scheme's full
   lifecycle: outline -> reserved matters -> S73/NMA amendments -> condition discharges.
"""
from __future__ import annotations

import re
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from developers import berkeley_brand  # noqa: E402
from refgraph import build_edges  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"

MAX_DEPTH = 6
MAX_KM = 2.0


def load_edges(df: pd.DataFrame) -> pd.DataFrame:
    path = DATA / "ref_edges.parquet"
    if path.exists():
        return pd.read_parquet(path)
    edges = build_edges(df)
    edges.to_parquet(path, index=False)
    return edges


def expand(seeds: list[int], edges: pd.DataFrame, lat: np.ndarray, lon: np.ndarray,
           blocked: set[int] = frozenset()) -> dict[int, tuple[int, int]]:
    """BFS from each seed over undirected reference edges. Returns node -> (seed, depth)."""
    adj: dict[int, list[int]] = {}
    for a, b in zip(edges.child.to_numpy(), edges.parent.to_numpy()):
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    found: dict[int, tuple[int, int]] = {}
    for s in seeds:
        if s in found and found[s][1] == 0:
            continue
        found[s] = (s, 0)
        dq = deque([(s, 0)])
        while dq:
            node, depth = dq.popleft()
            if depth >= MAX_DEPTH or node in blocked:
                continue
            for nb in adj.get(node, ()):
                if nb in found:
                    continue
                if not _within(lat, lon, s, nb):
                    continue
                found[nb] = (s, depth + 1)
                dq.append((nb, depth + 1))
    return found


def _within(lat, lon, a, b) -> bool:
    if np.isnan(lat[a]) or np.isnan(lat[b]):
        return True
    dy = (lat[a] - lat[b]) * 111.0
    dx = (lon[a] - lon[b]) * 111.0 * np.cos(np.radians(51.5))
    return dx * dx + dy * dy <= MAX_KM ** 2


def norm_regno(r: str | None) -> str | None:
    """Companies House numbers are recorded inconsistently ('04480928', '4480928', 'N/A')."""
    if not r:
        return None
    r = str(r).strip().upper().replace(" ", "")
    if r in {"N/A", "NA", "NONE", "0", "-"} or len(r) < 6:
        return None
    return r.lstrip("0") if r.isdigit() else r


def summarise_schemes(b: pd.DataFrame) -> pd.DataFrame:
    """One row per scheme: size, timing and the mix of application types."""
    def head_units(g: pd.DataFrame) -> float:
        heads = g[g.category.isin(["full", "outline", "full_or_other", "s73"])]
        return heads.units_proposed.max() if len(heads) else np.nan

    rows = []
    for sid, g in b.groupby("scheme"):
        rows.append({
            "scheme": sid,
            "lpa": g.lpa_name.mode().iat[0],
            "brand": g.scheme_brand.dropna().mode().iat[0] if g.scheme_brand.notna().any() else None,
            "apps": len(g),
            "first_valid": g.valid_date.min(),
            "last_valid": g.valid_date.max(),
            "max_scheme_units": head_units(g),
            "rm_units": g.loc[g.category == "reserved_matters", "units_proposed"].sum(),
            "n_new": g.category.isin(["full", "outline", "full_or_other"]).sum(),
            "n_rm": (g.category == "reserved_matters").sum(),
            "n_s73": (g.category == "s73").sum(),
            "n_nma": (g.category == "nma").sum(),
            "n_cond": (g.category == "conditions").sum(),
        })
    return pd.DataFrame(rows).sort_values("apps", ascending=False)


CONFIG = Path(__file__).resolve().parents[1] / "config"
ASSIGN_KM = 1.2


def _clean_dev(dev) -> str | None:
    """Treat placeholder values as blank (Islington records councillors in the developer field)."""
    if not isinstance(dev, str) or not dev.strip():
        return None
    if re.match(r"^\s*(councillor|cllr)\b", dev, re.I) or PLACEHOLDER_DEV.search(dev):
        return None
    return dev.strip()


# Council consultees that some boroughs record in the developer field
PLACEHOLDER_DEV = re.compile(
    r"public protection|design (and|&) conservation|access (and|&) inclusive|biodiversity|planning policy|"
    r"crime prevention|tree preservation|parks (and|&) open spaces|housing department|energy and energy|"
    r"^highways$|land use planning team|planning obligations|^n/?a$|^none$|^tbc$|^unknown$", re.I)

# Site keywords per scheme (used when lineage does not settle the scheme); borough-restricted
SCHEME_KEYWORDS = {
    ("Greenwich", "Kidbrooke Village"): r"kidbrooke|ferrier",
    ("Greenwich", "Royal Arsenal Riverside"): r"royal arsenal|ropeyard|victory parade|major draper|building 10|beresford street",
    ("Greenwich", "Lombard Square"): r"pettman|nathan way|lombard|hadden road",
    ("Hackney", "Woodberry Down"): r"woodberry|robin redmond|calico house|kss ?1",
    ("Hammersmith & Fulham", "White City Living"): r"white city|54 wood lane|m&s white",
    ("Hammersmith & Fulham", "King's Road Park (Fulham Gasworks)"): r"fulham gas|imperial road|gas ?holder|kings road park",
    ("Hammersmith & Fulham", "Fulham Reach"): r"fulham reach|distillery|frank banfield",
    ("Hammersmith & Fulham", "Chelsea Creek"): r"chelsea creek",
    ("Ealing", "The Green Quarter"): r"green quarter|the straight|southall gas|seva drive|southall waterside",
    ("Haringey", "Alexandra Gate (Clarendon)"): r"haringey heartlands|clarendon|olympia trading|hornsey park road|coburg road|gas holder station|western road",
    ("Brent", "Grand Union"): r"grand union|northfield|beresford avenue",
    ("Tower Hamlets", "London Dock"): r"london dock|news international|virginia st|pennington street",
    ("Tower Hamlets", "Heron Wharf (Poplar gasholders)"): r"poplar gas|leven road|poplar riverside",
    ("Tower Hamlets", "Bow Green (Bow Common gasworks)"): r"bow common",
    ("Tower Hamlets", "Regent's View"): r"marian place|bethnal green (gas|holder)|regent'?s view",
    ("Tower Hamlets", "South Quay Plaza"): r"south quay plaza|183-189 marsh wall",
    ("Tower Hamlets", "Goodman's Fields"): r"goodman",
    ("Newham", "TwelveTrees Park"): r"twelve ?trees|parcelforce|west ham station",
    ("Newham", "Bromley-by-Bow gasworks"): r"bromley.by.bow",
    ("Newham", "Beckton Riverside (Gallions Reach)"): r"land south of gallions|beckton gas",
    ("Newham", "East Ham gasholder"): r"east ham.{0,40}gas",
    ("Lambeth", "Oval Village"): r"oval village|gasholder station|kennington oval",
    ("Westminster", "West End Gate"): r"west end gate|14[37] to 147 church st|new church road",
    ("Westminster", "Trillium (Paddington Green)"): r"paddington green",
    ("Westminster", "Ergon House"): r"ergon house",
    ("Southwark", "Bermondsey Place"): r"malt street|bermondsey place",
    ("Southwark", "Borough Triangle"): r"borough triangle|newington causeway",
    ("Southwark", "Aylesham Centre"): r"aylesham",
    ("Enfield", "Trent Park"): r"trent park|middlesex university",
    ("Barnet", "Beaufort Park"): r"beaufort park|raf east camp",
    ("Barnet", "New Barnet gasworks"): r"british gas works|new barnet gas|ensemble road",
    ("Camden", "Camden Goods Yard"): r"camden goods|morrisons",
    ("Sutton", "Sutton Garden Square (former B&Q)"): r"chalk pit way|b and q|b&q",
    ("Kingston", "Motspur Park gasholders"): r"gas ?holder",
    ("Merton", "Seagas House"): r"seagas",
    ("Kensington & Chelsea", "Kensal Canalside (Kensal gasworks)"): r"kensal|canal way",
    ("Waltham Forest", "Lea Bridge"): r"lea bridge gas|perth road",
    ("Wandsworth", "Prince of Wales Drive"): r"prince of wales drive|battersea gas",
    ("Wandsworth", "Battersea Reach"): r"battersea reach",
    ("Wandsworth", "Riverlight"): r"riverlight",
    ("Wandsworth", "Wandsworth Mills (Ram Brewery)"): r"\bram (marketing|quarter|brewery)|former ram brewery",
    ("Islington", "250 City Road"): r"city forum|250 city road",
    ("Barnet", "Silkstream"): r"hyde estate road|silk ?stream|silk park",
    ("LLDC", "Stratford gasworks"): r"rick roberts|former gasworks",
    ("Newham", "Stratford gasworks"): r"rick roberts",
    ("Hounslow", "Syon Lane (Tesco Osterley)"): r"tesco extra|syon lane",
    ("Ealing", "Dickens Yard"): r"dickens yard",
}
_KW = {k: re.compile(v, re.I) for k, v in SCHEME_KEYWORDS.items()}


def keyword_scheme(lpa: str, text: str) -> str | None:
    for (kl, scheme), rx in _KW.items():
        if kl == lpa and rx.search(text):
            return scheme
    return None


def build(df: pd.DataFrame, edges: pd.DataFrame, min_conf=("high", "medium")) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    dev = df.developer.map(_clean_dev)
    df["brand"] = dev.map(berkeley_brand)
    df["reg"] = df.company_reg_no.map(lambda r: norm_regno(r) if isinstance(r, str) else None)
    regs = set(df.loc[df.brand.notna(), "reg"].dropna())
    named = (df.brand.notna() | df.reg.isin(regs)).to_numpy()

    cur = pd.read_csv(CONFIG / "berkeley_curated_heads.csv")
    cur = cur[cur.confidence.isin(min_conf)]
    key = df.lpa_name + "|" + df.lpa_app_no.fillna("")
    cur_map = {f"{r.lpa_name}|{r.lpa_app_no}": r for r in cur.itertuples()}
    curated = key.isin(cur_map).to_numpy()
    missing = set(cur_map) - set(key[curated])
    if missing:
        print("curated refs not found:", sorted(missing))

    other = (dev.notna() & df.brand.isna() & ~df.reg.isin(regs)).to_numpy()
    seeds = list(np.flatnonzero(named | curated))
    found = expand(seeds, edges, df.lat.to_numpy(), df.lon.to_numpy(), blocked=set(np.flatnonzero(other)))
    idx = np.array(sorted(found))
    b = df.iloc[idx].copy()
    b["depth"] = [found[i][1] for i in idx]
    b["tier"] = np.where(named[idx], "named", np.where(curated[idx], "curated", "linked"))
    b["other_developer"] = other[idx]

    # Scheme assignment, in order of evidence strength:
    #  1. curated head permissions carry their scheme;
    #  2. site keywords (borough-restricted) for named seeds and anything still unassigned;
    #  3. inherit the scheme of the BFS seed (reference-graph lineage);
    #  4. nearest scheme anchor in the same borough within ASSIGN_KM; else an "Other" site.
    b["seed"] = [found[i][0] for i in idx]
    text = (b.site_name.fillna("") + " " + b.description.fillna("").str[:300])
    b["scheme"] = [cur_map[k].scheme if k in cur_map else None for k in key.iloc[idx]]
    kw = [keyword_scheme(l, t) for l, t in zip(b.lpa_name, text)]
    b["scheme"] = b.scheme.fillna(pd.Series(kw, index=b.index))
    seed_scheme = dict(zip(b.index, b.scheme))
    b["scheme"] = b.scheme.fillna(b.seed.map(seed_scheme))
    anchors = b[b.scheme.notna() & b.lat.notna()].groupby(["scheme", "lpa_name"]).agg(
        lat=("lat", "median"), lon=("lon", "median")).reset_index()
    sch_brand = {r.scheme: r.brand for r in cur.itertuples()}
    # Schemes taken over part-way: drop records before Berkeley's involvement
    starts = {r.scheme: pd.Timestamp(r.from_date) for r in cur.itertuples()
              if isinstance(r.from_date, str) and r.from_date}
    for i in b.index[b.scheme.isna()]:
        r = b.loc[i]
        cands = anchors[anchors.lpa_name == r.lpa_name]
        if len(cands) and pd.notna(r.lat):
            d = np.hypot((cands.lat - r.lat) * 111.0, (cands.lon - r.lon) * 111.0 * np.cos(np.radians(51.5)))
            if d.min() <= ASSIGN_KM:
                b.at[i, "scheme"] = cands.scheme.iat[int(np.argmin(d.to_numpy()))]
                continue
        site = str(r.site_name if isinstance(r.site_name, str) else "").split("\n")[0].split("\r")[0][:40]
        b.at[i, "scheme"] = f"Other: {r.lpa_name} - {site}".strip(" -")
    for sch, start in starts.items():
        b = b[~((b.scheme == sch) & ~(b.valid_date >= start))]
    b["scheme_brand"] = b.scheme.map(sch_brand).fillna(b.brand)
    # St Edward is a 50:50 JV with M&G; St William was 50:50 with National Grid until Berkeley
    # bought National Grid's half on 15 Mar 2022
    b["jv"] = (b.scheme_brand == "St Edward") | ((b.scheme_brand == "St William") & (b.valid_date < "2022-03-15"))
    return b


def main() -> None:
    df = pd.read_parquet(DATA / "applications.parquet")
    edges = load_edges(df)
    b = build(df, edges)
    b.to_parquet(DATA / "berkeley_applications.parquet", index=False)
    print(f"Berkeley universe: {len(b):,} applications; tiers: {b.tier.value_counts().to_dict()}")
    print("other-named developers reached (excluded from Berkeley counts):")
    print(b.loc[b.other_developer, "developer"].value_counts().head(25).to_string())
    s = summarise_schemes(b[~b.other_developer])
    pd.set_option("display.width", 250)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
