"""London-wide context: volumes, major residential pipeline, outcomes, speed, tenure, height."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
OUT = Path(__file__).resolve().parents[1] / "output"
YEARS = list(range(2019, 2027))
AFFORDABLE = ["u_social_rent", "u_london_affordable_rent", "u_affordable_rent", "u_london_shared_ownership",
              "u_london_living_rent", "u_intermediate", "u_discount_market_rent",
              "u_discount_market_rent_charged_at_london_rents", "u_discount_market_sale", "u_shared_equity",
              "u_starter_homes"]


def load() -> pd.DataFrame:
    return pd.read_parquet(DATA / "applications.parquet")


def ytd_cutoff(df: pd.DataFrame) -> pd.Timestamp:
    """Last complete month before the most recent validation dates (reporting lag aware)."""
    latest = df.valid_date[df.valid_date <= pd.Timestamp("2026-10-09")].max()
    return (latest.to_period("M") - 1).to_timestamp(how="end").normalize()


def main() -> dict:
    df = load()
    res: dict = {}
    cut = ytd_cutoff(df)
    res["ytd_cutoff"] = str(cut.date())
    m = df[(df.valid_date >= "2019-01-01") & (df.valid_date <= cut)]
    # Same-period (Jan..cut month) comparisons for every year
    m_ytd = m[m.valid_date.dt.month <= cut.month]

    monthly = m.groupby(m.valid_date.dt.to_period("M")).size()
    res["monthly_all"] = {str(k): int(v) for k, v in monthly.items()}

    by_cat = m.pivot_table(index="valid_year", columns="category", values="id", aggfunc="count").fillna(0)
    res["by_year_category"] = by_cat.astype(int).to_dict()
    by_cat_ytd = m_ytd.pivot_table(index="valid_year", columns="category", values="id", aggfunc="count").fillna(0)
    res["by_year_category_ytd"] = by_cat_ytd.astype(int).to_dict()

    maj = m[m.major_resi]
    grp = maj.groupby("valid_year")
    res["major_resi"] = pd.DataFrame({
        "apps": grp.size(),
        "apps_50plus": grp.apply(lambda g: (g.units_best >= 50).sum()),
        "apps_150plus": grp.apply(lambda g: (g.units_best >= 150).sum()),
        "units": grp.units_best.sum(),
    }).astype(int).to_dict()
    maj_ytd = m_ytd[m_ytd.major_resi].groupby("valid_year")
    res["major_resi_ytd"] = pd.DataFrame({"apps": maj_ytd.size(), "units": maj_ytd.units_best.sum()}
                                         ).astype(int).to_dict()

    # Decisions on major residential schemes, by decision year
    dec = df[df.major_resi & df.decision_date.between("2019-01-01", cut)
             & df.outcome.isin(["approved", "refused", "split"])]
    g = dec.groupby("decision_year")
    res["major_decisions"] = pd.DataFrame({
        "decided": g.size(),
        "approved": g.apply(lambda x: (x.outcome == "approved").sum()),
        "approval_rate": g.apply(lambda x: (x.outcome == "approved").mean()),
        "units_approved": g.apply(lambda x: x.loc[x.outcome == "approved", "units_best"].sum()),
        "median_days": g.days_to_decision.median(),
        "p75_days": g.days_to_decision.quantile(0.75),
        "median_days_150plus": g.apply(lambda x: x.loc[x.units_best >= 150, "days_to_decision"].median()),
    }).round(3).to_dict()

    # Tenure mix of major schemes (applications), by valid year
    t = maj[maj.units_source == "field"].copy()   # tenure split only exists in the structured data
    t["affordable"] = t[AFFORDABLE].sum(axis=1)
    tg = t.groupby("valid_year")
    res["tenure_mix"] = pd.DataFrame({
        "units": tg.units_proposed.sum(),
        "market_sale": tg.u_market_for_sale.sum(),
        "market_rent_btr": tg.u_market_for_rent.sum(),
        "affordable": tg.affordable.sum(),
    }).astype(int).to_dict()

    # Height: share of major schemes with a building of 7+ storeys (~18m+, where the
    # Building Safety Act "higher-risk building" regime starts) - only where heights are filled
    h = maj[maj.max_storeys.notna()]
    hg = h.groupby("valid_year")
    res["height"] = pd.DataFrame({
        "with_heights": hg.size(),
        "share_7plus": hg.apply(lambda x: (x.max_storeys >= 7).mean()),
        "share_units_7plus": hg.apply(lambda x: x.loc[x.max_storeys >= 7, "units_proposed"].sum()
                                      / x.units_proposed.sum()),
        "median_max_storeys": hg.max_storeys.median(),
    }).round(3).to_dict()

    # Amendments: S73 + NMA volumes, and those on major schemes (>=10 units recorded)
    am = m[m.category.isin(["s73", "nma"])]
    res["amendments"] = am.pivot_table(index="valid_year", columns="category", values="id",
                                       aggfunc="count").fillna(0).astype(int).to_dict()
    res["amendments_ytd"] = m_ytd[m_ytd.category.isin(["s73", "nma"])].pivot_table(
        index="valid_year", columns="category", values="id", aggfunc="count").fillna(0).astype(int).to_dict()

    # Borough league table 2019+: units in major schemes applied for
    res["borough_units"] = maj.groupby("lpa_name").units_best.sum().sort_values(ascending=False
                                                                                    ).astype(int).to_dict()
    OUT.mkdir(exist_ok=True)
    (OUT / "london_overview.json").write_text(json.dumps(res, indent=1, default=str))
    return res


if __name__ == "__main__":
    r = main()
    for k, v in r.items():
        if k != "monthly_all":
            print("==", k)
            print(pd.DataFrame(v).to_string() if isinstance(v, dict) and v and isinstance(next(iter(v.values())), dict) else v)
