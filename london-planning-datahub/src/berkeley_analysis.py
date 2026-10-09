"""Berkeley Group: planning activity trends versus London, plus listed-peer comparison."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from developers import listed_builder  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
OUT = Path(__file__).resolve().parents[1] / "output"
ASOF = pd.Timestamp("2026-10-09")
YTD_END = pd.Timestamp("2026-09-30")
YEARS = list(range(2019, 2027))
warnings.filterwarnings("ignore")

GROUP = {
    "full": "New schemes and other full", "outline": "New schemes and other full",
    "full_or_other": "New schemes and other full", "reserved_matters": "Reserved matters",
    "s73": "Amendments (S73 and NMA)", "nma": "Amendments (S73 and NMA)",
    "conditions": "Condition discharges",
}
NOT_A_SCHEME = r"scoping|screening opinion|deed of variation|section 106|s106|legal agreement"


def group(cat: str) -> str:
    return GROUP.get(cat, "Other (adverts, listed bldg, trees, lawful dev)")


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(DATA / "applications.parquet")
    b = pd.read_parquet(DATA / "berkeley_applications.parquet")
    b = b[~b.other_developer & ~b.is_consultation & ~b.is_dup]
    return df, b


def by_year_group(x: pd.DataFrame, ytd: bool = False, last_month: int | None = None) -> pd.DataFrame:
    x = x[x.valid_date.between("2019-01-01", YTD_END)]
    if ytd:
        x = x[x.valid_date.dt.month <= (last_month or YTD_END.month)]
    return x.assign(g=x.category.map(group)).pivot_table(
        index="valid_year", columns="g", values="id", aggfunc="count").fillna(0).astype(int)


def originations(x: pd.DataFrame, min_homes: int) -> pd.DataFrame:
    """New residential schemes (full / outline / hybrid), excluding scoping and S106 records."""
    return x[x.category.isin(["full", "outline", "full_or_other"]) & (x.units_best >= min_homes) & ~x.is_scoping]


def main() -> dict:
    df, b = load()
    lon = df[~df.is_consultation & ~df.is_dup]
    res: dict = {"asof": str(ASOF.date()), "ytd_end": str(YTD_END.date())}

    # 1. Activity mix by year: Berkeley vs London (London excludes householder and prior approvals)
    lon_dev = lon[~lon.category.isin(["householder", "prior_approval"])]
    res["berkeley_by_year_group"] = by_year_group(b).to_dict()
    res["london_by_year_group"] = by_year_group(lon_dev).to_dict()
    res["berkeley_by_year_group_ytd"] = by_year_group(b, ytd=True).to_dict()
    res["london_by_year_group_ytd"] = by_year_group(lon_dev, ytd=True).to_dict()
    # Jan-Jun: the latest window not distorted by validation/reporting lags (Aug-Sep 2026 London
    # volumes run ~25% below normal) or by borough feeds that stopped updating in mid-2026
    res["berkeley_by_year_group_h1"] = by_year_group(b, ytd=True, last_month=6).to_dict()
    res["london_by_year_group_h1"] = by_year_group(lon_dev, ytd=True, last_month=6).to_dict()

    # 2. Originations: homes in new schemes (>=10 homes) applied for, Berkeley vs London
    b_or = originations(b, 10)
    l_or = originations(lon, 10)
    yr = lambda s: s[s.valid_date.between("2019-01-01", YTD_END)].groupby("valid_year")  # noqa: E731
    homes = pd.DataFrame({
        "berkeley_homes": yr(b_or).units_best.sum(),
        "berkeley_apps": yr(b_or).size(),
        "london_homes": yr(l_or).units_best.sum(),
        "london_apps": yr(l_or).size(),
    }).fillna(0)
    homes["berkeley_share"] = homes.berkeley_homes / homes.london_homes
    res["originations"] = homes.round(4).to_dict()

    # 3. Big Berkeley applications since 2019 (>=100 homes): new schemes, S73 re-plans, reserved matters
    big = b[(b.valid_date >= "2019-01-01") & b.category.isin(["full", "outline", "full_or_other", "s73", "reserved_matters"])
            & (b.units_best >= 100) & ~b.description.str.contains(NOT_A_SCHEME, case=False, na=False)].copy()
    big = big[~((big.category == "s73") & big.description.str.contains(r"non[- ]material", case=False, na=False))]
    big["kind"] = big.category.map({"full": "New scheme", "outline": "New scheme", "full_or_other": "New scheme",
                                    "s73": "S73 re-plan", "reserved_matters": "Reserved matters"})
    big["days_pending"] = np.where(big.outcome == "pending", (ASOF - big.valid_date).dt.days, np.nan)
    cols = ["scheme", "scheme_brand", "lpa_name", "lpa_app_no", "kind", "valid_date", "decision_date",
            "days_to_decision", "days_pending", "units_best", "outcome", "decision", "status", "tier"]
    big = big.sort_values("valid_date")[cols]
    res["big_apps"] = json.loads(big.to_json(orient="records", date_format="iso"))

    # 4. Homes by kind and validation year (Berkeley)
    res["big_homes_by_kind"] = big.pivot_table(index=big.valid_date.dt.year, columns="kind", values="units_best",
                                               aggfunc="sum").fillna(0).astype(int).to_dict()
    res["big_count_by_kind"] = big.pivot_table(index=big.valid_date.dt.year, columns="kind", values="units_best",
                                               aggfunc="count").fillna(0).astype(int).to_dict()

    # 5. Speed: median days to decision for >=100-home applications, by decision year
    def speed(x):
        d = x[x.decision_date.between("2019-01-01", YTD_END) & x.outcome.isin(["approved", "refused", "split"])
              & x.category.isin(["full", "outline", "full_or_other", "s73", "reserved_matters"])
              & (x.units_best >= 100) & ~x.description.str.contains(NOT_A_SCHEME, case=False, na=False)]
        d = d[~((d.category == "s73") & d.description.str.contains(r"non[- ]material", case=False, na=False))]
        return d.groupby("decision_year").days_to_decision.agg(["median", "count"])
    res["speed_berkeley"] = speed(b).to_dict()
    res["speed_london"] = speed(lon).to_dict()
    res["speed_london_ex_berkeley"] = speed(lon[~lon.id.isin(b.id)]).to_dict()

    # 6. Pending pipeline today
    pend = big[big.outcome == "pending"].copy()
    res["pending_homes"] = int(pend.units_best.sum())
    res["pending_count"] = int(len(pend))

    # 7. JV split (St William / St Edward) of activity
    yb = b[b.valid_date.between("2019-01-01", YTD_END)]
    res["brand_mix"] = (pd.crosstab(yb.valid_year, yb.scheme_brand, normalize="index") * 100).round(1).to_dict()
    res["jv_share_apps"] = yb.groupby("valid_year").jv.mean().round(3).to_dict()

    # 8. Condition discharges per active scheme (construction-activity proxy)
    c = b[(b.category == "conditions") & b.valid_date.between("2019-01-01", YTD_END)]
    res["conditions_by_scheme_year"] = c.pivot_table(index="scheme", columns="valid_year", values="id",
                                                     aggfunc="count").fillna(0).astype(int).to_dict()

    # 9. Building-safety redesign signal in amendment descriptions
    # Second-staircase / fire-safety redesigns (incidental mentions of stairs are excluded)
    stair = (r"(second|secondary|two|additional|2nd) (stair|staircase|core|stair core)|staircases|"
             r"fire (safety|strategy|regulations|requirement)|bs ?9991|building safety|"
             r"health and safety executive|gateway (?:one|two|1|2)")
    am_b = b[b.category.isin(["s73", "nma"]) & b.valid_date.between("2019-01-01", YTD_END)]
    am_l = lon[lon.category.isin(["s73", "nma"]) & lon.valid_date.between("2019-01-01", YTD_END)]
    res["safety_amend_berkeley_rows"] = json.loads(am_b[am_b.description.str.contains(stair, case=False, na=False)][
        ["scheme", "lpa_app_no", "valid_date", "category"]].sort_values("valid_date").to_json(orient="records", date_format="iso"))
    res["amend_berkeley_total"] = am_b.groupby("valid_year").size().to_dict()
    res["safety_amend_berkeley"] = am_b.groupby("valid_year").apply(
        lambda x: int(x.description.str.contains(stair, case=False, na=False).sum())).to_dict()
    res["safety_amend_london"] = am_l.groupby("valid_year").apply(
        lambda x: int(x.description.str.contains(stair, case=False, na=False).sum())).to_dict()
    res["amend_london_total"] = am_l.groupby("valid_year").size().to_dict()

    # 10. Tenure in Berkeley's structured records (where filled)
    t = big[big.kind != "S73 re-plan"].merge(b[["lpa_app_no", "lpa_name", "u_market_for_sale", "u_market_for_rent",
                                                "afford_proposed", "units_proposed", "max_storeys"]],
                                             on=["lpa_app_no", "lpa_name"], how="left")
    res["tenure_berkeley"] = {
        "homes_with_tenure": float(t.loc[t.units_proposed > 0, "units_proposed"].sum()),
        "market_sale": float(t.u_market_for_sale.sum()), "market_rent": float(t.u_market_for_rent.sum()),
        "affordable": float(t.afford_proposed.sum()),
        "median_max_storeys": float(t.max_storeys.median()) if t.max_storeys.notna().any() else None,
        "share_7plus_storeys": float((t.max_storeys >= 7).mean()) if t.max_storeys.notna().any() else None,
    }
    res["btr_rows"] = json.loads(t[t.u_market_for_rent > 0][["scheme", "lpa_app_no", "units_best",
                                                            "u_market_for_rent"]].to_json(orient="records"))

    # 11. Listed peers, like-for-like on the developer field only (2021+ when it is populated)
    p = lon[lon.valid_date.between("2021-01-01", YTD_END)].copy()
    p["builder"] = p.developer.map(listed_builder)
    pm = originations(p[p.builder.notna()], 10)
    res["peers_originations"] = pm.pivot_table(index="builder", columns="valid_year", values="units_best",
                                               aggfunc="sum").fillna(0).astype(int).to_dict()
    res["peers_apps_all"] = p[p.builder.notna()].pivot_table(index="builder", columns="valid_year", values="id",
                                                             aggfunc="count").fillna(0).astype(int).to_dict()
    res["dev_field_fill_major"] = originations(lon[lon.valid_date.between("2021-01-01", YTD_END)], 10).groupby(
        "valid_year").developer.apply(lambda s: s.notna().mean()).round(3).to_dict()

    # 12. Scheme table
    sc = b[b.valid_date.between("2019-01-01", YTD_END)]
    head = b[b.category.isin(["full", "outline", "full_or_other", "s73"])].groupby("scheme").units_best.max()
    tbl = sc.groupby("scheme").agg(brand=("scheme_brand", "first"), borough=("lpa_name", lambda s: s.mode().iat[0]),
                                   apps=("id", "size"), first=("valid_date", "min"), last=("valid_date", "max"))
    tbl["max_homes"] = head.reindex(tbl.index)
    tbl = tbl.sort_values("apps", ascending=False)
    res["schemes"] = json.loads(tbl.reset_index().to_json(orient="records", date_format="iso"))
    res["n_apps_2019"] = int(len(sc))
    res["tiers"] = sc.tier.value_counts().to_dict()

    OUT.mkdir(exist_ok=True)
    (OUT / "berkeley_analysis.json").write_text(json.dumps(res, indent=1, default=str))
    return res


if __name__ == "__main__":
    r = main()
    for k in ["berkeley_by_year_group_ytd", "originations", "brand_mix",
              "jv_share_apps", "safety_amend_berkeley", "amend_berkeley_total", "safety_amend_london", "amend_london_total",
              "peers_originations", "peers_apps_all", "dev_field_fill_major"]:
        print("==", k)
        v = r[k]
        print(pd.DataFrame(v).to_string() if isinstance(next(iter(v.values()), None), dict) else v)
    for k in ["pending_homes", "pending_count", "tenure_berkeley", "btr_rows", "n_apps_2019", "tiers"]:
        print("==", k, r[k])
