"""Write the datasets behind the dashboard (one JSON file of row objects each) to output/dashboard/.

Every figure on the dashboard, including the numbers quoted in its text, comes from these files,
which are computed from the pipeline outputs rather than typed in.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "output"
DASH = OUT / "dashboard"
ASOF = pd.Timestamp("2026-10-09")
YTD_END = pd.Timestamp("2026-09-30")
FIRE = (r"(second|secondary|two|additional|2nd) (stair|staircase|core|stair core)|staircases|"
        r"fire (safety|strategy|regulations|requirement)|bs ?9991|building safety|"
        r"health and safety executive|gateway (?:one|two|1|2)")
NEW = ["full", "outline", "full_or_other"]


def rows(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", date_format="iso"))


def write(name: str, data: list[dict]) -> None:
    (DASH / f"{name}.json").write_text(json.dumps(data, indent=1))


def main() -> None:
    DASH.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA / "applications.parquet")
    b = pd.read_parquet(DATA / "berkeley_applications.parquet")
    b = b[~b.other_developer & ~b.is_consultation & ~b.is_dup]
    lo = json.loads((OUT / "london_overview.json").read_text())
    ba = json.loads((OUT / "berkeley_analysis.json").read_text())
    lon = df[~df.is_consultation & ~df.is_dup]
    kf: list[dict] = []

    def fig(id_, value, unit, label):
        v = None if value is None else float(value)
        if v is not None:
            v = round(v, 1) if unit == "pct" else int(round(v))
        kf.append({"id": id_, "value": v, "unit": unit, "label": label})

    # ---- London pipeline -------------------------------------------------------------
    mr = pd.DataFrame(lo["major_resi"]).rename_axis("year").reset_index()
    mr["year"] = mr.year.astype(float).astype(int)
    mr["partial"] = mr.year == 2026
    write("london_pipeline", rows(mr.rename(columns={"units": "homes"})[
        ["year", "apps", "apps_50plus", "apps_150plus", "homes", "partial"]]))
    base = mr[mr.year.isin([2019, 2020, 2021])]
    fig("lon_apps_2019_21", base.apps.mean(), "count", "Major housing applications a year, 2019-21 average")
    fig("lon_apps_2025", mr.loc[mr.year == 2025, "apps"].iat[0], "count", "Major housing applications, 2025")
    fig("lon_apps_change", (mr.loc[mr.year == 2025, "apps"].iat[0] / base.apps.mean() - 1) * 100, "pct",
        "Change in major housing applications, 2025 vs 2019-21 average")
    fig("lon_homes_2019_21", base.homes.mean() if "homes" in base else base.units.mean(), "count",
        "Homes in major housing applications a year, 2019-21 average")
    fig("lon_homes_2025", mr.loc[mr.year == 2025, "units"].iat[0], "count", "Homes in major housing applications, 2025")

    dec = pd.DataFrame(lo["major_decisions"]).rename_axis("year").reset_index()
    dec["year"] = dec.year.astype(float).astype(int)
    dec["approval_rate"] = (dec.approval_rate * 100).round(1)
    dec["partial"] = dec.year == 2026
    write("london_decisions", rows(dec[["year", "decided", "approved", "approval_rate", "median_days",
                                        "median_days_150plus", "partial"]]))
    for y in (2019, 2025):
        fig(f"lon_days_{y}", dec.loc[dec.year == y, "median_days"].iat[0], "days", f"Median days to decide a major housing application, {y}")
        fig(f"lon_approval_{y}", dec.loc[dec.year == y, "approval_rate"].iat[0], "pct", f"Share of major housing decisions approved, {y}")
    fig("lon_approval_2023", dec.loc[dec.year == 2023, "approval_rate"].iat[0], "pct", "Share of major housing decisions approved, 2023")

    ten = pd.DataFrame(lo["tenure_mix"]).rename_axis("year").reset_index()
    ten["year"] = ten.year.astype(float).astype(int)
    ten["other"] = (ten.units - ten.market_sale - ten.market_rent_btr - ten.affordable).clip(lower=0)
    for c in ["market_sale", "market_rent_btr", "affordable", "other"]:
        ten[c + "_pct"] = (ten[c] / ten.units * 100).round(1)
    ten["partial"] = ten.year == 2026
    write("london_tenure", rows(ten.rename(columns={"units": "homes"})[
        ["year", "homes", "market_sale_pct", "market_rent_btr_pct", "affordable_pct", "other_pct", "partial"]]))
    for y in (2019, 2025):
        r = ten[ten.year == y].iloc[0]
        fig(f"lon_btr_{y}", r.market_rent_btr_pct, "pct", f"Build-to-rent share of homes in major applications, {y}")
        fig(f"lon_sale_{y}", r.market_sale_pct, "pct", f"Market-sale share of homes in major applications, {y}")

    # ---- Berkeley activity (Jan-Jun each year, like for like and clear of reporting lags) ----
    act = pd.DataFrame(ba["berkeley_by_year_group_h1"]).rename_axis("year").reset_index().melt(
        id_vars="year", var_name="type", value_name="applications")
    act["year"] = act.year.astype(float).astype(int)
    write("berkeley_activity", rows(act.sort_values(["year", "type"])))
    tot = act.groupby("year").applications.sum()
    cond = act[act.type == "Condition discharges"].set_index("year").applications
    peak = int(tot.loc[2019:2025].idxmax())
    fig("bkg_h1_2026", tot[2026], "count", "Applications on Berkeley sites, Jan-Jun 2026")
    fig("bkg_h1_2025", tot[2025], "count", "Applications on Berkeley sites, Jan-Jun 2025")
    fig("bkg_h1_peak", tot[peak], "count", f"Applications on Berkeley sites, Jan-Jun {peak} (peak)")
    fig("bkg_h1_peak_year", peak, "year", "Year of peak Jan-Jun Berkeley applications")
    fig("bkg_h1_change_peak", (tot[2026] / tot[peak] - 1) * 100, "pct", f"Change in Berkeley applications, Jan-Jun 2026 vs Jan-Jun {peak}")
    fig("bkg_cond_2026", cond[2026], "count", "Condition discharges on Berkeley sites, Jan-Jun 2026")
    fig("bkg_cond_peak", cond.loc[2019:2025].max(), "count", "Most condition discharges on Berkeley sites in any Jan-Jun, 2019-25")
    fig("bkg_cond_change", (cond[2026] / cond.loc[2019:2025].max() - 1) * 100, "pct",
        "Change in condition discharges, Jan-Jun 2026 vs the 2019-25 high")
    fig("bkg_cond_low", cond.loc[2019:2025].min(), "count", "Fewest condition discharges on Berkeley sites in any Jan-Jun, 2019-25")
    sep = ba["berkeley_by_year_group_ytd"]
    sep_tot = pd.DataFrame(sep).sum(axis=1)
    sep_tot.index = sep_tot.index.astype(float).astype(int)
    fig("bkg_ytd_2026", sep_tot[2026], "count", "Applications on Berkeley sites, Jan-Sep 2026 (affected by reporting lag)")
    fig("bkg_ytd_2025", sep_tot[2025], "count", "Applications on Berkeley sites, Jan-Sep 2025")
    mon = pd.Series(lo["monthly_all"])
    fig("lon_aug_sep_2026", mon[["2026-08", "2026-09"]].mean(), "count", "London applications a month, Aug-Sep 2026 (still being loaded)")
    fig("lon_monthly_2025", mon[[k for k in mon.index if k.startswith("2025")]].mean(), "count", "London applications a month, 2025 average")

    # ---- Berkeley originations vs London -----------------------------------------------------
    org = pd.DataFrame(ba["originations"]).rename_axis("year").reset_index()
    org["year"] = org.year.astype(float).astype(int)
    org["berkeley_share_pct"] = (org.berkeley_share * 100).round(1)
    org["partial"] = org.year == 2026
    write("berkeley_originations", rows(org[["year", "berkeley_homes", "berkeley_apps", "london_homes",
                                             "berkeley_share_pct", "partial"]]))
    early, late = org[org.year.between(2019, 2022)], org[org.year.between(2023, 2025)]
    fig("bkg_homes_2019_22", early.berkeley_homes.sum(), "count", "Homes in new Berkeley schemes applied for, 2019-22")
    fig("bkg_homes_2023_25", late.berkeley_homes.sum(), "count", "Homes in new Berkeley schemes applied for, 2023-25")
    fig("bkg_share_2019_22", early.berkeley_homes.sum() / early.london_homes.sum() * 100, "pct",
        "Berkeley share of homes in new London major applications, 2019-22")
    fig("bkg_share_2023_25", late.berkeley_homes.sum() / late.london_homes.sum() * 100, "pct",
        "Berkeley share of homes in new London major applications, 2023-25")

    # ---- Big Berkeley applications ----------------------------------------------------------------
    big = pd.DataFrame(ba["big_apps"])
    status = {"approved": "Approved", "pending": "Awaiting decision", "refused": "Not approved",
              "withdrawn": "Withdrawn", "split": "Split decision"}
    big["status"] = big.outcome.map(status).fillna("Other")
    big.loc[big.decision.fillna("").str.contains("Called in", case=False), "status"] = "Called in"
    big.loc[big.decision.fillna("").str.contains("Declined", case=False), "status"] = "Not approved"
    big["app_id"] = big.lpa_name + " " + big.lpa_app_no
    big["validated"] = big.valid_date.str[:10]
    big["decided"] = big.decision_date.str[:10]
    big = big.rename(columns={"scheme_brand": "brand", "lpa_name": "borough", "lpa_app_no": "reference",
                              "units_best": "homes"})
    write("berkeley_major_apps", rows(big[["app_id", "scheme", "brand", "borough", "reference", "kind", "validated",
                                           "decided", "days_to_decision", "days_pending", "homes", "status"]]))
    pend = big[big.status == "Awaiting decision"]
    fig("bkg_pending_apps", len(pend), "count", "Large Berkeley applications awaiting a decision")
    fig("bkg_pending_homes", pend.homes.sum(), "count", "Homes in large Berkeley applications awaiting a decision")
    beck = big[big.reference == "24/00989/OUT"].iloc[0]
    fig("beckton_homes", beck.homes, "count", "Homes in the Beckton Riverside outline application")
    fig("beckton_days", beck.days_pending, "days", "Days the Beckton Riverside outline has been awaiting a decision")

    # ---- Speed: >=100-home decisions -----------------------------------------------------------
    sb = pd.DataFrame(ba["speed_berkeley"]).rename_axis("year").reset_index()
    sl = pd.DataFrame(ba["speed_london_ex_berkeley"]).rename_axis("year").reset_index()
    sp = sb.merge(sl, on="year", suffixes=("_berkeley", "_london"))
    sp["year"] = sp.year.astype(float).astype(int)
    write("decision_speed", rows(sp.rename(columns={"median_berkeley": "berkeley_median_days",
                                                   "count_berkeley": "berkeley_decisions",
                                                   "median_london": "london_median_days",
                                                   "count_london": "london_decisions"})))

    # ---- Fire-safety / second-staircase redesigns ------------------------------------------------------
    fs = pd.DataFrame({"berkeley": ba["safety_amend_berkeley"], "berkeley_all": ba["amend_berkeley_total"],
                       "london": ba["safety_amend_london"], "london_all": ba["amend_london_total"]})
    fs.index = fs.index.astype(float).astype(int)
    fs = fs.rename_axis("year").reset_index()
    fs["berkeley_share_pct"] = (fs.berkeley / fs.london * 100).round(1)
    fs["partial"] = fs.year == 2026
    write("fire_safety", rows(fs))
    late_fs = fs[fs.year >= 2023]
    fig("fire_bkg_2023_26", late_fs.berkeley.sum(), "count", "Berkeley second-staircase / fire-safety amendments, 2023 to Sep 2026")
    fig("fire_bkg_2019_22", fs[fs.year < 2023].berkeley.sum(), "count", "Berkeley second-staircase / fire-safety amendments, 2019-22")
    fig("fire_bkg_share", late_fs.berkeley.sum() / late_fs.london.sum() * 100, "pct",
        "Berkeley share of London's second-staircase / fire-safety amendments, 2023 to Sep 2026")
    fig("amend_bkg_share", late_fs.berkeley_all.sum() / late_fs.london_all.sum() * 100, "pct",
        "Berkeley share of all London amendments (S73 and NMA), 2023 to Sep 2026")
    rows_fs = pd.DataFrame(ba["safety_amend_berkeley_rows"])
    fig("fire_bkg_schemes", rows_fs[rows_fs.valid_date >= "2023-01-01"].scheme.nunique(), "count",
        "Berkeley schemes with a second-staircase / fire-safety amendment since 2023")

    # Height: share of >=100-home schemes with a building of 7+ storeys (where heights are recorded)
    big_lon = lon[lon.major_resi & (lon.units_best >= 100) & lon.valid_date.between("2021-01-01", YTD_END)
                  & lon.max_storeys.notna()]
    big_bkg = b[b.category.isin(NEW + ["reserved_matters"]) & (b.units_best >= 100)
                & b.valid_date.between("2019-01-01", YTD_END) & b.max_storeys.notna() & ~b.is_scoping]
    fig("tall_london", (big_lon.max_storeys >= 7).mean() * 100, "pct",
        "Share of London 100+ home applications with a building of 7+ storeys (2021-26)")
    fig("tall_berkeley", (big_bkg.max_storeys >= 7).mean() * 100, "pct",
        "Share of Berkeley 100+ home applications with a building of 7+ storeys (2019-26)")

    # ---- Brand mix, schemes, peers -------------------------------------------------------------------
    bm = pd.DataFrame(ba["brand_mix"]).rename_axis("year").reset_index().melt(id_vars="year", var_name="brand",
                                                                             value_name="share_pct")
    bm["year"] = bm.year.astype(float).astype(int)
    write("brand_mix", rows(bm.sort_values(["year", "brand"])))
    sw = bm[(bm.brand == "St William") & bm.year.between(2019, 2026)]
    fig("st_william_share", sw.share_pct.mean(), "pct", "St William (former gasworks) share of Berkeley applications, 2019-26 average")

    sc = pd.DataFrame(ba["schemes"])
    sc = sc[~sc.scheme.str.startswith("Other")]
    sc["first"] = sc["first"].str[:10]
    sc["last"] = sc["last"].str[:10]
    write("berkeley_schemes", rows(sc.rename(columns={"apps": "applications", "first": "first_application",
                                                      "last": "latest_application",
                                                      "max_homes": "largest_permission_homes"})))
    fig("bkg_schemes", len(sc), "count", "Berkeley schemes with applications since 2019")

    pe = pd.DataFrame(ba["peers_originations"]).fillna(0)
    pe.columns = [f"y{int(float(c))}" for c in pe.columns]
    pe = pe.rename_axis("builder").reset_index()
    for y in range(2021, 2027):
        if f"y{y}" not in pe:
            pe[f"y{y}"] = 0
    pe["total"] = pe[[f"y{y}" for y in range(2021, 2027)]].sum(axis=1)
    for name in ["Crest Nicholson", "Persimmon"]:
        if name not in set(pe.builder):
            pe.loc[len(pe)] = [name] + [0] * (len(pe.columns) - 1)
    write("peers", rows(pe.sort_values("total", ascending=False)))
    yrs = [f"y{y}" for y in (2023, 2024, 2025)]
    bk = pe.loc[pe.builder == "Berkeley", yrs].sum(axis=1).iat[0]
    fig("peers_bkg_share", bk / pe[yrs].sum().sum() * 100, "pct",
        "Berkeley share of homes in named listed-builder major applications, 2023-25")

    # ---- Data quality ------------------------------------------------------------------------------
    fill = pd.read_csv(OUT / "dq_fill_rates.csv")
    fill = fill.rename(columns={"valid_year": "year"})
    maj = lon[lon.major_resi & lon.valid_date.between("2019-01-01", YTD_END)]
    mf = maj.groupby("valid_year").agg(developer=("developer", lambda s: s.notna().mean()),
                                       homes_field=("units_source", lambda s: (s == "field").mean()))
    fill = fill.merge(mf, left_on="year", right_index=True, how="left")
    fill["year"] = fill.year.astype(int)
    dq = pd.DataFrame({"year": fill.year,
                       "type_field_pct": (fill.application_type_full * 100).round(1),
                       "developer_on_majors_pct": (fill.developer * 100).round(1),
                       "homes_field_on_majors_pct": (fill.homes_field * 100).round(1)})
    write("data_quality", rows(dq))
    r25 = dq[dq.year == 2025].iloc[0]
    fig("dq_dev_2025", r25.developer_on_majors_pct, "pct", "Major housing applications naming the developer, 2025")
    fig("dq_homes_blank_2019", 100 - dq[dq.year == 2019].homes_field_on_majors_pct.iat[0], "pct",
        "Major housing applications with the homes field blank, 2019")
    fig("dq_homes_blank_2025", 100 - r25.homes_field_on_majors_pct, "pct", "Major housing applications with the homes field blank, 2025")
    fig("dq_dev_2019", dq[dq.year == 2019].developer_on_majors_pct.iat[0], "pct", "Major housing applications naming the developer, 2019")
    fig("dq_type_2019", dq[dq.year == 2019].type_field_pct.iat[0], "pct", "Applications with a detailed type, 2019")
    fig("dq_type_2025", r25.type_field_pct, "pct", "Applications with a detailed type, 2025")
    fig("dq_records", len(df), "count", "Records in the Datahub extract")
    fig("dq_lpas", df.lpa_name.nunique(), "count", "Planning authorities in the extract")
    raw = pd.concat([pd.read_parquet(p, columns=["lat", "lon", "decision", "borough"])
                     for p in sorted((DATA / "raw").glob("part_*.parquet"))])
    badc = raw.lat.notna() & ~(raw.lat.between(51.2, 51.8) & raw.lon.between(-0.6, 0.4))
    fig("dq_bad_coords", badc.sum(), "count", "Records with coordinates outside London")
    fig("dq_bad_coords_pct", badc.mean() * 100, "pct", "Share of records with coordinates outside London")
    fig("dq_decision_codes", raw.decision.replace({"nan": None}).dropna().nunique(), "count", "Distinct decision values")
    fig("dq_borough_names", raw.borough.replace({"nan": None}).dropna().nunique(), "count", "Distinct spellings in the borough field")
    fig("dq_consultations", df.is_consultation.sum(), "count", "Records that are another borough's consultation response")
    fig("dq_preapp", df.is_preapp.sum(), "count", "Pre-application advice records")

    fresh = pd.read_csv(OUT / "dq_freshness.csv", parse_dates=["latest_valid", "latest_sync"])
    fresh["days_behind"] = (ASOF - fresh.latest_valid).dt.days
    fresh = fresh.rename(columns={"lpa_name": "authority", "latest_valid": "latest_application", "n": "records"})
    fresh["latest_application"] = fresh.latest_application.dt.strftime("%Y-%m-%d")
    write("feed_freshness", rows(fresh[["authority", "latest_application", "days_behind", "records"]]
                                 .sort_values("days_behind", ascending=False)))

    # ---- Attribution and method ----------------------------------------------------------------------
    tiers = ba["tiers"]
    fig("tier_named", tiers.get("named", 0), "count", "Berkeley applications naming a Berkeley company")
    fig("tier_curated", tiers.get("curated", 0), "count", "Berkeley head permissions added from verified scheme list")
    fig("tier_linked", tiers.get("linked", 0), "count", "Applications linked to Berkeley permissions by reference number")
    fig("bkg_apps_2019", ba["n_apps_2019"], "count", "Applications on Berkeley sites since 2019")
    ents = pd.read_csv(ROOT / "config" / "berkeley_entities.csv")
    fig("entities", len(ents), "count", "Berkeley Group companies checked")
    prev = sum(len(re.findall(r"\(\d", str(n))) for n in ents.notes.fillna(""))
    fig("entity_prev_names", prev, "count", "Former company names checked")
    edges = pd.read_parquet(DATA / "ref_edges.parquet")
    fig("ref_links", len(edges), "count", "Reference-number links between applications")

    write("key_figures", kf)
    print(f"wrote {len(list(DASH.glob('*.json')))} datasets; {len(kf)} key figures")
    for k in kf:
        print(f"  {k['id']:22} {k['value']!s:>12} {k['unit']:5} {k['label']}")


if __name__ == "__main__":
    main()
