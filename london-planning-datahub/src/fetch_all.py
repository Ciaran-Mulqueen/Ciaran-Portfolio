"""Download every PLD application record (selected fields) to local parquet.

Nested residential/developer fields are flattened to one row per application.
Run: python src/fetch_all.py  (≈1.3m records, ~15-20 minutes)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from pld_client import count, iter_hits  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data" / "raw"

TOP_FIELDS = [
    "id", "lpa_name", "lpa_app_no", "borough", "ward", "postcode", "site_name", "street_name",
    "description", "application_type", "application_type_full", "development_type",
    "decision", "decision_date", "decision_process", "decision_target_date", "valid_date",
    "status", "appeal_decision", "appeal_decision_date", "appeal_status", "lapsed_date",
    "actual_commencement_date", "actual_completion_date", "centroid", "uprn", "pp_id",
    "bo_system", "last_synced", "last_updated", "url_planning_app",
]
RES = "application_details.residential_details"
TENURES = [
    "market_for_sale", "market_for_rent", "social_rent", "london_affordable_rent",
    "affordable_rent", "london_shared_ownership", "london_living_rent", "intermediate",
    "discount_market_rent", "discount_market_rent_charged_at_london_rents",
    "discount_market_sale", "shared_equity", "starter_homes", "self_build_and_custom_build",
]
DETAIL_FIELDS = [
    "application_details.lead_developer_company_name",
    "application_details.uk_company_registration_no",
    "application_details.scheme_name",
    "application_details.projected_cost_of_works",
    "application_details.site_area",
    "application_details.building_details",
    "application_details.phasing_details",
    "application_details.superseding_details",
    f"{RES}.total_no_proposed_residential_units",
    f"{RES}.total_no_existing_residential_units",
    f"{RES}.total_no_proposed_residential_affordable_units",
    f"{RES}.total_no_existing_residential_affordable_units",
    f"{RES}.total_no_proposed_bedrooms",
    f"{RES}.affordable_percentage",
] + [f"{RES}.total_no_proposed_residential_units_{t}" for t in TENURES]


def flatten(src: dict) -> dict:
    row = {k: src.get(k) for k in TOP_FIELDS}
    c = src.get("centroid") or {}
    row["lat"], row["lon"] = c.get("lat"), c.get("lon")
    row.pop("centroid", None)
    ad = src.get("application_details") or {}
    rd = ad.get("residential_details") or {}
    row["developer"] = ad.get("lead_developer_company_name")
    row["company_reg_no"] = ad.get("uk_company_registration_no")
    row["scheme_name"] = ad.get("scheme_name")
    row["cost_of_works"] = ad.get("projected_cost_of_works")
    row["site_area_ha"] = ad.get("site_area")
    row["has_resi_details"] = bool(rd)
    row["units_proposed"] = rd.get("total_no_proposed_residential_units")
    row["units_existing"] = rd.get("total_no_existing_residential_units")
    row["afford_proposed"] = rd.get("total_no_proposed_residential_affordable_units")
    row["afford_existing"] = rd.get("total_no_existing_residential_affordable_units")
    row["bedrooms_proposed"] = rd.get("total_no_proposed_bedrooms")
    row["affordable_pct"] = rd.get("affordable_percentage")
    for t in TENURES:
        row[f"u_{t}"] = rd.get(f"total_no_proposed_residential_units_{t}")
    storeys = [b.get("no_storeys") for b in (ad.get("building_details") or []) if isinstance(b, dict)]
    storeys = [s for s in storeys if isinstance(s, (int, float))]
    row["max_storeys"] = max(storeys) if storeys else None
    row["n_buildings"] = len(ad.get("building_details") or []) or None
    phases = ad.get("phasing_details") or []
    starts = [p.get("intended_commencement_date") for p in phases if isinstance(p, dict)]
    ends = [p.get("intended_completion_date") for p in phases if isinstance(p, dict)]
    row["intended_start"] = min([s for s in starts if s], default=None, key=_mmyyyy)
    row["intended_completion"] = max([e for e in ends if e], default=None, key=_mmyyyy)
    sup = ad.get("superseding_details") or []
    row["supersedes"] = ";".join(sorted({s.get("lpa_app_no") for s in sup
                                         if isinstance(s, dict) and s.get("lpa_app_no")})) or None
    return row


NUMERIC = ["lat", "lon", "site_area_ha", "units_proposed", "units_existing", "afford_proposed",
           "afford_existing", "bedrooms_proposed", "affordable_pct", "max_storeys",
           "n_buildings"] + [f"u_{t}" for t in TENURES]


def to_frame(rows: list[dict]) -> pd.DataFrame:
    """Coerce numerics and stringify everything else so parquet schemas stay stable."""
    df = pd.DataFrame(rows)
    for col in df.columns:
        if col in NUMERIC:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        elif col != "has_resi_details":
            df[col] = df[col].map(lambda v: None if v is None or (isinstance(v, float) and v != v) else str(v)).astype("object")
    return df


def _mmyyyy(s: str) -> tuple:
    try:
        m, y = s.split("/")[-2:]
        return int(y), int(m)
    except (ValueError, AttributeError):
        return (0, 0)


def main(limit: int | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    total = count()
    print(f"index holds {total:,} records", flush=True)
    buf, part, n, t0 = [], 0, 0, time.time()
    for src in iter_hits(source=TOP_FIELDS + DETAIL_FIELDS, page_size=5000):
        buf.append(flatten(src))
        n += 1
        if limit and n >= limit:
            break
        if len(buf) >= 100_000:
            to_frame(buf).to_parquet(OUT / f"part_{part:03d}.parquet", index=False)
            part += 1
            buf = []
            print(f"{n:,}/{total:,} ({time.time() - t0:.0f}s)", flush=True)
    if buf:
        to_frame(buf).to_parquet(OUT / f"part_{part:03d}.parquet", index=False)
    print(f"done: {n:,} records in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
