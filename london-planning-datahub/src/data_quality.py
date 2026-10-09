"""Data-quality profile of the PLD extract: coverage, freshness, fill rates, messiness."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
OUT = Path(__file__).resolve().parents[1] / "output"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = pd.read_parquet(DATA / "applications.parquet")
    lines = []
    p = lines.append
    p(f"records: {len(df):,}; unique ids: {df.id.nunique():,}; LPAs: {df.lpa_name.nunique()}")
    p(f"valid_date missing: {df.valid_date.isna().mean():.1%}")
    yr = df.valid_year
    p("records by valid year:\n" + yr.value_counts().sort_index().loc[2000:2026].to_string())

    # Coverage heatmap: records per LPA per year, relative to the LPA's own 2021-24 average
    cov = df[(yr >= 2015) & (yr <= 2026)].pivot_table(index="lpa_name", columns="valid_year",
                                                        values="id", aggfunc="count").fillna(0)
    base = cov[[2021, 2022, 2023, 2024]].mean(axis=1)
    rel = cov.div(base, axis=0)
    rel.round(2).to_csv(OUT / "dq_coverage_relative.csv")
    p("LPA-years 2019-2025 below 50% of own 2021-24 average:")
    weak = rel.loc[:, 2019:2025].stack()
    p(weak[weak < 0.5].round(2).to_string())

    # Freshness: latest validation date and sync per LPA
    fresh = df.groupby("lpa_name").agg(latest_valid=("valid_date", "max"), latest_sync=("last_synced", "max"),
                                       n=("id", "size")).sort_values("latest_valid")
    fresh.to_csv(OUT / "dq_freshness.csv")
    p("Freshness (oldest first):\n" + fresh.head(12).to_string())

    # Fill rates by year for 2019+
    d = df[(yr >= 2019) & (yr <= 2026)]
    fill = pd.DataFrame({
        "description": d.description.notna(),
        "decision_date": d.decision_date.notna(),
        "application_type_full": d.application_type_full.notna(),
        "centroid": d.lat.notna(),
        "resi_details": d.has_resi_details,
        "developer_name": d.developer.notna(),
    }).groupby(d.valid_year).mean()
    fill.round(3).to_csv(OUT / "dq_fill_rates.csv")
    p("Fill rates by valid year:\n" + fill.round(3).to_string())
    maj = d[d.major_resi]
    p("Developer name fill among major resi (>=10 units new schemes) by year:\n"
      + maj.groupby("valid_year").developer.apply(lambda s: s.notna().mean()).round(3).to_string())

    p(f"distinct raw decision values: {df.decision.nunique()}; distinct raw status values: {df.status.nunique()}")
    p(f"distinct 'borough' spellings: {df.borough.nunique()} vs lpa_name: {df.lpa_name.nunique()}")
    p("category mix 2019+:\n" + d.category.value_counts(normalize=True).round(3).to_string())
    p("outcome mix 2019+:\n" + d.outcome.value_counts(normalize=True).round(3).to_string())
    (OUT / "dq_report.txt").write_text("\n\n".join(lines))
    print("\n\n".join(lines))


if __name__ == "__main__":
    main()
