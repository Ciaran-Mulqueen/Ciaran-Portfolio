"""Static PNG charts for the project README (the interactive version is the dashboard)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DASH = ROOT / "output" / "dashboard"
IMG = ROOT / "images"

SURFACE, INK, INK2, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MUTED = "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 1,
    "xtick.bottom": False, "ytick.left": False,
})
thousands = FuncFormatter(lambda v, _: f"{v:,.0f}")


def load(name: str) -> pd.DataFrame:
    return pd.DataFrame(json.loads((DASH / f"{name}.json").read_text()))


def finish(fig, ax, title: str, note: str, path: str) -> None:
    ax.set_title(title, loc="left", pad=12)
    fig.text(0.01, 0.01, note, fontsize=8, color=INK2, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(IMG / path, dpi=160)
    plt.close(fig)


def london_pipeline() -> None:
    d = load("london_pipeline")
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [MUTED if p else SERIES[0] for p in d.partial]
    ax.bar(d.year.astype(str), d.apps, color=colors, width=0.55, zorder=2)
    for x, v in zip(d.year.astype(str), d.apps):
        ax.text(x, v + 8, f"{v:,}", ha="center", fontsize=8, color=INK2)
    ax.yaxis.set_major_formatter(thousands)
    finish(fig, ax, "London: major housing applications a year (10+ homes)",
           "New full, outline and hybrid applications, de-duplicated. 2026 = Jan-Sep and still loading. "
           "Source: GLA Planning London Datahub.", "london_major_applications.png")


def berkeley_activity() -> None:
    d = load("berkeley_activity")
    order = ["Condition discharges", "Amendments (S73 and NMA)", "New schemes and other full", "Reserved matters",
             "Other (adverts, listed bldg, trees, lawful dev)"]
    w = d.pivot_table(index="year", columns="type", values="applications", aggfunc="sum").reindex(columns=order).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bottom = pd.Series(0, index=w.index)
    for i, col in enumerate(order):
        ax.bar(w.index.astype(str), w[col], bottom=bottom, color=SERIES[i], width=0.55, zorder=2,
               edgecolor=SURFACE, linewidth=1, label=col.replace(" (adverts, listed bldg, trees, lawful dev)", ""))
        bottom += w[col]
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    finish(fig, ax, "Berkeley sites: applications by type, January to June each year",
           "Applications on Berkeley-linked sites (alias-aware attribution, see README). Source: GLA Planning London Datahub.",
           "berkeley_activity.png")


def berkeley_share() -> None:
    d = load("berkeley_originations")
    d = d[~d.partial]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.bar(d.year.astype(str), d.berkeley_homes, color=SERIES[0], width=0.55, zorder=2)
    a1.yaxis.set_major_formatter(thousands)
    a1.set_title("Homes in new Berkeley schemes", loc="left", pad=10)
    a2.plot(d.year.astype(str), d.berkeley_share_pct, color=SERIES[0], linewidth=2, marker="o",
            markersize=6, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    top = max(d.berkeley_share_pct) * 1.3
    vals = list(d.berkeley_share_pct)
    for i, (x, v) in enumerate(zip(d.year.astype(str), vals)):
        # a label sitting under a steep neighbouring segment goes below its point instead
        steep = any(0 <= j < len(vals) and vals[j] > v + 5 for j in (i - 1, i + 1))
        a2.annotate(f"{v:.1f}%", (x, v), xytext=(0, -15 if steep and v > 2 else 9), textcoords="offset points",
                    ha="center", fontsize=8, color=INK2)
    a2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    a2.set_ylim(0, top)
    a2.set_title("Berkeley share of London's new-scheme homes", loc="left", pad=10)
    fig.text(0.01, 0.01, "New full, outline and hybrid applications of 10+ homes, by year validated. "
             "Source: GLA Planning London Datahub.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(IMG / "berkeley_share.png", dpi=160)
    plt.close(fig)


def fire_safety() -> None:
    d = load("fire_safety")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    labels = [f"{y}{'*' if p else ''}" for y, p in zip(d.year, d.partial)]
    a1.bar(labels, d.berkeley, color=SERIES[0], width=0.55, zorder=2)
    a1.set_title("Berkeley sites", loc="left", pad=10)
    a2.bar(labels, d.london, color=MUTED, width=0.55, zorder=2)
    a2.set_title("All London", loc="left", pad=10)
    fig.suptitle("Amendments citing second staircases or fire safety", x=0.01, ha="left", fontweight="bold")
    fig.text(0.01, 0.01, "S73 and non-material amendments citing second staircases/cores, fire strategy, BS 9991 or HSE "
             "guidance. *2026 = Jan-Sep. Source: GLA Planning London Datahub.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(IMG / "fire_safety_amendments.png", dpi=160)
    plt.close(fig)


def main() -> None:
    IMG.mkdir(exist_ok=True)
    london_pipeline()
    berkeley_activity()
    berkeley_share()
    fire_safety()
    print("charts:", sorted(p.name for p in IMG.glob("*.png")))


if __name__ == "__main__":
    main()
