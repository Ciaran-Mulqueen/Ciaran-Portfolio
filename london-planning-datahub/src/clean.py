"""Clean the raw PLD extract: parse dates, normalise decisions, classify application types.

Output: data/applications.parquet (one row per application record).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
TODAY = pd.Timestamp("2026-10-09")

DATE_COLS = ["valid_date", "decision_date", "decision_target_date", "appeal_decision_date",
             "lapsed_date", "actual_commencement_date", "actual_completion_date"]

# --- application category -------------------------------------------------------------
# Order of precedence: explicit application_type_full where it is specific, otherwise the
# EARLIEST matching phrase in the description (so "Reserved matters pursuant to condition 3"
# is RM, while "Details pursuant to condition 12 of reserved matters approval" is conditions),
# then the reference-number suffix.
ATF_MAP = [
    (r"^non-material amendment", "nma"),
    (r"^s73 minor material amendment|^removal/variation of a condition", "s73"),
    (r"^approval of details reserved by a condition", "conditions"),
    (r"^approval of reserved matters", "reserved_matters"),
    (r"^outline planning permission", "outline"),
    (r"^householder|enlargement of a dwellinghouse", "householder"),
    (r"^prior approval", "prior_approval"),
    (r"^lawful development", "lawful_dev"),
    (r"^tree works|^dangerous tree|^hedgerow", "trees"),
    (r"^consent to display an advertisement|^advert", "advert"),
    (r"^listed building consent|^demolition in a conservation area", "listed"),
    (r"^s106 deed of variation|^legal agreement|^circular 18/84|^limited period", "other"),
    (r"^full planning", "full"),
]
DESC_PATTERNS = [
    ("nma", r"non[\s-]*material\s+(minor\s+)?amendment|\bs\.?\s?96\s?a\b|section\s+96\s?a"),
    ("s73", r"variation\s+of\s+(planning\s+)?condition|removal\s+of\s+(planning\s+)?condition|"
            r"\bvary(ing)?\s+(of\s+)?(planning\s+)?condition|section\s+73\b|\bs\.?\s?73\b|"
            r"minor\s+material\s+amendment|removal/variation|variation/removal|amendment\s+to\s+condition"),
    ("reserved_matters", r"reserved\s+matters"),
    ("conditions", r"details\s+(pursuant|required|reserved|submitted|in\s+relation)\s+(to|by|under|for|pursuant)|"
                   r"discharge\s+(of\s+)?(planning\s+)?condition|approval\s+of\s+details|submission\s+of\s+details|"
                   r"details\s+of\s+condition|^\s*(details|discharge)\b|compliance\s+with\s+condition|"
                   r"pursuant\s+to\s+condition|^\s*condition\s*\d"),
    ("outline", r"\boutline\s+(planning|application|permission)|\bhybrid\b|\bin\s+outline\b"),
    ("lawful_dev", r"lawful\s+development|certificate\s+of\s+lawful|lawfulness"),
    ("prior_approval", r"prior\s+approval|prior\s+notification"),
    ("listed", r"listed\s+building\s+consent"),
    ("advert", r"advertisement\s+consent|display\s+of\s+(an?\s+)?(internally|externally|non)?[\s-]*illuminated|"
               r"^\s*(display|installation)\s+of\s+.{0,40}(signage|signs|advertisement)"),
    ("trees", r"tree\s+preservation|\btpo\b|\bfell(ing)?\b|\bpollard|crown\s+(reduc|thin|lift|raise)|tree\s+works"),
]
DESC_RE = [(cat, re.compile(p, re.I)) for cat, p in DESC_PATTERNS]
ATF_RE = [(re.compile(p, re.I), cat) for p, cat in ATF_MAP]
SUFFIX_MAP = {
    "OUT": "outline", "OUTM": "outline", "OUTMAJ": "outline",
    "REM": "reserved_matters", "RES": "reserved_matters", "RMA": "reserved_matters", "RM": "reserved_matters",
    "VAR": "s73", "S73": "s73", "MMA": "s73", "VOC": "s73", "S73A": "s73",
    "NMA": "nma", "NONMAT": "nma", "S96A": "nma", "NMAMD": "nma",
    "CON": "conditions", "COND": "conditions", "DIS": "conditions", "DISCON": "conditions",
    "DOC": "conditions", "AOD": "conditions", "ADFULL": "conditions", "SD": "conditions", "DET": "conditions",
    "ADV": "advert", "LBC": "listed", "TPO": "trees", "TCA": "trees", "TRE": "trees", "PTT": "trees",
    "HSE": "householder", "HH": "householder", "HOU": "householder", "HHA": "householder",
    "CLE": "lawful_dev", "CLP": "lawful_dev", "CLUP": "lawful_dev", "CLUD": "lawful_dev", "LDC": "lawful_dev",
    "LDCP": "lawful_dev", "LDCE": "lawful_dev", "CLOPUD": "lawful_dev", "CLEUD": "lawful_dev",
    "PRIOR": "prior_approval", "PNH": "prior_approval", "PRIEXT": "prior_approval",
    "PREDEM": "prior_approval", "PRETEL": "prior_approval",
    "FUL": "full", "FULL": "full",
}


NMA_SUFFIXES = {"NMA", "NONMAT", "NMAT", "NMC", "NM", "S96A", "NMAMD"}


def ref_suffix(ref: str) -> str:
    ref = (ref or "").upper().strip()
    m = re.search(r"/([A-Z][A-Z0-9]*)$", ref) or re.search(r"\d([A-Z]{2,7})$", ref)
    return m.group(1) if m else ""


def classify(atf: str | None, coarse: str | None, desc: str | None, ref: str | None) -> str:
    atf_cat = None
    if atf:
        for rx, cat in ATF_RE:
            if rx.search(atf):
                atf_cat = cat
                break
    # Specific ATF values are trusted; generic "full planning" is checked against the text,
    # because several boroughs file S73s / RMs / condition details under "Full".
    if atf_cat and atf_cat not in ("full", "other"):
        return atf_cat
    text = (desc or "")[:300]
    best, best_pos = None, 10 ** 6
    for cat, rx in DESC_RE:
        m = rx.search(text)
        if m and m.start() < best_pos:
            best, best_pos = cat, m.start()
    suffix = ref_suffix(ref)
    if suffix in NMA_SUFFIXES and best in (None, "s73", "conditions"):
        return "nma"   # e.g. Westminster '/NMA' records worded as "Amendments to ... Variation of Condition 1"
    if best in ("nma", "s73", "reserved_matters", "conditions", "outline"):
        return best
    coarse = (coarse or "").lower()
    if coarse == "householder":
        return "householder"
    if coarse == "prior approval":
        return "prior_approval"
    if atf_cat:
        return atf_cat if best is None or atf_cat == "other" else best
    if best:
        return best
    return SUFFIX_MAP.get(ref_suffix(ref), "full_or_other")


# --- decisions --------------------------------------------------------------------------
APPROVED = {"approved", "approve", "gra", "grant", "grantd", "granted", "permitted development", "allowed",
            "pdv", "cond", "parg", "gwla", "gwgc", "tcg", "prior approval granted", "approved with conditions",
            "gsuu", "cds106"}  # Ealing: grant subject to unilateral undertaking / conditional on S106
REFUSED = {"refused", "ref", "refuse", "dismissed"}
WITHDRAWN = {"withdrawn", "wdn", "wthdrw", "wdna", "wna", "disposed", "disp", "closed"}
SPLIT = {"split decision"}


def norm_decision(decision: str | None, status: str | None) -> str:
    d = (decision or "").strip().lower()
    s = (status or "").strip().lower()
    for val in (d, s):
        if val in APPROVED:
            return "approved"
        if val in REFUSED:
            return "refused"
        if val in WITHDRAWN:
            return "withdrawn"
        if val in SPLIT:
            return "split"
    if s in {"application under consideration", "application received", "appeal in progress",
             "appeal received", "pending legacy record", "not yet decided"} or (not d and not s):
        return "pending"
    if s in {"completed", "commenced", "lapsed", "superseded"}:
        return "approved"   # lifecycle states that only follow a grant
    return "other"


UNITS_RE = re.compile(
    r"(\d{1,2},\d{3}|\d{2,5})\s*(?:no\.?\s*|x\s*)?(?:new\s+|net\s+|additional\s+)?(?:self[- ]contained\s+)?"
    r"(?:residential\s+(?:units|dwellings|homes|apartments|flats)|dwellings?|homes|flats|apartments)", re.I)


def units_from_text(desc) -> float:
    """Largest 'N homes/dwellings/flats/residential units' figure quoted in a description."""
    if not isinstance(desc, str):
        return np.nan
    vals = [int(m.group(1).replace(",", "")) for m in UNITS_RE.finditer(desc[:1500])]
    vals = [v for v in vals if v < 20000]
    return float(max(vals)) if vals else np.nan


NOT_A_SCHEME = r"scoping|screening opinion|deed of variation|section 106|s106|legal agreement"
OUTCOME_RANK = {"approved": 0, "split": 1, "refused": 2, "withdrawn": 3, "other": 4, "pending": 5}


def flag_consultations(df: pd.DataFrame) -> pd.Series:
    """Records where the LPA was only consulted on a neighbouring authority's application."""
    dec = (df.decision.fillna("") + " " + df.status.fillna("")).str.lower()
    ref = df.lpa_app_no.fillna("").str.upper()
    return (dec.str.contains(r"obs only|comment issued|observation", regex=True)
            | ref.str.contains(r"/OB/|/OBS\d*$|/OBS/|/CONSLT|/LBC?CON$", regex=True))


def flag_duplicates(df: pd.DataFrame) -> pd.Series:
    """Duplicate records of the same scheme (keeps one canonical record per scheme).

    1. Same LPA + same reference once the type suffix is stripped (Tower Hamlets '/A1',
       Merton/Bexley twin records) with the same description.
    2. Same description and homes figure in another LPA within 60 days (cross-boundary
       sites recorded by both authorities).
    Only applied to records quoting 10+ homes, where double counting distorts totals.
    """
    dup = pd.Series(False, index=df.index)
    big = df[(df.units_best >= 10) & df.is_new_scheme].copy()
    big["base"] = big.lpa_app_no.fillna("").str.upper().str.replace(r"/[A-Z][A-Z0-9]*$", "", regex=True)
    big["dkey"] = big.description.fillna("").str.lower().str.replace(r"[^a-z0-9]", "", regex=True).str[:150]
    big["rank"] = big.outcome.map(OUTCOME_RANK).fillna(9)
    big = big.sort_values(["rank", "valid_date"])
    dup.loc[big[big.duplicated(["lpa_name", "base", "dkey"])].index] = True
    rest = big[~dup.loc[big.index]].sort_values("valid_date")
    for _, g in rest[rest.dkey.str.len() >= 40].groupby(["dkey", "units_best"]):
        if len(g) < 2:
            continue
        kept = []
        for idx, vd in zip(g.index, g.valid_date):
            if any(pd.notna(vd) and pd.notna(k) and abs((vd - k).days) <= 60 for k in kept):
                dup.loc[idx] = True
            else:
                kept.append(vd)
    return dup


def _s(v) -> str | None:
    return v if isinstance(v, str) else None


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for c in DATE_COLS:
        d = pd.to_datetime(df[c], format="%d/%m/%Y", errors="coerce")
        df[c] = d.where((d >= "1980-01-01") & (d <= TODAY + pd.Timedelta(days=3650)))
    for c in ["last_synced", "last_updated"]:
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=True).dt.tz_localize(None)
    return df


def main() -> None:
    parts = sorted((DATA / "raw").glob("part_*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    print("raw rows", len(df))
    df = df.drop_duplicates("id")
    # Older extracts stringified missing values; normalise them back to nulls
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            df[col] = df[col].where(~df[col].isin(["nan", "None", "NaN", ""]))
    df = parse_dates(df)
    # Records with easting/northing of 0 geocode to the British National Grid origin (49.77N, 7.56W);
    # treat anything outside a Greater London bounding box as missing
    bad = ~(df.lat.between(51.2, 51.8) & df.lon.between(-0.6, 0.4))
    print(f"coordinates outside London set to null: {int((bad & df.lat.notna()).sum()):,}")
    df.loc[bad, ["lat", "lon"]] = np.nan
    df["category"] = [classify(_s(a), _s(c), _s(d), _s(r)) for a, c, d, r in
                      zip(df.application_type_full, df.application_type, df.description, df.lpa_app_no)]
    df["outcome"] = [norm_decision(_s(d), _s(st)) for d, st in zip(df.decision, df.status)]
    df["valid_year"] = df.valid_date.dt.year
    df["decision_year"] = df.decision_date.dt.year
    dd = (df.decision_date - df.valid_date).dt.days
    df["days_to_decision"] = dd.where((dd >= 0) & (dd < 3650))
    df["units_net"] = df.units_proposed.fillna(0) - df.units_existing.fillna(0)
    df.loc[df.units_proposed.isna() & df.units_existing.isna(), "units_net"] = np.nan
    df["is_new_scheme"] = df.category.isin(["full", "outline", "full_or_other"])
    # The structured units field is blank for ~30-55% of major schemes (worse for refused /
    # withdrawn / pending ones), so fall back to the figure quoted in the description.
    df["units_desc"] = df.description.map(units_from_text)
    # Reserved matters descriptions usually quote the parent masterplan's total, so for RMs only
    # the structured field is trusted.
    desc_ok = df.units_desc.where(df.category != "reserved_matters")
    df["units_best"] = df.units_proposed.where(df.units_proposed.notna() & (df.units_proposed > 0), desc_ok)
    df["units_source"] = np.where(df.units_proposed.notna() & (df.units_proposed > 0), "field",
                                  np.where(df.units_desc.notna(), "description", None))
    df["is_consultation"] = flag_consultations(df)
    df["is_dup"] = flag_duplicates(df)
    # EIA scoping/screening requests and S106 deeds quote homes figures but are not applications for a scheme
    df["is_scoping"] = df.description.str.contains(NOT_A_SCHEME, case=False, na=False)
    # Some boroughs (e.g. Hounslow DRP/ and PPA/ references) publish pre-application advice cases
    dec = (df.decision.fillna("") + " " + df.status.fillna("")).str.lower()
    df["is_preapp"] = (dec.str.contains(r"pre-?app", regex=True)
                       | df.lpa_app_no.fillna("").str.upper().str.match(r"^(DRP|PPA|PRE|PREAPP|ENQ)/"))
    df["is_scoping"] = df.is_scoping | df.is_preapp
    df["major_resi"] = (df.is_new_scheme & (df.units_best >= 10) & ~df.is_dup & ~df.is_consultation
                        & ~df.is_scoping)
    df.to_parquet(DATA / "applications.parquet", index=False)
    print(df.category.value_counts().to_string())
    print(df.outcome.value_counts().to_string())


if __name__ == "__main__":
    main()
