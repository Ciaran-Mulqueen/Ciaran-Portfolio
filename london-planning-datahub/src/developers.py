"""Map free-text `lead_developer_company_name` values to listed housebuilders.

The field is self-reported and messy ("Berkeley Homes (East Thames) Ltd ", "St George ",
"BDW Trading Ltd"), so matching is regex based with explicit exclusions.
"""
from __future__ import annotations

import re

BERKELEY_BRANDS = [
    ("Berkeley Homes", r"\bberkeley\s*(homes?|group|st\.?\s*joseph|capital|first|partnerships?|modular|"
                       r"developments?|residential|plc|ltd|limited)\b|\bberkeley\s*\(|^\s*berkeley\s*$"),
    ("St George", r"\bst\.?\s*george\s*(plc|west|south|north|central|city|northfields|limited|ltd|homes|\()|"
                  r"^\s*st\.?\s*george\s*$"),
    ("St James", r"\bst\.?\s*james\s*(group|homes)\b"),
    ("St Edward", r"\bst\.?\s*edward\s*(homes|ltd|limited)\b"),
    ("St William", r"\bst\.?\s*williams?\s*(homes|llp|ltd|limited)\b|^\s*st\.?\s*williams?\s*$"),
    ("St Joseph", r"\bst\.?\s*joseph\s*(homes|ltd|limited)\b"),
]
BERKELEY_EXCLUDE = re.compile(r"square\s+holdings|maybourne|court\s+management|hotel|de\s*ve+r|"
                              r"berkeley\s+(square|street|road|avenue|estate)", re.I)
_BRAND_RE = [(name, re.compile(p, re.I)) for name, p in BERKELEY_BRANDS]


def berkeley_brand(dev: str | None) -> str | None:
    if not isinstance(dev, str) or not dev or BERKELEY_EXCLUDE.search(dev):
        return None
    for name, rx in _BRAND_RE:
        if rx.search(dev):
            return name
    return None


PEERS = {
    "Berkeley": None,  # handled by berkeley_brand
    "Barratt Redrow": r"\bbarratt|\bbdw\b|\bbdw\s+trading|\bredrow\b|david\s+wilson\s+homes",
    "Taylor Wimpey": r"taylor\s*wimpey",
    "Bellway": r"\bbellway\b",
    "Vistry": r"\bvistry\b|\bcountryside\s+(partnerships|properties|homes|developments)|\blinden\s+(homes|ltd|limited)\b",
    "Crest Nicholson": r"crest\s*nicholson",
    "Persimmon": r"\bpersimmon\b|charles\s+church",
}
_PEER_RE = {k: re.compile(v, re.I) for k, v in PEERS.items() if v}


def listed_builder(dev: str | None) -> str | None:
    if not isinstance(dev, str) or not dev:
        return None
    if berkeley_brand(dev):
        return "Berkeley"
    for name, rx in _PEER_RE.items():
        if rx.search(dev):
            return name
    return None
