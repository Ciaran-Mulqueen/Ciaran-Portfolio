"""Minimal client for the Planning London Datahub (PLD) public Elasticsearch API.

The guest endpoint needs no API key. It exposes a single index (`applications`)
and accepts the standard Elasticsearch query DSL. Text fields have `.raw`
keyword sub-fields for aggregations; dates are stored as dd/MM/yyyy.
"""
from __future__ import annotations

import json
import time
from typing import Iterator

import requests

SEARCH_URL = "https://planningdata.london.gov.uk/api-guest/applications/_search"

_session = requests.Session()


def search(body: dict, retries: int = 5) -> dict:
    """POST a query body to the search endpoint, retrying transient failures."""
    for attempt in range(retries):
        try:
            r = _session.post(SEARCH_URL, json=body, timeout=300)
            data = r.json()
            if "error" in data:
                raise RuntimeError(json.dumps(data["error"])[:1000])
            return data
        except (requests.RequestException, ValueError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("unreachable")


def count(query: dict | None = None) -> int:
    body = {"size": 0, "track_total_hits": True}
    if query:
        body["query"] = query
    return search(body)["hits"]["total"]["value"]


def terms(field: str, size: int = 50, query: dict | None = None) -> list[tuple[str, int]]:
    body = {"size": 0, "aggs": {"a": {"terms": {"field": field, "size": size}}}}
    if query:
        body["query"] = query
    return [(b["key"], b["doc_count"]) for b in search(body)["aggregations"]["a"]["buckets"]]


def iter_hits(query: dict | None = None, source: list[str] | None = None,
              page_size: int = 5000) -> Iterator[dict]:
    """Yield every matching document's _source using search_after on _id."""
    body: dict = {"size": page_size, "sort": [{"_id": "asc"}], "track_total_hits": False}
    if query:
        body["query"] = query
    if source:
        body["_source"] = source
    after = None
    while True:
        if after is not None:
            body["search_after"] = after
        hits = search(body)["hits"]["hits"]
        if not hits:
            return
        for h in hits:
            yield h["_source"]
        after = hits[-1]["sort"]
