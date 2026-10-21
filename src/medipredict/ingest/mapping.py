"""English/French canonical-column mapping suggestions."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Iterable


CANONICAL_COLUMNS = (
    "timestamp",
    "district",
    "blood_group",
    "stock_units",
    "demand_units",
    "supply_received",
    "expired_units",
    "unmet_demand_units",
    "facility_id",
)
REQUIRED_COLUMNS = ("timestamp", "district", "blood_group", "stock_units")

SYNONYMS: dict[str, tuple[str, ...]] = {
    "timestamp": (
        "timestamp",
        "date",
        "day",
        "jour",
        "date du jour",
        "date de lecture",
        "datetime",
        "horodatage",
    ),
    "district": (
        "district",
        "district name",
        "nom district",
        "nom du district",
        "district administratif",
        "district_name",
    ),
    "blood_group": (
        "blood group",
        "blood type",
        "groupe sanguin",
        "groupe de sang",
        "groupe",
        "type sanguin",
    ),
    "stock_units": (
        "stock",
        "stock units",
        "closing stock",
        "closing inventory",
        "inventory",
        "inventory units",
        "stock de cloture",
        "stock final",
        "stock actuel",
        "stock disponible",
    ),
    "demand_units": (
        "demand",
        "demand units",
        "consommation",
        "units used",
        "used",
        "issued",
        "issued units",
        "sorties",
        "units issued",
    ),
    "supply_received": (
        "supply",
        "supply received",
        "received",
        "received units",
        "delivery",
        "deliveries",
        "livraison",
        "livraisons",
        "quantite recue",
        "units received",
    ),
    "expired_units": (
        "expired",
        "expired units",
        "expiry",
        "expired stock",
        "perime",
        "perimes",
        "units expired",
        "pertes par peremption",
    ),
    "unmet_demand_units": (
        "unmet",
        "unmet demand",
        "unmet demand units",
        "shortage",
        "shortfall",
        "manquant",
        "besoin non satisfait",
        "demande non satisfaite",
    ),
    "facility_id": (
        "facility id",
        "facility code",
        "facility",
        "hospital id",
        "hospital code",
        "etablissement",
        "code formation sanitaire",
        "code fosa",
        "formation sanitaire",
        "id hopital",
    ),
}


def normalize_header(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


_NORMALIZED_SYNONYMS = {
    canonical: tuple(normalize_header(item) for item in values)
    for canonical, values in SYNONYMS.items()
}


def suggest_mappings(columns: Iterable[Any]) -> dict[str, dict[str, Any]]:
    """Suggest one distinct source column per canonical field with confidence."""
    names = [str(column) for column in columns]
    normalized = {name: normalize_header(name) for name in names}
    suggestions: dict[str, dict[str, Any]] = {}
    used: set[str] = set()
    for canonical in CANONICAL_COLUMNS:
        candidates: list[tuple[float, int, str, str]] = []
        for column, header in normalized.items():
            if column in used or not header:
                continue
            for synonym in _NORMALIZED_SYNONYMS[canonical]:
                if header == synonym:
                    candidates.append((1.0, len(synonym), column, synonym))
                else:
                    similarity = SequenceMatcher(None, header, synonym).ratio()
                    if similarity >= 0.72:
                        candidates.append((similarity, len(synonym), column, synonym))
        if candidates:
            confidence, _, column, matched = max(
                candidates, key=lambda item: (item[0], item[1], item[2])
            )
            confidence = round(confidence, 3)
            suggestions[canonical] = {
                "column": column,
                "confidence": confidence,
                "matched_synonym": matched,
                "required": canonical in REQUIRED_COLUMNS,
            }
            used.add(column)
        else:
            suggestions[canonical] = {
                "column": None,
                "confidence": 0.0,
                "matched_synonym": None,
                "required": canonical in REQUIRED_COLUMNS,
            }
    return suggestions


def header_quality(columns: Iterable[Any]) -> tuple[int, float]:
    suggestions = suggest_mappings(columns)
    confident = sum(
        item["column"] is not None and item["confidence"] >= 0.9
        for item in suggestions.values()
    )
    score = sum(float(item["confidence"]) for item in suggestions.values())
    return int(confident), score
