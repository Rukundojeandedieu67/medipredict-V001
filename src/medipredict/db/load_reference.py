"""Load the provenance-tagged reference CSVs into the migrated PostgreSQL database."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text
import yaml


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REFERENCE = ROOT / "data" / "reference"
DEFAULT_DATABASE_URL = "postgresql+psycopg://medipredict:medipredict@localhost:5432/medipredict"
ALLOWED_SOURCES = {"real", "derived", "simulated", "entered"}
RBC_FILES = (
    ("collection_by_group_rcbt", "rbc_2024_collected_by_group_rcbt.csv"),
    ("national_indicator", "rbc_2024_national_indicators.csv"),
    ("transfusion_indication", "rbc_2024_transfusion_indications.csv"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        if "data_source" not in reader.fieldnames:
            raise ValueError(f"Missing required data_source column: {path}")
        rows = list(reader)
    for line_number, row in enumerate(rows, start=2):
        if row["data_source"] not in ALLOWED_SOURCES:
            raise ValueError(f"Invalid data_source at {path}:{line_number}")
    return rows


def _json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _nullable_float(value: str | None) -> float | None:
    return float(value) if value not in (None, "") else None


def _nullable_int(value: str | None) -> int | None:
    return int(value) if value not in (None, "") else None


def load_reference(
    reference_dir: Path,
    database_url: str,
    *,
    include_kigali_referral_hospitals: bool | None = None,
) -> dict[str, int]:
    districts = read_csv(reference_dir / "dim_district.csv")
    facilities = read_csv(reference_dir / "dim_facility_osm.csv")
    referral_nodes = read_csv(reference_dir / "kigali_referral_nodes.csv")
    distance_matrix = read_csv(reference_dir / "district_distance_km.csv")
    district_pairs = read_csv(reference_dir / "district_pairs.csv")
    hospitalisation = read_csv(reference_dir / "fact_hospitalisation.csv")
    blood_usage = read_csv(reference_dir / "fact_blood_usage.csv")
    collection_mix = read_csv(reference_dir / "rbc_2024_collected_by_group_rcbt.csv")
    national_indicators = read_csv(reference_dir / "rbc_2024_national_indicators.csv")
    indications = read_csv(reference_dir / "rbc_2024_transfusion_indications.csv")
    mix_audit = read_csv(reference_dir / "blood_group_mix_comparison.csv")

    district_code = {row["district"]: row["adm2_pcode"] for row in districts}
    config_path = Path(__file__).parents[1] / "sim" / "config" / "sim.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    referral_nodes_enabled = (
        bool(config.get("add_kigali_referral_hospitals", False))
        if include_kigali_referral_hospitals is None
        else include_kigali_referral_hospitals
    )
    engine = create_engine(database_url, future=True)
    counts: dict[str, int] = {}
    try:
        with engine.begin() as connection:
            connection.execute(text("DELETE FROM ref_hospitalisation"))
            connection.execute(text("DELETE FROM ref_blood_usage"))
            for row in districts:
                connection.execute(
                    text(
                        """
                        INSERT INTO districts
                            (district_id, name, province, geom, attributes, data_source)
                        VALUES
                            (:district_id, :name, :province,
                             ST_SetSRID(ST_MakePoint(:lon, :lat), 4326),
                             CAST(:attributes AS jsonb), CAST(:data_source AS data_source_enum))
                        ON CONFLICT (district_id) DO UPDATE SET
                            name = EXCLUDED.name,
                            province = EXCLUDED.province,
                            geom = EXCLUDED.geom,
                            attributes = EXCLUDED.attributes,
                            data_source = EXCLUDED.data_source
                        """
                    ),
                    {
                        "district_id": row["adm2_pcode"],
                        "name": row["district"],
                        "province": row["province"],
                        "lon": float(row["lon"]),
                        "lat": float(row["lat"]),
                        "attributes": _json(row),
                        "data_source": row["data_source"],
                    },
                )
            counts["districts"] = len(districts)

            for row in facilities:
                if row["district"] not in district_code:
                    raise ValueError(f"Facility references unknown district: {row['district']}")
                connection.execute(
                    text(
                        """
                        INSERT INTO facilities
                            (facility_id, district_id, name, tier, amenity, healthcare, geom,
                             attributes, data_source)
                        VALUES
                            (:facility_id, :district_id, :name, :tier, :amenity, :healthcare,
                             CASE WHEN :lat IS NULL THEN NULL
                                  ELSE ST_SetSRID(ST_MakePoint(:lon, :lat), 4326) END,
                             CAST(:attributes AS jsonb), CAST(:data_source AS data_source_enum))
                        ON CONFLICT (facility_id) DO UPDATE SET
                            district_id = EXCLUDED.district_id,
                            name = EXCLUDED.name,
                            tier = EXCLUDED.tier,
                            amenity = EXCLUDED.amenity,
                            healthcare = EXCLUDED.healthcare,
                            geom = EXCLUDED.geom,
                            attributes = EXCLUDED.attributes,
                            data_source = EXCLUDED.data_source
                        """
                    ),
                    {
                        "facility_id": row["id"],
                        "district_id": district_code[row["district"]],
                        "name": row.get("name_clean") or None,
                        "tier": row.get("tier") or None,
                        "amenity": row.get("amenity") or None,
                        "healthcare": row.get("healthcare") or None,
                        "lon": _nullable_float(row.get("lon")),
                        "lat": _nullable_float(row.get("lat")),
                        "attributes": _json(row),
                        "data_source": row["data_source"],
                    },
                )
            counts["facilities"] = len(facilities)

            for row in districts:
                connection.execute(
                    text(
                        """
                        INSERT INTO nodes
                            (node_id, district_id, facility_id, name, node_type, enabled,
                             attributes, data_source)
                        VALUES (:node_id, :district_id, NULL, :name, 'district_blood_bank', true,
                                CAST(:attributes AS jsonb), CAST(:data_source AS data_source_enum))
                        ON CONFLICT (node_id) DO UPDATE SET
                            district_id = EXCLUDED.district_id,
                            name = EXCLUDED.name,
                            node_type = EXCLUDED.node_type,
                            enabled = EXCLUDED.enabled,
                            attributes = EXCLUDED.attributes,
                            data_source = EXCLUDED.data_source
                        """
                    ),
                    {
                        "node_id": f"district:{row['adm2_pcode']}",
                        "district_id": row["adm2_pcode"],
                        "name": f"District blood bank ({row['district']})",
                        "attributes": _json({"district": row["district"]}),
                        "data_source": "derived",
                    },
                )
            for row in referral_nodes:
                if row["district"] not in district_code:
                    raise ValueError(f"Referral node references unknown district: {row['district']}")
                connection.execute(
                    text(
                        """
                        INSERT INTO nodes
                            (node_id, district_id, facility_id, name, node_type, enabled,
                             attributes, data_source)
                        VALUES (:node_id, :district_id, NULL, :name, 'kigali_referral_hospital',
                                :enabled, CAST(:attributes AS jsonb),
                                CAST(:data_source AS data_source_enum))
                        ON CONFLICT (node_id) DO UPDATE SET
                            district_id = EXCLUDED.district_id,
                            name = EXCLUDED.name,
                            node_type = EXCLUDED.node_type,
                            enabled = EXCLUDED.enabled,
                            attributes = EXCLUDED.attributes,
                            data_source = EXCLUDED.data_source
                        """
                    ),
                    {
                        "node_id": row["node_id"],
                        "district_id": district_code[row["district"]],
                        "name": row["facility_name"],
                        "enabled": referral_nodes_enabled,
                        "attributes": _json(row),
                        "data_source": row["data_source"],
                    },
                )
            counts["nodes"] = len(districts) + len(referral_nodes)

            for row in distance_matrix:
                connection.execute(
                    text(
                        """
                        INSERT INTO district_distance
                            (from_district, to_district, straight_km, est_road_km,
                             est_transit_hours, data_source)
                        VALUES (:origin, :destination, :straight_km, NULL, NULL,
                                CAST(:data_source AS data_source_enum))
                        ON CONFLICT (from_district, to_district) DO UPDATE SET
                            straight_km = EXCLUDED.straight_km,
                            data_source = EXCLUDED.data_source
                        """
                    ),
                    {
                        "origin": district_code[row["from_district"]],
                        "destination": district_code[row["to_district"]],
                        "straight_km": float(row["straight_km"]),
                        "data_source": row["data_source"],
                    },
                )
            for row in district_pairs:
                connection.execute(
                    text(
                        """
                        INSERT INTO district_distance
                            (from_district, to_district, straight_km, est_road_km,
                             est_transit_hours, data_source)
                        VALUES (:origin, :destination, :straight_km, :est_road_km,
                                :est_transit_hours, CAST(:data_source AS data_source_enum))
                        ON CONFLICT (from_district, to_district) DO UPDATE SET
                            straight_km = EXCLUDED.straight_km,
                            est_road_km = EXCLUDED.est_road_km,
                            est_transit_hours = EXCLUDED.est_transit_hours,
                            data_source = EXCLUDED.data_source
                        """
                    ),
                    {
                        "origin": district_code[row["from_district"]],
                        "destination": district_code[row["to_district"]],
                        "straight_km": float(row["straight_km"]),
                        "est_road_km": float(row["est_road_km"]),
                        "est_transit_hours": float(row["est_transit_hours"]),
                        "data_source": row["data_source"],
                    },
                )
            counts["district_distance"] = len(distance_matrix)

            totals: dict[str, int] = {}
            for row in collection_mix:
                group = row["blood_group"]
                totals[group] = totals.get(group, 0) + int(row["units_collected"])
            collection_total = sum(totals.values())
            for group, units in totals.items():
                connection.execute(
                    text(
                        """
                        INSERT INTO blood_groups (code, rbc_2024_share, data_source)
                        VALUES (:code, :share, 'real')
                        ON CONFLICT (code) DO UPDATE SET
                            rbc_2024_share = EXCLUDED.rbc_2024_share,
                            data_source = EXCLUDED.data_source
                        """
                    ),
                    {"code": group, "share": units / collection_total},
                )
            counts["blood_groups"] = len(totals)

            for row in hospitalisation:
                connection.execute(
                    text(
                        """
                        INSERT INTO ref_hospitalisation
                            (district_id, facility_type, admissions, avg_los_days,
                             source_period, attributes, data_source)
                        VALUES (:district_id, :facility_type, :admissions, :avg_los_days,
                                NULL, CAST(:attributes AS jsonb), CAST(:data_source AS data_source_enum))
                        """
                    ),
                    {
                        "district_id": district_code[row["district"]],
                        "facility_type": row["facility_type"],
                        "admissions": int(row["admissions"]),
                        "avg_los_days": _nullable_float(row["avg_los_days"]),
                        "attributes": _json(row),
                        "data_source": row["data_source"],
                    },
                )
            counts["ref_hospitalisation"] = len(hospitalisation)

            for row in blood_usage:
                connection.execute(
                    text(
                        """
                        INSERT INTO ref_blood_usage
                            (district_id, blood_group, units, source_period, attributes, data_source)
                        VALUES (:district_id, :blood_group, :units, 'period_unspecified',
                                CAST(:attributes AS jsonb), CAST(:data_source AS data_source_enum))
                        """
                    ),
                    {
                        "district_id": district_code[row["district"]],
                        "blood_group": row["blood_group"],
                        "units": int(row["units"]),
                        "attributes": _json(row),
                        "data_source": row["data_source"],
                    },
                )
            counts["ref_blood_usage"] = len(blood_usage)

            rbc_rows = (
                [("collection_by_group_rcbt", row) for row in collection_mix]
                + [("national_indicator", row) for row in national_indicators]
                + [("transfusion_indication", row) for row in indications]
            )
            for record_type, row in rbc_rows:
                numeric_value = (
                    row.get("units_collected")
                    or row.get("value")
                    or row.get("total_components")
                )
                component_values = {
                    key: int(row[key])
                    for key in ("rbc", "platelets", "ffp", "cryo")
                    if row.get(key) not in (None, "")
                }
                connection.execute(
                    text(
                        """
                        INSERT INTO ref_rbc_2024
                            (record_type, record_key, year, blood_group, region, rcbt,
                             indicator, indication, units, components, source, attributes, data_source)
                        VALUES (:record_type, :record_key, :year, :blood_group, :region, :rcbt,
                                :indicator, :indication, :units, CAST(:components AS jsonb),
                                :source, CAST(:attributes AS jsonb), CAST(:data_source AS data_source_enum))
                        ON CONFLICT (record_type, record_key) DO UPDATE SET
                            year = EXCLUDED.year,
                            blood_group = EXCLUDED.blood_group,
                            region = EXCLUDED.region,
                            rcbt = EXCLUDED.rcbt,
                            indicator = EXCLUDED.indicator,
                            indication = EXCLUDED.indication,
                            units = EXCLUDED.units,
                            components = EXCLUDED.components,
                            source = EXCLUDED.source,
                            attributes = EXCLUDED.attributes,
                            data_source = EXCLUDED.data_source
                        """
                    ),
                    {
                        "record_type": record_type,
                        "record_key": f"{record_type}:{row.get('year', 2024)}:{row.get('rcbt') or row.get('indicator') or row.get('indication')}:{row.get('blood_group', '')}",
                        "year": int(row.get("year") or 2024),
                        "blood_group": row.get("blood_group") or None,
                        "region": row.get("region") or None,
                        "rcbt": row.get("rcbt") or None,
                        "indicator": row.get("indicator") or None,
                        "indication": row.get("indication") or None,
                        "units": _nullable_int(numeric_value),
                        "components": _json(component_values),
                        "source": row.get("source") or row.get("table") or SRC_FALLBACK,
                        "attributes": _json(row),
                        "data_source": row["data_source"],
                    },
                )
            counts["ref_rbc_2024"] = len(rbc_rows)

            for row in mix_audit:
                connection.execute(
                    text(
                        """
                        INSERT INTO ref_blood_group_mix_audit (blood_group, attributes, data_source)
                        VALUES (:group, CAST(:attributes AS jsonb),
                                CAST(:data_source AS data_source_enum))
                        ON CONFLICT (blood_group) DO UPDATE SET
                            attributes = EXCLUDED.attributes,
                            data_source = EXCLUDED.data_source
                        """
                    ),
                    {
                        "group": row["blood_group"],
                        "attributes": _json(row),
                        "data_source": row["data_source"],
                    },
                )
            counts["ref_blood_group_mix_audit"] = len(mix_audit)
    finally:
        engine.dispose()
    return counts


SRC_FALLBACK = "RBC Blood Transfusion Division Annual Report 2024"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL),
    )
    parser.add_argument(
        "--include-kigali-referral-hospitals",
        action="store_true",
        default=None,
        help="Enable the four official Kigali referral-hospital nodes in the database.",
    )
    args = parser.parse_args()
    counts = load_reference(
        args.reference_dir,
        args.database_url,
        include_kigali_referral_hospitals=args.include_kigali_referral_hospitals,
    )
    for table, count in counts.items():
        print(f"{table}: {count} source rows loaded")


if __name__ == "__main__":
    main()
