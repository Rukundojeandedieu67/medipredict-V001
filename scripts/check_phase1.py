"""Print and enforce Phase 1 data, provenance, and reproducibility checks."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path

from medipredict.simulator import DEFAULT_CONFIG, DEFAULT_OUTPUT, DEFAULT_REFERENCE
from medipredict.simulator import load_config, run_simulation


GROUPS = ("O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    failures: list[str] = []
    districts = read_csv(args.reference_dir / "dim_district.csv")
    facilities = read_csv(args.reference_dir / "dim_facility_osm.csv")
    pairs = read_csv(args.reference_dir / "district_pairs.csv")
    matrix = read_csv(args.reference_dir / "district_distance_km.csv")
    hospitalisation = read_csv(args.reference_dir / "fact_hospitalisation.csv")
    blood_usage = read_csv(args.reference_dir / "fact_blood_usage.csv")
    referral_nodes = read_csv(args.reference_dir / "kigali_referral_nodes.csv")
    collections = read_csv(args.reference_dir / "rbc_2024_collected_by_group_rcbt.csv")
    district_names = {row["district"] for row in districts}
    district_codes = {row["adm2_pcode"] for row in districts}
    province_by_district = {row["district"]: row["province"] for row in districts}
    group_totals = {group: 0 for group in GROUPS}
    for row in collections:
        group_totals[row["blood_group"]] += int(row["units_collected"])
    collection_total = sum(group_totals.values())
    expected_shares = {group: group_totals[group] / collection_total for group in GROUPS}

    key_checks = {
        "districts": len(district_names) == 30
        and len(district_codes) == 30
        and all(row["district"] and row["adm2_pcode"] for row in districts),
        "facilities": len({row["id"] for row in facilities}) == len(facilities)
        and all(row["id"] and row["district"] for row in facilities),
        "district_pairs": all(row["from_district"] and row["to_district"] for row in pairs),
        "distance_matrix": all(row["from_district"] and row["to_district"] for row in matrix),
        "hospitalisation": all(row["district"] and row["facility_type"] for row in hospitalisation),
        "blood_usage": all(row["district"] and row["blood_group"] for row in blood_usage),
        "rbc_mix": all(row["blood_group"] and row["rcbt"] for row in collections),
        "kigali_referral_nodes": all(
            row["node_id"] and row["facility_name"] and row["district_code"] for row in referral_nodes
        ),
    }
    for name, passed in key_checks.items():
        print(f"No null keys ({name}): {'PASS' if passed else 'FAIL'}")
        if not passed:
            failures.append(f"null or duplicate key in {name}")

    foreign_key_checks = {
        "facility district": all(row["district"] in district_names for row in facilities),
        "distance endpoints": all(
            row["from_district"] in district_names and row["to_district"] in district_names
            for row in pairs + matrix
        ),
        "hospitalisation district": all(row["district"] in district_names for row in hospitalisation),
        "blood usage district/group": all(
            row["district"] in district_names and row["blood_group"] in GROUPS
            for row in blood_usage
        ),
        "Kigali referral district": all(
            row["district"] in district_names
            and province_by_district.get(row["district"]) == "Kigali City"
            and row["facility_type"] == "Referral Hospital"
            for row in referral_nodes
        ),
    }
    for name, passed in foreign_key_checks.items():
        print(f"Reference FK values ({name}): {'PASS' if passed else 'FAIL'}")
        if not passed:
            failures.append(f"invalid reference foreign key: {name}")

    reference_files = sorted(args.reference_dir.glob("*.csv"))
    provenance_ok = True
    for path in reference_files:
        rows = read_csv(path)
        if not rows or any(row.get("data_source") not in {"real", "derived"} for row in rows):
            provenance_ok = False
            break
    print(
        f"Reference provenance tags ({len(reference_files)} CSV files): "
        f"{'PASS' if provenance_ok else 'FAIL'}"
    )
    if not provenance_ok:
        failures.append("missing or invalid reference data_source")

    summary_path = args.output_dir / "simulation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_rows = int(summary["node_count"]) * 8 * int(summary["period_days"])
    if summary["frequency"] == "hourly":
        expected_rows *= 24
    row_count_ok = summary["rows"] == expected_rows
    print(
        f"Simulation row count: {summary['rows']:,} / {expected_rows:,} "
        f"{'PASS' if row_count_ok else 'FAIL'}"
    )
    if not row_count_ok:
        failures.append("wrong simulated row count")

    stock_path = args.output_dir / summary["stock_readings_file"]
    simulated_rows = 0
    sim_keys_ok = True
    expected_node_ids = {f"district:{row['adm2_pcode']}" for row in districts}
    expected_node_ids.update(row["node_id"] for row in referral_nodes)
    with stock_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            simulated_rows += 1
            if not (
                row["timestamp"]
                and row["node_id"] in expected_node_ids
                and row["district_code"] in district_codes
                and row["blood_group"] in GROUPS
                and row["data_source"] == "simulated"
            ):
                sim_keys_ok = False
                break
    sim_keys_ok = sim_keys_ok and simulated_rows == expected_rows
    print(f"Simulated keys and provenance: {'PASS' if sim_keys_ok else 'FAIL'}")
    if not sim_keys_ok:
        failures.append("invalid simulated key or data_source")

    print("\nSimulated demand mix vs real RBC 2024 collection mix:")
    print("| Blood group | Simulated units | Simulated share | RBC 2024 share | Difference (pp) |")
    print("|---|---:|---:|---:|---:|")
    mix_ok = True
    for group, values in summary["demand_mix_by_group"].items():
        delta = (values["simulated_share"] - expected_shares[group]) * 100
        mix_ok = mix_ok and abs(delta) <= 1
        print(
            f"| {group} | {values['demand_units']:,} | {values['simulated_share'] * 100:.3f}% | "
            f"{expected_shares[group] * 100:.3f}% | {delta:+.3f} |"
        )
    if not mix_ok:
        failures.append("simulated blood group mix differs by more than 1 percentage point")

    unmet_ok = 0.18 <= summary["unmet_demand_pct"] <= 0.38
    expiry_ok = 2 <= summary["expiry_pct_of_supply"] <= 6
    print(
        f"\nUnmet demand: {summary['unmet_demand_units']:,} / {summary['demand_units']:,} "
        f"= {summary['unmet_demand_pct']:.3f}% {'PASS' if unmet_ok else 'FAIL'}"
    )
    print(
        f"Expiry: {summary['expired_units']:,} / {summary['supply_received_units']:,} "
        f"= {summary['expiry_pct_of_supply']:.3f}% {'PASS' if expiry_ok else 'FAIL'}"
    )
    print("Contiguous stockout episodes by blood group (simulated):")
    for group in GROUPS:
        print(f"  {group}: {summary['stockout_events_by_group'][group]}")
    if not unmet_ok:
        failures.append("simulated unmet demand is outside the target range")
    if not expiry_ok:
        failures.append("simulated expiry is outside the target range")

    config = load_config(args.config)
    config["period_days"] = 3
    with tempfile.TemporaryDirectory(prefix="medipredict-repro-") as temporary:
        first = run_simulation(config, args.reference_dir, Path(temporary) / "first", seed=851)
        second = run_simulation(config, args.reference_dir, Path(temporary) / "second", seed=851)
    reproducible = first["sha256"] == second["sha256"]
    print(f"Same seed reproduces identical simulated readings: {'PASS' if reproducible else 'FAIL'}")
    if not reproducible:
        failures.append("seed reproducibility failed")

    if failures:
        print("\nPhase 1 acceptance: FAIL")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("\nPhase 1 acceptance: PASS")


if __name__ == "__main__":
    main()
