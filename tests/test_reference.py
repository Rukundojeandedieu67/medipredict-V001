import csv
from pathlib import Path

from medipredict.simulator import GROUPS, load_config, load_reference


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data" / "reference"


def test_reference_csv_rows_are_provenance_tagged():
    files = sorted(REFERENCE.glob("*.csv"))
    assert len(files) == 11
    for path in files:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            assert "data_source" in (reader.fieldnames or [])
            rows = list(reader)
        assert rows, path.name
        assert all(row["data_source"] in {"real", "derived"} for row in rows), path.name


def test_reference_cardinalities_and_group_mix_source():
    config = load_config(ROOT / "src" / "medipredict" / "sim" / "config" / "sim.yaml")
    districts, shares = load_reference(REFERENCE, config)
    assert len(districts) == 30
    assert set(shares) == set(GROUPS)
    assert abs(sum(shares.values()) - 1.0) < 1e-12
    assert all(district.transit_hours >= 0 for district in districts)


def test_blood_bank_comparison_is_audit_only():
    path = REFERENCE / "blood_group_mix_comparison.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert all(row["use_restriction"] == "audit_only_not_for_model_or_simulation" for row in rows)


def test_reference_foreign_key_values_and_key_fields_are_valid():
    def rows(filename):
        with (REFERENCE / filename).open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    districts = rows("dim_district.csv")
    district_names = {row["district"] for row in districts}
    district_codes = {row["adm2_pcode"] for row in districts}
    groups = set(GROUPS)
    assert len(district_names) == len(district_codes) == 30
    assert all(row["district"] and row["adm2_pcode"] for row in districts)

    for filename in ("dim_facility_osm.csv", "fact_hospitalisation.csv", "fact_blood_usage.csv"):
        assert all(row["district"] in district_names for row in rows(filename))
    assert all(row["id"] for row in rows("dim_facility_osm.csv"))
    assert all(row["blood_group"] in groups for row in rows("fact_blood_usage.csv"))
    for filename in ("district_pairs.csv", "district_distance_km.csv"):
        assert all(
            row["from_district"] in district_names and row["to_district"] in district_names
            for row in rows(filename)
        )
    assert district_codes == {row["adm2_pcode"] for row in districts}


def test_optional_kigali_referral_nodes_are_official_and_weight_preserving():
    config = load_config(ROOT / "src" / "medipredict" / "sim" / "config" / "sim.yaml")
    config["add_kigali_referral_hospitals"] = True
    nodes, _ = load_reference(REFERENCE, config)
    referral_nodes = [node for node in nodes if node.node_type == "kigali_referral_hospital"]
    assert len(nodes) == 34
    assert len(referral_nodes) == 4
    assert all(node.province == "Kigali City" for node in referral_nodes)
    base = load_config(ROOT / "src" / "medipredict" / "sim" / "config" / "sim.yaml")
    base_nodes, _ = load_reference(REFERENCE, base)
    assert abs(sum(node.admissions for node in nodes) - sum(node.admissions for node in base_nodes)) < 1e-8
