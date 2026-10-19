import csv
from pathlib import Path

from medipredict.simulator import Inventory, load_config, run_simulation


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data" / "reference"
CONFIG = ROOT / "src" / "medipredict" / "sim" / "config" / "sim.yaml"


def test_inventory_expires_and_consumes_fefo():
    inventory = Inventory()
    inventory.receive(0, 5, 2)
    inventory.receive(2, 5, 3)
    assert inventory.expire(4) == 0
    assert inventory.consume(1) == 0
    assert inventory.units == 4
    assert inventory.expire(5) == 1
    assert inventory.units == 3
    assert inventory.expire(7) == 3
    assert inventory.units == 0


def test_seed_reproduces_identical_stock_data(tmp_path):
    config = load_config(CONFIG)
    config["period_days"] = 7
    first = run_simulation(config, REFERENCE, tmp_path / "first", seed=1234)
    second = run_simulation(config, REFERENCE, tmp_path / "second", seed=1234)
    assert first["sha256"] == second["sha256"]
    assert (tmp_path / "first" / "simulation_daily.csv").read_bytes() == (
        tmp_path / "second" / "simulation_daily.csv"
    ).read_bytes()
    assert (tmp_path / "first" / "demand_events.csv").read_bytes() == (
        tmp_path / "second" / "demand_events.csv"
    ).read_bytes()


def test_reading_row_count_and_provenance(tmp_path):
    config = load_config(CONFIG)
    config["period_days"] = 3
    report = run_simulation(config, REFERENCE, tmp_path, seed=12)
    assert report["rows"] == 30 * 8 * 3
    with (tmp_path / "simulation_daily.csv").open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == report["rows"]
    assert all(row["data_source"] == "simulated" for row in rows)
    assert all(row["unmet_demand_units"] != "" for row in rows)
    assert all(row["expiring_units_7d"] != "" for row in rows)
    assert all(row["recent_surge_flag"] in {"0", "1"} for row in rows)


def test_hourly_output_uses_hourly_row_count_and_provenance(tmp_path):
    config = load_config(CONFIG)
    config["period_days"] = 2
    config["frequency"] = "hourly"
    report = run_simulation(config, REFERENCE, tmp_path, seed=12)
    assert report["rows"] == 30 * 8 * 2 * 24
    assert Path(tmp_path / "simulation_hourly.csv").exists()


def test_optional_referral_nodes_add_rows_without_double_counting_districts(tmp_path):
    config = load_config(CONFIG)
    config["period_days"] = 2
    config["add_kigali_referral_hospitals"] = True
    report = run_simulation(config, REFERENCE, tmp_path, seed=12)
    assert report["district_count"] == 30
    assert report["node_count"] == 34
    assert report["rows"] == 34 * 8 * 2


def test_rare_group_uses_configured_zero_to_two_stock_and_common_groups_have_no_floor(tmp_path):
    config = load_config(CONFIG)
    config["period_days"] = 1
    config["rare_group_local_stock_units"] = 0
    report = run_simulation(config, REFERENCE, tmp_path, seed=18)
    with (tmp_path / "simulation_daily.csv").open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    a_negative = [row for row in rows if row["blood_group"] == "A-"]
    a_positive = [row for row in rows if row["blood_group"] == "A+"]
    assert all(int(row["stock_open_units"]) == 0 for row in a_negative)
    assert all(int(row["stock_open_units"]) >= 0 for row in a_positive)
    assert "inventory_by_group" in report
    assert set(report["inventory_by_group"]) == {
        "O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"
    }
