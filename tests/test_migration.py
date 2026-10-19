from pathlib import Path


MIGRATION = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "medipredict"
    / "db"
    / "migrations"
    / "versions"
    / "0001_initial_schema.py"
)


def test_migration_has_all_required_tables_and_provenance_enum():
    sql = MIGRATION.read_text(encoding="utf-8").lower()
    required = (
        "districts",
        "facilities",
        "nodes",
        "district_distance",
        "blood_groups",
        "stock_readings",
        "demand_events",
        "transfers",
        "predictions",
        "recommendations",
        "decisions_audit",
        "ref_rbc_2024",
    )
    assert all(f"create table {name}" in sql for name in required)
    assert "create type data_source_enum as enum ('real', 'derived', 'simulated', 'entered')" in sql
    assert "create extension if not exists postgis" in sql
    assert "check (unmet_demand_units >= 0)" in sql
    assert "using gist (geom)" in sql
