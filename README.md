# MediPredict Blood Network

Phase 1 establishes provenance-tagged reference data, a seeded inventory simulator, and a PostgreSQL/PostGIS schema for Rwanda's 30 districts and eight blood groups. Phase 2 adds a pooled, simulated-data early-warning model with independent seed and time holdouts. Rebalancing, API, and dashboard phases are not implemented yet.

## Architecture

```text
Supplied source files
        |
        v
scripts/prepare_reference.py ---> data/reference/*.csv
        |                                |
        v                                +--> Alembic / PostGIS reference loader
seeded demand + inventory simulator              |
        |                                        v
        +--> data/generated/seed_XX/*.csv     PostgreSQL/PostGIS
        |          |
        |          +--> leakage-aware features --> pooled 24h/48h models
        |                                       --> metrics + model card
```

## Setup

Requires Python 3.11+. Install this Phase 1 package and dependencies:

```powershell
python -m pip install -e ".[dev]"
```

Generate the canonical reference CSVs, search the documented safety-stock grid, write the calibration report, and create the default 730-day daily simulation:

```powershell
python scripts/prepare_reference.py
python scripts/calibrate_simulation.py
python scripts/check_phase1.py
python scripts/report_phase1_diagnostics.py
```

`data/reference/` contains source-tagged data prepared from `data/files/`. To generate a shorter test run or choose a seed:

```powershell
python -m medipredict.simulator --days 30 --seed 42 --output-dir data/generated/demo
python -m medipredict.simulator --frequency hourly --days 2 --seed 42
python -m medipredict.simulator --days 30 --include-kigali-referral-hospitals
```

The optional flag adds the four Ministry-listed Kigali referral hospitals to the 30 district nodes. Their district admissions weight is split equally across the district blood bank and referral nodes because facility-level admissions are unavailable.

The generated summary contains actual counts and rates for that run. It is not a model-performance report.

Run tests:

```powershell
python -m pytest
```

## Phase 2: early warning

Install the modeling dependencies through the editable package install above, then generate 20 independent 730-day simulator runs (`seed_00` through `seed_19`, actual seeds 2026-2045) and train/evaluate:

```powershell
python scripts/generate_seeds.py --count 20 --start-seed 2026
python scripts/train_model.py
```

Or run both with `make phase2`. Seed directories are split 60%/20%/20% into train/validation/test; within each seed, time windows are also chronological with a 3-day gap. The pooled model includes blood group as a categorical feature. Labels are observed unmet demand only in rows t+1 through t+24h/t+48h. Results and positive counts are written to `docs/metrics.json`, with the model card in `docs/MODEL_CARD.md` and fitted estimator bundle in `models/phase2_model.joblib`. All reported model metrics are from held-out **simulated** data; they are not field performance claims.

## Database

The Phase 1 local service is PostgreSQL 16 with PostGIS:

```powershell
docker compose up -d db
python -m pip install -e .
alembic upgrade head
python -m medipredict.db.load_reference
```

Set `DATABASE_URL` to connect to another PostgreSQL/PostGIS database. The reference loader is repeatable and loads each canonical reference CSV into its dimension/reference table. All schema tables include a constrained `data_source` value (`real`, `derived`, `simulated`, or `entered`).

`make data`, `make test`, and `make db-setup` targets are also provided where GNU Make is installed.

## What is real vs simulated

- **Real reference data:** OSM facilities; supplied Rwanda hospitalisation and blood-use extracts (their reporting periods are not supplied); RBC Blood Transfusion Division Annual Report 2024.
- **Derived data:** normalized district dimensions, distance estimates, calculated RBC indicators, and the explicitly audit-only comparison with the undated blood-bank extract.
- **Simulated data:** all generated stock readings, demand, shipments, expiries, unmet demand, and surge events.
- **Entered data:** reserved for future user/API inputs. No such operational readings are included in Phase 1.

The simulation uses only the RBC 2024 collection mix. It does not use blood-group shares from `fact_blood_usage.csv` or `transfusion.data` (the Taiwanese UCI donor dataset). It also does not claim that generated rates represent actual Rwanda inventory performance.

See [ASSUMPTIONS.md](./ASSUMPTIONS.md), [docs/DATA_CARD.md](./docs/DATA_CARD.md), [docs/DATA_DICTIONARY.md](./docs/DATA_DICTIONARY.md), and [docs/CALIBRATION.md](./docs/CALIBRATION.md).
