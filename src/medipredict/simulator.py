"""Generate reproducible, explicitly simulated Rwanda blood-stock readings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).parent / "sim" / "config" / "sim.yaml"
DEFAULT_REFERENCE = ROOT / "data" / "reference"
DEFAULT_OUTPUT = ROOT / "data" / "generated"
GROUPS = ("O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-")
STOCK_FIELDS = (
    "timestamp",
    "node_id",
    "node_name",
    "node_type",
    "district",
    "district_code",
    "province",
    "blood_group",
    "stock_open_units",
    "stock_units",
    "expiring_units_7d",
    "demand_units",
    "supply_received",
    "expired_units",
    "unmet_demand_units",
    "recent_surge_flag",
    "lead_time_hours",
    "delivery_mode",
    "data_source",
)
EVENT_FIELDS = (
    "event_id",
    "timestamp",
    "node_id",
    "district",
    "blood_group",
    "event_type",
    "units",
    "data_source",
)


@dataclass(frozen=True)
class District:
    node_id: str
    node_name: str
    node_type: str
    name: str
    code: str
    province: str
    admissions: float
    transit_hours: float


@dataclass
class Inventory:
    batches: list[list[int]] = field(default_factory=list)

    @property
    def units(self) -> int:
        return sum(batch[1] for batch in self.batches)

    def expire(self, step: int) -> int:
        expired = sum(units for expiry, units in self.batches if expiry <= step)
        self.batches = [batch for batch in self.batches if batch[0] > step and batch[1] > 0]
        return expired

    def receive(self, step: int, shelf_life_steps: int, units: int) -> None:
        if units > 0:
            self.batches.append([step + shelf_life_steps, units])

    def consume(self, units: int) -> int:
        remaining = units
        self.batches.sort(key=lambda batch: batch[0])
        for batch in self.batches:
            used = min(batch[1], remaining)
            batch[1] -= used
            remaining -= used
            if remaining == 0:
                break
        self.batches = [batch for batch in self.batches if batch[1] > 0]
        return remaining


@dataclass
class SimulationSummary:
    rows: int = 0
    demand_units: int = 0
    unmet_demand_units: int = 0
    supply_received_units: int = 0
    expired_units: int = 0
    demand_by_group: Counter[str] = field(default_factory=Counter)
    stockout_events_by_group: Counter[str] = field(default_factory=Counter)
    stockout_periods_by_group: Counter[str] = field(default_factory=Counter)
    closing_stock_by_group: Counter[str] = field(default_factory=Counter)
    periods_by_group: Counter[str] = field(default_factory=Counter)
    supply_by_group: Counter[str] = field(default_factory=Counter)
    expired_by_group: Counter[str] = field(default_factory=Counter)
    event_count: int = 0
    previous_stockout: dict[tuple[str, str], bool] = field(default_factory=dict)
    sha256: str = ""

    def as_dict(self, expected_shares: dict[str, float]) -> dict[str, Any]:
        demand_total = sum(self.demand_by_group.values())
        shares = {
            group: {
                "demand_units": self.demand_by_group[group],
                "simulated_share": (self.demand_by_group[group] / demand_total if demand_total else 0.0),
                "rbc_2024_share": expected_shares[group],
            }
            for group in GROUPS
        }
        group_inventory = {
            group: {
                "stockout_episodes": self.stockout_events_by_group[group],
                "district_group_periods": self.periods_by_group[group],
                "stockout_rate_pct": (
                    100 * self.stockout_periods_by_group[group] / self.periods_by_group[group]
                    if self.periods_by_group[group]
                    else 0.0
                ),
                "mean_closing_stock_units": (
                    self.closing_stock_by_group[group] / self.periods_by_group[group]
                    if self.periods_by_group[group]
                    else 0.0
                ),
                "supply_received_units": self.supply_by_group[group],
                "expired_units": self.expired_by_group[group],
                "expiry_pct_of_supply": (
                    100 * self.expired_by_group[group] / self.supply_by_group[group]
                    if self.supply_by_group[group]
                    else 0.0
                ),
            }
            for group in GROUPS
        }
        return {
            "rows": self.rows,
            "demand_units": self.demand_units,
            "unmet_demand_units": self.unmet_demand_units,
            "unmet_demand_pct": (
                100 * self.unmet_demand_units / self.demand_units if self.demand_units else 0.0
            ),
            "supply_received_units": self.supply_received_units,
            "expired_units": self.expired_units,
            "expiry_pct_of_supply": (
                100 * self.expired_units / self.supply_received_units
                if self.supply_received_units
                else 0.0
            ),
            "event_count": self.event_count,
            "stockout_events_by_group": {
                group: self.stockout_events_by_group[group] for group in GROUPS
            },
            "inventory_by_group": group_inventory,
            "demand_mix_by_group": shares,
            "sha256": self.sha256,
            "data_source": "simulated",
        }


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Simulation configuration must be a YAML mapping: {path}")
    return config


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_reference(reference_dir: Path, config: dict[str, Any]) -> tuple[list[District], dict[str, float]]:
    district_rows = _read_csv(reference_dir / "dim_district.csv")
    pair_rows = _read_csv(reference_dir / "district_pairs.csv")
    mix_rows = _read_csv(reference_dir / "rbc_2024_collected_by_group_rcbt.csv")

    pair_hours = {
        (row["from_district"], row["to_district"]): float(row["est_transit_hours"])
        for row in pair_rows
    }
    hubs = config["rcbt_hub_district"]
    rcbt_by_province = config["rcbt_by_province"]
    raw_nodes: list[District] = []
    for row in district_rows:
        name = row["district"]
        province = row["province"]
        try:
            hub = hubs[rcbt_by_province[province]]
            transit = 0.0 if name == hub else pair_hours[(name, hub)]
            admissions = float(row["hospital_admissions"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid district reference row for {name!r}: {error}") from error
        if admissions < 0:
            raise ValueError(f"Negative admissions for district {name}")
        raw_nodes.append(
            District(
                node_id=f"district:{row['adm2_pcode']}",
                node_name=f"District blood bank ({name})",
                node_type="district_blood_bank",
                name=name,
                code=row["adm2_pcode"],
                province=province,
                admissions=admissions,
                transit_hours=transit,
            )
        )

    if bool(config.get("add_kigali_referral_hospitals", False)):
        referral_rows = _read_csv(reference_dir / "kigali_referral_nodes.csv")
        districts_by_name = {district.name: district for district in raw_nodes}
        for row in referral_rows:
            if row["data_source"] != "real" or row["facility_type"] != "Referral Hospital":
                raise ValueError("Optional Kigali nodes must be official, real referral hospitals")
            district = districts_by_name.get(row["district"])
            if district is None or district.province != "Kigali City":
                raise ValueError(f"Invalid district for Kigali referral node: {row['district']}")
            if district.code != row["district_code"]:
                raise ValueError(f"District code mismatch for referral node {row['node_id']}")
            raw_nodes.append(
                District(
                    node_id=row["node_id"],
                    node_name=row["facility_name"],
                    node_type="kigali_referral_hospital",
                    name=district.name,
                    code=district.code,
                    province=district.province,
                    admissions=district.admissions,
                    transit_hours=district.transit_hours,
                )
            )
    nodes_per_district = Counter(node.name for node in raw_nodes)
    districts = [
        District(
            node_id=node.node_id,
            node_name=node.node_name,
            node_type=node.node_type,
            name=node.name,
            code=node.code,
            province=node.province,
            admissions=node.admissions / nodes_per_district[node.name],
            transit_hours=node.transit_hours,
        )
        for node in raw_nodes
    ]

    totals = {group: 0 for group in GROUPS}
    for row in mix_rows:
        if row["data_source"] != "real":
            raise ValueError("RBC group mix must be sourced from real RBC 2024 data")
        group = row["blood_group"]
        if group not in totals:
            raise ValueError(f"Unexpected blood group in RBC reference: {group}")
        totals[group] += int(row["units_collected"])
    collection_total = sum(totals.values())
    if collection_total <= 0:
        raise ValueError("RBC 2024 reference has no collected units")
    return districts, {group: totals[group] / collection_total for group in GROUPS}


def _poisson(rng: random.Random, mean: float) -> int:
    if mean <= 0:
        return 0
    threshold = math.exp(-mean)
    product = 1.0
    count = 0
    while product > threshold:
        count += 1
        product *= rng.random()
    return count - 1


def negative_binomial(rng: random.Random, mean: float, dispersion: float) -> int:
    """Sample NB2 via its Gamma-Poisson representation (variance = mean + mean²/r)."""
    if mean <= 0:
        return 0
    rate = rng.gammavariate(dispersion, mean / dispersion)
    return _poisson(rng, rate)


def _step_timestamp(start: date, index: int, frequency: str) -> datetime:
    initial = datetime.combine(start, time.min, tzinfo=timezone.utc)
    delta = timedelta(days=index + 1) if frequency == "daily" else timedelta(hours=index + 1)
    return initial + delta


def _ceil_units(units: float) -> int:
    return max(0, math.ceil(units - 1e-12))


def _baseline_lead_hours(district: District, config: dict[str, Any]) -> float:
    road = float(config["processing_delay_hours"]) + district.transit_hours
    drone = float(config["processing_delay_hours"]) + (
        district.transit_hours * float(config["drone_route_time_multiplier"])
    )
    share = float(config["drone_delivery_share"])
    return road * (1.0 - share) + drone * share


def _iter_readings(
    config: dict[str, Any],
    districts: list[District],
    group_shares: dict[str, float],
    seed: int,
    summary: SimulationSummary,
) -> Iterator[tuple[dict[str, Any], list[dict[str, Any]]]]:
    demand_rng = random.Random(seed)
    supply_rng = random.Random(seed ^ 0x51A7E)
    surge_rng = random.Random(seed ^ 0xA1F0)
    period_days = int(config["period_days"])
    frequency = config["frequency"]
    if frequency not in {"daily", "hourly"}:
        raise ValueError("frequency must be 'daily' or 'hourly'")
    steps_per_day = 1 if frequency == "daily" else 24
    periods = period_days * steps_per_day
    period_hours = 24 if frequency == "daily" else 1
    shelf_life_steps = int(config["shelf_life_days"]) * steps_per_day
    dispersion = float(config["negative_binomial_dispersion"])
    if dispersion <= 0:
        raise ValueError("negative_binomial_dispersion must be positive")
    admission_total = sum(district.admissions for district in districts)
    if admission_total <= 0:
        raise ValueError("District admissions must sum to a positive number")
    blend = float(config["district_weight_admissions_blend"])
    national_daily = float(config["national_daily_demand_units"])
    mean_by_cell: dict[tuple[str, str], float] = {}
    for district in districts:
        admission_share = district.admissions / admission_total
        district_weight = blend * admission_share + (1.0 - blend) / len(districts)
        for group in GROUPS:
            mean_by_cell[(district.node_id, group)] = (
                national_daily * district_weight * group_shares[group] / steps_per_day
            )

    inventory = {
        (district.node_id, group): Inventory()
        for district in districts
        for group in GROUPS
    }
    rare_threshold = float(config["rare_group_daily_mean_threshold_units"])
    default_rare_stock = int(config["rare_group_local_stock_units"])
    rare_stock_by_group = {
        str(group): int(units)
        for group, units in config.get("rare_group_local_stock_units_by_group", {}).items()
    }
    if (
        rare_threshold < 0
        or not 0 <= default_rare_stock <= 2
        or any(group not in GROUPS or not 0 <= units <= 2 for group, units in rare_stock_by_group.items())
    ):
        raise ValueError("Rare-group threshold must be non-negative and local stock must be 0-2 units")
    rare_stock_by_cell: dict[tuple[str, str], int] = {}
    rare_by_cell: dict[tuple[str, str], bool] = {}
    opening_stock: dict[tuple[str, str], int] = {}
    # Initial stock is a simulated opening inventory snapshot, not an observed reading.
    for district in districts:
        for group in GROUPS:
            mean = mean_by_cell[(district.node_id, group)]
            is_rare = mean * steps_per_day < rare_threshold
            rare_by_cell[(district.node_id, group)] = is_rare
            local_rare_stock = rare_stock_by_group.get(group, default_rare_stock)
            rare_stock_by_cell[(district.node_id, group)] = local_rare_stock
            opening = (
                local_rare_stock
                if is_rare
                else _ceil_units(mean * steps_per_day * float(config["initial_stock_cover_days"]))
            )
            opening_stock[(district.node_id, group)] = opening
            inventory[(district.node_id, group)].receive(0, shelf_life_steps, opening)

    arrivals: dict[int, list[tuple[tuple[str, str], int, float, str]]] = defaultdict(list)
    in_transit: Counter[tuple[str, str]] = Counter()
    rbc_by_province = config["rcbt_by_province"]
    weekly = config["weekly_seasonality"]
    weekday_names = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
    start = date.fromisoformat(str(config["start_date"]))
    surge_probability = float(config["surge_probability_per_district_day"])
    if not 0 <= surge_probability <= 1:
        raise ValueError("surge_probability_per_district_day must be between 0 and 1")
    event_counter = 0
    surge_additions: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)

    for step in range(periods):
        timestamp = _step_timestamp(start, step, frequency)
        day = timestamp.date()
        if frequency == "daily" or timestamp.hour == 0:
            for district in districts:
                if surge_rng.random() < surge_probability:
                    event_counter += 1
                    event_base = f"sim-{seed}-{event_counter:06d}"
                    event_type = surge_rng.choice(config["surge_types"])
                    target_step = step
                    if frequency == "hourly":
                        target_step += surge_rng.randrange(steps_per_day)
                    positive = surge_rng.randint(*config["surge_o_positive_units"])
                    negative = surge_rng.randint(*config["surge_o_negative_units"])
                    surge_additions[(district.node_id, "O+", target_step)].append(
                        {
                            "event_id": event_base + "-Oplus",
                            "timestamp": _step_timestamp(start, target_step, frequency).isoformat(),
                            "node_id": district.node_id,
                            "district": district.name,
                            "blood_group": "O+",
                            "event_type": event_type,
                            "units": positive,
                            "data_source": "simulated",
                        }
                    )
                    surge_additions[(district.node_id, "O-", target_step)].append(
                        {
                            "event_id": event_base + "-Ominus",
                            "timestamp": _step_timestamp(start, target_step, frequency).isoformat(),
                            "node_id": district.node_id,
                            "district": district.name,
                            "blood_group": "O-",
                            "event_type": event_type,
                            "units": negative,
                            "data_source": "simulated",
                        }
                    )

        seasonality = float(weekly[weekday_names[day.weekday()]])
        uplift_probability = float(config["rainfall_road_trauma_uplift_probability"])
        uplift_fraction = float(config["rainfall_road_trauma_uplift_fraction"])
        uplift_multiplier = (
            1.0 + uplift_fraction
            if uplift_probability > 0 and surge_rng.random() < uplift_probability
            else 1.0
        )

        due_by_cell: dict[tuple[str, str], list[tuple[int, float, str]]] = defaultdict(list)
        for incoming_key, units, lead_hours, mode in arrivals.pop(step, []):
            due_by_cell[incoming_key].append((units, lead_hours, mode))
            in_transit[incoming_key] -= units

        for district in districts:
            for group in GROUPS:
                key = (district.node_id, group)
                store = inventory[key]
                expired = store.expire(step)
                received = opening_stock[key] if step == 0 else 0
                actual_lead = _baseline_lead_hours(district, config)
                actual_mode = "opening" if received else "none"
                for units, lead_hours, mode in due_by_cell[key]:
                    store.receive(step, shelf_life_steps, units)
                    received += units
                    actual_lead = lead_hours
                    actual_mode = mode

                stock_at_open = store.units
                mean = mean_by_cell[key] * seasonality * uplift_multiplier
                base_demand = negative_binomial(demand_rng, mean, dispersion)
                events = surge_additions.pop((district.node_id, group, step), [])
                surge_units = sum(int(event["units"]) for event in events)
                demand = base_demand + surge_units
                unmet = store.consume(demand)
                summary.rows += 1
                summary.demand_units += demand
                summary.unmet_demand_units += unmet
                summary.supply_received_units += received
                summary.expired_units += expired
                summary.demand_by_group[group] += demand
                summary.closing_stock_by_group[group] += store.units
                summary.periods_by_group[group] += 1
                summary.supply_by_group[group] += received
                summary.expired_by_group[group] += expired
                state_key = (district.node_id, group)
                had_stockout = unmet > 0
                if had_stockout:
                    summary.stockout_periods_by_group[group] += 1
                if had_stockout and not summary.previous_stockout.get(state_key, False):
                    summary.stockout_events_by_group[group] += 1
                summary.previous_stockout[state_key] = had_stockout
                summary.event_count += len(events)

                row = {
                    "timestamp": timestamp.isoformat(),
                    "node_id": district.node_id,
                    "node_name": district.node_name,
                    "node_type": district.node_type,
                    "district": district.name,
                    "district_code": district.code,
                    "province": district.province,
                    "blood_group": group,
                    "stock_open_units": stock_at_open,
                    "stock_units": store.units,
                    "expiring_units_7d": sum(
                        units
                        for expiry_step, units in store.batches
                        if step < expiry_step <= step + 7 * steps_per_day
                    ),
                    "demand_units": demand,
                    "supply_received": received,
                    "expired_units": expired,
                    "unmet_demand_units": unmet,
                    "recent_surge_flag": int(bool(events)),
                    "lead_time_hours": round(actual_lead, 3),
                    "delivery_mode": actual_mode,
                    "data_source": "simulated",
                }
                yield row, events

                inventory_position = store.units + in_transit[key]
                if rare_by_cell[key]:
                    reorder_point = rare_stock_by_cell[key]
                    target = rare_stock_by_cell[key]
                else:
                    reorder_point = _ceil_units(
                        mean_by_cell[key]
                        * steps_per_day
                        * float(config["reorder_point_cover_days"])
                    )
                    target = _ceil_units(
                        mean_by_cell[key]
                        * steps_per_day
                        * float(config["reorder_up_to_cover_days"])
                    )
                on_demand_rare_order = (
                    rare_by_cell[key]
                    and rare_stock_by_cell[key] == 0
                    and unmet > 0
                    and inventory_position == 0
                )
                if (
                    inventory_position < reorder_point and inventory_position < target
                ) or on_demand_rare_order:
                    order_units = (
                        max(target - inventory_position, 1)
                        if rare_by_cell[key] and target == 0
                        else target - inventory_position
                    )
                    noise = float(config["reorder_quantity_noise_fraction"])
                    if noise > 0:
                        order_units = max(
                            1, round(order_units * supply_rng.uniform(1 - noise, 1 + noise))
                        )
                    drone = supply_rng.random() < float(config["drone_delivery_share"])
                    mode = "drone" if drone else "road"
                    lead_hours = float(config["processing_delay_hours"]) + district.transit_hours * (
                        float(config["drone_route_time_multiplier"]) if drone else 1.0
                    )
                    delay = max(1, math.ceil(lead_hours / period_hours))
                    arrival_step = step + delay
                    if arrival_step < periods:
                        arrivals[arrival_step].append((key, order_units, lead_hours, mode))
                        in_transit[key] += order_units


def run_simulation(
    config: dict[str, Any],
    reference_dir: Path,
    output_dir: Path,
    seed: int | None = None,
) -> dict[str, Any]:
    districts, group_shares = load_reference(reference_dir, config)
    effective_seed = int(config["seed"] if seed is None else seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    stock_path = output_dir / "simulation_daily.csv"
    if config["frequency"] == "hourly":
        stock_path = output_dir / "simulation_hourly.csv"
    events_path = output_dir / "demand_events.csv"
    summary = SimulationSummary()
    digest = hashlib.sha256()
    with stock_path.open("w", encoding="utf-8", newline="") as stock_stream, events_path.open(
        "w", encoding="utf-8", newline=""
    ) as events_stream:
        stock_writer = csv.DictWriter(stock_stream, fieldnames=STOCK_FIELDS)
        event_writer = csv.DictWriter(events_stream, fieldnames=EVENT_FIELDS)
        stock_writer.writeheader()
        event_writer.writeheader()
        for row, events in _iter_readings(config, districts, group_shares, effective_seed, summary):
            stock_writer.writerow(row)
            digest.update((",".join(str(row[name]) for name in STOCK_FIELDS) + "\n").encode("utf-8"))
            if events:
                event_writer.writerows(events)
    summary.sha256 = digest.hexdigest()
    report = summary.as_dict(group_shares)
    report.update(
        {
            "seed": effective_seed,
            "frequency": config["frequency"],
            "period_days": int(config["period_days"]),
            "district_count": len({district.name for district in districts}),
            "node_count": len(districts),
            "blood_group_count": len(GROUPS),
            "stock_readings_file": stock_path.name,
            "demand_events_file": events_path.name,
        }
    )
    (output_dir / "simulation_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def measure_simulation(
    config: dict[str, Any], reference_dir: Path, seed: int
) -> dict[str, Any]:
    """Measure a candidate calibration without writing the large row files."""
    districts, group_shares = load_reference(reference_dir, config)
    summary = SimulationSummary()
    for _ in _iter_readings(config, districts, group_shares, seed, summary):
        pass
    return summary.as_dict(group_shares)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate seeded simulated blood inventory readings.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--days", type=int, help="Override the configured simulation length.")
    parser.add_argument("--frequency", choices=("daily", "hourly"))
    parser.add_argument(
        "--include-kigali-referral-hospitals",
        action="store_true",
        help="Add the four Ministry-listed Kigali referral hospitals as additional nodes.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.days is not None:
        if args.days <= 0:
            parser.error("--days must be positive")
        config["period_days"] = args.days
    if args.frequency:
        config["frequency"] = args.frequency
    if args.include_kigali_referral_hospitals:
        config["add_kigali_referral_hospitals"] = True
    report = run_simulation(config, args.reference_dir, args.output_dir, args.seed)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
