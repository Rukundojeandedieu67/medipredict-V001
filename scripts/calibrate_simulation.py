"""Calibrate the rare-group policy against all-seed network targets."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from medipredict.simulator import DEFAULT_CONFIG, DEFAULT_REFERENCE, GROUPS, load_config
from medipredict.simulator import measure_simulation


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "CALIBRATION.md"
DELAYS = (6, 12, 24, 48)
RARE_STOCK_LEVELS = (0, 1, 2)
UNMET_TARGET = 0.28
UNMET_TOLERANCE = 0.1
EXPIRY_MAX = 10.0
MIN_EPISODES_PER_GROUP = 30


def _aggregate(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    demand = sum(item["demand_units"] for item in metrics)
    unmet = sum(item["unmet_demand_units"] for item in metrics)
    supply = sum(item["supply_received_units"] for item in metrics)
    expired = sum(item["expired_units"] for item in metrics)
    groups = {}
    for group in GROUPS:
        rows = [item["inventory_by_group"][group] for item in metrics]
        periods = sum(row["district_group_periods"] for row in rows)
        group_supply = sum(row["supply_received_units"] for row in rows)
        group_expired = sum(row["expired_units"] for row in rows)
        groups[group] = {
            "stockout_episodes": sum(row["stockout_episodes"] for row in rows),
            "stockout_periods": round(
                sum(
                    row["stockout_rate_pct"] * row["district_group_periods"] / 100
                    for row in rows
                )
            ),
            "stockout_rate_pct": (
                100
                * sum(
                    row["stockout_rate_pct"] * row["district_group_periods"] / 100
                    for row in rows
                )
                / periods
            ),
            "mean_closing_stock_units": sum(
                row["mean_closing_stock_units"] * row["district_group_periods"]
                for row in rows
            )
            / periods,
            "supply_received_units": group_supply,
            "expired_units": group_expired,
            "expiry_pct_of_supply": (
                100 * group_expired / group_supply if group_supply else 0.0
            ),
        }
    return {
        "seed_count": len(metrics),
        "demand_units": demand,
        "unmet_demand_units": unmet,
        "unmet_demand_pct": 100 * unmet / demand if demand else 0.0,
        "supply_received_units": supply,
        "expired_units": expired,
        "expiry_pct_of_supply": 100 * expired / supply if supply else 0.0,
        "inventory_by_group": groups,
    }


def _score(metrics: dict[str, Any]) -> tuple[bool, float]:
    unmet_pct = metrics["unmet_demand_pct"]
    groups = metrics["inventory_by_group"]
    unmet_ok = abs(unmet_pct - UNMET_TARGET) <= UNMET_TOLERANCE
    expiry_ok = all(row["expiry_pct_of_supply"] < EXPIRY_MAX for row in groups.values())
    episodes_ok = all(
        row["stockout_episodes"] >= MIN_EPISODES_PER_GROUP for row in groups.values()
    )
    score = abs(unmet_pct - UNMET_TARGET) / UNMET_TOLERANCE
    score += sum(
        max(0.0, row["expiry_pct_of_supply"] - EXPIRY_MAX) / EXPIRY_MAX
        for row in groups.values()
    )
    score += sum(
        max(0, MIN_EPISODES_PER_GROUP - row["stockout_episodes"])
        / MIN_EPISODES_PER_GROUP
        for row in groups.values()
    )
    return unmet_ok and expiry_ok and episodes_ok, score


def _candidate_summary(
    config: dict[str, Any],
    reference_dir: Path,
    stock_level: int,
    delay: int,
    seeds: range,
) -> dict[str, Any]:
    candidate = dict(config)
    candidate["rare_group_local_stock_units"] = stock_level
    candidate["processing_delay_hours"] = delay
    return _aggregate(
        [measure_simulation(candidate, reference_dir, seed) for seed in seeds]
    )


def _markdown(
    seed_start: int,
    seed_count: int,
    candidates: list[dict[str, Any]],
    selected: dict[str, Any],
) -> str:
    lines = [
        "# Simulator calibration report",
        "",
        "All values are computed from simulated data and are not observed Rwanda inventory outcomes.",
        "",
        f"- Seed runs: `{seed_count}` sequential seeds beginning at `{seed_start}`.",
        "- Rare-group threshold: node/group mean daily demand below 0.5 units.",
        f"- Selected policy: rare local target `{selected['rare_stock']}` units, processing delay `{selected['delay']}` hours; common groups retain cover-day ordering.",
        f"- National unmet demand: `{selected['metrics']['unmet_demand_pct']:.3f}%` (target 0.18%-0.38%).",
        f"- Combined expiry: `{selected['metrics']['expiry_pct_of_supply']:.3f}%`; per-group limit is strictly below 10%.",
        "",
        "## Recalibration search",
        "",
        "| Rare local target | Processing delay (h) | Unmet demand % | O+ expiry | O- expiry | A+ expiry | A- expiry | B+ expiry | B- expiry | AB+ expiry | AB- expiry | Min episodes/group | Meets all targets |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for candidate in candidates:
        groups = candidate["metrics"]["inventory_by_group"]
        min_episodes = min(row["stockout_episodes"] for row in groups.values())
        lines.append(
            f"| {candidate['rare_stock']} | {candidate['delay']} | "
            f"{candidate['metrics']['unmet_demand_pct']:.3f} | "
            + " | ".join(f"{groups[group]['expiry_pct_of_supply']:.2f}%" for group in GROUPS)
            + f" | {min_episodes} | {'yes' if candidate['meets_targets'] else 'no'} |"
        )
    sensitivity = sorted(
        (
            candidate
            for candidate in candidates
            if candidate["rare_stock"] == selected["rare_stock"]
        ),
        key=lambda candidate: candidate["delay"],
    )
    lines.extend(
        [
            "",
            f"## Processing-delay sensitivity at rare local target {selected['rare_stock']} units",
            "",
            "All entries aggregate the same 20 configured seeds (or the seed count above) and vary only processing delay. Daily simulation rounds total route plus processing lead time up to whole daily steps, so sub-day delay changes can be muted where they do not cross a day boundary.",
            "",
            "| Processing delay (h) | Unmet demand | Combined expiry | "
            + " | ".join(f"{group} expiry" for group in GROUPS)
            + " | Min stockout episodes/group |",
            "|---:|---:|---:|"
            + "---:|" * (len(GROUPS) + 1),
        ]
    )
    for candidate in sensitivity:
        groups = candidate["metrics"]["inventory_by_group"]
        lines.append(
            f"| {candidate['delay']} | {candidate['metrics']['unmet_demand_pct']:.3f}% | "
            f"{candidate['metrics']['expiry_pct_of_supply']:.3f}% | "
            + " | ".join(f"{groups[group]['expiry_pct_of_supply']:.2f}%" for group in GROUPS)
            + f" | {min(row['stockout_episodes'] for row in groups.values())} |"
        )
    lines.extend(
        [
            "",
            "## Selected policy results by blood group",
            "",
            "| Group | Stockout episodes | Stockout rate | Mean closing stock | Supply | Expired | Expiry / supply |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group, values in selected["metrics"]["inventory_by_group"].items():
        lines.append(
            f"| {group} | {values['stockout_episodes']:,} | "
            f"{values['stockout_rate_pct']:.3f}% | "
            f"{values['mean_closing_stock_units']:.3f} | "
            f"{values['supply_received_units']:,} | {values['expired_units']:,} | "
            f"{values['expiry_pct_of_supply']:.3f}% |"
        )
    unmet_ok = abs(selected["metrics"]["unmet_demand_pct"] - UNMET_TARGET) <= UNMET_TOLERANCE
    expiry_ok = all(
        row["expiry_pct_of_supply"] < EXPIRY_MAX
        for row in selected["metrics"]["inventory_by_group"].values()
    )
    episodes_ok = all(
        row["stockout_episodes"] >= MIN_EPISODES_PER_GROUP
        for row in selected["metrics"]["inventory_by_group"].values()
    )
    lines.extend(
        [
            "",
            "## Acceptance",
            "",
            f"- National unmet demand within 0.28% ±0.1 percentage point: **{'PASS' if unmet_ok else 'FAIL'}**.",
            f"- Every group's expiry below 10%: **{'PASS' if expiry_ok else 'FAIL'}**.",
            f"- At least 30 stockout episodes per group across these seeds: **{'PASS' if episodes_ok else 'FAIL'}**.",
            "",
        ]
    )
    if not (unmet_ok and expiry_ok and episodes_ok):
        lines.append(
            "No candidate in this bounded search satisfies all three constraints simultaneously. The selected candidate minimizes the documented normalized unmet-demand, expiry-excess, and episode-deficit score; failed targets are reported rather than hidden."
        )
        lines.append("")
    return "\n".join(lines)


def calibrate(
    config_path: Path, reference_dir: Path, seed_start: int, seed_count: int
) -> dict[str, Any]:
    if seed_count < 1:
        raise ValueError("seed_count must be positive")
    config = load_config(config_path)
    seeds = range(seed_start, seed_start + seed_count)
    candidates = []
    for stock_level in RARE_STOCK_LEVELS:
        for delay in DELAYS:
            metrics = _candidate_summary(
                config, reference_dir, stock_level, delay, seeds
            )
            meets_targets, score = _score(metrics)
            candidates.append(
                {
                    "rare_stock": stock_level,
                    "delay": delay,
                    "metrics": metrics,
                    "meets_targets": meets_targets,
                    "score": score,
                }
            )
            print(
                f"rare_stock={stock_level} delay={delay}h "
                f"unmet={metrics['unmet_demand_pct']:.3f}% "
                f"max_expiry={max(row['expiry_pct_of_supply'] for row in metrics['inventory_by_group'].values()):.2f}% "
                f"min_episodes={min(row['stockout_episodes'] for row in metrics['inventory_by_group'].values())}"
            )
    selected = min(
        (candidate for candidate in candidates if candidate["meets_targets"]),
        key=lambda candidate: candidate["score"],
        default=min(candidates, key=lambda candidate: candidate["score"]),
    )
    config["rare_group_local_stock_units"] = selected["rare_stock"]
    config["processing_delay_hours"] = float(selected["delay"])
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    REPORT_PATH.write_text(
        _markdown(seed_start, seed_count, candidates, selected), encoding="utf-8"
    )
    return {"candidates": candidates, "selected": selected}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--seed-start", type=int, default=2026)
    parser.add_argument("--seed-count", type=int, default=20)
    args = parser.parse_args()
    result = calibrate(
        args.config, args.reference_dir, args.seed_start, args.seed_count
    )
    selected = result["selected"]
    print(
        f"Selected rare stock={selected['rare_stock']} delay={selected['delay']}h; "
        f"all_targets={selected['meets_targets']}"
    )
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
