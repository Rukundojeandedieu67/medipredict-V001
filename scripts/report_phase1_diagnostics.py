"""Write per-district and pooled per-group Phase 1 inventory diagnostics."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from medipredict.simulator import DEFAULT_OUTPUT, GROUPS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "docs" / "PHASE1_GROUP_DIAGNOSTICS.md"


def diagnostics(stock_path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    cells: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"days": 0, "stock": 0, "stockout_days": 0, "supply": 0, "expired": 0}
    )
    with stock_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            cell = cells[(row["district"], row["blood_group"])]
            cell["days"] += 1
            cell["stock"] += int(row["stock_units"])
            cell["stockout_days"] += int(int(row["unmet_demand_units"]) > 0)
            cell["supply"] += int(row["supply_received"])
            cell["expired"] += int(row["expired_units"])

    table = []
    group_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"days": 0, "stock": 0, "stockout_days": 0, "supply": 0, "expired": 0}
    )
    for district, group in sorted(cells):
        cell = cells[(district, group)]
        group_totals[group] = {
            key: group_totals[group].get(key, 0) + value for key, value in cell.items()
        }
        table.append(
            {
                "district": district,
                "blood_group": group,
                "district_days": int(cell["days"]),
                "stockout_days": int(cell["stockout_days"]),
                "stockout_rate_pct": 100 * cell["stockout_days"] / cell["days"],
                "mean_stock_units": cell["stock"] / cell["days"],
                "supply_units": int(cell["supply"]),
                "expired_units": int(cell["expired"]),
                "expiry_pct_supply": (
                    100 * cell["expired"] / cell["supply"] if cell["supply"] else 0.0
                ),
            }
        )
    pooled = {}
    for group in GROUPS:
        values = group_totals[group]
        pooled[group] = {
            "district_days": int(values["days"]),
            "stockout_days": int(values["stockout_days"]),
            "stockout_rate_pct": (
                100 * values["stockout_days"] / values["days"] if values["days"] else 0.0
            ),
            "mean_stock_units": values["stock"] / values["days"] if values["days"] else 0.0,
            "supply_units": int(values["supply"]),
            "expired_units": int(values["expired"]),
            "expiry_pct_supply": (
                100 * values["expired"] / values["supply"] if values["supply"] else 0.0
            ),
        }
    return table, pooled


def write_report(stock_path: Path, report_path: Path) -> dict[str, dict[str, float]]:
    table, pooled = diagnostics(stock_path)
    relative_input = Path(os.path.relpath(stock_path, report_path.parent)).as_posix()
    flags = []
    for group, values in pooled.items():
        if values["expiry_pct_supply"] > 10:
            flags.append(f"{group}: expiry above 10% ({values['expiry_pct_supply']:.3f}%)")
        if values["stockout_days"] == 0:
            flags.append(f"{group}: zero stockouts")
    lines = [
        "# Phase 1 group and district diagnostics",
        "",
        f"Input: `{relative_input}`. All figures below are computed from simulated data; each `district_days` count is one district/group/day row.",
        "",
        "## Pooled by blood group",
        "",
        "| Blood group | District-days | Stockout days | Stockout rate | Mean closing stock | Supply | Expired | Expiry / supply | Flags |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for group, values in pooled.items():
        group_flags = []
        if values["expiry_pct_supply"] > 10:
            group_flags.append("expiry >10%")
        if values["stockout_days"] == 0:
            group_flags.append("zero stockouts")
        lines.append(
            f"| {group} | {values['district_days']:,} | {values['stockout_days']:,} | "
            f"{values['stockout_rate_pct']:.3f}% | {values['mean_stock_units']:.3f} | "
            f"{values['supply_units']:,} | {values['expired_units']:,} | "
            f"{values['expiry_pct_supply']:.3f}% | {', '.join(group_flags) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Per district and blood group",
            "",
            "| District | Blood group | District-days | Stockout days | Stockout rate | Mean closing stock | Supply | Expired | Expiry / supply |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in table:
        lines.append(
            f"| {row['district']} | {row['blood_group']} | {row['district_days']:,} | "
            f"{row['stockout_days']:,} | {row['stockout_rate_pct']:.3f}% | "
            f"{row['mean_stock_units']:.3f} | {row['supply_units']:,} | "
            f"{row['expired_units']:,} | {row['expiry_pct_supply']:.3f}% |"
        )
    lines.extend(
        [
            "",
            "## Flags",
            "",
        ]
    )
    lines.extend([f"- {flag}" for flag in flags] or ["- No pooled group exceeded the specified flag conditions."])
    lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return pooled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--input", type=Path, help="Specific daily simulation CSV.")
    args = parser.parse_args()
    stock_path = args.input or args.output_dir / "simulation_daily.csv"
    pooled = write_report(stock_path, args.report)
    print(f"Report: {args.report}")
    for group, values in pooled.items():
        flags = []
        if values["expiry_pct_supply"] > 10:
            flags.append("expiry >10%")
        if values["stockout_days"] == 0:
            flags.append("zero stockouts")
        print(
            f"{group}: stockout_rate={values['stockout_rate_pct']:.3f}% "
            f"mean_stock={values['mean_stock_units']:.3f} "
            f"expiry={values['expiry_pct_supply']:.3f}% "
            f"flags={', '.join(flags) if flags else 'none'}"
        )


if __name__ == "__main__":
    main()
