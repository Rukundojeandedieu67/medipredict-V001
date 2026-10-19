"""Generate a reproducible collection of independent simulated seed runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from medipredict.simulator import DEFAULT_CONFIG, DEFAULT_OUTPUT, DEFAULT_REFERENCE
from medipredict.simulator import load_config, run_simulation


def generate(
    count: int,
    start_seed: int,
    config_path: Path,
    reference_dir: Path,
    output_dir: Path,
) -> list[dict[str, object]]:
    if count < 1:
        raise ValueError("count must be at least 1")
    config = load_config(config_path)
    summaries = []
    for index in range(count):
        seed = start_seed + index
        destination = output_dir / f"seed_{index:02d}"
        summary = run_simulation(config, reference_dir, destination, seed=seed)
        summaries.append(summary)
        print(
            f"{destination}: seed={seed}, rows={summary['rows']:,}, "
            f"unmet={summary['unmet_demand_pct']:.3f}%"
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--start-seed", type=int, default=2026)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summaries = generate(
        args.count,
        args.start_seed,
        args.config,
        args.reference_dir,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "seed_count": len(summaries),
                "seeds": [summary["seed"] for summary in summaries],
                "data_source": "simulated",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
