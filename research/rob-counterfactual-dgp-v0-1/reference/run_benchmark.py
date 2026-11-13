from __future__ import annotations
import argparse
import json
from pathlib import Path
from dgp import DGPConfig, run_all

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the synthetic MCR² v17 known-truth DGP")
    parser.add_argument("--blocks", type=int, default=10_000, help="matched blocks per scenario (even, >=2)")
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--write-data", action="store_true", help="write full observed and oracle JSONL datasets")
    parser.add_argument("--output", type=Path, default=ROOT / "report.json")
    args = parser.parse_args()
    report, datasets = run_all(DGPConfig(seed=args.seed, blocks=args.blocks))
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.write_data:
        generated = args.output.parent / "generated"
        generated.mkdir(parents=True, exist_ok=True)
        for scenario, (observed, oracle) in datasets.items():
            (generated / f"{scenario}.observed.jsonl").write_text(
                "".join(json.dumps(x, sort_keys=True) + "\n" for x in observed), encoding="utf-8"
            )
            (generated / f"{scenario}.oracle-only.jsonl").write_text(
                "".join(json.dumps(x, sort_keys=True) + "\n" for x in oracle), encoding="utf-8"
            )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
