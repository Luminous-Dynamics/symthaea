from __future__ import annotations
import json
from pathlib import Path
from causal_reflex_benchmark import compare, from_dict

ROOT = Path(__file__).resolve().parent
raw = json.loads((ROOT / "scenarios.json").read_text())
scenarios = [from_dict(item) for item in raw["scenarios"]]
report = compare(scenarios)
(ROOT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
