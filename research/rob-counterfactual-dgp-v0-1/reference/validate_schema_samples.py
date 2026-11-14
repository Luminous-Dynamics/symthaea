from __future__ import annotations
import json
from pathlib import Path
import jsonschema

ROOT = Path(__file__).resolve().parent
SCHEMAS = {
    "observed": json.loads((ROOT / "observed.schema.json").read_text(encoding="utf-8")),
    "oracle": json.loads((ROOT / "oracle.schema.json").read_text(encoding="utf-8")),
}
counts = {"observed": 0, "oracle": 0}
for path in sorted((ROOT / "generated").glob("*.jsonl")):
    kind = "oracle" if ".oracle-only." in path.name else "observed"
    schema = SCHEMAS[kind]
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                jsonschema.validate(json.loads(line), schema)
            except Exception as exc:  # noqa: BLE001 - keep the failing artifact context
                raise SystemExit(f"{path.name}:{line_number}: {exc}") from exc
            counts[kind] += 1
if not counts["observed"] or not counts["oracle"]:
    raise SystemExit(f"expected both artifact types, got {counts}")
print(f"JSON_SCHEMA_VALIDATION_PASS observed={counts['observed']} oracle={counts['oracle']}")
