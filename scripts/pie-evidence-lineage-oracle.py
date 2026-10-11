#!/usr/bin/env python3
"""PIE-009B independent campaign evidence-lineage oracle.

Freezes one evidence bundle, then derives pessimistic/nominal/optimistic
scenario values from that same bundle. Synthetic fixtures only.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EvidenceRecord:
    record_id: str
    family_id: str
    version: str
    process_id: str
    parameter_id: str
    low: float
    nominal: float
    high: float
    superseded_by: str | None = None


@dataclass(frozen=True)
class FrozenBundle:
    bundle_id: str
    records: tuple[EvidenceRecord, ...]


@dataclass(frozen=True)
class ScenarioValue:
    record_id: str
    family_id: str
    version: str
    process_id: str
    parameter_id: str
    mode: str
    value: float


def validate_record(record: EvidenceRecord) -> None:
    if not all(
        (
            record.record_id,
            record.family_id,
            record.version,
            record.process_id,
            record.parameter_id,
        )
    ):
        raise ValueError("evidence identifiers must be non-empty")
    if not all(math.isfinite(x) for x in (record.low, record.nominal, record.high)):
        raise ValueError("evidence values must be finite")
    if record.low > record.nominal or record.nominal > record.high:
        raise ValueError("evidence envelope must satisfy low <= nominal <= high")
    if record.superseded_by == record.record_id:
        raise ValueError("record cannot supersede itself")


def freeze_bundle(bundle_id: str, records: list[EvidenceRecord]) -> FrozenBundle:
    if not bundle_id:
        raise ValueError("bundle_id required")
    if not records:
        raise ValueError("bundle cannot be empty")

    ids: set[str] = set()
    slots: set[tuple[str, str]] = set()
    family_versions: dict[str, str] = {}

    for record in records:
        validate_record(record)
        if record.record_id in ids:
            raise ValueError("duplicate evidence record ID")
        ids.add(record.record_id)

        slot = (record.process_id, record.parameter_id)
        if slot in slots:
            raise ValueError("duplicate active process/parameter slot")
        slots.add(slot)

        if (
            record.family_id in family_versions
            and family_versions[record.family_id] != record.version
        ):
            raise ValueError("mixed versions from one evidence family")
        family_versions[record.family_id] = record.version

    for record in records:
        if record.superseded_by is not None:
            raise ValueError("superseded records cannot be active in a frozen bundle")

    return FrozenBundle(bundle_id, tuple(sorted(records, key=lambda x: x.record_id)))


def scenario(bundle: FrozenBundle, mode: str) -> tuple[ScenarioValue, ...]:
    if mode not in {"pessimistic", "nominal", "optimistic"}:
        raise ValueError("invalid scenario mode")

    values: list[ScenarioValue] = []
    for record in bundle.records:
        value = {
            "pessimistic": record.low,
            "nominal": record.nominal,
            "optimistic": record.high,
        }[mode]
        values.append(
            ScenarioValue(
                record.record_id,
                record.family_id,
                record.version,
                record.process_id,
                record.parameter_id,
                mode,
                value,
            )
        )
    return tuple(values)


def same_lineage(
    a: tuple[ScenarioValue, ...], b: tuple[ScenarioValue, ...]
) -> bool:
    sig_a = [
        (x.record_id, x.family_id, x.version, x.process_id, x.parameter_id) for x in a
    ]
    sig_b = [
        (x.record_id, x.family_id, x.version, x.process_id, x.parameter_id) for x in b
    ]
    return sig_a == sig_b


def self_test() -> None:
    records = [
        EvidenceRecord(
            "r1", "family-a", "v2", "excavation", "throughput", 8, 10, 12
        ),
        EvidenceRecord("r2", "family-b", "v1", "refining", "yield", 0.55, 0.65, 0.72),
        EvidenceRecord(
            "r3", "family-c", "2026-09", "power", "availability", 0.80, 0.90, 0.96
        ),
    ]
    bundle = freeze_bundle("bundle-001", records)
    pessimistic = scenario(bundle, "pessimistic")
    nominal = scenario(bundle, "nominal")
    optimistic = scenario(bundle, "optimistic")

    assert same_lineage(pessimistic, nominal)
    assert same_lineage(nominal, optimistic)
    assert [x.value for x in pessimistic] == [8, 0.55, 0.80]
    assert [x.value for x in nominal] == [10, 0.65, 0.90]
    assert [x.value for x in optimistic] == [12, 0.72, 0.96]

    try:
        freeze_bundle(
            "bad-slot",
            records
            + [
                EvidenceRecord(
                    "r4", "family-x", "v1", "refining", "yield", 0.70, 0.75, 0.80
                )
            ],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate active parameter slot must fail")

    try:
        freeze_bundle(
            "bad-version",
            [
                EvidenceRecord("a1", "same-family", "v1", "p1", "x", 1, 2, 3),
                EvidenceRecord("a2", "same-family", "v2", "p2", "y", 1, 2, 3),
            ],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("mixed family versions must fail")

    try:
        freeze_bundle(
            "bad-superseded",
            [EvidenceRecord("old", "fam", "v1", "p", "x", 1, 2, 3, superseded_by="new")],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("superseded active record must fail")

    new_bundle = freeze_bundle(
        "bundle-002",
        [
            EvidenceRecord(
                "r1-new", "family-a", "v3", "excavation", "throughput", 9, 11, 13
            ),
            records[1],
            records[2],
        ],
    )
    assert bundle.bundle_id != new_bundle.bundle_id
    assert not same_lineage(
        scenario(bundle, "nominal"), scenario(new_bundle, "nominal")
    )

    try:
        freeze_bundle(
            "bad-range", [EvidenceRecord("z", "f", "v1", "p", "x", 3, 2, 1)]
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid envelope must fail")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("ok")
        return

    bundle = freeze_bundle(
        "example",
        [
            EvidenceRecord(
                "r1", "family-a", "v2", "excavation", "throughput", 8, 10, 12
            ),
            EvidenceRecord(
                "r2", "family-b", "v1", "refining", "yield", 0.55, 0.65, 0.72
            ),
        ],
    )
    print(
        json.dumps(
            {
                mode: [asdict(x) for x in scenario(bundle, mode)]
                for mode in ("pessimistic", "nominal", "optimistic")
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
