#!/usr/bin/env python3
"""SPINE-000B-N2 integrated/lowering-domain classifier.

Measurement-only. Validates canonical IntegratedBits and the deterministic scalar
operands that Phase C would lower across its application boundary. It does not
claim that an application executed or that any subsystem caused a state change.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

REGISTRY = Path("docs/research/SPINE_000B_PHASE_C_APPLICATION_REGISTRY_V1.json")
I1_VERIFIER = Path("scripts/verify_spine_000b_observer_noninterference_v1.py")
REGISTRY_SCHEMA = "symthaea.spine.000b.phase-c-application-registry.v1"

FIELDS = ("confidence_delta", "lr_modulation", "exploration_delta", "arousal_delta", "valence_delta")
FIELD_BITS = {name: 1 << i for i, name in enumerate(FIELDS)}
EXPECTED_LOWERING = {
    "CONFIDENCE_DELTA": ("confidence_delta", "F32_BITS", "integrated.confidence_delta as f32", 0.0),
    "LR_MODULATION": ("lr_modulation", "F32_BITS", "integrated.lr_modulation as f32", 1.0),
    "EXPLORATION_DELTA": ("exploration_delta", "F32_BITS", "integrated.exploration_delta as f32", 0.0),
    "AROUSAL_DELTA": ("arousal_delta", "F32_BITS", "integrated.arousal_delta", 0.0),
    "VALENCE_DELTA": ("valence_delta", "F32_BITS", "integrated.valence_delta", 0.0),
}
CLASSES = {
    "INVALID_INTEGRATED_NONFINITE",
    "INVALID_LOWERED_NONFINITE",
    "INVALID_CONTRIBUTOR_DOMAIN",
    "UNSUPPORTED_INTEGRATED_FLAGS",
    "INVALID_EMPTY_COLLECTOR_IDENTITY",
    "QUALIFIED_IDENTITY",
    "QUALIFIED_APPLICATION_FINITE",
}


def f64_from_bits(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def f64_bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def rust_f64_to_f32(value: float) -> float:
    try:
        return f32_from_bits(f32_bits(value))
    except OverflowError:
        return math.copysign(math.inf, value)


def rust_f64_to_f32_bits(value: float) -> int:
    return f32_bits(rust_f64_to_f32(value))


def rust_f32_add(a: float, b: float) -> float:
    return rust_f64_to_f32(float(a) + float(b))


def observer_capacity() -> int:
    text = I1_VERIFIER.read_text(encoding="utf-8")
    match = re.search(r"^CAPACITY\s*=\s*(\d+)\s*$", text, re.MULTILINE)
    if not match:
        raise ValueError("could not bind I1 Stage-A capacity")
    value = int(match.group(1))
    if value <= 0 or value > 0xFFFF_FFFF:
        raise ValueError("invalid I1 Stage-A capacity")
    return value


def registry_contract() -> tuple[int, dict[str, tuple[str, str, str, float]]]:
    raw = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if raw.get("schema") != REGISTRY_SCHEMA:
        raise ValueError("Phase-C registry schema drifted")
    scalars = {str(item["source_tag"]): item for item in raw.get("scalar_sources", [])}
    if set(scalars) != set(EXPECTED_LOWERING):
        raise ValueError("scalar source set drifted")
    for tag, (field, kind, derivation, _identity) in EXPECTED_LOWERING.items():
        source = scalars[tag]
        if source.get("field") != field or source.get("condition") != "SCALAR_NON_IDENTITY":
            raise ValueError(f"source mapping drifted for {tag}")
        apps = source.get("applications")
        if not isinstance(apps, list) or len(apps) != 1:
            raise ValueError(f"v1 expects one scalar application for {tag}")
        app = apps[0]
        if app.get("applied_argument_kind") != kind or app.get("applied_argument_derivation") != derivation:
            raise ValueError(f"lowering rule drifted for {tag}")
    mask = 0
    for entry in raw.get("flag_sources", []):
        value = entry.get("value")
        if not isinstance(value, int) or value <= 0 or value & (value - 1):
            raise ValueError("invalid registry flag value")
        mask |= value
    return mask, EXPECTED_LOWERING


@dataclass(frozen=True)
class IntegratedBits:
    confidence_delta_bits: int = f64_bits(0.0)
    lr_modulation_bits: int = f64_bits(1.0)
    exploration_delta_bits: int = f64_bits(0.0)
    arousal_delta_bits: int = f32_bits(0.0)
    valence_delta_bits: int = f32_bits(0.0)
    flags: int = 0
    n_contributors: int = 0

    def values(self) -> dict[str, float]:
        return {
            "confidence_delta": f64_from_bits(self.confidence_delta_bits),
            "lr_modulation": f64_from_bits(self.lr_modulation_bits),
            "exploration_delta": f64_from_bits(self.exploration_delta_bits),
            "arousal_delta": f32_from_bits(self.arousal_delta_bits),
            "valence_delta": f32_from_bits(self.valence_delta_bits),
        }


@dataclass(frozen=True)
class Classification:
    primary_class: str
    integrated_nonfinite_field_mask: int
    lowered_nonfinite_field_mask: int
    triggered_scalar_mask: int
    lowered_f32_bits: dict[str, int]
    unsupported_integrated_flag_bits: int
    contributor_count: int
    max_qualified_contributors: int
    empty_collector_identity_mismatch: bool
    numeric_identity: bool


def classify(bits: IntegratedBits) -> Classification:
    known_flags, _ = registry_contract()
    capacity = observer_capacity()
    values = bits.values()

    native_nonfinite = 0
    lowered_nonfinite = 0
    triggered = 0
    lowered: dict[str, int] = {}

    for _tag, (field, _kind, _derivation, identity) in EXPECTED_LOWERING.items():
        value = values[field]
        field_bit = FIELD_BITS[field]
        if not math.isfinite(value):
            native_nonfinite |= field_bit
        is_triggered = value != identity
        if is_triggered:
            triggered |= field_bit
        if field in {"confidence_delta", "lr_modulation", "exploration_delta"}:
            lowered_value = rust_f64_to_f32(value)
        else:
            lowered_value = value
        lowered[field] = f32_bits(lowered_value)
        if is_triggered and not math.isfinite(lowered_value):
            lowered_nonfinite |= field_bit

    unsupported_flags = bits.flags & ~known_flags & 0xFFFF_FFFF
    bad_count = bits.n_contributors < 0 or bits.n_contributors > capacity

    empty_mismatch = False
    if bits.n_contributors == 0:
        empty_mismatch = not (
            bits.confidence_delta_bits == f64_bits(0.0)
            and bits.lr_modulation_bits == f64_bits(1.0)
            and bits.exploration_delta_bits == f64_bits(0.0)
            and bits.arousal_delta_bits == f32_bits(0.0)
            and bits.valence_delta_bits == f32_bits(0.0)
            and bits.flags == 0
        )

    numeric_identity = triggered == 0 and bits.flags == 0

    if native_nonfinite:
        primary = "INVALID_INTEGRATED_NONFINITE"
    elif lowered_nonfinite:
        primary = "INVALID_LOWERED_NONFINITE"
    elif bad_count:
        primary = "INVALID_CONTRIBUTOR_DOMAIN"
    elif unsupported_flags:
        primary = "UNSUPPORTED_INTEGRATED_FLAGS"
    elif empty_mismatch:
        primary = "INVALID_EMPTY_COLLECTOR_IDENTITY"
    elif numeric_identity:
        primary = "QUALIFIED_IDENTITY"
    else:
        primary = "QUALIFIED_APPLICATION_FINITE"

    if primary not in CLASSES:
        raise AssertionError(primary)
    return Classification(
        primary_class=primary,
        integrated_nonfinite_field_mask=native_nonfinite,
        lowered_nonfinite_field_mask=lowered_nonfinite,
        triggered_scalar_mask=triggered,
        lowered_f32_bits=lowered,
        unsupported_integrated_flag_bits=unsupported_flags,
        contributor_count=bits.n_contributors,
        max_qualified_contributors=capacity,
        empty_collector_identity_mismatch=empty_mismatch,
        numeric_identity=numeric_identity,
    )


def self_test() -> None:
    known_flags, _ = registry_contract()
    assert known_flags == sum(1 << bit for bit in range(9))
    assert observer_capacity() == 64

    classification = classify(IntegratedBits())
    assert classification.primary_class == "QUALIFIED_IDENTITY" and classification.numeric_identity

    classification = classify(IntegratedBits(confidence_delta_bits=f64_bits(0.25), n_contributors=1))
    assert classification.primary_class == "QUALIFIED_APPLICATION_FINITE"
    assert classification.triggered_scalar_mask & FIELD_BITS["confidence_delta"]

    classification = classify(
        IntegratedBits(
            confidence_delta_bits=f64_bits(-0.0),
            arousal_delta_bits=f32_bits(-0.0),
            n_contributors=1,
        )
    )
    assert classification.primary_class == "QUALIFIED_IDENTITY" and classification.numeric_identity

    classification = classify(IntegratedBits(confidence_delta_bits=f64_bits(-0.0), n_contributors=0))
    assert classification.primary_class == "INVALID_EMPTY_COLLECTOR_IDENTITY"

    classification = classify(
        IntegratedBits(confidence_delta_bits=f64_bits(float("inf")), n_contributors=1)
    )
    assert classification.primary_class == "INVALID_INTEGRATED_NONFINITE"

    f32_max = f32_from_bits(0x7F7F_FFFF)
    below = f32_max + 2.0**102
    above = f32_max + 2.0**103
    assert math.isfinite(below) and math.isfinite(above)
    assert math.isfinite(rust_f64_to_f32(below))
    assert math.isinf(rust_f64_to_f32(above))
    classification = classify(IntegratedBits(confidence_delta_bits=f64_bits(above), n_contributors=1))
    assert classification.primary_class == "INVALID_LOWERED_NONFINITE"

    assert math.isfinite(1e308)
    assert math.isinf(1e308 + 1e308)
    classification = classify(
        IntegratedBits(exploration_delta_bits=f64_bits(1e308 + 1e308), n_contributors=2)
    )
    assert classification.primary_class == "INVALID_INTEGRATED_NONFINITE"

    max_f32 = f32_from_bits(0x7F7F_FFFF)
    assert math.isfinite(max_f32)
    overflowed_f32_sum = rust_f32_add(max_f32, max_f32)
    assert math.isinf(overflowed_f32_sum)
    classification = classify(
        IntegratedBits(arousal_delta_bits=f32_bits(overflowed_f32_sum), n_contributors=2)
    )
    assert classification.primary_class == "INVALID_INTEGRATED_NONFINITE"

    classification = classify(IntegratedBits(n_contributors=65))
    assert classification.primary_class == "INVALID_CONTRIBUTOR_DOMAIN"

    classification = classify(IntegratedBits(flags=1 << 31, n_contributors=1))
    assert classification.primary_class == "UNSUPPORTED_INTEGRATED_FLAGS"

    a = 1.0 + 2.0**-30
    b = 1.0 + 2.0**-29
    assert f64_bits(a) != f64_bits(b)
    assert rust_f64_to_f32_bits(a) == rust_f64_to_f32_bits(b)

    d = 1.0 + 2.0**-20
    assert rust_f64_to_f32_bits(a) != rust_f64_to_f32_bits(d)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--classify-json", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("SPINE-000B-N2 integration/lowering-domain self-test: PASS")
        print(f"max_qualified_contributors={observer_capacity()}")
        print("authority=measurement-only")
        if not args.classify_json:
            return 0
    if args.classify_json:
        raw = json.loads(args.classify_json.read_text(encoding="utf-8"))
        print(json.dumps(asdict(classify(IntegratedBits(**raw))), indent=2, sort_keys=True))
        return 0
    parser.error("request --self-test and/or --classify-json")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
