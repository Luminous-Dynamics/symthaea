#!/usr/bin/env python3
"""SPINE-000B-N1 proposal-domain classifier and negative controls.

Measurement-only. This script does not mutate cognition or redefine production
OutputCollector behavior. It classifies exact proposal bit patterns for evidence
qualification against the frozen Phase-C application registry.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass, asdict
from pathlib import Path

REGISTRY = Path("docs/research/SPINE_000B_PHASE_C_APPLICATION_REGISTRY_V1.json")
PRODUCTION = Path("src/cognitive_loop/subsystem_trait.rs")

FIELDS = (
    "confidence_delta",
    "lr_modulation",
    "exploration_delta",
    "arousal_delta",
    "valence_delta",
)
FIELD_BITS = {name: 1 << index for index, name in enumerate(FIELDS)}

CLASSES = {
    "INVALID_NONFINITE",
    "INVALID_LR_NONPOSITIVE",
    "UNSUPPORTED_FLAG_BITS",
    "RESERVED_NONZERO",
    "QUALIFIED_WITH_LR_CLAMP",
    "QUALIFIED_NEUTRAL",
    "QUALIFIED_NON_NEUTRAL",
}


def f64_from_bits(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def f64_bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def known_flag_mask() -> int:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    flags = registry.get("flag_sources")
    if not isinstance(flags, list) or not flags:
        raise ValueError("registry flag_sources missing")
    values = []
    for entry in flags:
        value = entry.get("value")
        bit_index = entry.get("bit_index")
        if not isinstance(value, int) or value <= 0 or value & (value - 1):
            raise ValueError("registry flag value must be positive power of two")
        if not isinstance(bit_index, int) or value != 1 << bit_index:
            raise ValueError("registry flag bit_index/value mismatch")
        values.append(value)
    if len(values) != len(set(values)):
        raise ValueError("duplicate registry flag value")
    mask = 0
    for value in values:
        mask |= value
    return mask


@dataclass(frozen=True)
class ProposalBits:
    confidence_delta_bits: int = f64_bits(0.0)
    lr_modulation_bits: int = f64_bits(1.0)
    exploration_delta_bits: int = f64_bits(0.0)
    arousal_delta_bits: int = f32_bits(0.0)
    valence_delta_bits: int = f32_bits(0.0)
    flags: int = 0
    reserved: int = 0

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
    nonfinite_field_mask: int
    lr_nonpositive: bool
    lr_clamped_by_integration: bool
    unsupported_flag_bits: int
    reserved_nonzero: bool
    production_neutral: bool
    negative_zero_field_mask: int
    known_registry_flag_mask: int


def is_negative_zero(value: float) -> bool:
    return value == 0.0 and math.copysign(1.0, value) < 0.0


def classify(proposal: ProposalBits, *, registry_mask: int | None = None) -> Classification:
    mask = known_flag_mask() if registry_mask is None else registry_mask
    values = proposal.values()

    nonfinite_mask = 0
    negative_zero_mask = 0
    for name, value in values.items():
        if not math.isfinite(value):
            nonfinite_mask |= FIELD_BITS[name]
        if is_negative_zero(value):
            negative_zero_mask |= FIELD_BITS[name]

    lr = values["lr_modulation"]
    lr_nonpositive = math.isfinite(lr) and lr <= 0.0
    lr_clamped = math.isfinite(lr) and 0.0 < lr < 0.01
    unsupported = proposal.flags & ~mask & 0xFFFF_FFFF
    reserved_nonzero = proposal.reserved != 0

    production_neutral = (
        values["confidence_delta"] == 0.0
        and lr == 1.0
        and values["exploration_delta"] == 0.0
        and values["arousal_delta"] == 0.0
        and values["valence_delta"] == 0.0
        and proposal.flags == 0
    )

    if nonfinite_mask:
        primary = "INVALID_NONFINITE"
    elif lr_nonpositive:
        primary = "INVALID_LR_NONPOSITIVE"
    elif unsupported:
        primary = "UNSUPPORTED_FLAG_BITS"
    elif reserved_nonzero:
        primary = "RESERVED_NONZERO"
    elif lr_clamped:
        primary = "QUALIFIED_WITH_LR_CLAMP"
    elif production_neutral:
        primary = "QUALIFIED_NEUTRAL"
    else:
        primary = "QUALIFIED_NON_NEUTRAL"

    if primary not in CLASSES:
        raise AssertionError("unreachable proposal class")

    return Classification(
        primary_class=primary,
        nonfinite_field_mask=nonfinite_mask,
        lr_nonpositive=lr_nonpositive,
        lr_clamped_by_integration=lr_clamped,
        unsupported_flag_bits=unsupported,
        reserved_nonzero=reserved_nonzero,
        production_neutral=production_neutral,
        negative_zero_field_mask=negative_zero_mask,
        known_registry_flag_mask=mask,
    )


def production_source_preflight() -> None:
    source = PRODUCTION.read_text(encoding="utf-8")
    required = (
        "if !output.is_neutral()",
        "self.outputs.push((name, output));",
        ".map(|(_, o)| o.lr_modulation.max(0.01).ln())",
    )
    for phrase in required:
        if phrase not in source:
            raise ValueError(f"production admission/integration semantics drifted: {phrase}")


def self_test() -> None:
    production_source_preflight()
    mask = known_flag_mask()
    assert mask == sum(1 << bit for bit in range(9)), "unexpected v1 registry flag mask"

    neutral = ProposalBits()
    c = classify(neutral, registry_mask=mask)
    assert c.primary_class == "QUALIFIED_NEUTRAL"
    assert c.production_neutral

    ordinary = ProposalBits(confidence_delta_bits=f64_bits(0.25), flags=1)
    assert classify(ordinary, registry_mask=mask).primary_class == "QUALIFIED_NON_NEUTRAL"

    # Two distinct NaN payloads remain invalid without canonicalization.
    qnan = ProposalBits(confidence_delta_bits=0x7FF8_0000_0000_0001)
    snan_like = ProposalBits(confidence_delta_bits=0x7FF0_0000_0000_0001)
    assert classify(qnan, registry_mask=mask).primary_class == "INVALID_NONFINITE"
    assert classify(snan_like, registry_mask=mask).primary_class == "INVALID_NONFINITE"
    assert qnan.confidence_delta_bits != snan_like.confidence_delta_bits

    for infinity_bits in (f64_bits(float("inf")), f64_bits(float("-inf"))):
        assert classify(
            ProposalBits(exploration_delta_bits=infinity_bits), registry_mask=mask
        ).primary_class == "INVALID_NONFINITE"

    for lr in (0.0, -0.0, -1.0):
        result = classify(ProposalBits(lr_modulation_bits=f64_bits(lr)), registry_mask=mask)
        assert result.primary_class == "INVALID_LR_NONPOSITIVE"
        assert result.lr_nonpositive

    below = classify(ProposalBits(lr_modulation_bits=f64_bits(0.009)), registry_mask=mask)
    assert below.primary_class == "QUALIFIED_WITH_LR_CLAMP"
    assert below.lr_clamped_by_integration

    exact_floor = classify(ProposalBits(lr_modulation_bits=f64_bits(0.01)), registry_mask=mask)
    assert exact_floor.primary_class == "QUALIFIED_NON_NEUTRAL"
    assert not exact_floor.lr_clamped_by_integration

    unknown = classify(ProposalBits(flags=1 << 31), registry_mask=mask)
    assert unknown.primary_class == "UNSUPPORTED_FLAG_BITS"
    assert unknown.unsupported_flag_bits == 1 << 31

    reserved = classify(ProposalBits(reserved=1), registry_mask=mask)
    assert reserved.primary_class == "RESERVED_NONZERO"
    assert reserved.reserved_nonzero

    neg_zero = classify(
        ProposalBits(confidence_delta_bits=f64_bits(-0.0), arousal_delta_bits=f32_bits(-0.0)),
        registry_mask=mask,
    )
    assert neg_zero.primary_class == "QUALIFIED_NEUTRAL"
    assert neg_zero.production_neutral
    assert neg_zero.negative_zero_field_mask & FIELD_BITS["confidence_delta"]
    assert neg_zero.negative_zero_field_mask & FIELD_BITS["arousal_delta"]

    # Precedence: nonfinite outranks LR, unknown flags, and reserved diagnostics.
    mixed_invalid = classify(
        ProposalBits(
            confidence_delta_bits=0x7FF8_0000_0000_0001,
            lr_modulation_bits=f64_bits(-1.0),
            flags=1 << 31,
            reserved=9,
        ),
        registry_mask=mask,
    )
    assert mixed_invalid.primary_class == "INVALID_NONFINITE"
    assert mixed_invalid.lr_nonpositive
    assert mixed_invalid.unsupported_flag_bits
    assert mixed_invalid.reserved_nonzero


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--classify-json", type=Path)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("SPINE-000B-N1 proposal-domain self-test: PASS")
        print(f"known_registry_flag_mask=0x{known_flag_mask():08x}")
        print("authority=measurement-only")
        if not args.classify_json:
            return 0

    if args.classify_json:
        raw = json.loads(args.classify_json.read_text(encoding="utf-8"))
        proposal = ProposalBits(**raw)
        print(json.dumps(asdict(classify(proposal)), indent=2, sort_keys=True))
        return 0

    parser.error("request --self-test and/or --classify-json")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
