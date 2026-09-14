#!/usr/bin/env python3
"""SPINE-000B independent proposal-influence oracle.

This script is measurement-only. It does not execute Symthaea cognition and it is
not the production integration implementation. Its purpose is to provide an
independent executable oracle for the preregistered leave-one-out semantics in
SPINE_000B_PROPOSAL_INFLUENCE_CONTRACT.md before runtime instrumentation is
wired into CognitiveLoopService.

Input JSON schema:
{
  "cycle_number": 1,
  "proposals": [
    {
      "subsystem_name": "manager_a",
      "confidence_delta": 0.1,
      "lr_modulation": 1.2,
      "exploration_delta": 0.0,
      "arousal_delta": 0.0,
      "valence_delta": 0.0,
      "flags": 0
    }
  ]
}

Only non-neutral proposals should be supplied as admitted proposals. Neutral,
skipped, panic, and health-disabled execution outcomes belong to the runtime
receipt layer and are deliberately outside this oracle's scope.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SCHEMA = "symthaea.spine.000b.influence-oracle.v1"
AUTHORITY = "measurement-only"


def f32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", float(value)))[0]


def f32_bits(value: float) -> int:
    return struct.unpack("!I", struct.pack("!f", f32(value)))[0]


def f64_bits(value: float) -> int:
    return struct.unpack("!Q", struct.pack("!d", float(value)))[0]


@dataclass(frozen=True)
class Proposal:
    subsystem_name: str
    confidence_delta: float = 0.0
    lr_modulation: float = 1.0
    exploration_delta: float = 0.0
    arousal_delta: float = 0.0
    valence_delta: float = 0.0
    flags: int = 0

    @classmethod
    def from_json(cls, raw: dict[str, object]) -> "Proposal":
        name = str(raw["subsystem_name"])
        if not name:
            raise ValueError("subsystem_name must be non-empty")
        flags = int(raw.get("flags", 0))
        if flags < 0 or flags > 0xFFFF_FFFF:
            raise ValueError(f"flags out of u32 range for {name}")
        proposal = cls(
            subsystem_name=name,
            confidence_delta=float(raw.get("confidence_delta", 0.0)),
            lr_modulation=float(raw.get("lr_modulation", 1.0)),
            exploration_delta=float(raw.get("exploration_delta", 0.0)),
            arousal_delta=f32(float(raw.get("arousal_delta", 0.0))),
            valence_delta=f32(float(raw.get("valence_delta", 0.0))),
            flags=flags,
        )
        proposal.validate()
        return proposal

    def validate(self) -> None:
        scalars = {
            "confidence_delta": self.confidence_delta,
            "lr_modulation": self.lr_modulation,
            "exploration_delta": self.exploration_delta,
            "arousal_delta": self.arousal_delta,
            "valence_delta": self.valence_delta,
        }
        for field, value in scalars.items():
            if not math.isfinite(value):
                raise ValueError(f"non-finite {field} for {self.subsystem_name}")

    def is_neutral(self) -> bool:
        return (
            self.confidence_delta == 0.0
            and self.lr_modulation == 1.0
            and self.exploration_delta == 0.0
            and self.arousal_delta == 0.0
            and self.valence_delta == 0.0
            and self.flags == 0
        )

    def exact_bits(self) -> dict[str, int | bool]:
        return {
            "confidence_delta_bits": f64_bits(self.confidence_delta),
            "lr_modulation_bits": f64_bits(self.lr_modulation),
            "exploration_delta_bits": f64_bits(self.exploration_delta),
            "arousal_delta_bits": f32_bits(self.arousal_delta),
            "valence_delta_bits": f32_bits(self.valence_delta),
            "flags": self.flags,
            "is_neutral": self.is_neutral(),
        }


@dataclass(frozen=True)
class Integrated:
    confidence_delta: float
    lr_modulation: float
    exploration_delta: float
    arousal_delta: float
    valence_delta: float
    flags: int
    n_contributors: int

    def exact_bits(self) -> dict[str, int]:
        return {
            "confidence_delta_bits": f64_bits(self.confidence_delta),
            "lr_modulation_bits": f64_bits(self.lr_modulation),
            "exploration_delta_bits": f64_bits(self.exploration_delta),
            "arousal_delta_bits": f32_bits(self.arousal_delta),
            "valence_delta_bits": f32_bits(self.valence_delta),
            "flags": self.flags,
            "n_contributors": self.n_contributors,
        }


def f32_sum(values: Iterable[float]) -> float:
    acc = f32(0.0)
    for value in values:
        acc = f32(acc + f32(value))
    return acc


def integrate(proposals: list[Proposal]) -> Integrated:
    """Mirror the current OutputCollector integration equations.

    This is intentionally an independent implementation. Production SPINE-000B
    must compute its runtime leave-one-out result through the Rust collector's
    own integration path and compare that result against this oracle in tests.
    """
    if not proposals:
        return Integrated(0.0, 1.0, 0.0, 0.0, 0.0, 0, 0)

    n = len(proposals)
    nf64 = float(n)
    nf32 = f32(float(n))

    confidence_delta = sum(p.confidence_delta for p in proposals) / nf64
    exploration_delta = sum(p.exploration_delta for p in proposals) / nf64
    arousal_delta = f32(f32_sum(p.arousal_delta for p in proposals) / nf32)
    valence_delta = f32(f32_sum(p.valence_delta for p in proposals) / nf32)

    log_sum = sum(math.log(max(p.lr_modulation, 0.01)) for p in proposals)
    lr_modulation = math.exp(log_sum / nf64)

    flags = 0
    for proposal in proposals:
        flags |= proposal.flags

    return Integrated(
        confidence_delta=confidence_delta,
        lr_modulation=lr_modulation,
        exploration_delta=exploration_delta,
        arousal_delta=arousal_delta,
        valence_delta=valence_delta,
        flags=flags,
        n_contributors=n,
    )


def influence_report(cycle_number: int, proposals: list[Proposal]) -> dict[str, object]:
    names = [p.subsystem_name for p in proposals]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError("duplicate subsystem names: " + ", ".join(duplicates))
    neutral = [p.subsystem_name for p in proposals if p.is_neutral()]
    if neutral:
        raise ValueError(
            "oracle input is admitted proposals only; neutral proposals supplied by: "
            + ", ".join(neutral)
        )

    integrated_all = integrate(proposals)
    all_bits = integrated_all.exact_bits()
    receipts: list[dict[str, object]] = []

    bit_channels = (
        "confidence_delta_bits",
        "lr_modulation_bits",
        "exploration_delta_bits",
        "arousal_delta_bits",
        "valence_delta_bits",
    )

    for index, proposal in enumerate(proposals):
        without = proposals[:index] + proposals[index + 1 :]
        integrated_without = integrate(without)
        without_bits = integrated_without.exact_bits()
        changed_channels = [
            channel for channel in bit_channels if all_bits[channel] != without_bits[channel]
        ]
        unique_flags = int(all_bits["flags"]) & ~int(without_bits["flags"]) & 0xFFFF_FFFF
        integration_changed = bool(changed_channels or unique_flags)

        receipts.append(
            {
                "subsystem_name": proposal.subsystem_name,
                "proposal": proposal.exact_bits(),
                "integrated_all": all_bits,
                "integrated_without_subject": without_bits,
                "changed_channels": changed_channels,
                "uniquely_contributed_flags": unique_flags,
                "integration_changed": integration_changed,
            }
        )

    receipts.sort(key=lambda item: str(item["subsystem_name"]))
    return {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "causal_load_claimed": False,
        "runtime_evidence_claimed": False,
        "cycle_number": cycle_number,
        "admitted_proposal_count": len(proposals),
        "integrated_all": all_bits,
        "receipts": receipts,
    }


def self_test() -> None:
    # Empty admission set has collector identity semantics.
    assert integrate([]).exact_bits() == {
        "confidence_delta_bits": f64_bits(0.0),
        "lr_modulation_bits": f64_bits(1.0),
        "exploration_delta_bits": f64_bits(0.0),
        "arousal_delta_bits": f32_bits(0.0),
        "valence_delta_bits": f32_bits(0.0),
        "flags": 0,
        "n_contributors": 0,
    }

    # Unique scalar proposal changes only that scalar plus contributor count;
    # contributor count is metadata, not a proposal-value influence channel.
    one = influence_report(1, [Proposal("a", confidence_delta=0.25)])
    r = one["receipts"][0]
    assert r["changed_channels"] == ["confidence_delta_bits"]
    assert r["uniquely_contributed_flags"] == 0
    assert r["integration_changed"] is True

    # Equal duplicated scalar proposals still have leave-one-out influence only
    # if averaging semantics changes the canonical result. For identical values,
    # the integrated scalar remains identical, so neither proposal is credited.
    duplicate_scalar = influence_report(
        2,
        [Proposal("a", confidence_delta=0.25), Proposal("b", confidence_delta=0.25)],
    )
    assert all(
        "confidence_delta_bits" not in receipt["changed_channels"]
        for receipt in duplicate_scalar["receipts"]
    )

    # Unequal scalar proposals both influence the current average.
    unequal_scalar = influence_report(
        3,
        [Proposal("a", confidence_delta=0.25), Proposal("b", confidence_delta=-0.25)],
    )
    assert all(
        "confidence_delta_bits" in receipt["changed_channels"]
        for receipt in unequal_scalar["receipts"]
    )

    # Unique flag is uniquely attributable.
    unique_flag = influence_report(4, [Proposal("a", flags=0x4)])
    assert unique_flag["receipts"][0]["uniquely_contributed_flags"] == 0x4
    assert unique_flag["receipts"][0]["integration_changed"] is True

    # Duplicated flag is admitted/emitted but has no unique flag influence.
    duplicate_flag = influence_report(5, [Proposal("a", flags=0x4), Proposal("b", flags=0x4)])
    assert all(r["uniquely_contributed_flags"] == 0 for r in duplicate_flag["receipts"])

    # Mixed flags preserve only subject-unique bits.
    mixed_flag = influence_report(6, [Proposal("a", flags=0x5), Proposal("b", flags=0x4)])
    by_name = {r["subsystem_name"]: r for r in mixed_flag["receipts"]}
    assert by_name["a"]["uniquely_contributed_flags"] == 0x1
    assert by_name["b"]["uniquely_contributed_flags"] == 0

    # Neutral proposals are not valid admitted-oracle inputs.
    try:
        influence_report(7, [Proposal("neutral")])
    except ValueError as exc:
        assert "neutral proposals" in str(exc)
    else:
        raise AssertionError("neutral proposal must be rejected by admitted-only oracle")

    # Duplicate subsystem identities fail closed.
    try:
        influence_report(8, [Proposal("dup", flags=1), Proposal("dup", flags=2)])
    except ValueError as exc:
        assert "duplicate subsystem names" in str(exc)
    else:
        raise AssertionError("duplicate subsystem names must fail closed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("SPINE-000B independent influence oracle self-test: PASS")
        if not args.input:
            return 0

    if not args.input:
        parser.error("--input is required unless only --self-test is requested")

    raw = json.loads(args.input.read_text(encoding="utf-8"))
    cycle_number = int(raw["cycle_number"])
    proposals = [Proposal.from_json(item) for item in raw.get("proposals", [])]
    report = influence_report(cycle_number, proposals)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
