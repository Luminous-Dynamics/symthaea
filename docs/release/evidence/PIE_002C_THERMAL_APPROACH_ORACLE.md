# PIE-002C Thermal Approach-Temperature Oracle

Status: independent reference semantics; synthetic fixtures only.

## Purpose

`scripts/pie-thermal-approach-oracle.py` freezes a conservative first-order theorem for thermal temperature headroom without importing Symthaea production code.

PIE-002A already separates thermal energy quantity from source/process temperature compatibility. PIE-002C tightens the temperature side by requiring an explicit approach-temperature envelope whenever the claim is intended to represent a finite thermal driving force.

It does not rewrite PIE-002A history and does not upgrade the qualified PIE-002 production kernel implicitly.

## Core residual

For source temperature `S`, required process temperature `D`, and declared minimum approach `A`, PIE-002C exposes the conservative interval:

```text
thermal_headroom = S - D - A

headroom_min = S_min - D_max - A_max
headroom_max = S_max - D_min - A_min
```

Classification is then mechanical:

- `Guaranteed` iff `headroom_min >= 0`;
- `Impossible` iff `headroom_max < 0`;
- otherwise `Possible`.

The equivalent required-source envelope `D + A` is also reported explicitly.

All derived arithmetic must remain finite.

## Claim bases

Two claim bases are distinct:

### `AlgebraicEnvelope`

Allows a declared zero approach. It can reproduce a pure source-vs-demand temperature-envelope screen, but **never** supports a finite-driving-force claim. Equality may therefore be `Guaranteed` only for the weaker algebraic proposition.

### `FiniteDrivingForce`

Requires `minimum_approach_k.min > 0`. Missing or zero-inclusive approach data fail closed. A source equal to the required process temperature cannot satisfy a positive approach requirement.

This distinction prevents a zero-temperature-difference boundary from silently becoming evidence of realizable heat transfer.

## Synthetic fixtures

The self-test covers:

- 520 K source, 500 K demand, 10 K approach -> `Guaranteed`, +10 K headroom;
- 500 K source, 500 K demand, 10 K approach -> `Impossible`;
- 500 K source, 500 K demand, zero algebraic approach -> weaker `Guaranteed` algebraic screen, but no finite-driving-force authority;
- overlapping source/demand/approach intervals -> `Possible`;
- widening uncertainty weakens `Guaranteed` to `Possible` rather than strengthening it;
- clearly insufficient source temperature -> `Impossible`;
- an interval that can exactly meet zero headroom at one realization remains `Possible`, not `Impossible`;
- negative, reversed, non-finite, zero-inclusive finite-driving-force approaches, and non-finite derived sums fail closed.

## Separation from energy quantity

PIE-002C does not decide whether enough joules exist. Energy quantity remains a separate PIE-002 screen. A thermal source may have adequate energy but fail temperature/approach quality, or have adequate temperature headroom but insufficient energy.

Neither proposition should overwrite the other.

## Boundaries

This oracle is **not**:

- entropy or exergy analysis;
- a second-law proof;
- UA/LMTD/NTU analysis;
- heat-exchanger sizing;
- phase-change modeling;
- pressure-dependent thermodynamics;
- reaction enthalpy or kinetics;
- heat-loss modeling;
- temporal heat-network dispatch;
- equipment qualification;
- process feasibility;
- hardware/control authority.

A positive approach temperature is necessary for the intended first-order driving-force proposition, but it is not sufficient to prove a realizable heat exchanger or industrial process.

## Promotion rule

Independent executable semantics first. Production Rust may mirror this theorem only as a separate subject with its own qualification lineage; PIE-002A and the already-qualified production PIE-002 subject do not inherit this claim automatically.

Tracks #2726, #1610, #1647, and master #1604.
