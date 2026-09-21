# EPI-AUTH-FLOW-001 R5R4 — producer → certainty → ordinal/cube generation authority discovery

## Purpose

R5R4 is a fresh measurement-only replacement for R5R3. R5R3 correctly added the
`EpistemicStatus::Certain -> epistemic_ordinal -> EpistemicGate` path, but a deeper
consumer audit found a second independent production path:

```text
legacy / typed E/N/M
        ↓
ThoughtChannels cube
        ↓
EpistemicCubeGate
        ↓
token-level assertion / hedging / social / temporal framing
```

The per-axis gate currently gives strong coordinates direct language effects. In
particular, E3 boosts assertion tokens, E4 further boosts assertions and suppresses
hedges, N3 boosts axiomatic language, and M3 boosts foundational/permanent language.
The strict-code gate also treats `e > 1` as high certainty and returns without its
low-certainty hallucination suppression.

R5R4 therefore discovers both authority consumers plus the explicit controls that
can disable or attenuate them.

## Core negative theorem

```text
internal confidence / familiarity / runtime maturity / similarity
!= proposition-specific admitted authority
```

and:

```text
legacy E/N/M coordinate
!= permission to change assertion strength
!= permission to suppress hedging
!= permission to emit axiomatic/foundational framing
```

and:

```text
heuristic Certain
!= permission to bypass the ordinal epistemic gate
```

## Distinct consumer families

R5R4 keeps these surfaces separate rather than collapsing them into one "Broca"
bucket:

1. **1D ordinal gate**
   - `EpistemicStatus -> epistemic_ordinal`
   - `Certain/Probable` currently bypass epistemic modification.

2. **Per-axis cube gate**
   - E controls assertion/hedging.
   - N controls personal/communal/network/axiomatic framing.
   - M controls ephemeral/persistent/foundational framing.
   - H modulates distribution depth.

3. **Strict-code gate**
   - low E applies identifier/path suppression;
   - `e > 1` currently returns as "high certainty".

4. **Explicit controls**
   - `enable_epistemic_gate`;
   - `enable_epistemic_cube_gate`;
   - `bypass_gating`;
   - familiarity-based hedge attenuation.

Explicit diagnostic bypasses are inventoried, not automatically classified as
defects. The audit asks where authority-sensitive behavior can be disabled or
attenuated and whether those controls are product-, evaluation-, or training-facing.

## Discovery output

The executable emits deterministic sorted production-Rust path sets and SHA-256
set digests for producer, transport, certainty, ordinal, cube-gate, strict-code,
and gate-control surfaces.

Standalone `examples/`, `tests/`, and `benches/` are excluded from discovered path
sets so the future frozen inventory describes product/runtime source. Production
`src/bin` utilities remain visible because they are real executable surfaces.

## Required witness chains

R5R4 fails if it loses representative evidence for:

```text
runtime metrics -> numeric E/N/M cube -> carryover -> Broca
```

```text
typed E/N/M -> Certain -> ordinal 0 -> ordinal gate bypass
```

```text
typed / numeric E/N/M -> EpistemicCubeGate
                     -> assertion / hedge / axiomatic / foundational token effects
```

```text
E tier -> strict-code gate -> high-tier early return
```

and:

```text
bypass_gating / enable_epistemic_gate / enable_epistemic_cube_gate
-> generation control surface
```

The audit also retains witnesses for heuristic certainty producers, synthesis
certainty, Mycelix numeric producers, HDC statistical-tier producers, and physics
catalog authority-shaped outputs.

## Qualification semantics

An exact-head qualifier may claim only:

```text
schema=epi-auth-flow-001-r5r4-discovery-v1
authority_scope=measurement-only-producer-certainty-ordinal-cube-gates-and-controls
result=PASS_DISCOVERY
```

`PASS_DISCOVERY` means the configured discovery program executed successfully on
the exact subject. It is not a frozen inventory and does not establish semantic
correctness.

It grants no cryptographic verification, reproducibility, consensus, axiomatic
truth, foundational status, scientific truth, action authority, direct-expression
authority, hedging-suppression authority, or gate-bypass authority.

## Base and successor

Prepared only as a fresh direct child of:

`adb69f11fa8068b019cc5bb598d0c7726a197fc9`

base tree:

`35a6c5fdba319556af9bb487838734f67c8ac0d6`

A successful exact R5R4 run is intended to mechanically seed R6 as another fresh
direct child of this unchanged `main`, freezing the emitted path sets, digests,
and witnesses. Only executable R6 `PASS_INVENTORY` unlocks semantic repair.

R6 must feed both:

- #5334 — producer-side proxy-to-authority removal;
- #5385 — Broca consumer firewall using source-admitted MEL-EPI claim semantics.
