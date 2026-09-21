# EPI-AUTH-FLOW-001 R5R3 — cube → certainty → ordinal gate → behavior discovery

## Purpose

R5R3 is the replacement measurement-only subject after R5R2 was superseded before qualification.

R5R2 added the missing `ETier::E3/E4 -> EpistemicStatus::Certain` authority layer. Further preflight found that `Certain` is then encoded as epistemic ordinal `0`, and the Broca epistemic gate deliberately returns without modification for `Certain` or `Probable` (`ordinal < 1.5`). Incorrectly minted certainty can therefore bypass a generation-time factual/hedging safety mechanism.

R5R3 discovers the complete bounded flow:

```text
E/N/M and strong typed tiers
        ↓
EpistemicStatus / Certain
        ↓
epistemic ordinal
        ↓
Broca gate / generator / decoder
        ↓
language behavior
```

## Core negative theorem

```text
internal tier / confidence / familiarity / prediction signal
!= proposition-specific certainty
```

and:

```text
heuristic Certain
!= authority to bypass epistemic generation gating
```

The gate itself is not classified as defective merely because it trusts its input. The authority problem is that upstream heuristics can mint an input whose downstream meaning is stronger than the evidence establishes.

## Discovery surfaces

The executable emits deterministic sorted production-Rust path sets and SHA-256 set digests for:

- typed strong empirical authority (`ETier::E3/E4`);
- typed high N/M authority (`NTier::N3`, `MTier::M3`);
- typed cube definitions;
- `EpistemicStatus` definitions;
- `EpistemicStatus::Certain` uses;
- `EpistemicCode::Confident` uses;
- `epistemic_ordinal` transport/consumers;
- Broca epistemic-gating vocabulary/configuration;
- legacy E3/E4 proof/reproduction vocabulary;
- legacy N3/M3 axiomatic/foundational vocabulary;
- numeric cube transport;
- scalar E/N/M collapse;
- Broca cube sinks.

Standalone examples/tests/benches are excluded from path discovery.

## Required end-to-end witnesses

The audit fails if it loses representative witnesses for:

```text
runtime metrics -> numeric cube -> carryover -> Broca cube channels
```

```text
StructuredThought cube -> E/N/M integers -> Broca
```

```text
ETier::E3/E4 -> Certain -> ordinal 0 -> Broca gate bypass
```

The exact Broca witness requires the current documented ordinal contract:

```text
0=Certain, 1=Probable, 2=Uncertain, 3=Unknown, 4=OutOfDomain
```

and the current bypass:

```text
if epistemic_ordinal < 1.5
    -> Certain or Probable: no modification
```

R5R3 also witnesses generator/decoder/Liquid-Mamba ordinal propagation, translation hedging, backend dispatch, certainty scalarization, heuristic certainty producers, synthesis certainty, Mycelix, HDC statistical retrieval, and physics-product authority producers.

## Qualification semantics

An exact-head qualifier may claim only:

```text
schema=epi-auth-flow-001-r5r3-discovery-v1
authority_scope=measurement-only-cube-certainty-ordinal-broca-authority-flow
result=PASS_DISCOVERY
```

`PASS_DISCOVERY` means executable discovery on the exact subject, not semantic correctness and not a frozen inventory.

It specifically does not establish valid E3/E4 authority, proposition-specific certainty, cryptographic verification, public reproduction, axiomatic truth, foundational materiality, statistical/causal validity, scientific truth, action authority, hedging suppression authority, or Broca gate-bypass authority.

## Base and next step

Prepared only as a fresh direct child of:

`adb69f11fa8068b019cc5bb598d0c7726a197fc9`

base tree:

`35a6c5fdba319556af9bb487838734f67c8ac0d6`

A successful exact R5R3 run should mechanically seed R6: a fresh direct child of unchanged `main` that freezes the exact discovered path sets, digests, and witness contract. Only executable R6 `PASS_INVENTORY` should unlock semantic repairs.
