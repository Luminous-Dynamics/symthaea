# EPI-AUTH-FLOW-001 R5R2 — cube → certainty → behavior discovery

## Purpose

R5R2 is the replacement measurement-only discovery subject after R5 was superseded before qualification.

R5 correctly expanded the audit from isolated E/N/M producers into producer → transport → Broca flow, but preflight exposed a separate and stronger authority layer that R5 did not enumerate:

```text
ETier::E3 / ETier::E4
        ↓
EpistemicStatus::Certain
        ↓
translation / dispatch / scalar confidence / synthesis behavior
```

The concrete trigger was the runtime projection in `src/symthaea/mod.rs`:

```text
ETier::E4 | ETier::E3 -> EpistemicStatus::Certain
```

R5R2 therefore discovers the end-to-end **cube + certainty + Broca** authority surface before any frozen inventory is claimed.

## Core negative theorem

```text
strong internal tier / confidence / familiarity / prediction signal
!=
proposition-specific certainty
```

and:

```text
legacy/high authority metadata may be carried or displayed
!=
permission to suppress hedging, select a more authoritative backend,
raise scalar confidence, or condition generated language
```

## Discovery surfaces

R5R2 deterministically emits sorted paths and SHA-256 set digests for:

1. typed strong empirical authority (`ETier::E3`, `ETier::E4`);
2. typed high N/M authority (`NTier::N3`, `MTier::M3`);
3. typed cube definitions;
4. every production `pub enum EpistemicStatus` definition;
5. every production `EpistemicStatus::Certain` use;
6. `EpistemicCode::Confident` use;
7. legacy E3/E4 proof/reproduction vocabulary;
8. legacy N3/M3 axiomatic/foundational vocabulary;
9. raw numeric cube transport and carryover;
10. weighted E/N/M scalar collapse;
11. Broca cube conditioning sinks.

Standalone examples/tests/benches are excluded from discovery path sets so the emitted inventory represents product/runtime source rather than demonstration-only code. Embedded tests inside product files remain part of those product paths.

## Required authority-flow witnesses

The discovery fails if representative source witnesses disappear from any of these paths:

```text
runtime internal metrics
  -> numeric E/N/M cube
  -> carryover
  -> Broca conditioning
```

```text
StructuredThought ETier/NTier/MTier
  -> raw E/N/M + weighted quality
  -> Broca channels
```

```text
ETier::E3 or E4
  -> EpistemicStatus::Certain
  -> no language hedge
  -> certainty channel / backend selection
```

It also freezes witness reachability for:

- `Certain -> 0.95` scalarization in MAGI;
- prediction-confidence -> Certain in coding experience;
- familiarity/novelty heuristic -> Certain in intent classification;
- Gaia prediction trust/confidence -> Certain;
- `EpistemicCode::Confident -> Certain`;
- the independent core synthesis `EpistemicStatus` family and its local conversion;
- Mycelix numeric authority producers;
- HDC statistical E3/E4 producer;
- physics product E/N/M producer.

## Qualification semantics

The exact-head qualifier may claim only:

```text
schema=epi-auth-flow-001-r5r2-discovery-v1
authority_scope=measurement-only-cube-certainty-broca-authority-flow
result=PASS_DISCOVERY
```

`PASS_DISCOVERY` means the exact measurement program ran, the configured source surfaces were enumerated and hashed, minimum sanity counts were met, and all reviewed representative witnesses remained present.

It does **not** mean the inventory is frozen or semantically correct.

In particular:

```text
PASS_DISCOVERY
!= valid E3/E4 authority
!= proposition-specific certainty
!= public reproduction
!= cryptographic verification
!= axiomatic truth
!= foundational materiality
!= statistical or causal validity
!= scientific truth
!= action authority
!= permission to suppress hedging
!= permission to condition Broca
```

## Base

This subject is prepared only as a fresh direct child of:

`adb69f11fa8068b019cc5bb598d0c7726a197fc9`

base tree:

`35a6c5fdba319556af9bb487838734f67c8ac0d6`

If `main` moves before freeze, rebuild rather than silently rebasing.

## Next step

A successful exact R5R2 discovery run becomes evidence for **R6**, which should be a fresh direct child of unchanged `main` and freeze the exact emitted path sets, set digests, and witness chain.

Only executable R6 `PASS_INVENTORY` should unlock semantic repair work.
