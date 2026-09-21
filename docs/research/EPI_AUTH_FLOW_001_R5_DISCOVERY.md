# EPI-AUTH-FLOW-001 R5 — measurement-only authority-flow discovery

## Purpose

R5 is a discovery subject, not a semantic repair and not a frozen inventory.

Earlier EPI-SEM inventory iterations were superseded when deeper preflight exposed authority surfaces they could not fail closed on. R5 changes the method: discover the current end-to-end authority flow mechanically first, then use its exact executable output to construct the next frozen inventory subject.

## Core theorem

```text
legacy/high epistemic value may be readable or describable
!=
legacy/high epistemic value may authorize a claim or condition generated certainty
```

The current review has shown that epistemic authority can flow through more than one representation family:

```text
consciousness EmpiricalTier / EpistemicCoordinate
Mycelix E/N/M bridge representations
raw u8 runtime cube transport
StructuredThought ETier / NTier / MTier
Broca one-hot epistemic cube channels
```

R5 therefore measures producer → transport → carryover → language/Broca sinks rather than only enum spellings.

## Discovery surfaces

The script emits deterministic sorted file sets and SHA-256 set digests for:

1. `typed_top_authority` — `ETier::E4`, `NTier::N3`, `MTier::M3`;
2. `typed_cube_definition` — the StructuredThought cube definitions;
3. `legacy_empirical_authority` — E3/E4 proof/reproduction aliases;
4. `legacy_high_axis_authority` — N3/M3 axiomatic/foundational aliases;
5. `numeric_cube_transport` — raw cube integers, injection, carryover and setters;
6. `scalar_cube_collapse` — weighted E/N/M scalar recombination;
7. `broca_cube_sink` — Broca cube channel sinks.

Each surface prints:

```text
SURFACE <name> count=<n> digest=<sha256>
FILE <name> <repo-relative-path>
```

The exact output is intended to be copied mechanically into the subsequent frozen inventory subject rather than inferred from GitHub search snippets.

## Representative witness chain

R5 also requires exact source witnesses for the currently known critical flow:

```text
cycle runtime heuristics
    -> raw E/N/M cube
    -> carryover
    -> cycle training signals
    -> broca_bridge
    -> ThoughtChannels::set_epistemic_cube
    -> E/N/M one-hot generation conditioning
```

and separately:

```text
StructuredThought EpistemicCube
    -> ssm_backend typed-to-numeric projection
    -> weighted quality scalar
    -> Broca cube channels
```

It also keeps representative Mycelix, HDC statistical-retrieval, and physics-catalog producers in the discovery boundary.

## Qualification semantics

A successful exact-head run may report only:

```text
result=PASS_DISCOVERY
```

That means:

- the R5 discovery source compiled;
- the reviewed representative witnesses were present;
- every configured surface was mechanically enumerated;
- configured minimum discovery counts were met;
- exact set digests were emitted for the next freeze.

It does **not** mean:

```text
PASS_DISCOVERY
!= frozen inventory
!= semantic correctness
!= valid cryptographic proof
!= public reproduction
!= normative consensus or axiomatic truth
!= foundational materiality
!= statistical calibration
!= causal validity
!= scientific truth
!= action authority
!= Broca trust authorization
```

## Current base

R5 is prepared only for main:

`adb69f11fa8068b019cc5bb598d0c7726a197fc9`

with tree:

`35a6c5fdba319556af9bb487838734f67c8ac0d6`

If main changes before freezing the R5 subject, the subject must be rebuilt from the new base rather than silently rebased.

## Next step after executable discovery

Use the exact R5 output to create a fresh direct child of main that freezes:

- every discovered file set;
- every set digest;
- the critical producer/transport/sink witnesses;
- negative authority invariants for future semantic repair.

Only that later frozen subject may gate implementation work.
