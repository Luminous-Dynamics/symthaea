# GEOM-003D0A4 — Matched-arm equivalence seal

Status: implementation candidate. No target GWT lesion outcome was executed or inspected to define this tranche.

Parent authority: #3157  
Implementation issue: #3185  
Exact predecessor: `b585c90b9e4542d0cf0af852b38e4f5807770aa8`

## Purpose

D0A1–D0A3 can make an individual run reproducible. That is not enough for a causal comparison.

Four individually valid seals could still hide a second changed configuration field. D0A4 therefore seals the **relationship among arms** before execution.

## Fixed primary roles

The first GEOM GWT campaign has exactly four roles and exact arm identifiers:

| role | arm id | `enable_gwt` | B3 gate plan |
|---|---|---:|---|
| A intact | `A-intact` | true | `InstalledEnabled` |
| A' sham intact | `A-prime-sham-intact` | true | `InstalledEnabled` |
| B broad GWT off | `B-broad-gwt-off` | false | `AbsentBecauseGwtDisabled` |
| C handler delivery blocked | `C-handler-delivery-blocked` | true | `InstalledBlocked` |

No free-form intervention label can substitute for these typed roles in the primary protocol.

## Full-config comparison

The authority serializes the complete `CognitiveLoopConfig` for every arm. It does not maintain a hand-copied list of ordinary cognitive fields.

Per-arm persistence paths are the only normalization before comparison:

- every D0A2 canonical persistence root must be pairwise distinct;
- `memory_db_path` and `epistemic_auditor_db_path` must be `None` for this first campaign;
- `aesthetic_memory_path` must exactly match the accepted D0A2 report, be strictly inside that arm's canonical root, and have the same relative layout in all four arms;
- the absolute aesthetic path is replaced by a fixed persistence-root token before config comparison.

After that normalization:

- A' must have zero differences from A;
- C must have zero differences from A;
- B must have exactly one recursive JSON-pointer difference: `/enable_gwt`;
- A's value must be `true` and B's value must be `false`.

Any second field difference is a protocol failure, even if both individual arm seals are valid.

## Cross-binding to D0A2/D0A3

For each arm, D0A4 independently checks that:

- the supplied D0A3 sealed-run commitment recomputes from its snapshot;
- the supplied full config recomputes to the D0A3 config commitment under D0A3's frozen config domain;
- the supplied D0A2 report recomputes to the D0A3 preflight commitment under D0A3's frozen preflight domain;
- the D0A2 persistence root equals the D0A3 persistence root;
- the D0A2 fixed UTC hour equals the D0A3 campaign fixed UTC hour;
- the D0A2 report still represents the primary clean state: explicit genesis present, adaptive training off, persistence initially empty, and every forbidden ambient channel present exactly once and marked absent.

This prevents a caller from pairing one clean seal with a different config/preflight object.

## Shared scientific identity

Across A, A', B and C the following must match:

- normalized exact source identity and lock/toolchain commitments;
- normalized runtime/toolchain/platform/hardware/thread identity and Cargo feature set;
- campaign id;
- analysis-authority revision;
- input-schedule commitment;
- arm-order commitment;
- fixed UTC hour;
- process-environment commitment.

Expected per-arm fields are deliberately excluded from the shared comparison only after their separate rules are checked: arm id, config/preflight commitment, and isolated persistence root.

## Matched-set commitment

The output contains ordered arm records plus a domain-separated commitment under:

`symthaea:geom:matched-arm-set:v1`

Each arm record binds its fixed role/id, typed intervention plan, exact D0A3 run commitment, exact D0A3 config/preflight commitments, normalized config commitment, canonical persistence root, and normalized aesthetic relative path.

The set also binds a shared-environment commitment. Any arm reconstruction after this seal creates a new lineage.

## Claim boundary

A passing D0A4 seal establishes only:

> The preregistered four-arm set was constructed under the same declared scientific environment and differed only by the authorized GWT intervention dimensions plus isolated persistence locations.

It does not establish that the interventions actually fired during execution. D1 manipulation checks must separately verify GWT absence/presence and B3 invocation/blocked counters.

It does not establish a geometric effect, consciousness, GWT as a biological theory, causal emergence, or gravity coupling.
