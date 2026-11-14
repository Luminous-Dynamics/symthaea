# PLASMA-TRANSFER-COMPOSED-001 — actual transfer decision requires typed endpoint evidence

**Date:** 2026-11-14  
**Issue:** [#12138](https://github.com/Luminous-Dynamics/symthaea/issues/12138)  
**Branch:** `research/plasma-transfer-composed-v0-2`  
**Stack:** claim-scope #12028 → composition #12051 → design #12070 → method ontology #12074 → composed transfer

## Why this increment exists

The transfer audit, method-scope audit, evidence-composition audit and equivalence-design audit previously existed as separate commands. Passing them independently did not prove that the *actual transfer decision* consumed the same method receipts, design, specimens and interval as the transfer bridge.

This increment changes the architecture from **adjacent evidence reports** to **evidence composition inside the transfer decision**.

The legacy bounded-transfer logic remains intact: exact baseline profile digest, qualification scope, changed-factor discovery, single-factor bridges, exact full-vector interaction bridge, artifact uniqueness and per-endpoint equivalence/confidence bounds are still checked. The new gate is additional and fail-closed.

## What is now enforced

### 1. Baseline qualification is method-scoped

Every baseline qualification receipt now includes a D3163 method-scope receipt for the primary lap-shear endpoint. The transfer audit checks:
- exact D3163-01R23 edition and measurement evidence type;
- rigid-plastic pair / single-lap geometry;
- apparent-shear-strength endpoint and comparative-screening claim;
- protocol and raw-data digest shape;
- exact match to the baseline qualification specimen identities;
- material scope derived from the fixture's explicit substrate-grade mapping.

The unchanged-profile case is still eligible to pass without a new bridge, but it must now have a method-scoped baseline strength receipt.

### 2. Each changed-factor bridge embeds composition/design evidence

Every bridge in a changed transfer case now carries a `composition_evidence` envelope containing:
- the exact transfer bridge artifact SHA-256;
- baseline/candidate method-qualified D3163 observation receipts;
- exact baseline/candidate specimen IDs;
- a cluster ID for each specimen, not merely two unrelated lists of IDs;
- the cluster-aware equivalence design receipt;
- the analysis interval and outcome bound to the same plan and sample manifest.

The actual transfer audit calls `audit_transfer_bridge` for every bridge after its existing transfer-evidence checks. It rejects or blocks:
- absent composition/design envelope;
- transfer artifact digest substitution;
- specimen or sample-to-cluster mapping mismatch;
- method/endpoint/material/geometry mismatch;
- a design receipt for another transfer bridge;
- design interval that differs from the transfer bridge's initial lap-shear interval.

The embedded design receipt binds:
- baseline/candidate D3163 method receipts;
- specimen and independent-lot mappings;
- the transfer interval's margin, confidence and absolute upper difference bound;
- frozen plan, power-planning assumptions and analysis receipt;
- `planning_only` power claim ceiling.

The transfer fixture's material-scope mapping is explicit and versioned. The current PP profile maps to `rigid_plastic_pair` and `single_lap`.

### 3. Existing transfer gates remain mandatory

The new evidence composition does not replace:
- baseline profile hash verification;
- qualified factor scope;
- exact change-factor mapping;
- one independent single-factor bridge for every changed factor;
- full joint bridge for the exact set of two-or-more simultaneous changes;
- endpoint presence, interval shape, margin, confidence and artifact uniqueness.

A design receipt can be structurally valid while the interval fails the transfer equivalence boundary. In that case the transfer decision remains BLOCKED or REJECTED under the pre-existing transfer rules. A positive design audit does not override a negative transfer result.

## Important remaining endpoint boundary

The legacy transfer fixture declares three endpoints:
- `initial_lap_shear_mpa`;
- `aged_strength_retention`;
- `failure_mode_fraction`.

This first end-to-end increment fully composes **initial lap-shear strength** with the D3163 method and cluster-aware design receipt.

The other two endpoints remain under the existing transfer audit's interval/evidence checks, but are **not yet fully method-composed**:
- aged strength retention still needs a typed relationship between aged and baseline strength observations, the conditioning protocol, and the survival/aging owner (#11971);
- failure-mode fraction needs the typed D5573 classification observation and an explicit analysis/acceptance separation; the D5573 classification itself does not determine acceptability.

No endpoint is silently promoted to a stronger claim. The transfer result must not be described as fully aged-life or failure-acceptability qualified until those two evidence paths are integrated.

## Frozen fixture

Fixture: `data/research/plasma_transfer_fixture_v0_2.json`  
Command:

```bash
cargo run -p xtask -- plasma-transfer-audit
```

The fixture schema is `plasma-transfer-fixture-v0.2`, with an explicit `material_scope_map`.

| Expected transfer disposition | Cases |
|---|---:|
| PASS | 5 |
| BLOCKED | 11 |
| REJECTED | 16 |
| **Total** | **32** |

There are 42 changed-factor/joint bridge records. All bridges in the fixture carry an embedded composition envelope. The adversarial cases include:
- no bridge for a changed material lot;
- absent exact joint evidence for multi-factor changes;
- intervals crossing or exceeding the existing equivalence margins;
- missing aged-retention or failure-mode endpoints;
- confidence below the transfer profile threshold;
- overlapping specimen or independence groups;
- malformed or duplicated artifact digests;
- baseline profile digest mismatch and absent baseline qualification;
- missing embedded composition;
- bridge artifact SHA substitution;
- specimen-manifest substitution;
- specimen-to-cluster mapping substitution;
- a design result interval different from the transfer interval;
- a design receipt copied from another bridge.

The original overlap negative control was retained after fixture migration. Its baseline and candidate independence-group sets still overlap, so the existing transfer validator must REJECT it rather than silently normalizing it.

## Current validation evidence

Exact branch readback confirms:
- transfer source and command wiring are present;
- fixture v0.2 contains 32 cases / 42 bridges;
- baseline method-scope receipts are present wherever a baseline qualification receipt exists;
- 16 transfer Rust tests are authored;
- seven design tests and seven composition tests remain authored in their owner modules.

An independent JavaScript replay of the transfer contract before the final overlap-fixture restoration matched 31/32 cases; the one mismatch was the intentionally overlapping independence-group negative control that had been weakened by fixture migration. That control has since been restored, and its exact group-overlap predicate now returns REJECTED by the existing transfer validator. The full latest 32-case replay has not been rerun after that correction, so **32/32 is not claimed yet**.

Rust tests, formatting, compilation, Clippy and exact-head hosted CI are **not verified**. The current execution environment has no Rust/Cargo toolchain; hosted PR workflows on adjacent stacked work have previously completed as skipped rather than passed.

## Industry / SOTA position

The industry comparison remains:
- plasma equipment suppliers (including Plasmatreat and Surfx) already offer process control, monitoring and traceability;
- ASTM test methods define scoped ways to measure particular material/joint responses;
- NASA process specifications demonstrate material-combination-specific bonding process qualification and requalification after relevant changes;
- statistical packages already provide TOST, ICC, cluster covariance and power-planning tools.

Luminous should not clone any of those capabilities. The differentiator to prove is that a vendor-neutral qualification-transfer decision consumes exact method/edition, material/geometry, lot/cluster, process, calibration, preregistered design, analysis result and transfer evidence as one replayable chain.

Prospective benchmarks:
- zero transfer PASS if any bridge lacks a design/composition receipt;
- 100% detection of bridge-artifact, sample-manifest and cluster-map substitution;
- zero transfer PASS when an exact multi-factor joint bridge is absent;
- zero transfer PASS when a primary lap-shear interval exceeds its frozen margin;
- separate false-BLOCKED, false-REJECTED, false-PASS, reviewer effort and missing-evidence rates;
- separate coverage for method-composed endpoints versus legacy interval-only endpoints.

These are software-contract targets, not physical performance results.

## Claim ceiling

S0 PASS now means the legacy transfer contract and the primary initial-lap-shear D3163 method/design composition are structurally consistent. It does not establish:
- numerical validity of the underlying power calculation or TOST result;
- actual laboratory execution, calibration or artifact authenticity;
- aged service-life equivalence;
- failure-mode acceptability;
- physical material equivalence or robustness to new suppliers/lines/sites;
- product qualification, safety or production release authority.

The transfer is still a research audit, not a production interlock or certificate.
