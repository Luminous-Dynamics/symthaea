# PLASMA-TRANSFER-COMPOSITION-001 — composed transfer decision

**Date:** 2026-11-14  
**Issue:** [#12081](https://github.com/Luminous-Dynamics/symthaea/issues/12081)  
**Branch:** `research/plasma-transfer-composed-v0-1`  
**Parent:** [draft PR #12074](https://github.com/Luminous-Dynamics/symthaea/pull/12074)

## Executive result

The transfer decision now consumes method-qualified endpoint evidence and preregistered cluster-aware study-design receipts. These are not sibling reports: a transfer case that passes the old bounded-profile and interval checks still cannot PASS if its endpoint composition is absent, belongs to another transfer case, is reused across bridge dimensions, or does not establish equivalence.

The existing transfer hard gates remain: baseline qualification, factor scope, exact from/to mappings, individual factor bridges, full joint interaction evidence for multi-factor changes, specimen/group non-overlap, evidence-artifact uniqueness, confidence and equivalence intervals.

## What changed

### Exact profile and bridge identity

The integration resolves material-pair class and joint geometry from a versioned `material_scope_registry` keyed by substrate grade. For each bridge it starts from the baseline profile and applies only the bridge's declared `dimensions[*].to` values, then hashes the resulting exact candidate profile.

A single-factor bridge is therefore bound to a profile where only that factor changed. A multi-factor bridge is bound to the exact joint vector it claims to qualify. The full multi-factor candidate profile cannot be substituted for a single-factor bridge. The unchanged-profile path also requires its own baseline method/design evidence.

Each bridge carries a unique `composition_case_id`. The referenced composition case must bind:
- exact transfer case ID;
- exact bridge ID derived from sorted dimension keys;
- baseline profile digest and bridge-specific candidate profile digest;
- material-pair class and geometry;
- the exact baseline/candidate method-qualified observation receipts;
- matching equivalence design and result receipt.

A composition case cannot be reused across two bridges in one transfer decision. Missing evidence BLOCKS; contradictory identities REJECT.

### Transfer requires equivalence, not merely a valid study

The standalone design auditor intentionally permits a structurally valid study whose interval does not establish equivalence. That preserves the study as a valid record.

The transfer-only binding is stricter: the design receipt must report `EQUIVALENCE_ESTABLISHED`. A structurally sound design whose interval misses the frozen margin returns BLOCKED for transfer. An inconsistent reported outcome remains REJECTED.

### Fail-closed command path

The CLI pins transfer fixture schema v0.2, requires composition fixture schema v0.1 and rejects duplicate transfer case IDs and duplicate composition identities. The command runs legacy factor/interval gates and the evidence composition/design checks in one decision path:

```bash
cargo run -p xtask -- plasma-transfer-audit
```

Defaults:
- `data/research/plasma_transfer_fixture_v0_2.json`
- `data/research/plasma_transfer_composition_fixture_v0_1.json`

## Frozen fixtures

Transfer fixture v0.2 contains 32 cases:

| Disposition | Cases |
|---|---:|
| PASS | 5 |
| BLOCKED | 13 |
| REJECTED | 14 |
| **Total** | **32** |

New cases cover interval-only transfer without composition, composition from another transfer, composition reuse across bridges, joint-to-single-factor substitution, unchanged profile without baseline evidence, and a structurally valid but non-equivalent design.

The transfer composition fixture contains 11 records: ten structural-positive records for the current positive paths and one structurally valid non-equivalent record used to prove that the transfer layer still blocks.

Exact-branch source and fixtures were fetched back and independently replayed in JavaScript:
- composed transfer: **32/32** expected dispositions matched;
- composition: **11/11** structural records passed;
- no duplicate transfer case IDs.

This is synthetic contract replay, not Rust execution or physical qualification.

## Current endpoint boundary

This increment closes the composition gap for the **primary ASTM D3163 apparent-shear endpoint** on the current PP-GRADE-A / rigid-plastic / single-lap fixture profile.

The legacy transfer fixture still has three interval gates—initial lap shear, aged strength retention and failure-mode fraction—but aged retention and failure-mode fraction are **not yet fully typed derived-endpoint compositions**. They remain legacy interval evidence. A PASS here must not be promoted to a claim that all three endpoint derivation chains have been completed.

The next typed-endpoint increment is tracked in [#12085](https://github.com/Luminous-Dynamics/symthaea/issues/12085): initial/post-conditioned observations plus D1151/D1183 conditioning for retention, and D5573 classifications plus a complete denominator for failure-mode distributions. D5573 classification must never be interpreted as failure acceptability.

## Industry comparison

Plasmatreat already documents process monitoring of plasma generation, gas flow, power, pressure, optical monitoring/stability, recipes and production data. Surfx documents real-time RF power, gas flow, head temperature monitoring, controls and traceability. Fraunhofer IFAM describes pre-process, inline and post-process QA, destructive/NDT testing and aging assessment. Generic monitoring dashboards are therefore a weak differentiator.

The Luminous position to prove is a vendor-neutral qualification and transfer evidence layer: exact method/edition, specimens, independent lot, calibration, process profile, frozen study design, interval and subsequent invalidation when upstream evidence changes.

Relevant public references:
- Plasmatreat process control: https://www.plasmatreat.com/en/products-and-services/process-control
- Surfx Atomflo-INT: https://www.surfxtechnologies.com/products/atomflo-int/
- Fraunhofer IFAM quality assurance: https://www.ifam.fraunhofer.de/content/dam/ifam/en/documents/Adhesive_Bonding_Surfaces/Adhesion-and-Interface-Research/quality-assurance-fraunhofer-ifam.pdf
- NASA PRC-1001 Rev G: https://www.nasa.gov/wp-content/uploads/2023/03/prc-1001-current.pdf

## Research implications and paradigm shifts

- **Recipe equivalence → bounded profile-space qualification.** A successful nominal coupon is not a recipe-wide transfer certificate.
- **Coupon counts → independent manufacturing units.** Lots, batches, plasma heads, production days, lines and sites matter; repeated coupons from one lot do not become independent lots.
- **Interval-only → method + design + result composition.** The exact method, specimen cohort, independent unit, frozen design and interval travel together.
- **Individual changes → joint interaction evidence.** Individually qualified changes do not automatically qualify their combination.
- **Maximum wettability → constrained surface integrity.** A treatment that improves wetting but degrades barrier or mechanical properties is not an optimum.
- **Static qualification → event-reconciled currentness.** Calibration, supplier, material, process and source changes must invalidate only dependent claims.

Current research supports these distinctions: the October 2026 five-polymer study found that lower contact angle did not consistently yield greater bond strength; a February 2026 polyethylene study reported increased water-vapour and oxygen permeability after plasma activation; and a November 2026 aged CFRP study reported non-monotonic degradation and right-censored fatigue run-outs. These effects are material- and study-specific, not universal performance laws.

## Benchmarks

Prospective targets, not achieved results:
- zero PASS for interval-only transfer or missing composition/design evidence;
- 100% detection of wrong transfer-case, bridge, profile, specimen and result substitution;
- 100% detection of missing full-vector joint evidence;
- zero PASS when the design interval fails its equivalence margin;
- zero reuse of one composition case across bridge vectors;
- deterministic replay under fixture-record permutation;
- external grouped holdouts by substrate lot, adhesive batch, plasma head, line and site;
- report false-PASS, false-BLOCKED, non-equivalence, bridge cost, destructive-test count, review latency and time-to-transfer separately.

## Validation boundary

- Independent synthetic replay: **32/32** transfer cases and **11/11** composition records.
- Eighteen Rust tests in the transfer module, eight in the design module and seven in composition are authored but **not executed**.
- Rust formatting, compilation, Clippy and exact-head hosted CI are **not verified**.
- SHA-256/profile references are structurally bound; referenced artifact bytes, signatures, laboratory custody, clock trust, calibration and lot authenticity are not authenticated.
- The current physical profile scope is only PP-GRADE-A / rigid-plastic / single-lap.
- Aged-retention and failure-mode derived endpoint composition remains open in #12085.
- No physical equivalence, bond lifetime, failure acceptability, product qualification, safety, standards compliance or production-release authority is claimed.
