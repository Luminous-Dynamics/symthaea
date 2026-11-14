# PLASMA-TRANSFER-COMPOSITION-001 — composed transfer decision

**Date:** 2026-11-14  
**Issue:** [#12081](https://github.com/Luminous-Dynamics/symthaea/issues/12081)  
**Branch:** `research/plasma-transfer-composed-v0-1`  
**Parent:** [draft PR #12074](https://github.com/Luminous-Dynamics/symthaea/pull/12074)  
**Related:** #12002/#12006 qualification transfer, #12026/#12028 method scope, #12033/#12051 endpoint composition, #12064/#12070 equivalence design, #12072/#12074 method ontology.

## Executive result

The transfer decision now consumes method-qualified endpoint evidence and a preregistered, cluster-aware study-design receipt. These are not sibling reports. A transfer case that passes the old bounded-profile and interval checks still cannot PASS if its endpoint composition is absent, belongs to another transfer case, was reused across bridge dimensions, or does not establish equivalence.

The existing transfer hard gates are retained: baseline qualification, factor scope, exact from/to mappings, individual factor bridges, full joint interaction evidence for multi-factor changes, specimen/group non-overlap, evidence artifact uniqueness, confidence and equivalence intervals.

## What changed

### 1. Exact profile identity

The integration resolves the material-pair class and joint geometry from a versioned `material_scope_registry` keyed by the substrate grade. Both baseline and candidate bridge profiles must remain in the same declared applicability class for this S0 path.

The existing transfer audit already verifies that the baseline qualification receipt's profile digest matches the serialized baseline profile. The composed path now also computes a digest for the candidate bridge profile.

For each bridge, it starts with the baseline profile and applies only that bridge's declared `dimensions[*].to` values. Therefore:
- a one-factor bridge is bound to a profile in which only that factor changed;
- a multi-factor bridge is bound to the exact joint vector it claims to qualify;
- a full candidate profile with several changed factors cannot be substituted for a single-factor bridge;
- the unchanged-profile path requires its own baseline method/design evidence.

### 2. Exact bridge-to-composition identity

Every bridge carries a unique `composition_case_id`. The referenced composition case must bind:
- the exact transfer case ID;
- the exact bridge ID derived from its sorted dimension set;
- the baseline profile SHA-256;
- the bridge-specific candidate profile SHA-256;
- the material-pair class and geometry resolved by the transfer registry;
- the same exact baseline and candidate method-qualified observation receipts used by the endpoint composition;
- the matching equivalence design and result receipt.

A composition case cannot be reused across two bridges in one transfer decision. It cannot be substituted from another transfer case or a different factor vector.

### 3. Equivalence must actually be established

The standalone equivalence-design auditor intentionally allows:

`PASS / EQUIVALENCE_NOT_ESTABLISHED`

That means the study design and result linkage are structurally valid, but the result interval misses the frozen equivalence margin. This is useful as a study record; it is not a transferable process.

The transfer-only binding is stricter: the design receipt must say `EQUIVALENCE_ESTABLISHED`. A structurally sound but non-equivalent result returns BLOCKED for transfer. A contradictory reported outcome or mismatched interval/result remains REJECTED by the design auditor.

### 4. Fail-closed schema and identity checks

The transfer command pins fixture schema v0.2, requires the transfer composition fixture schema v0.1, rejects duplicate transfer case IDs and rejects duplicate composition fixture IDs. Missing or unknown composition evidence blocks; contradictory profile, bridge, design or evidence identities reject.

Command:

```bash
cargo run -p xtask -- plasma-transfer-audit
```

The command now accepts:
- `data/research/plasma_transfer_fixture_v0_2.json`;
- `data/research/plasma_transfer_composition_fixture_v0_1.json`.

It runs the old transfer gates and the composition/design checks as one decision path.

## Frozen synthetic fixture

### Transfer fixture

`data/research/plasma_transfer_fixture_v0_2.json`

| Disposition | Cases |
|---|---:|
| PASS | 5 |
| BLOCKED | 13 |
| REJECTED | 14 |
| **Total** | **32** |

The 32 cases retain the original 26 transfer cases and add controls for:
- interval-only bridge with no composition → BLOCKED;
- composition receipt from a different transfer case → REJECTED;
- same composition receipt reused across different bridge sets → REJECTED;
- multi-factor joint design substituted for a single-factor bridge → REJECTED;
- unchanged profile without a current baseline composition → BLOCKED;
- structurally valid design whose interval does not establish equivalence → BLOCKED at transfer.

### Transfer composition fixture

`data/research/plasma_transfer_composition_fixture_v0_1.json`

It contains 11 exact-profile-bound D3163 apparent-shear composition records. Ten correspond to the five positive legacy transfer paths and their individual/joint bridge studies; the eleventh is a structurally valid but non-equivalent design used to verify that transfer is blocked.

### Independent replay

Fetched the latest transfer source, both fixtures, design module, composition module and command wiring back from the exact branch. Independently reimplemented the transfer, method-scope, design/result and profile-binding decisions in JavaScript:

- Composed transfer fixture: **32/32** expected dispositions matched.
- Transfer composition fixture: **11/11** records passed their structural method/design audit.
- Transfer case IDs: no duplicates.
- The deliberately non-equivalent result remains a valid composition/design record but blocks transfer.

This is synthetic contract replay, not Rust execution or physical qualification.

## Important current boundary: primary endpoint only

This integration closes the composition gap for the **primary D3163 apparent-shear endpoint** on the current PP-GRADE-A / rigid-plastic / single-lap fixture profile.

The original transfer fixture still retains all three legacy interval gates:
- initial lap-shear strength;
- aged strength retention;
- failure-mode fraction.

However, aged strength retention and failure-mode fraction are **not yet fully typed derived-endpoint compositions** in this increment. They remain legacy interval evidence. The command's output explicitly states this limitation. A PASS from this S0 fixture must not be promoted to a claim that all three endpoints have completed the new method-qualified derivation chain.

The next work is tracked in [#12085 — typed aged-retention and failure-mode derived endpoints](https://github.com/Luminous-Dynamics/symthaea/issues/12085). That work must bind initial and conditioned D3163 measurements plus D1151/D1183 conditioning for retention, and D5573 classifications plus their complete denominator for failure-mode distributions. D5573 classification must never be treated as acceptability.

## Research and competitive position

### What industry already does well

**Plasmatreat** documents PCU monitoring of plasma generation, gas flow, power, pressure, optical plasma monitoring and stability, with recipe storage, process data and remote service.  
Source: https://www.plasmatreat.com/en/products-and-services/process-control

**Surfx Technologies** documents real-time monitoring/control of forward and reflected RF power, gas flow and head temperature, plus automated inline treatment and process traceability.  
Sources: https://www.surfxtechnologies.com/products/atomflo-int/ and https://www.surfxtechnologies.com/wp-content/uploads/2023/08/AFC-600-Datasheet-1.4.pdf

**Fraunhofer IFAM** describes adhesive/surface quality assurance across pre-process, inline and post-process stages, including surface monitoring, process monitoring, destructive and non-destructive joint testing, and aging evaluation.  
Source: https://www.ifam.fraunhofer.de/content/dam/ifam/en/documents/Adhesive_Bonding_Surfaces/Adhesion-and-Interface-Research/quality-assurance-fraunhofer-ifam.pdf

**NASA PRC-1001 Rev G** illustrates material-combination-specific process qualification and requalification after changes to procedure or materials. The publicly linked PDF says the qualification path uses at least ten room-temperature lap-shear specimens under the stated assumptions. Verify the controlled current revision before applying any detailed requirement.  
Source: https://www.nasa.gov/wp-content/uploads/2023/03/prc-1001-current.pdf

Therefore, a generic process dashboard, data log or recipe alarm is not a defensible differentiator. Luminous should consume vendor telemetry and connect it to method-qualified downstream outcomes, independent study designs, exact profile transfer and later evidence invalidation.

### Current research that changes our priorities

- The 20 October 2026 five-polymer argon/nitrogen study reports that lower water contact angle does not consistently produce higher bond strength; polymer, gas, treatment severity and adhesive chemistry interact. It also reports a wider argon process window under its tested conditions.  
  https://link.springer.com/article/10.1007/s44245-026-00396-4
- The February 2026 TPO/silicone BIPV paper reports plasma-enhanced bond and lap-shear results and a generalized Eyring lifetime model with a paper-reported 98% accuracy. That figure is specific to the study's material, data and validation setup; it is not a universal lifetime-prediction SOTA threshold.  
  https://www.sciencedirect.com/science/article/abs/pii/S0257897226000162
- A November 2026 CFRP adhesive-joint study reports non-monotonic hygrothermal aging behaviour, Weibull regression for fatigue initiation and right-censored run-outs. Static fracture, initiation resistance and crack-growth response evolve differently.  
  https://www.mdpi.com/2076-3417/16/21/10810
- A February 2026 polyethylene cold-plasma study reports better wettability but increases in water-vapour and oxygen permeability, plus partial hydrophobic recovery. This motivates an explicit surface-integrity budget, not maximum activation.  
  https://doi.org/10.1002/app.70533

## Paradigm shifts

1. **Recipe equivalence → profile-space qualification.** A transfer is a bounded evidence-backed region, not a matching recipe name or a successful nominal coupon.
2. **Coupon count → independent manufacturing units.** Lots, batches, heads, days, lines and sites are the units that matter for transfer; repeated coupons do not create independence.
3. **Interval-only evidence → method + design + result composition.** The exact method, specimen cohort, independent cluster, frozen design and result must travel together.
4. **Single-factor checks → explicit interaction bridges.** A collection of individually tested changes does not prove the combined change.
5. **Static qualification → event-reconciled currentness.** Calibration, supplier, material, process and source changes must later invalidate only the claims that depend on them.
6. **Maximum wettability → constrained surface integrity.** A treatment that improves wetting but damages the product's barrier or mechanical properties is not an optimum.

## Benchmark targets

Prospective targets—not achieved results or industry standards:
- zero PASS for interval-only transfer or missing method/design composition;
- 100% detection of wrong transfer-case, bridge, profile, specimen and result substitution;
- 100% detection of missing full-vector joint evidence;
- zero PASS when the design interval fails its frozen equivalence margin;
- zero reuse of one composition case across multiple bridge vectors;
- deterministic exact replay under fixture record permutation;
- grouped external holdouts by substrate lot, adhesive batch, plasma head, line and site;
- report false-PASS, false-BLOCKED, non-equivalence, bridge cost, test count, review latency and transfer time separately.

Physical campaign targets should add durability, barrier/surface integrity, aging, failure-mode and lifecycle outcomes. Those are not measured in this software fixture.

## Validation boundary

- Independent synthetic replay: **32/32** transfer cases and **11/11** composition records.
- Eighteen Rust unit tests are authored in the transfer module, eight in the design module and seven in the composition module; they are **not executed**.
- Rust formatting, compilation, Clippy and exact-head hosted CI are **not verified**.
- The profile/material-scope registry and SHA-256 references are structurally bound; referenced artifact bytes, signatures, laboratory custody, clock trust and lot authenticity are not independently authenticated.
- The transfer fixture covers PP-GRADE-A / rigid-plastic / single-lap scope only.
- Aged-retention and failure-mode derived endpoint composition remains open (#12085).
- No physical material equivalence, bond lifetime, failure acceptability, product qualification, safety, standard compliance or production release authority is claimed.
