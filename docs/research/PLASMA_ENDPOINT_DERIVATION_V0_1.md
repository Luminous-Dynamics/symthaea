# PLASMA-ENDPOINT-DERIVATION-001 — measured observations and derived joint estimands

**Date:** 2026-11-14  
**Issue:** [#12166](https://github.com/Luminous-Dynamics/symthaea/issues/12166)  
**Branch:** `research/plasma-endpoint-derivation-v0-1`  
**Stacked on:** [draft #12074](https://github.com/Luminous-Dynamics/symthaea/pull/12074) → [draft #12070](https://github.com/Luminous-Dynamics/symthaea/pull/12070) → [draft #12051](https://github.com/Luminous-Dynamics/symthaea/pull/12051).

## Executive finding

A transfer endpoint must say whether it is a **directly measured observation** or a **derived estimand**. The existing plasma transfer fixture contains `initial_lap_shear_mpa`, `aged_strength_retention` and `failure_mode_fraction`, but its old evidence contract largely treated each as an endpoint label plus an interval.

That is not sufficient:
- lap-shear strength can be a direct method-scoped observation;
- retained-strength percentage is derived from initial and aged joint-strength measurements under an explicit conditioning profile and estimand;
- failure-mode fraction is derived from specimen-level classifications under an explicit taxonomy, denominator and aggregation rule.

An ASTM method receipt for one of the input measurements does not magically become a direct method receipt for the derived endpoint.

## Research and standards position

### Plasma results reinforce endpoint separation

Péczi-Kovács and Kókai's 20 October 2026 study compares atmospheric argon and nitrogen treatment across PP, HIPS, PVC, PC and PET. Both gases improved wettability and adhesive performance relative to untreated specimens; nitrogen generally produced lower contact angles, while argon produced a wider processing window and comparable or higher bond strength for several polymer/adhesive systems. The lowest contact angle did not always correspond to the highest bond strength.

Source: https://link.springer.com/article/10.1007/s44245-026-00396-4

That result makes a scalar `bond_quality` score scientifically unsafe. Wettability, joint strength, aging retention and failure-mode fractions must remain separately typed.

### Retained strength already has a standards basis

ASTM D1151-00(2022) defines continuous moisture/temperature exposure conditions for adhesive bonds and describes performance as a percentage based on strength retained after exposure relative to original strength. The standard does not replace the strength test method used before and after conditioning.

Source: https://store.astm.org/standards/d1151

ASTM D1183-03R19 covers cyclic temperature/humidity accelerated aging. ASTM explicitly cautions that no accelerated procedure correlates perfectly with actual service and that a small set of laboratory conditions cannot simulate every service environment. Therefore an aging receipt must preserve the exposure profile and must not silently turn an accelerated screening result into a service-life claim.

Source: https://store.astm.org/d1183-03r19.html

The contribution here is **not inventing retained-strength percentage**. It is binding the exact initial and aged destructive cohorts, independent manufacturing clusters, strength-method receipts, conditioning receipt, selected estimand, result and uncertainty receipt into one replayable derivation record.

### Failure-mode classification is descriptive and material-specific

ASTM D5573-99R19 covers classifying failure modes in adhesively bonded FRP joints. Its development material was glass-fibre / thermoset-polyester SMC. It may describe observations from methods such as D3163/D5868, but explicitly does not address whether a failure mode is acceptable.

Source: https://store.astm.org/d5573-99r19.html

**Important material-scope correction:** the existing reference transfer profile in `plasma_transfer_fixture_v0_1.json` is `PP-GRADE-A`, not an SMC FRP joint. D5573 cannot be used to qualify a PP failure-mode fraction. The new fixture therefore includes a D5573-on-polypropylene BLOCKED negative control. The current generic `failure_mode_fraction` endpoint remains unqualified for that PP application until an appropriate polymer-joint classification method and taxonomy have been reviewed and registered. Do not invent an ASTM-equivalent taxonomy.

## Incumbent comparison

Plasmatreat's public PCU documentation already covers process gas flow, power, pressure, plasma stability, optical monitoring, motion/rotation, recipe storage, process data and production traceability. Its own documentation also says the surface effect is not usually directly visible or verified inline; the result often appears in subsequent bonding/coating operations.

Sources:
- https://www.plasmatreat.com/en/products-and-services/process-control
- https://www.plasmatreat.com/en/products-and-services/openair-plasmar-process-monitoring

Surfx's Atomflo-INT literature describes inline cleaning/activation and real-time process monitoring; its controller datasheet describes RF power, gas-flow, source-temperature control, recipes, out-of-spec warnings and data logging.

Sources:
- https://www.surfxtechnologies.com/products/atomflo-int/
- https://www.surfxtechnologies.com/wp-content/uploads/2023/08/AFC-600-Datasheet-1.4.pdf

These are substantive competitors, not straw men. Luminous should not rebuild their plasma-generator controls or claim they lack traceability. The differentiated layer to prove is the downstream evidence graph: exact method and material scope → observation → derivation → independent study design → transfer → lifecycle invalidation.

A 2026 systematic review of debond-on-demand adhesives benchmarks more than 40 studies and identifies multiple thermal, electrical, photothermal/UV and magnetic trigger families. Its abstract highlights strong reported pre-debond lap-shear results in selected systems but also finds no standardized DoD test protocol, comprehensive long-term durability data or comprehensive lifecycle assessment. That strengthens the circular-bond opportunity already tracked by #11976; it does not justify duplicating that workstream.

Source: https://www.sciencedirect.com/science/article/pii/S0143749626002149

## Paradigm shift

### 1. Type the endpoint before evaluating it

The v0.1 audit has three payloads:
- `direct_observation`;
- `aged_strength_retention`;
- `failure_mode_fraction`.

A direct observation must exactly match its method receipt's endpoint and claim scope. A derived endpoint must name its formula/taxonomy, input receipts, sample manifests, group structure, analysis plan, result and claim ceiling.

### 2. Destructive aging requires distinct specimens and explicit cluster pairing

For `ratio_of_means_v1`, the initial and aged cohorts contain distinct specimen IDs because the strength test is destructive. Within each arm, initial and aged cohorts must be paired by the exact independent cluster set. Baseline and candidate manufacturing clusters must not overlap. The aged cohort must cite ASTM D1151 conditioning evidence and a matching conditioning-profile ID/digest.

The result is checked arithmetically:

`retention = aged_mean_strength / initial_mean_strength`

`transfer_difference = candidate_retention - baseline_retention`

The audit explicitly does not treat `ratio_of_means` as interchangeable with `mean_of_ratios`. It checks the reported arithmetic but does not recompute the means from raw data or certify the uncertainty estimator.

### 3. A failure-mode fraction has a real denominator

The v0.1 SMC FRP profile uses specimen-level D5573 classifications and fixes the denominator to `all_tested_specimens_including_unknown`. Unknown/unclassifiable specimens remain in the denominator. Multiple raters require an adjudication receipt. The numerator is reconciled to the selected specimen category.

The output claim ceiling is `descriptive_fraction_only`. It cannot determine acceptability, product disposition or safe use.

### 4. Uncertainty belongs to a source-bound analysis owner

The derivation receipt requires an uncertainty-method ID and immutable uncertainty receipt digest. It does not run TOST, cluster bootstrap or power calculations. Those remain with Symthaea Statistics and the experiment-design owners. A structurally correct interval can still be statistically wrong; this audit does not certify it.

## Implementation

Files:
- `xtask/src/plasma_endpoint_derivation.rs`
- `data/research/plasma_endpoint_derivation_fixture_v0_1.json`
- `xtask/src/main.rs`

Command:

```bash
cargo run -p xtask -- plasma-endpoint-derivation-audit
```

The module reuses the shared #12026 method-scope evaluator rather than duplicating a method registry. It validates:
- direct versus derived endpoint types;
- exact method/edition/material/geometry/endpoint/claim scope;
- destructive cohort non-overlap and independent cluster pairing;
- matching conditioning profile and D1151 evidence;
- ratio-of-means arithmetic, result/formula/plan/input-manifest linkage;
- uncertainty method and receipt linkage;
- D5573 taxonomy/material scope, specimen-level class membership and denominator accounting;
- unknown/unclassifiable inclusion, rater adjudication and no failure-acceptability escalation.

## Frozen adversarial fixture

Fixture: `data/research/plasma_endpoint_derivation_fixture_v0_1.json`

The 32 synthetic cases currently comprise:
- **4 PASS** — direct D3163 observation, valid ratio-of-means retention, valid D5573 SMC classification fraction with unknown specimens counted, and valid multi-rater classification with adjudication;
- **8 BLOCKED** — missing conditioning, too few independent lots, missing plan digest, D5573 outside FRP/SMC scope, unknown taxonomy, unresolved rater adjudication and related missing-evidence controls;
- **20 REJECTED** — endpoint substitution, destructive-cohort specimen overlap, formula relabeling, material/cluster mismatch, retention arithmetic mismatch, zero denominator, unknowns omitted, acceptability escalation, duplicate classifications, numerator mismatch, taxonomy/result digest mismatch, artifact reuse and interval-domain violations.

Latest source and fixture were fetched back from the exact branch and independently replayed in JavaScript: **32/32** expected dispositions and outcome labels matched.

## Benchmarks

Compare:
1. endpoint name + confidence interval only;
2. direct method-scope validation only;
3. typed direct/derived receipt with sample and method lineage;
4. typed receipt plus independent clusters, unknown/exclusion accounting, uncertainty receipt and lifecycle invalidation.

Prospective software-contract targets:
- zero direct/derived endpoint substitution;
- 100% detection of missing/mismatched result, formula, taxonomy and input-manifest lineage;
- 100% detection of destructive specimen overlap and cluster crossover;
- zero coupon/pixel pseudo-replication promoted to independent manufacturing units;
- 100% reconciliation of failure-mode numerator/denominator with specimen-level records;
- zero silent removal of unknown/unclassifiable failure surfaces;
- zero D5573 application to non-SMC PP/CFRP/FRP outside registered scope;
- exact deterministic replay of frozen endpoint receipts.

These are regression-contract targets, not physical performance or statistically powered benchmarking results.

## Next integration sequence

1. Extend the existing transfer bridge schema to require typed receipts for all required endpoints—not just endpoint labels and intervals.
2. Replace the generic PP `failure_mode_fraction` requirement with a specifically reviewed PP-compatible classification method, or remove it from the PP qualification claim set with explicit rationale. D5573 is not that method.
3. Bind every derived endpoint to the complete #12070 equivalence-design receipt, not only a plan/digest reference.
4. Qualify uncertainty calculations through the existing Statistics owner and real independent lot cohorts.
5. Run the first physical PP/adhesive/plasma/aging campaign with preregistered controls and raw-data custody.
6. Keep circular release/recovery qualification under #11976; use the 2026 DoD review to design matched trigger/rebond benchmarks without creating a duplicate workstream.

## Validation boundary

- Independent synthetic replay: **32/32**.
- Seven Rust tests are authored but **not executed**.
- Rust formatting, compilation, Clippy and exact-head hosted CI are **not verified**.
- Referenced digest fields are syntactically checked and cross-bound; source bytes and signatures are not authenticated.
- Summary means and statistical intervals are not recomputed from raw observations.
- No physical bond strength, aged retention, failure-mode prevalence, failure acceptability, process transfer, service-life, product qualification or release authority is claimed.
