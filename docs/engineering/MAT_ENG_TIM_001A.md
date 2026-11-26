# MAT-ENG-TIM-001A — Synthetic Thermal-Interface Benchmark

Issue: #6191  
Parent: #6190  
Reference corpus: `docs/engineering/data/mat_eng_tim_001a_reference_v1.json`  
Canonical corpus SHA-256: `b975e03078278879e2907d72c92b3f39d8e59e92ae970076aa9f8347726e046f`

## Status

This is a docs/data-only synthetic benchmark for the engineering↔materials bridge. It establishes no real TIM property, interface/package performance, manufacturability, reliability, safety, certification, procurement/build, or deployment authority.

This exact generation supersedes all earlier pre-qualification source heads. No qualification result transfers across source generations.

## Analytical model

The frozen reference model is:

`R_total = R_c1 + t/(k*A) + R_c2 + R_sink`

`DeltaT = Qdot * R_total`

`T_hot = T_sink + DeltaT`

Units are frozen in the corpus. The model intentionally omits spreading-detail, nonlinear contact behavior, anisotropy, phase change, complex transients and real package geometry.

Core boundary:

`high bulk k != low contact resistance != low bond-line resistance != low package resistance != durable/manufacturable interface != best system intervention`.

## Explicit nominal reference

`NOMINAL_TIM_PATH_V1` is frozen at:

- `R_c1 = 0.10 K/W`
- `t = 0.0005 m`
- `k = 5 W/(m*K)`
- `A = 0.001 m^2`
- `R_c2 = 0.10 K/W`
- `R_sink = 0.20 K/W`
- `Qdot = 50 W`
- `T_sink = 40 degC`

This gives `R_tim=0.10 K/W`, `R_total=0.50 K/W`, `DeltaT=25 K`, `T_hot=65 degC`.

All intervention improvements reference this object explicitly; there is no hidden baseline.

Fractional improvement is `(baseline_R_total-candidate_R_total)/baseline_R_total`.

## Bottleneck attribution

Attribution is group-level:

- `bulk = R_tim`
- `contact = R_c1 + R_c2`
- `sink = R_sink`

A unique dominant group requires both:

- largest group fraction `> 0.50`;
- `largest_group_fraction - second_largest_group_fraction > 0.15`.

Research mapping is frozen in machine-readable semantics:

- `BulkTIMDominant -> BulkMaterialDiscoveryJustified`
- `ContactDominant -> InterfaceInnovationJustified`
- `SinkDominant -> ArchitectureAlternativeCompetitive`
- `Unresolved -> EvidenceActionRequired`.

No causal claim follows from this synthetic contribution rule.

## Uncertainty and migration

T13 uses group means/sigmas with `mean ± 1 sigma`. If the nominal top two group intervals overlap, the final disposition is `Unresolved` and `EvidenceActionRequired`.

T14 is intentionally unambiguous:

- before: bulk/contact/sink fractions = `.60/.20/.20` -> `BulkTIMDominant`;
- after: `.20/.20/.60` -> `SinkDominant`;
- frozen migration mapping -> `MigratedToSink`.

## Headroom semantics

T11 compares feasible thickness and conductivity interventions on the same base path. If thickness headroom produces the larger fractional system-resistance reduction, the frozen disposition is `ProcessInnovationJustified`.

T15 freezes `max_system_gain_fraction=(R_tim_before-R_tim_after)/baseline_R`; this corpus labels gain `<= 0.20` as `DiminishingBulkKReturn`.

T20 emits `NominalGainErased` only when computed `R_total` equals the nominal-reference total within absolute tolerance `1e-12`.

These thresholds are benchmark semantics, not universal engineering laws.

## Candidate and robustness rules

Candidate `k` explicitly overrides path `k` when supplied.

Hard constraints are non-compensatory. Their frozen violation disposition is `Violated`; a violating candidate is `Blocked`.

Required form absent from `available_forms` yields process `Blocked` and candidate `Blocked`.

Lifecycle `initial_pass=true` with `end_of_life_pass=false` yields lifecycle `Blocked` and candidate `Blocked`.

Robust conductivity uses:

`lower_tail_k = mean_k - lower_tail_sigma*sigma_k`.

It passes only when `lower_tail_k >= thermal_pass_requires_k_min`; dispositions are explicitly `Pass` or `Blocked`.

## Input and numeric domain

All required numeric inputs must be finite. The frozen domain requires:

- `k > 0`;
- `A > 0`;
- `t >= 0`;
- declared resistance terms `>= 0`;
- `Qdot >= 0`.

Invalid arithmetic inputs yield `RejectInvalidInput`. Declared equality comparisons use absolute tolerance `1e-12`.

## Evidence, applicability and currentness

The V1 evidence cases freeze only these narrow mappings:

- `HandbookBulkK -> ProcessConditionedInterfaceProperty = EvidenceLimited`;
- `BulkThermalConductivity -> ContactThermalResistance = QuantityMismatch`.

Coupon→article applicability compares `pressure_kpa`, `thickness_um`, and `process`; any mismatch yields `ApplicabilityMismatch` and `ReviewRequired`.

Fresh evidence that changes a ranking requires `NewGenerationRequired` while preserving `PreservePriorRanking` for the historical generation.

`R_total` alone with bulk/contact inseparability yields `Unresolved` + `EvidenceActionRequired`.

Synthetic success cannot promote `ProductQualified`; the frozen authority disposition is `RejectPromotion`.

## Corpus and qualification

The corpus contains exactly 28 ordered cases `T01`–`T28`. Claim-critical disposition names and derivation rules live in the machine-readable `semantics` object; an independent qualifier must not invent hidden policy or branch on case ID for answers.

Qualification order:

`source freeze -> independent stdlib oracle -> exact-head hosted qualification -> typed MAT-ENG/TIM adapters`.

A qualifier must rederive expected fields from raw inputs plus frozen top-level semantics before comparison.

## Claim ceiling

A future exact-head PASS may establish only deterministic software semantics and simple analytical known answers for this exact synthetic corpus. It does not establish any real material, interface, package, product, reliability, safety, manufacturing, certification, procurement, or deployment claim.
