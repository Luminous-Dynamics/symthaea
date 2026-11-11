# Home service metrology study-design contract v0.1

Parent: Symthaea [HOME-SVC-METROLOGY-001 #10341](https://github.com/Luminous-Dynamics/symthaea/issues/10341).

## Why this exists

IEC TR 63250:2021 explicitly treats repeatability, reproducibility and measurement uncertainty for appliance test methods, but does not cover production-unit variability or how closely a method represents normal household use. A single prototype/lab result is therefore not enough to support product superiority.

This contract keeps four evidence strata separate:
1. within-unit repeatability;
2. between-laboratory reproducibility;
3. production-unit/lot variation;
4. household ecological validity.

## Design principle

Sample size is not a fixed magic number. Pilot variance components inform a preregistered power/precision analysis. The manifest deliberately says NOT_SELECTED / NOT_PERFORMED until a product endpoint, variance pilot, meaningful effect and independent review exist. The synthetic minimums (two clusters for an estimable stratum and one independent confirmatory holdout) are structural feasibility checks, not sufficient confirmatory sample sizes.

## Fair comparison

Every arm must use the same protocol digest, primary endpoint and functional unit, exogenous weather/demand/draw traces, operating conditions, exact model/revision and frozen adaptation/compute/human-time budgets. Run order is randomized or blocked; assessors are blinded or adjudication is independent; missingness, exclusions and outliers are preregistered. A confirmatory lab, lot and household/site remain untouched by adaptation.

## Statistical contract

Use a justified hierarchical/cluster-aware analysis. Preserve lab, lot, unit, operator and household/site levels; do not count repeated samples, repeated photos/items or repeated time windows as independent products/households. Report variance components and uncertainty separately. Keep worst-stratum service failures visible.

## Safety and rights

Consent, withdrawal, purpose limitation and privacy are hard gates. Safety, service, accessibility and consent failures cannot be compensated by savings in energy or cost. The four metric/evidence strata are not collapsed into one replication badge.

## Claim ceiling

This is a preregistration gate only. The manifest is not a study protocol, product selection, performance result, lab qualification or household approval. All results are synthetic. Its maximum outcome is `ELIGIBLE_FOR_PREREGISTRATION_REVIEW_ONLY`.
