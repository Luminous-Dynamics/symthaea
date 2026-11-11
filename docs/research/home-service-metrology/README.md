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

## Current standards crosswalk — 11 November 2026

- **ISO 5725-1:2023** supplies the general trueness/precision principles, definitions and study organization. It concerns continuous-scale methods yielding a single reported result and points to other standards for proficiency testing and reference-material production.
- **ISO 5725-2:2025 (Edition 3)** is the current published basic interlaboratory method for repeatability/reproducibility of a standard measurement method. ISO 5725-2:2019 is withdrawn and must not be described as current.
- **ISO 5725-3:2023** covers intermediate precision and alternative collaborative-study designs; use where day/operator/equipment/time effects require intermediate-precision analysis.
- **ISO/IEC 17043:2023** covers competence and impartiality of proficiency-testing providers. **ISO 13528:2022**, together with **Amendment 1:2026**, provides statistical methods for proficiency testing by interlaboratory comparison.
- **IEC TR 63250:2021** remains a household-appliance test-method reference for repeatability, reproducibility, uncertainty and round-robin testing; its scope explicitly excludes production-unit variability and the representativeness of normal household use.

These standards serve different questions. A proficiency test evaluates participant performance against an assigned value or consensus criterion; it is not itself a crossed product reproducibility study. A successful method round-robin does not establish manufactured-population variation or household ecological validity. Therefore the four strata remain distinct, and laboratory proficiency/scope evidence is a separate applicability/competence gate rather than a fifth replication label.

Official references: [ISO 5725-1:2023](https://www.iso.org/standard/69418.html), [ISO 5725-2:2025](https://www.iso.org/standard/90054.html), [ISO 5725-3:2023](https://www.iso.org/ics/17.020/x/), [ISO/IEC 17043:2023](https://www.iso.org/standard/80864.html), [ISO 13528:2022/Amd 1:2026](https://www.iso.org/standard/90057.html), [IEC TR 63250:2021](https://webstore.iec.ch/en/publication/65090).