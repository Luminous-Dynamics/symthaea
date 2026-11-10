# EV SOH Benchmark Evidence Manifest v0.1

**Status:** Executable protocol/provenance gate prototype. It does not train or validate an SOH estimator.
**Owner:** [EV-SOH-REF-001 #11342](https://github.com/Luminous-Dynamics/symthaea/issues/11342).

## Research basis

Li and Wang, *Interpreting Public-Dataset SOH Benchmarks: A Systematic Review of Evaluation Protocols and Reproducibility Evidence*, Batteries 12(11), 462, published 4 November 2026: https://doi.org/10.3390/batteries12110462.

The review mapped 408 studies and examined a nested, non-representative detailed subset of 168. All four core interpretability elements were simultaneously identifiable in 95/168 (56.5%); independent physical-battery generalization evidence in 70/168 (41.7%); uncertainty evidence in 20/168 (11.9%); and explicit public implementation evidence in 2/168 (1.2%). These are reporting frequencies in that nested subset, not population-wide estimates or direct study-quality scores.

Rizanov et al., *Toward Trustworthy and Transferable SOH/RUL Estimation for Lithium-Ion Batteries*, Batteries 12(7), 255, 15 July 2026, recommends blind physical holdouts, worst-domain performance, calibration, pack-realism tests and staged evidence from synthetic and controlled cells through pack and fleet shadow: https://doi.org/10.3390/batteries12070255.

## Paradigm shift

Benchmark results are protocol-conditioned evidence, not free-floating MAE scores. This manifest binds dataset lineage, operational target, physical train/validation/test relationships, preprocessing fit scope, target adaptation, metric aggregation, uncertainty and compute environment. Hashed physical identities permit overlap checks without disclosing raw battery, vehicle or supplier identifiers.

## Physical evaluation modes

| Mode | What it supports | What it does not support |
|---|---|---|
| same_cell_temporal | Future observations on cells represented during development | Unseen-cell transfer |
| unseen_cell | Transfer to cells absent from train/validation | Unseen packs/vehicles unless those are also held out |
| unseen_module | Module-level holdout | Generalization to all packs or fleets |
| unseen_pack | Pack-level holdout | Unseen vehicle/manufacturer if still represented |
| unseen_vehicle | Vehicle-level holdout | Unseen climate/chemistry unless separately held out |
| unseen_lab | Lab/protocol context holdout | Generality beyond declared labs |
| unseen_domain | Declared domain holdout | Untested domains not represented in the manifest |

All three partitions carry entity-level sample and hierarchy digests. The selected group level must match the evaluation mode. For unseen-* modes, group identities must be pairwise disjoint across train, validation and test. Sample digests must always be disjoint. For same-cell temporal evaluation, every validation/test cell must have training history and the train, validation and test windows must be strictly chronological and non-overlapping.

## Leakage and adaptation controls

Preprocessing records name the transform, fit scope and transform digest. Validation/test-label fitting is prohibited. A fixed predeclared transform additionally binds its predeclaration time and digest, and that time must precede the frozen split timestamp. Label-free target adaptation is allowed only when explicitly declared with a nonzero unlabeled target budget and exactly zero target-label budget. Target label adaptation is not part of v0.1.

The manifest distinguishes the fitted transform from the learned estimator. A future extension should bind feature/label extraction, model selection, hyperparameter search, calibration and test-set access to a single workflow receipt, with target-label access separately budgeted.

## Target definition

The operational target binds a target-definition digest, numerator, denominator, reference test protocol, reference conditions, unit and label uncertainty evidence. “SOH” alone is not a complete target: nominal capacity versus measured initial capacity, reference protocol and condition can change its meaning and scale.

## Metrics, aggregation and uncertainty

Every metric has a formula, unit, aggregation rule, sample count and result digest. Stratified metrics must cover the exact physical identities in the held-out test partition, and stratum sample counts must sum to the reported count. Macro values are replay-checked from the strata; nearest-rank p95 and worst-domain values are also recomputed. Worst-domain error metrics use the maximum stratum error, while coverage and R² use the minimum. A worst-domain result requires at least two named domain strata.

Uncertainty status is explicit. A reported interval method requires nominal levels, a held-out validation or external calibration source, and matching 90%/95% coverage metrics stratified by domain. Train-only calibration cannot substantiate held-out coverage. Not-reported uncertainty cannot be disguised as an empty but otherwise successful record.

## Reproducibility envelope

Bind source commit, dependency-lock digest, toolchain and digest, hardware, seeds, wall time, peak memory, optional energy, environment digest, dataset version/digest, subset digest and split-manifest digest. This is enough to replay the declared experiment when source and data rights permit; the manifest does not imply that third parties can access restricted datasets.

## Frozen corpus

Fifteen synthetic software cases: clean unseen-cell; shared-cell leakage in unseen-cell; validation/test-label preprocessing leakage; valid same-cell temporal; duplicate sample identity across splits; explicit label-free target adaptation; zero-budget target adaptation; aggregation/group-level mismatch; valid fixed predeclared transform; late predeclaration; metric strata outside the test groups; metric stratum count mismatch; incorrect aggregate value; train-only uncertainty calibration; and reported uncertainty without held-out domain coverage. Four manifests are accepted and eleven adversarial controls rejected. These fixtures test protocol gates only and are not physical battery evidence.

## Benchmark tournament

Use identical datasets, target definitions and split manifests for physical/statistical and equivalent-circuit baselines, ridge/SVR and tree ensembles, TCN/LSTM, Transformer and MoE, domain adaptation, test-time adaptation and abstention baselines. Compare adaptation with identical target-label or label-free budgets.

Report per-cell/per-pack/per-vehicle and worst-domain MAE/RMSE, R², interval coverage/score, selective risk/coverage, OOD detection, data required to adapt, inference latency, memory, energy and test burden. Do not claim SOTA by comparing non-matched splits.

## Claim ceiling

The auditor validates structure and internal consistency. It does not verify raw dataset authenticity, independently certify hashes, prove declared splits match the underlying data, evaluate a model, or establish battery/vehicle safety. Physical group identity digests are producer-declared commitments; independent audit still requires access to the corresponding identity mapping or a trusted external attestation. Accordingly, the current report labels physical evidence as producer-declared protocol evidence, not independently attested physical generalization.
