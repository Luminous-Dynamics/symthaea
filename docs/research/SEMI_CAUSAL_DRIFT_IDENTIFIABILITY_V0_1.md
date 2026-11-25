# SEMI causal-drift identifiability benchmark v0.1

**Issue:** #13616  
**Implementation:** `benchmark::causal_drift_interventions`  
**Status:** deterministic synthetic intervention generator, evidence-qualified channel contract, single-cause signature evaluator and authored tests; exact-head validation pending.

## Why this benchmark exists

A residual change says that the predictive relationship moved. It does not identify whether the physical process, the sensor, the calibration/reference chain, the product/recipe context, or the sampling mechanism changed.

The benchmark deliberately separates:
- **detection:** did an observable regime change?
- **localization:** which channel or tool/chamber changed?
- **identification:** does the available evidence distinguish a physical cause from competing explanations?
- **recommendation:** what additional observation would discriminate between remaining hypotheses?

The v0.1 implementation addresses only a deliberately small synthetic identifiability problem. It is not a general causal discovery algorithm.

## Structural measurement model

The challenge panel exposes four streams:

| Stream | Meaning | Role |
|---|---|---|
| Primary process sensor | Production sensor used by the VM | Target channel |
| Independent process sensor | Separately sourced process/sensor channel | Negative control for single-sensor drift |
| Physical metrology | Qualified metrology measurement | Physical target evidence |
| Reference check standard | Check-standard result through the reference measurement chain | Negative control for reference-chain drift |

Each stream contains baseline and post-intervention values, a source-artifact digest, transform digest, qualification-receipt digest, declared source independence, current qualification state and availability time.

The generator uses a common latent process shift, a primary-sensor additive bias and a reference-chain additive bias:

- primary sensor = process state + primary sensor bias + noise;
- independent sensor = process state + noise;
- physical metrology = process state + reference bias + noise;
- check standard = reference bias + noise.

The check-standard channel is deliberately part of the reference measurement chain: a reference-only calibration shift moves both physical metrology and the check-standard observation, while a process-only change moves both process sensors and the physical target but leaves the check standard unchanged.

## Identifiability rules

Only these predeclared single-cause signatures can be identified in v0.1:

- **Process-state shift:** primary sensor, independent sensor and physical metrology shift together; check standard remains stable.
- **Primary-sensor drift:** primary sensor moves; independent sensor, physical metrology and check standard remain stable.
- **Metrology/reference drift:** physical metrology and check standard move together; both process sensor streams remain stable.
- **Context-only shift:** the context key changes but all measurement channels remain stable. This is classified as a context transition, not a physical process fault.

Any missing required channel, insufficient support, unqualified evidence, future/unavailable evidence, malformed source/transform/qualification digest or non-finite observation prevents a causal claim. Multiple concurrent shifts or a pattern outside the declared signatures returns UNKNOWN rather than guessing.

The output explicitly sets:
- `source_authenticity_proven = false`;
- `physical_root_cause_proven = false`;
- `intervention_executed = false`;
- `equipment_control_authority = false`.

A SHA-256 digest binds the input evidence and diagnosis. It does not prove the underlying source is authentic or that the declared independent-channel relationship is physically true.

## Implementation and tests

The Rust implementation includes:
1. deterministic, seed-bound paired intervention fixture generation;
2. manifest-to-ground-truth signature validation, so fixture labels cannot contradict their own intervention deltas;
3. four typed measurement-channel roles and evidence/availability metadata;
4. fail-closed input validation;
5. deterministic channel-delta calculation and exact single-cause pattern checks;
6. explicit UNKNOWN on non-identifiable/confounded cases;
7. canonical channel ordering for input/diagnosis digest order invariance.

Ten authored tests cover process-only, sensor-only, reference-only and context-only interventions; missing negative control; unqualified/future evidence; confounding; ambiguous paired patterns; channel order invariance; malformed source digests; and false manifest labels.

## Benchmark tournament

Compare:
- SPC/EWMA/CUSUM and multivariate SPC;
- PCA/DPCA monitoring;
- M2D2-style multimodal drift detection and feature attribution;
- Bayesian change-point/HMM regime detection;
- residual + SHAP/attention explanation;
- causal hypothesis tests on the synthetic intervention SCM;
- evidence-only monitoring (qualification and event lineage without causal inference).

Score separately:
- detection delay and false alarms per 1,000 wafers/tool-week;
- cause macro-F1 only on identifiable fixtures;
- false causal-attribution rate on confounded/non-identifiable fixtures;
- UNKNOWN recall on missing-control, unknown-cause and unavailable-evidence fixtures;
- negative-control false-effect rate;
- intervention effect sign/magnitude error when the SCM identifies it;
- next-measurement information gain and metrology cost.

Do not score the correct UNKNOWN response as a missed cause. For a non-identifiable observation distribution, abstention is the desired result.

## Relation to literature

- Industrial causal fault diagnosis survey (Engineering Applications of Artificial Intelligence, 2025): https://doi.org/10.1016/j.engappai.2025.111376
- M2D2 multimodal multi-machine drift detection and explanation (Applied Sciences, 2025, 15(12), 6500): https://www.mdpi.com/2076-3417/15/12/6500
- Bayesian VM framework connecting recipe, chamber, sensor and wafer variables with concept drift and observability error (IEEE TSM, 2022): https://doi.org/10.1109/TSM.2022.3170270
- NIST online GP dynamic CMP sampling (IEEE TSM, 2025): https://doi.org/10.1109/TSM.2025.3531920
- NIST SI for Manufacturing / TIDAL integration-test-bench and human-machine teaming work: https://www.nist.gov/programs-projects/si-manufacturing

These are related comparators, not proofs that synthetic structural equations recover a real fab's causal structure.

## Claim ceiling and next step

The intervention generator proves only that the declared software contract distinguishes its own declared synthetic scenarios. It does not prove real process causality, physical independence of sensors, traceability, yield improvement, or safe process correction.

The next extension should add paired observationally equivalent fixtures in which different causal worlds produce the same observed marginals, then introduce a declared intervention or independent physical check that breaks the equivalence. Only after those fixtures pass should a causal hypothesis-ranking or measurement-value-of-information policy be added. No recipe, APC/R2R, wafer disposition or equipment actuation is authorized.
