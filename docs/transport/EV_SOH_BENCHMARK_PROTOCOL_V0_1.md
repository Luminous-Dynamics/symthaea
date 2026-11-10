# EV SOH Benchmark Protocol v0.1

**Status:** Evidence-qualified benchmark manifest and split/leakage audit. Not an SOH estimator or a claim that one model is SOTA.
**Owner:** [EV-SOH-REF-001 #11342](https://github.com/Luminous-Dynamics/symthaea/issues/11342).

## Research basis

Li & Wang, *Interpreting Public-Dataset SOH Benchmarks: A Systematic Review of Evaluation Protocols and Reproducibility Evidence*, Batteries 2026, 12(11), 462, published 4 November 2026: https://doi.org/10.3390/batteries12110462

The review mapped 408 studies and analyzed a nested 168-study subset. It reports that the four Core interpretability elements were simultaneously identifiable in 95/168 (56.5%); preprocessing fitting scope was assessable in 40/168 (23.8%); leakage assessability in 64/168 (38.1%); uncertainty evidence in 20/168 (11.9%); and explicit article-reported public implementation evidence in 2/168 (1.2%). These are descriptive reporting frequencies in a nested, non-prespecified representative subset, not population prevalence, study-quality scores or proof of actual leakage. The review's main implication is that target, physical train/test relation, preprocessing fitting scope, metric aggregation and uncertainty must accompany the headline error.

## Protocol principles

A benchmark result is a reproducible experiment specification, not a headline MAE. The v0.1 manifest binds:

- dataset name/version, source/license, retrieval date and data digest;
- explicit SOH/capacity target, numerator, denominator basis, reference protocol and target-definition digest;
- physical evaluation relation and split manifest digest;
- observation-to-cell/pack/vehicle/lab/domain identity digests, partition and time index;
- preprocessing fit scope and model-selection/test-label isolation;
- MAE/RMSE, unit, aggregation, sample count and result digest;
- uncertainty method, nominal and observed interval coverage, calibration partition and group-disjointness evidence;
- source commit, toolchain, dependencies, hardware, wall time, memory and energy.

## Physical evaluation relations

These are different scientific claims and must not be collapsed:

| Relation | What it tests | Required rule |
|---|---|---|
| Same-cell temporal | Later trajectory for an already represented cell | For each cell: train time < validation time < test time |
| Unseen-cell | Transfer to new physical cells | Cell identity digests disjoint across train/validation/test |
| Unseen-pack | Transfer to packs absent from development | Pack identity digests disjoint |
| Unseen-vehicle | Transfer to new vehicles | Vehicle identity digests disjoint |
| Unseen-lab | Transfer to an independent lab | Lab identity digests disjoint |
| Unseen-domain | Transfer to a held-out domain | Domain identity digests disjoint |
| Cross-fidelity | Transfer between levels such as lab cell, module/pack or fleet | Domain/fidelity identity digests disjoint and fidelity declared |

A same-cell temporal test is valid when correctly ordered, but it does not establish unseen-cell generalization. The protocol auditor verifies this distinction from the identity digests; it does not infer physical identity from sample rows.

## Hard gates

- Every observation references a declared dataset digest and unique observation ID.
- Group-disjoint relations have zero overlap on their declared physical identity.
- Same-cell temporal relations have complete train/validation/test trajectories with strictly ordered time boundaries.
- Fitted preprocessing and feature selection may use train-only data or fixed predeclared transforms; validation/test/all-data fitting is rejected.
- Test labels may not influence feature selection, model selection or tuning.
- SOH requires an explicit nominal or initial-measured-capacity denominator; direct capacity must be reported in Ah without pretending it is SOH.
- Results require MAE and RMSE plus macro-per-cell and worst-domain aggregation; sample-pooled error alone is insufficient.
- An uncertainty claim requires nominal and observed coverage plus validation calibration provenance and stated group disjointness.
- Dataset/target/split/result/execution digests are retained in the deterministic report.

## Frozen synthetic corpus

Eleven cases: valid unseen-cell and same-cell temporal designs; group-overlap leakage; preprocessing fit on all data; test-label selection; undeclared dataset digest; pooled-only metrics; relation/key mismatch; missing SOH denominator; duplicate observation ID; and unsupported uncertainty calibration.

The corpus is synthetic protocol testing. It does not contain real battery measurements or train a model.

## Run

`cargo fmt --manifest-path tools/ev-soh-benchmark-protocol-audit/Cargo.toml -- --check`

`cargo test --manifest-path tools/ev-soh-benchmark-protocol-audit/Cargo.toml`

`cargo run --manifest-path tools/ev-soh-benchmark-protocol-audit/Cargo.toml -- --schema schemas/transport/ev_soh_benchmark_protocol_v0_1.schema.json --corpus fixtures/ev_battery/soh_benchmark_protocol_scenarios_v0_1.json`

## Benchmark tournament

For every benchmark relation, compare a physical/statistical baseline, SVR/tree models, TCN/LSTM, Transformer, MoE/domain adaptation and a simple sample-count/early-trajectory baseline on exactly the same train/validation/test asset identities. Model selection must be label-free on the held-out target domain unless a declared adaptation protocol permits a fixed, budgeted amount of target data.

Report:
- MAE and RMSE in the declared target unit;
- per-cell/per-pack and worst-domain errors, plus sample-pooled values as secondary summaries;
- 90%/95% interval coverage, interval score or NLL, calibration by domain and selective risk/coverage;
- OOD detection and abstention, plus false reassurance near maintenance/replacement thresholds;
- label/measurement acquisition cost, wall time, energy and peak memory;
- performance-vs-data curves at 1/2/5/10/20/50% training data with training-side-only curation.

A target such as ≤2 percentage-point MAE is meaningful only for a named target, dataset, physical split, domain and aggregation. We should not compare non-matched published headline errors as if they ranked model families.

## Claim ceiling

This audit verifies whether a benchmark manifest supports the declared protocol claim. It does not verify the truth of labels, prove sensor calibration, establish physical accuracy, certify a BMS or validate transfer to a field fleet. Synthetic fixtures cannot substantiate real battery performance.
