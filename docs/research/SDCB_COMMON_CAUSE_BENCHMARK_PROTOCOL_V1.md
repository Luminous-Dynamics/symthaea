# SDCB common-cause and correlated-measurement benchmark protocol v1

**Status:** preregistration-style specification; no benchmark tournament has been run.  
**Reference:** `scripts/research/sdcb_common_cause_measurement.rs`  
**Fixture:** `SDCB_COMMON_CAUSE_MEASUREMENT_V1.example.json`

## 1. Claim under test

Test whether a scientific decision agent can identify when multiple sensors share systematic uncertainty and select experiments that discriminate scientific hypotheses from shared environment, batch, reference-chain, or acquisition-system effects.

Target claim:

> At matched evidence access and resource budgets, common-cause-aware experiment selection reduces held-out scientific decision regret and false independent-confirmation claims relative to i.i.d. sensor fusion and same-device/cross-device repeat policies, without degrading well-specified independent-sensor tasks.

This is not a claim that shared-error modelling or hierarchical sensor calibration is novel. The proposed contribution is integrating these models into an evidence-gated, decision-focused experiment policy and evaluating their effect on scientific conclusions.

## 2. Task families

| Family | Hidden data-generating process | Required behaviour |
|---|---|---|
| C0: independent sensors | Independent local errors; no common factor | Do not invent common-cause uncertainty or over-calibrate |
| C1: shared offset | All devices share a latent additive/binary offset | Distinguish scientific state from common bias |
| C2: shared environmental drift | Common state changes between stages | Discount stale reference observations and plan refresh/check experiments |
| C3: local plus common bias | Shared factor plus device-specific random effects | Attribute uncertainty to shared and local sources |
| C4: shared reference-chain error | Devices were calibrated through the same uncertain reference | Do not count shared traceability-chain evidence as independent |
| C5: batch/preparation effect | Measurements share sample preparation, operator, or lot-level effect | Design controls and blocked randomization to identify the batch term |
| C6: shared electronics/firmware | Sensors share clock, power, firmware or acquisition software | Detect correlated channel failures despite different device IDs |
| C7: correlated measurand | Sensors correctly observe a spatially/temporally correlated physical field | Distinguish genuine measurand correlation from correlated sensor error |
| C8: non-identifiability | Common state and scientific mechanism induce equivalent observations | Abstain or seek an intervention/reference that breaks equivalence |
| C9: calibration-transfer failure | Calibration is transferred from one device/cohort to another under shift | Preserve uncertainty and detect transfer invalidity |

C0–C3 are represented by current finite binary references. C4–C7 require richer reference-chain, batch, and measurand metadata or continuous random-effects models. C8 can be built as exact paired worlds. C9 requires held-out device cohorts and calibration transfer models.

## 3. Matched comparator arms

All methods receive the same scientific hypothesis set, device registry, reference-chain metadata, candidate experiments, resource vector, task seeds, and stopping rules.

1. No experiment / best current action.
2. Same-device repeat policy.
3. Cross-device majority/agreement policy.
4. I.i.d. sensor-noise Bayesian fusion (deliberate negative-control model).
5. Shared-reference-first and fixed calibration schedules.
6. Check-standard / control-chart trigger policy.
7. Random-effects or hierarchical common/local bias model.
8. Bayesian sensor-network calibration / co-calibration baseline.
9. EIG and greedy EIG-per-cost.
10. One-step EVSI and exact finite-horizon decision oracle on small cases.
11. R-I/R-IDeA where representativeness/proxy assumptions fit.
12. Robust finite-scenario maximin policy.
13. Common-cause-aware sequential policy.

Relevant literature baselines include the variational calibration-network method of *Modelling calibration uncertainty in networks of environmental sensors*, sensor-error correlation models, in-situ calibration/co-calibration methods, and NIST Measurement Assurance Program practices. Pin exact papers, code release, task generator, optimizer budget and preprocessing before tournament runs.

Do not compare acquisition values across EIG, EVSI, R-IDeA and maximin utility as if they were on the same scale. Keep the exact finite oracle labelled as an evaluator/reference, not a deployable model.

## 4. Negative controls

Freeze these before tuning:

- Different device IDs with the same unobserved common offset must not count as independent confirmation.
- Repeated samples under a shared reference chain must not be assigned independent calibration evidence.
- An instrument calibration likelihood cannot depend on the scientific hypothesis conditional on common and local instrument state.
- In a genuinely independent sensor task, common-cause modelling must not cause material decision-quality regression or wasteful reference measurement.
- Correlation in the measurand must not be misclassified as correlation in sensor error.
- A reference observation must lose value when the shared state drifts before the science measurement.
- Calibration data from one reference-chain identity must not silently update a different chain.
- A shared batch effect requires blocked/control measurements or an explicitly declared model; it must not be hidden in the residual noise.
- An observationally non-identifiable science/common-cause pair must not generate an unsupported mechanism claim.
- Every contingent policy branch must obey hard cost/safety limits.
- Invalid, stale, duplicate, revoked, misbound, or wrong-chain evidence must cause zero scientific- and instrument-belief mutation after production EGBT integration.

Include positive controls with valid evidence. Reject-all is not acceptable.

## 5. Metrics

Report separate families.

**Scientific decision quality**
- Bayes decision regret, worst-scenario regret, robust value of information;
- held-out observable prediction log score/Brier score;
- mechanism recovery conditioned on identifiability;
- false mechanistic or causal claim rate;
- calibrated abstention and selective risk.

**Common/local measurement inference**
- common-state and device-local posterior Brier/log scores;
- interval coverage for common and local effects;
- cross-device residual correlation estimation error;
- common-cause false-positive and false-negative rates;
- calibration-transfer error by device cohort/reference chain;
- common-state drift detection delay and false alarms.

**Experiment and metrology efficiency**
- science measurements, reference measurements and device replications;
- cost/time to first justified scientific decision;
- maximum branch cost and scenario-specific expected cost;
- calibration and measurement uncertainty budget;
- full cost/latency/energy/compute Pareto frontier where telemetry is available.

**Evidence assurance**
- wrong-device/reference-chain/batch identity acceptance;
- stale/duplicate/revoked receipt acceptance;
- false rejection of valid verified controls;
- zero-mutation rejection rate;
- exact posterior replay agreement;
- provenance completeness and traceability-chain completeness.

Do not reward estimation of the common state unless it improves a declared scientific or measurement-quality target. Do not collapse all dimensions into one weighted score.

## 6. Statistical protocol and competitive gates

- Start with 20 paired seeds per synthetic family; increase to achieve preregistered precision.
- Hold out complete common-cause mechanisms, device cohorts, reference chains, and batch structures.
- Freeze random-effects priors, transition matrices, calibration thresholds and acquisition hyperparameters on validation tasks only.
- Use task-family clustered bootstrap confidence intervals and paired comparisons with multiplicity correction.
- Publish failures, rejected receipts, consumed budget and all resource dimensions.
- Reproduce selected tasks with a second implementation or independent evaluator.

Proposed targets, not achievements:
1. 100% agreement with the exact finite oracle on frozen known-answer cases.
2. Zero false independent-confirmation claims in the shared-cause negative controls.
3. Zero calibration-leakage and hard-budget violations.
4. At least 10% lower held-out decision regret than the i.i.d./repeat-only baseline on preregistered C1–C3 tasks, with paired 95% confidence interval excluding zero.
5. Non-inferior decision quality on C0 tasks, where errors truly are independent.
6. At least 10% lower resource cost to matched decision quality on one shared-reference/calibration-sensitive task.
7. Zero unsupported mechanism claims on C8 observational-equivalence controls.
8. Independent reproduction on held-out device/reference cohorts.

The 10% thresholds are internal project targets, not industry standards.

## 7. External evaluation

Preserve the native metrics of Petri-bench, ActiveSciBench, Autonomous Science Bench, ASI-Bench and AstaBench. These suites do not all expose device identities, calibration chains, or controllable common-cause shifts. Add a measurement-assurance subtrack only where the task genuinely supports it; do not manufacture a common-cause score from unrelated tasks.

For actual sensor-network data, seek datasets with reference co-location, device IDs, timestamps, environmental context, and calibration/quality-control records. Report the limits of any public dataset that lacks an independent reference or cannot distinguish measurand correlation from sensor-error correlation.

## 8. Implementation limits

Current Rust reference: binary common state, binary device-local states, at most three devices, four hypotheses/scenarios, three candidate experiments, binary outcomes, two stages, one common Markov transition, and local transitions conditionally independent given old/new common state.

Not implemented: continuous Gaussian/random-field models, multiple common factors, cross-device residual dependence after conditioning on the common state, multiple batches, spatial/temporal covariance kernels, active calibration resets, multi-stage decision trees, EGBT receipt integration, HDC/CfC integration, or physical metrology.

The next generalization should replace the single binary (g) with a low-rank or structured continuous common-factor model, retain a small exact finite oracle for verification, and evaluate approximate HDC/CfC proposals against it.
