# Drift-aware, multi-instrument sequential scientific design v1

**Status:** bounded exact two-stage research reference; not a production planner.  
**Implementation:** [`scripts/research/sdcb_drift_replication.rs`](../../scripts/research/sdcb_drift_replication.rs)  
**Schema:** [`SDCB_DRIFT_REPLICATION_V1.schema.json`](SDCB_DRIFT_REPLICATION_V1.schema.json)  
**Example:** [`SDCB_DRIFT_REPLICATION_V1.example.json`](SDCB_DRIFT_REPLICATION_V1.example.json)  
**Expected result:** [`SDCB_DRIFT_REPLICATION_V1.expected.json`](SDCB_DRIFT_REPLICATION_V1.expected.json)  
**Benchmark protocol:** [`SDCB_DRIFT_REPLICATION_BENCHMARK_PROTOCOL_V1.md`](SDCB_DRIFT_REPLICATION_BENCHMARK_PROTOCOL_V1.md)

## 1. Research question

A calibration result is conditional on the instrument state at the time it was measured. If an instrument changes between calibration and scientific measurement, the old calibration is not automatically valid. Likewise, two observations from one device share its latent systematic bias; measurements from separate devices have separate bias states, although they may still share common-cause errors.

The reference adds two explicit dimensions to SDCB:
- **Time:** a Markov transition for each instrument state between two experiment stages.
- **Identity:** every experiment names the instrument whose measurement channel generated its outcome.

The scientific hypothesis (h) and the binary instrument states (c_{d,t}) form a joint latent state. The initial prior can express independent or correlated instrument offsets. Each complete ambiguity scenario supplies a utility schedule, per-device transition matrices, and experiment likelihoods.

## 2. Exact two-stage decision contract

The contingent policy is (pi=(e_1,e_2(y_1),a(y_1,y_2))). At stage one, the first experiment observes the initial instrument state. Before stage two, each instrument evolves according to the scenario's Markov transition. The second experiment observes the transitioned state of its named device.

For the v1 conditional-independence model:

[
T_s(c'_1,ldots,c'_Dmid c_1,ldots,c_D)
=prod_{d=1}^{D} T_{s,d}(c'_dmid c_d).
]

The exact robust policy solves:

[
max_{pi}min_{sin S}
sum_{h,c_0,c_1,y_1,y_2}
p(h,c_0),
p_s(y_1mid h,c_{0,d(e_1)},e_1),
T_s(c_1mid c_0),
p_s(y_2mid h,c_{1,d(e_2(y_1))},e_2(y_1)),
U_s(h,a(y_1,y_2)).
]

The same contingent policy is evaluated in every scenario; it cannot read the hidden scenario identifier. Every first-outcome branch must fit the hard cumulative budget. Expected resource cost is reported separately from the maximum branch cost.

A `Calibration` experiment is checked for non-leakage: when the instrument state is held fixed, its likelihood must not depend on the scientific hypothesis. This prevents a nominal calibration operation from acting as an undeclared science measurement.

## 3. Frozen analytical control

The fixture has:
- Two equally likely scientific hypotheses.
- Three devices, A/B/C.
- Device A has an unknown binary offset and substantial drift between stages.
- Devices B and C have independent binary offsets, each with prior probability 0.10 of the flipped state.
- Measurement noise is conditionally independent, with 0.80 accuracy on B/C.
- A calibration standard and science measurement on A each have 0.95 accuracy.
- The decision set includes choose-h0, choose-h1, or abstain for utility 0.60.

Two ambiguity scenarios differ only in device-A transition probability:
- Nominal A-state flip probability: 0.02.
- Shifted A-state flip probability: 0.25.
- B and C remain stable between stages.

The exact finite maximin policy is:

1. Measure using device B.
2. Measure using device C, regardless of the first binary outcome.
3. Choose h0 if both readings are 0, choose h1 if both are 1, and abstain if they disagree.

| Policy/control | Nominal value | Shifted value | Worst-case value |
|---|---:|---:|---:|
| No experiment; abstain | 0.60000 | 0.60000 | 0.60000 |
| Repeat device B twice, with abstention on disagreement | 0.77200 | 0.77200 | 0.77200 |
| Independent devices B then C | 0.77848 | 0.77848 | 0.77848 |
| Calibrate A then measure A | 0.88880 | 0.70250 | 0.70250 |

The selected cross-device policy has robust gain 0.17848 over the no-experiment baseline. It exceeds same-device repetition by 0.00648 and avoids the calibration-staleness failure of device A under the shifted scenario.

These are independently enumerated analytical fixture values, not real sensor data, laboratory measurements, or benchmark results. The task's utility schedule intentionally gives abstention positive value; this is necessary to expose the value of a second independent measurement when two binary observations disagree.

## 4. What this adds to the instrument-aware v1 reference

The preceding instrument-aware reference assumed one static instrument state shared by both stages. This reference adds:
- Multiple named instruments with distinct latent states.
- A separate prior for each device while allowing an arbitrary joint prior over the finite state space.
- Scenario-specific Markov state transitions between the two stages.
- Calibration freshness as a consequence of elapsed-state transitions rather than a manually assigned confidence discount.
- Independent-device replication as a first-class experiment choice.
- Explicit robust comparison between cross-device replication, same-device repetition, and calibration followed by science.

The v1 transition model factorizes across devices conditional on their previous states and scenario. It cannot represent correlated post-stage common-cause transitions, continuous drift, or a calibration intervention that changes the device state. Those require later extensions.

## 5. Scientific context

The direction is connected to established work, rather than claiming that drift-aware calibration or independent replication are new in themselves:

- Oliveira, Sejdinovic, Howard & Bonilla, [Bayesian Adaptive Calibration and Optimal Design](https://papers.neurips.cc/paper_files/paper/2024/hash/673b9f8cba4ee9f15b88656a69761631-Abstract-Conference.html), NeurIPS 2024, jointly optimizes calibration inference and batch-sequential design.
- Yang, Dong & Wu, [Bayesian experimental design for model discrepancy calibration](https://doi.org/10.1016/j.jcp.2025.114469), *Journal of Computational Physics*, 15 January 2026, addresses design in the presence of model discrepancy.
- Li & Ma, [An Update on Measurement Error Modeling](https://doi.org/10.1146/annurev-statistics-040722-043616), *Annual Review of Statistics and Its Application* 2024, surveys measurement-error models and their assumptions.
- The 2026 measurement-science roadmap [From Signals to Measurands](https://doi.org/10.1021/acsmeasuresciau.6c00198) discusses independent-method reproducibility, measurement routes, and uncertainty contributions that may be hidden by shared standards or nuisance effects.

SDCB's contribution is a bounded decision-oracle synthesis: calibration, drift, device identity, independent replication, abstention, scenario robustness, and branch-wise resource limits are compared under one explicit policy contract. It does not replace metrology standards or the cited statistical methods.

## 6. Evidence and trust boundary

This reference deliberately does not mutate a persistent scientific belief ledger. It only evaluates a finite policy under supplied priors, likelihoods, transitions, and utilities.

A future EGBT integration must:
- bind the observation to the exact device ID, serial/asset identity, firmware/configuration digest, experiment proposal, calibration certificate, and capture interval;
- bind the applied calibration state/version and the state-transition assumption used by the likelihood;
- distinguish instrument authentication from metrological traceability and scientific validity;
- reject stale, duplicate, misbound, revoked, or unauthorized receipts with zero mutation of both scientific and instrument beliefs;
- preserve accepted and rejected evidence attempts for independent replay.

EGBT v1's `independently_verified` flag remains a test seam, not production attestation. A valid signature or receipt can establish provenance and integrity; it cannot by itself prove measurement quality or a scientific conclusion.

## 7. Limits

The Rust oracle is bounded to:
- at most four scientific hypotheses;
- at most three binary-state devices;
- at most four ambiguity scenarios;
- at most three terminal actions;
- at most four candidate experiments;
- binary outcomes and exactly two experiment stages;
- one fixed Markov transition between the two stages;
- conditionally independent device transitions given the old state and scenario;
- conditionally independent observation noise given hypothesis, device state, and scenario.

Not implemented: continuous-time or action-dependent drift, device-to-device common-cause transitions after stage one, calibration interventions that repair/reset the sensor, multi-stage policies, continuous observations, EGBT receipt integration, external benchmark adapters, HDC/CfC integration, or physical metrology.

**Evidence boundary:** six Rust tests are authored but not compiled or executed in this environment. The JSON fixture and exact result are independently validated outside Rust. No physical experiment, external benchmark tournament, or SOTA result is claimed.
