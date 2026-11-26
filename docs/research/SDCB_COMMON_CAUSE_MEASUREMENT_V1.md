# Common-cause measurement uncertainty and shared-reference experiment design v1

**Status:** bounded exact two-stage reference; not a production scientific or metrology system.  
**Implementation:** [`scripts/research/sdcb_common_cause_measurement.rs`](../../scripts/research/sdcb_common_cause_measurement.rs)  
**Schema:** [`SDCB_COMMON_CAUSE_MEASUREMENT_V1.schema.json`](SDCB_COMMON_CAUSE_MEASUREMENT_V1.schema.json)  
**Example:** [`SDCB_COMMON_CAUSE_MEASUREMENT_V1.example.json`](SDCB_COMMON_CAUSE_MEASUREMENT_V1.example.json)  
**Expected result:** [`SDCB_COMMON_CAUSE_MEASUREMENT_V1.expected.json`](SDCB_COMMON_CAUSE_MEASUREMENT_V1.expected.json)  
**Benchmark protocol:** [`SDCB_COMMON_CAUSE_BENCHMARK_PROTOCOL_V1.md`](SDCB_COMMON_CAUSE_BENCHMARK_PROTOCOL_V1.md)

## 1. Research question

Different device IDs do not imply independent evidence. Sensors can share:
- a laboratory environment, temperature field, power supply, clock, firmware, or acquisition pipeline;
- a batch effect or preparation protocol;
- a common reference standard, calibration transfer, or traceability-chain link;
- a shared unobserved influence acting on every device.

When a common latent cause affects multiple measurement channels, cross-device agreement can repeat the same systematic error. A planner that treats device identity as independence can overstate confidence and choose the wrong experiment.

This reference represents a shared common-cause state (g) plus a local state (c_d) for each device (d). The scientific hypothesis is (h). The latent state is the full joint distribution:

[
p(h,g,c_1,ldots,c_D).
]

A measurement on device (d) depends on (h), the shared state (g), and that device's local state (c_d). A shared reference-calibration experiment is conditionally independent of (h) given (g,c_d).

## 2. Common-cause transition model

The common state evolves once between stage one and stage two:

[
P(g'mid g,s).
]

The local state of each device evolves conditional on the old and new common state:

[
P(c'_1,ldots,c'_Dmid c_1,ldots,c_D,g,g',s)
=
prod_{d=1}^{D}P(c'_dmid c_d,g,g',s).
]

This structure makes a common shock couple all devices through (g'), while retaining device-local residual state. It supports arbitrary initial correlation through the declared joint prior. The transition factorization is still an assumption: correlated residual local transitions after conditioning on the common state are not represented in v1.

Each experiment likelihood is (P(ymid h,g,c_d,e)). One policy ((e_1,e_2(y_1),a(y_1,y_2))) is evaluated over all complete ambiguity scenarios, with a hard maximum path budget. The exact maximin selector cannot observe the hidden scenario identity.

## 3. Analytic negative control: three device IDs, one shared bias

The fixture has two equally likely scientific hypotheses, three device channels, and one shared binary common-cause state. Device-local offsets begin known at zero. The shared state (g) is equally likely to be zero or one.

All three devices have 95% conditional measurement accuracy:
- science-A and science-B report (hoplus g);
- the shared-reference standard reports (g), not (h).

The common state is stable in one scenario and has a 0.20 flip probability between stages in the shifted scenario. Local device states remain stable in this fixture.

| Strategy/control | Stable common state | 0.20 common-state flip |
|---|---:|---:|
| No experiment; best action | 0.500 | 0.500 |
| Repeat science-A twice | 0.500 | 0.500 |
| Science-A then science-B | 0.500 | 0.500 |
| Shared-reference calibration then science-A | **0.905** | **0.743** |

The exact robust policy first measures the shared reference, then measures science-A. It chooses the scientific hypothesis by XOR of the two outcomes. Its worst-case value is 0.743, for a robust gain of 0.243 over the no-experiment baseline.

The equality of the two science-only controls is the important negative control: using different instruments does not resolve a bias they share. A reference measurement that directly observes the common cause can resolve the confounding, but its value falls when the shared state changes between the two stages.

These are exact finite analytical results from the frozen model, not physical sensor measurements or benchmark results.

## 4. What this adds to the drift-aware reference

The preceding multi-instrument reference #13793 contains independent device Markov transitions. This v1 extension adds:
- an explicit common environment/reference state shared by all instruments;
- common-state transition dynamics that couple device channels;
- device-local state transitions conditional on the common-state transition;
- likelihoods that depend on scientific, common-cause, and local device states;
- a shared-reference calibration experiment;
- negative controls for same-device repetition and cross-device agreement under a shared cause.

The central distinction is:

[
	ext{different device identity}
otRightarrow	ext{independent error}.
]

The model can represent a common batch effect, environmental shift, common reference-chain error, or shared electronics/firmware effect as a common latent state. It does not identify which physical mechanism is responsible; that requires domain-specific metadata and measurement design.

## 5. Connection to existing metrology and statistical work

This is a systems-level synthesis of established measurement-science principles, not a claim to invent random-effects models or sensor-network calibration.

- NIST's [Measurement Assurance Programs](https://www.nist.gov/calibrations/policies) compare transfer standards across a reference laboratory and a customer's measurement process, and recommend in-house check standards and statistical control. NIST emphasizes that a calibrated instrument alone does not make a particular measurement result traceable; the chain and uncertainty contributions must be documented.
- The [NIST measurement-process-control handbook](https://www.itl.nist.gov/div898/handbook/mpc/section2/mpc21.htm) distinguishes variation in the measurand from measurement-system errors and describes check-standard monitoring for changes in bias and variability.
- [Measurement Uncertainty Evaluation for Sensor Network Metrology](https://www.mdpi.com/2673-8244/5/1/3) specifically discusses temporal and cross-sensor error correlations, including shared ambient conditions and shared reference instruments.
- [Modelling calibration uncertainty in networks of environmental sensors](https://academic.oup.com/jrsssc/article/72/5/1187/7250331) proposes uncertainty-quantified calibration across sensor networks and reports improvement over a multi-hop calibration baseline on synthetic and real air-quality data.
- [Modelling and determining correlations in sensor networks](https://doi.org/10.1016/j.measen.2024.101793) shows why i.i.d. sensor-error assumptions can change uncertainty estimates for aggregate/interpolated measurements.
- [Automation in sensor network metrology](https://doi.org/10.1016/j.measen.2024.101799) surveys in-situ calibration and co-calibration approaches.

SDCB's proposed contribution is to make common-cause uncertainty affect **which experiment is chosen and whether a scientific decision is justified**, while retaining hard resource and evidence-transition contracts. These established statistical and calibration approaches must be included as comparators.

## 6. Benchmark and baseline requirements

Do not compare only against naive i.i.d. sensor fusion. At minimum include:
1. i.i.d. device-noise baseline (negative control showing the danger of the assumption);
2. same-device repeat policy;
3. cross-device agreement policy;
4. shared-reference-first policy;
5. fixed and adaptive calibration schedules;
6. random-effects or hierarchical shared-bias model;
7. multi-sensor Bayesian calibration / co-calibration;
8. the exact finite common-cause oracle on small episodes;
9. robust maximin and decision-focused EVSI policies;
10. R-I/R-IDeA and multi-fidelity BO where assumptions and task objectives match.

Keep the upstream benchmark metric unchanged. Report scientific prediction loss, decision regret, common-state and local-state inference, cross-device residual correlation, false independent-confirmation rate, calibration resource expenditure, abstention, and evidence replay separately.

## 7. Limits and next work

The Rust reference is bounded to four hypotheses, three devices, one binary common state, one binary local state per device, four scenarios, three actions, three experiment candidates, binary outcomes, and exactly two experiment stages.

The common state follows one Markov transition between stages. Device-local transitions are conditionally independent given old/new common state and old local state. The reference does not support multiple common factors, continuous Gaussian/random-field effects, spatial correlation, multiple stages, learning transition parameters, sensor-network co-calibration from continuous observations, EGBT receipt integration, or physical metrology.

Before any deployment claim:
- add multiple common factors and correlated residual device transitions;
- use source-bound environment/batch/power/reference metadata;
- estimate cross-sensor residual correlation on control standards and independent reference campaigns;
- distinguish shared measurement error from genuine correlation in the measurand;
- bind device identity, common-cause context, calibration chain, uncertainty budget and capture interval into EGBT;
- test on sensor-network calibration data and held-out physical devices.

**Evidence boundary:** eight Rust test functions are authored but not compiled or executed in the current environment. The frozen fixture and policy values were independently exhaustively enumerated outside Rust. No physical test, external benchmark tournament, or SOTA result is claimed.
