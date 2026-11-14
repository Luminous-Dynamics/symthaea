# AIRROAD-DECISION-DIRECTED-MEASUREMENT-001 — solver-witness-aware evidence acquisition

**Status:** V0.1 exact-rational policy discriminator  
**Tracking issue:** [#11958](https://github.com/Luminous-Dynamics/symthaea/issues/11958)  
**Composes with:** #11946 matched IQC benchmark, #11948 Rust IQC solver, #11923/#11924 adaptive mesh, #11887 adaptive excitation, #11914 shared calibration dependencies

> This is a policy-discrimination reference. It does not claim that a real measurement guarantees an uncertainty contraction or that a physical excitation is safe.

## 1. Research question

Can AIRROAD choose the next evidence action based on its effect on the robust engineering decision, rather than the size of an uncertainty interval or interpolation remainder?

This is not a claim to invent active experiment design. Robust and control-oriented experiment design have a substantial literature. The AIRROAD target is narrower: use the actual continuous-frequency robust-control bottleneck, solver witness, shared nuisance coordinates and evidence lineage to rank feasible next actions.

## 2. Why maximum uncertainty is not enough

The largest uncertainty interval need not dominate the robust stability/performance margin. Likewise, a statistically informative measurement may not change the current engineering disposition.

A solver-witness-aware policy should combine:
- active frequency interval and limiting IQC constraint;
- dual sensitivity or another independently checkable margin sensitivity;
- worst-case uncertainty direction;
- affected shared calibration/acquisition coordinates;
- candidate action cost, excitation feasibility and risk;
- the action's guaranteed or outcome-conditioned set update.

A witness is stale if any bound model, topology, operating point, frequency range, port order, uncertainty identity, calibration, or solver input digest changes. Stale witnesses must not drive action ranking.

## 3. Exact-rational discriminator

V0.1 uses two nonnegative hard uncertainty coordinates:

[
x\in[0,1],\qquad y\in[0,4/5].
]

The synthetic robust-margin lower bound is:

[
m_{\mathrm{lower}}
=1-\frac{1}{10}\,\overline{x}-2\,\overline{y}.
]

The initial margin is exactly (-7/10).

Two candidate actions are supplied. Both have unit cost and risk (1/2), under a synthetic risk budget of (1):

| Action | Contracted hard set | Resulting lower margin | Margin gain |
|---|---|---:|---:|
| Refine x | x in [0, 1/10], y unchanged | -61/100 | 9/100 |
| Calibrate y | y in [0, 1/10], x unchanged | 7/10 | 7/5 |

A width-only policy selects x because its initial interval width is (1), larger than y's (4/5).

A robust-margin policy selects y because its exact margin improvement is (7/5), versus (9/100) for x. The y action moves this synthetic case to a positive robust lower margin, whereas the x action does not.

This fixture is intentionally simple: it isolates the failure mode of ranking evidence solely by uncertainty width. The contractions are stipulated synthetic guaranteed updates. They are not empirical claims about a sensor, physical test, or calibration procedure.

## 4. Hard-set and statistical semantics

### Hard-set acquisition

For a real hard-set measurement action, do not simply shrink an interval. Define:
- the prior admitted joint set;
- the action's excitation and hardware constraints;
- measurement error / calibration bounds;
- the full set of possible observations;
- the posterior set for every possible observation;
- the resulting worst-case robust margin and disposition;
- cost, risk and authority.

The action is useful only if the complete outcome partition is covered. If any outcome remains unresolved, preserve NOT_ESTABLISHED / UNKNOWN. If no guaranteed contraction follows from the action model, report no guaranteed progress.

### Statistical acquisition

An expected-value or information-gain policy requires a declared statistical observation model. Report calibration, empirical coverage, proper scoring rules and false-safe risk. Do not use a covariance matrix as a deterministic hard-set certificate.

The hard-set and statistical policies must be evaluated separately, with distinct metrics.

## 5. Solver-witness action loop

1. Solve the matched robust-control problem and preserve solver status, active constraints, multipliers, residuals, model identity and input digest.
2. Identify uncertain coordinates and frequency/port channels contributing to the limiting margin.
3. Generate candidate actions from an approved action library.
4. Reject actions that cannot excite the limiting channel, exceed risk/cost/authority limits, or rely on stale evidence.
5. Predict the action's outcome-conditioned posterior sets or its certified calibration improvement.
6. Rank the remaining actions by worst-case decision improvement per cost for hard sets, or by calibrated expected decision value for statistical models.
7. Execute only under a separately approved physical test authority.
8. Bind the observation to immutable receipts, update the joint uncertainty model, and solve again.

This should be compared to uniform frequency sweeps, largest-remainder refinement (#11924), largest-width selection, information-gain/D-optimal selection where justified, and model-discrimination selection (#11887).

## 6. Benchmark metrics

- Robust-margin improvement per action and per unit cost.
- Budget required to reach a definitive disposition.
- Number of frequency knots, port excitations, repeats and calibration actions.
- Fraction of actions that change the disposition versus only improve generic model fit.
- False robust-pass and false-abstention rates.
- Worst-case margin over all hard-set observation outcomes.
- Statistical expected value of sample information and proper-score/calibration results, reported separately.
- Solver time, action-scoring time and witness stability.
- Shared-coordinate preservation across all affected frequency/port records.
- Rejected actions due to physical feasibility, risk, authority or stale evidence.

## 7. Adversarial campaign

1. Largest-width policy chooses the wrong coordinate in the exact fixture.
2. High solver sensitivity but no admissible excitation exists for that port.
3. A measurement changes a marginal projection but not the shared joint coordinate.
4. One calibration action updates several knots and is not counted as independent measurements at each knot.
5. Observations are inconsistent with the current model; model expansion or abstention is required.
6. Statistical expected-value logic is supplied only hard-set inputs.
7. Deterministic guarantees are requested from uncalibrated covariance only.
8. Cost/risk limits exclude the action with the largest nominal value.
9. No action has a certified contraction.
10. A solver witness or evidence digest is stale.

## 8. Claim ceiling

V0.1 demonstrates only that uncertainty width and robust-margin decision value can rank two stipulated synthetic evidence actions differently. It does not show real measurement efficiency, safe excitation, a working IQC solver, physical FRF validity, aircraft robustness, airworthiness or certification.

## References

- Rojas et al. (2012), *Robustness in Experiment Design*: https://people.kth.se/~crro/pubs/rojas_aguero_welsh_goodwin_feuer2012.pdf
- *Least costly identification experiment for control* (2006): https://www.sciencedirect.com/science/article/abs/pii/S0005109806002238
- Xie and Bemporad (2025), *Online design of experiments by active learning for nonlinear system identification*: https://arxiv.org/abs/2506.21754
- AIRROAD matched IQC benchmark: https://github.com/Luminous-Dynamics/symthaea/issues/11946
