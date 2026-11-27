# SEMI-QUALIFIED-RISK-001 — population risk under selective and qualified metrology v0.1

**Issue:** #13980  
**Implementation:** symthaea-semiconductor::benchmark::qualified_observation_risk  
**Status:** Rust bounded-loss risk ledger with authored tests; exact-head validation pending.

## Problem formulation

Physical measurement and online learning are not the same observation process. For each wafer/prediction round record the stages separately:

- \(S\): selected for physical metrology;
- \(R\): physical result received, conditional on selection;
- \(Q\): result passes K1 numeric/evidence qualification, conditional on receipt;
- \(A\): admitted to online tracking, conditional on selection;
- \(V\): qualified feedback available by the declared model-update frontier;
- \(U\): actually used by the online update.

The physical observation probability is represented as:
\[
\pi_i^{physical} =
P(S_i=1)\,
P(R_i=1\mid S_i=1)\,
P(Q_i=1\mid R_i=1,S_i=1)
\]
under the declared covariate and conditional-probability factorization.

The online-update inclusion probability additionally includes tracking and availability:
\[
\pi_i^{online} =
\pi_i^{physical}
P(A_i=1\mid S_i=1)
P(V_i=1\mid Q_i=1)
\]
This is a declared factorization contract; it must not be assumed true merely because each factor has a numeric field.

## Three different estimands

1. **All-unit population risk:** average bounded loss over every wafer in the declared chronological population.
2. **Physically observed/qualified risk:** average over results with usable K1-qualified physical labels.
3. **Online-update risk:** average over labels that actually entered the online conformal/model update.

A physically measured but not online-tracked wafer still contributes to offline population-risk estimation. It does not retroactively become an online update.

## Identification modes

### Complete physical census

When every population unit has a qualified bounded loss, report exact all-unit risk. This is a census, not an IPW estimate.

### Known MAR / identified observation

Emit a Horvitz–Thompson finite-population mean estimate only when:
- selection, physical-delivery and qualification propensities are known, strictly positive and valid;
- the stated conditional factorization is supported;
- conditional ignorability is supported for the target estimand;
- the probability evidence is bound to an assessment receipt.

The physical-observation estimator uses only the physical observation stages. The online-update estimator additionally requires known positive tracking and frontier-availability propensities.

Report minimum joint inclusion probability, maximum inverse weight and effective sample size. V0.1 does not truncate weights silently; truncation changes the estimator and requires its own sensitivity analysis.

### Unknown or MNAR qualification

When qualification may depend on the unobserved residual/error, a plug-in qualification propensity does not identify all-unit risk. In that case, suppress IPW and report the deterministic finite-population bound for bounded loss \(L_i\in[0,1]\):
\[
\frac{\sum_{i\in O}L_i}{N}
\leq
R_N
\leq
\frac{\sum_{i\in O}L_i+(N-|O|)}{N}
\]
where \(O\) is the set of physically observed and K1-qualified outcomes. This is an identification bound, not a confidence interval.

Unknown, blocked and revoked labels are counted separately. Not-selected, not-received and qualified-but-unavailable-by-frontier are also reported separately. A wide interval is an honest representation of insufficient evidence, not a software failure to be hidden.

## Assumption registry and trust boundary

Each input explicitly records Satisfied / Violated / Unknown for:
- conditional-probability factorization;
- conditional ignorability;
- positivity.

A digest identifies the supplied assumption assessment but is not a signature and does not establish that an assumption is true. The estimator only emits IPW under the declared Satisfied states and valid complete propensities. A reviewer must assess the supporting evidence independently.

The report never claims a theorem or physical guarantee. The robust interval always remains available, regardless of assumptions.

## Implementation invariants

- A unit cannot be tracked unless selected.
- A physical result cannot exist without selection.
- A result cannot be available before receipt.
- A Qualified status requires a received result and finite loss in [0,1].
- NotSelected / NotReceived / Unknown / Blocked / Revoked cannot expose a qualified loss.
- An online-used outcome must be selected, tracked, received, K1-qualified and available by the frontier.
- Population rounds must be unique and consecutive; sample IDs must be unique.
- All probability values are validated in (0,1]; missing or zero required propensities suppress IPW.

## Authored tests

The initial test suite covers:
1. complete census returns exact population risk, not an IPW claim;
2. selected-only observations produce conservative [lower, upper] bounds;
3. known MAR propensities enable physical and online IPW;
4. physical risk IPW does not require online tracking propensity;
5. potential MNAR qualification failure suppresses IPW;
6. physically measured but not online-tracked units still enter offline population risk;
7. unknown/blocked/revoked labels widen bounds and remain separately counted;
8. zero or missing propensity fails closed;
9. online use without all observation stages is rejected;
10. duplicate units and out-of-range losses are rejected;
11. an all-missing population returns [0,1].

Tests are authored; exact-head rustfmt, strict Clippy, crate tests and SEMI-BENCH-001 reference validation are pending.

## Benchmark matrix

- Complete-feedback oracle.
- Random selection with known delivery/qualification propensities.
- Risk-directed/EVSI selection with randomized audit channel.
- MAR qualification missingness with known probabilities.
- Propensity estimation error / positivity failure.
- MNAR qualification failure correlated with high residuals.
- Calibration revocation after previous updates.
- Tool/chamber-specific common-reference failures.
- Tracking capacity saturation while offline measurement remains available.

Report true all-unit risk, observed-only mean, physical IPW, online-update IPW, robust risk interval, interval width, selection/delivery/qualification/tracking/availability fractions, ESS and maximum weight. Measure false IPW eligibility (target zero) and robust-bound coverage over seeded synthetic populations.

## Claim ceiling

V0.1 supplies exact census calculations, declared-assumption IPW estimates and deterministic worst-case bounds for synthetic bounded losses. It does not derive a new theorem for MNAR qualification or selective measurement, authenticate physical source data, or authorize model/equipment control.