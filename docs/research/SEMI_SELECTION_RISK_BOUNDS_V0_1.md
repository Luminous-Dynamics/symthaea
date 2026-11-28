# SEMI-SELECTION-RISK-001 — population risk under selective metrology qualification v0.1

**Issue:** #13646  
**Implementation:** symthaea-semiconductor::benchmark::selection_risk_bounds  
**Status:** Rust partial-identification/audit estimator with authored tests; exact-head validation pending.

## Problem statement

A metrology result can be missing from model feedback because it was not measured, arrived late, or failed qualification. Those mechanisms are not interchangeable.

The operational model must distinguish:
- qualified labels used for learning/calibration;
- UNKNOWN qualification outcomes;
- BLOCKED outcomes;
- predictions for which no physical label was obtained.

Coverage over the qualified subset is not automatically coverage over the full production population. If missingness depends on tool condition, drift regime, residual magnitude or calibration failure, the qualified subset can be systematically easier than the population.

## Partial-identification interval

Let N be the total prediction population, Q the qualified outcomes, and M=N-|Q| the outcomes with UNKNOWN, BLOCKED or unmeasured status. Let S_Q be the sum of qualified losses, with each loss in [0,1].

Without assumptions on the missing outcomes, the sharp population-risk interval is:

- lower = S_Q / N
- upper = (S_Q + M) / N

The qualified-subset risk S_Q/|Q| is reported separately. It is never substituted for the population risk unless M=0.

This interval makes no missing-at-random or propensity assumption. It may be wide; that is an honest consequence of unobserved physical truth.

## Independent randomized audit

The optional audit estimator assumes each nonqualified population unit is independently sampled by a Bernoulli design with declared inclusion probability pi_i, bounded below by pi_min>0. The audit measurement must use a traceable reference method independent of the operational failure that caused the original outcome to be UNKNOWN/BLOCKED/unmeasured.

The Horvitz-Thompson estimate is:

R_hat = (S_Q + sum_{i in audit} L_i/pi_i) / N.

For independent Bernoulli sampling, the implementation reports the unbiased variance estimate:

Var_hat(R_hat) = (1/N^2) * sum_{i in audit} (1-pi_i)*L_i^2/pi_i^2.

It also reports effective sample size based on inverse-probability weights.

For a conservative finite-sample interval, it uses Hoeffding on the missing population: with M missing units, loss in [0,1], and all pi_i>=pi_min, radius is

sqrt(M * ln(2/delta)/2) / (N*pi_min).

The normal 95% interval is descriptive only; the Hoeffding interval is the declared conservative bound under the independent Bernoulli audit assumptions. The estimator is not claimed when the design is not independent or positivity fails.

## Implemented validation

- The population partition must cover every prediction round exactly once as Qualified, Unknown, Blocked or Unmeasured.
- Each qualified round has exactly one finite loss and a valid evidence receipt digest.
- Audit observations must refer to unique nonqualified population rounds.
- Audit inclusion probabilities must be at least the declared minimum and no greater than one.
- Audit measurement receipt digests, design identity, randomization provenance and reference-method identity must be present.
- The audit design must explicitly declare independence from operational qualification.
- No IPW estimate is emitted without the audit design.
- No population point estimate is emitted from a qualified-only subset when missing outcomes remain.

## Required adversarial benchmark matrix

MCAR, MAR and MNAR selection; tool/chamber-only qualification failure; high-residual-dependent failure; maintenance and calibration-revocation regimes; zero-propensity strata; propensity overlap failure; audit independence failure; audit inclusion-probability misspecification; and delayed labels layered on top of qualification missingness.

Metrics include qualified-subset risk, full-population ground truth in synthetic experiments, worst-case identification interval width, audit-estimator bias/error, coverage of the Hoeffding interval, audit effective sample size and the false-reassurance rate where qualified-only risk appears acceptable while true population risk is not.

## Claim ceiling

This is statistical identification software. It cannot recover an unavailable physical outcome without an assumption or independent audit measurement. Inverse-probability weighting cannot make an invalid/revoked calibration record trustworthy. An audit measurement is only as credible as its independent physical reference process.
