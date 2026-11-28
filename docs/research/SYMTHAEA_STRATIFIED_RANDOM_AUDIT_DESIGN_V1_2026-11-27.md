# Symthaea stratified randomized audit design v1

Status: research implementation; exact-head Rust qualification pending. The module builds a sampling plan from externally generated priorities; it does not produce entropy or execute adjudication.

## Why this is needed

The propensity-aware estimator in `selective_observation.rs` needs an inclusion probability for every record in the frozen candidate cohort. Merely asking a human reviewer to adjudicate some claims does not produce a defensible sampling design. The system needs a pre-label randomized audit protocol that covers all relevant cases, including low-confidence and abstained cases.

This increment provides one simple design with transparent first- and second-order inclusion probabilities: stratified simple random sampling without replacement (SRSWOR).

## Protocol

1. Freeze the complete pre-label prediction cohort and its stratum assignments.
2. Assign each record a 128-bit random priority using an independent custodian's random generator. The custodian preserves the randomization input and cohort/strata/quota manifests.
3. Declare a positive integer audit quota (n_h) for every nonempty stratum (h), with (1 leq n_h leq N_h).
4. Select the (n_h) records with the lowest priorities in each stratum.
5. Independently adjudicate every selected record, including cases the candidate abstained on. Do not show the candidate the audit selection, hidden labels or evaluator state.
6. Join the complete cohort, selected status and inclusion probabilities into the propensity-aware evaluation input. Nonselected records remain present with pending labels.

The module rejects missing or extra stratum quotas, zero quotas, overlarge quotas, duplicate record IDs and duplicate random priorities within a stratum. It does not break priority ties using record IDs because that would create a non-random tie preference.

## Inclusion probabilities

Conditional on independent uniform random priorities within and across strata, selecting the smallest (n_h) priorities is SRSWOR within each stratum.

For any unit (i) in stratum (h):

`pi_i = n_h / N_h`.

For two distinct units (i,j) in the same stratum:

`pi_ij = n_h(n_h - 1) / [N_h(N_h - 1)]`.

For records in different strata, independent stratum randomization gives:

`pi_ij = pi_i * pi_j`.

For the same unit, `pi_ii = pi_i`.

These probabilities are returned with the plan and can be joined directly to `SelectiveObservationRecord.adjudication_inclusion_probability`. They provide the design information missing from a generic propensity-only input.

## Design-based variance follow-up

For a fixed candidate acceptance set (A), define the full-population variable (x_i(A)) to be 1 when the candidate answers and is wrong, and 0 otherwise. The accepted population count (N_A) is known from the frozen predictions. Under stratified SRSWOR, the Horvitz-Thompson error total is the sum of the stratum estimates.

If each stratum has (n_h geq 2), the standard unbiased design-based variance estimator for the total is:

`estimated_V(T_A) = sum_h N_h^2 * (1 - n_h/N_h) * s_h^2(A) / n_h`,

where (s_h^2(A)) is the sample variance of (x_i(A)) among the independently adjudicated random-audit units in stratum (h). The corresponding risk variance divides by (N_A^2). A full-census stratum contributes zero sampling variance. A non-census stratum with only one audited unit does not provide an estimable within-stratum sample variance; the plan marks this rather than silently treating the variance as zero.

The design-based variance formula is grounded in standard Horvitz-Thompson/SRSWOR sampling theory. This PR exposes the exact `pi_i` and `pi_ij` design, but does **not yet** implement the variance estimator, confidence interval, or finite-sample risk-control procedure. Those are follow-on work after qualification of this sampling boundary.

## Authority and security boundaries

- Random priorities are supplied by an independent custodian and must be generated uniformly with a cryptographically secure random source.
- This module does not choose the cohort, define the scientific strata, set audit quotas, generate randomness, select reviewers, authenticate the randomization manifest, or reveal sealed truth.
- The candidate must not receive the random priorities or audit membership before predictions and policy outputs are frozen.
- Selection must be independent of the adjudication outcome conditional on the declared pre-label sampling frame.
- Stratum randomization streams must be independent for the cross-stratum joint-inclusion product formula.
- A deterministic plan replay proves the same priorities and quotas produce the same sample; it does not prove the priorities were random or the custodian acted independently.

## Tests authored

The production module includes tests for exact quotas and positive first-order inclusion probabilities, within-stratum and cross-stratum joint inclusion probabilities, input-order invariance, random-priority tie rejection, missing/duplicate/unknown quotas, invalid quotas, census inclusion probabilities, design-based variance-support diagnostics, duplicate record IDs and invalid design identity.

Tests are imported by path into the standalone Rust 1.96 epistemic adjudication harness. They are authored, not claimed as executed until the exact-head workflow result is observed.
