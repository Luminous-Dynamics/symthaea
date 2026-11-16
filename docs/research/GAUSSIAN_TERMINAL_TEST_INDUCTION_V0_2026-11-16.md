# Gaussian terminal-test induction reference v0

**Date:** 2026-11-16  
**Owner:** Symthaea #12426  
**Implementation:** standalone falsification-power-reference crate  
**Composes with:** #12402 / PR #12405; generic induction reference #9820; method league #12271

## 1. Why this reference exists

The Gaussian e-process in PR #12405 has an anytime-valid likelihood-ratio boundary. It is useful for a predictable alternative and can be updated after each observation. The first exact-head synthetic calibration produced lower-than-nominal finite-horizon crossing rates (2.635% fixed null and 3.300% predictable adaptive-variance null at nominal 5%) and 78.060% power by look 100 on the declared adaptive alternative. That is one frozen generator, not a universal calibration result.

A distinct sequential construction is available when the design and terminal test are fixed: induce a [0,1]-valued test process by taking the null conditional expectation of the terminal Neyman–Pearson test. Koning & van Meer (JRSS B, 2026) show that this preserves the terminal test exactly while remaining valid at bounded stopping times. This does not require treating e-values as the only route to optional-stopping validity.

## 2. Frozen simple-vs-simple Gaussian design

At each design index t=1..N, the observation residual has:

    H0: Y_t ~ Normal(0, sigma_t^2)
    H1: Y_t ~ Normal(d_t, sigma_t^2)

The complete sequence d_t and sigma_t is deterministic and frozen before data. Sigma_t is strictly positive and known. Define:

    I = sum_t d_t^2 / sigma_t^2
    L_t = sum_(s<=t) [d_s * Y_s / sigma_s^2 - d_s^2 / (2*sigma_s^2)]

Under H0, the terminal log likelihood ratio is Normal(-I/2, I). The level-alpha one-sided Neyman–Pearson test uses:

    c_alpha = -I/2 + sqrt(I) * Phi_inverse(1-alpha)

and rejects at horizon N when L_N >= c_alpha.

For an observed prefix t<N, future log-likelihood increments are independent Gaussian under H0 with mean -I_remaining/2 and variance I_remaining, where:

    I_remaining = sum_(s>t) d_s^2 / sigma_s^2

The induced process is:

    phi_t = P0(L_N >= c_alpha | F_t)
          = Phi((L_t - c_alpha - I_remaining/2) / sqrt(I_remaining))

At t=N it equals the terminal 0/1 decision. At t=0 it equals alpha.

The process phi_t is a [0,1]-valued null martingale. For any bounded stopping time tau, E0[phi_tau]=alpha. If an independent U~Uniform(0,1) is drawn at the stopping time and the evaluator rejects exactly when U<phi_tau, the resulting randomized test has null size alpha. At the fixed horizon, the decision is exactly the terminal Neyman–Pearson test and therefore has its terminal power.

**phi_t is a conditional rejection probability, not a p-value.** It must not be interpreted as a conventional p-value threshold or compared directly with an e-value boundary.

## 3. Implementation contract

The Rust module gaussian_terminal_induction accepts frozen per-step discrepancy and sigma arrays and provides:

- total design information and the terminal NP critical log-LR;
- evaluator-side prefix log-LR computation;
- the induced conditional rejection probability at each prefix;
- fixed-horizon terminal power;
- an explicit independent-uniform randomization helper.

It rejects empty/unequal designs, non-finite discrepancies/observations, nonpositive variances, invalid alpha, zero-information designs and overlong prefixes.

The current tests include:
- initial induced value equals nominal alpha;
- terminal prefix equals the frozen NP decision;
- external known-answer terminal power;
- a numerical prefix-martingale check;
- an optional-stopping null-size smoke simulation with a separate randomizer stream;
- malformed input and randomization validation.

## 4. Why this is a comparator, not a replacement

The construction relies on the remaining terminal likelihood under H0 being known conditional on the current information. The v0 implementation achieves this with a fully frozen deterministic design and a simple-vs-simple Gaussian model.

It does not automatically extend to:
- future actions or sensor precisions selected from future observations;
- arbitrary adaptive experiment policies whose remaining information is path-dependent;
- unknown or estimated variance;
- composite nulls or nuisance parameters;
- serial dependence, heavy tails or unmodelled measurement shifts;
- a growing or post-hoc selected alternative family.

The predictable Gaussian e-process in #12405 has a different scope: its one-step predictions and known sigma_t may adapt to prior observations, but its test is a simple conditional-null LR reference. The induced terminal process preserves a fixed terminal NP test and is a specialist fixed-design comparator. Compare the two empirically under compatible conditions; do not claim either universally dominates.

## 5. Matched benchmark

On identical fixed Gaussian designs and common outcome streams, compare:

1. Fixed-horizon NP test (terminal reference).
2. Induced terminal-test process under prespecified bounded stopping rules, with independent randomization at stop.
3. Gaussian e-process with Ville boundary.
4. Naive repeated fixed-time Gaussian test (deliberately invalid negative control).

Report:
- null rejection rate and episode-level confidence interval;
- terminal power and exact terminal-decision equality;
- power at each separated alternative;
- early stopping frequency and stopping-look distribution;
- expected cost and compute at decision;
- unresolved / no-crossing rate;
- paired differences with uncertainty.

The terminal NP test is an oracle comparator because it knows the correct alternative and frozen design. Do not present it as a deployable unknown-mechanism discovery policy.

## 6. Research comparison and source

Koning & van Meer, *Anytime validity is free: inducing sequential tests*, Journal of the Royal Statistical Society Series B 88(4), 1366–1384 (published 21 February 2026), DOI: https://doi.org/10.1093/jrsssb/qkag050.

Their result is general for a fixed valid terminal test; this module implements only the deterministic simple-Gaussian case. The paper also discusses optional continuation. The generic simple-null Bernoulli implementation remains under #9820, and generic anytime-valid confidence-sequence/e-process comparisons remain owned by AOBench #12197 / #12242 / #12271.

## 7. Claim boundary

A passing Rust workflow qualifies implementation behavior on the exact source head. A simulation estimates operating characteristics only for the frozen generator and stopping rules. Neither proves universal type-I control, model adequacy, external benchmark superiority, causal identification or physical validity.
