# AR-5B12: Anytime-Valid Test Induction for Adaptive Replication v0.1

**Status:** finite reference implementation draft; tests authored but not compiled or executed.  
**Stack:** #9429 → #9718 → #9719 → #9721 → #9754 → #9782 → #9785 → #9787 → #9807 → #9814 → this branch.  
**Research anchor:** Koning & van Meer, “Anytime validity is free: inducing sequential tests,” JRSSB 88(4), 1366–1384, published 21 February 2026. https://doi.org/10.1093/jrsssb/qkag050  
**Claim ceiling:** exact finite-horizon, simple-null Bernoulli test induction using a randomized terminal Neyman–Pearson test. The induced conditional-expectation process is a bounded null martingale up to floating-point error. This is not a general confidence sequence for composite bounded-mean hypotheses or a simultaneous multi-objective Pareto guarantee.

## 1. Why add this after the noisy-Pareto intervals?

The current noisy Pareto analyzer uses a summable alpha-spending schedule across candidate/objective coordinates and replicate prefixes. That is a conservative, time-uniform construction under its declared bounded-IID assumptions.

A different reference is useful: start with a fixed-horizon test whose terminal size and power are known, then induce a sequential test by taking its conditional expectation under the null. This gives an anytime-valid [0,1]-valued process without changing the terminal test at the planned horizon.

The 2026 JRSSB paper by Koning and van Meer formalizes this induction principle and also studies optional continuation. This PR implements a deliberately small finite Bernoulli reference before attempting more general statistics.

## 2. Finite test contract

The plan freezes:
- a simple Bernoulli null (H_0: X_i \sim \operatorname{Bernoulli}(p_0)), IID;
- a one-sided simple alternative (H_1: X_i \sim \operatorname{Bernoulli}(p_1)), with (p_1>p_0);
- a fixed terminal horizon (N\le512);
- a significance level (\alpha\in(0,1)).

The null probability must lie strictly inside (0,1). The alternative may include (p_1=1), which is handled as a degenerate binomial distribution.

For the terminal count (S_N=\sum_{i=1}^N X_i), the likelihood ratio is monotone in (S_N). The Neyman–Pearson test is therefore an upper-tail test:
- reject with probability 1 above a critical success count;
- reject with a randomized probability \(\gamma\in[0,1]\) at the boundary;
- reject with probability 0 below the boundary.

The randomized boundary is calculated so that the terminal null size equals the declared alpha, subject to f64 tolerance. Alternative power is reported separately.

## 3. Induced prefix process

Let \(\phi_N(S_N)\in[0,1]\) be the terminal randomized test. For a prefix with (n) observations and (s) successes, define:

```
phi_n(s) = E_0[ phi_N(s + T) ]
T ~ Binomial(N-n, p0)
```

Equivalently, in filtration notation:

```
Phi_n = E_0[ phi_N | F_n ]
```

By the tower property:

```
E_0[ Phi_(n+1) | F_n ] = Phi_n
```

Thus `Phi_n` is a bounded [0,1]-valued martingale under the frozen simple null. For any bounded stopping time tau <= N:

```
E_0[ Phi_tau ] = E_0[ Phi_0 ] = E_0[ phi_N ] = alpha
```

This is an anytime-valid test process in the finite simple-null setting. Its terminal value at N is exactly the original fixed-horizon test, including the randomized boundary.

The implementation computes every reachable (n,s) prefix value, checks the maximum martingale conditional-expectation residual, checks terminal equality, and reconstructs a digest-bound report.

## 4. Numerical method

- Binomial PMFs are evaluated in log space with log-sum-exp normalization.
- Degenerate (p=0) and (p=1) binomial endpoint distributions are handled explicitly; the test plan still requires (0<p_0<1).
- Compensated summation is used for null size, power and conditional expectations.
- The horizon is capped at 512 to bound the finite prefix table and calculations.
- The terminal test and report are deterministically reconstructed from the frozen plan.

All calculations use f64; equality claims are numerical within stated tolerances, not symbolic rational proofs.

## 5. Relation to existing Symthaea noisy-Pareto inference

The existing `pareto_noisy_observations.rs` analyzer uses a summable per-prefix alpha allocation:

```
alpha_(candidate, objective, n)
  = familywise_alpha / (max_candidates * objective_count * n * (n+1))
```

and bounded Hoeffding intervals. This gives a direct simultaneous confidence-sequence-style guarantee under its bounded-IID sampling contract.

The induced-test reference answers a different question: can a fixed-horizon test be preserved exactly as an anytime sequence? The Bernoulli induction result does not automatically yield:
- confidence intervals for unknown bounded means;
- a composite null (H_0: \mu\le\mu_0);
- family-wise control over multiple candidates and objectives;
- valid Pareto dominance decisions;
- validity under dependence, drift, or informative missingness.

Do not replace the current noisy-Pareto interval method with this simple-null test. Use this implementation as a calibration benchmark for future composite bounded-mean and multi-objective inductions.

## 6. Optional-stopping benchmark

The unit tests enumerate all (2^N) binary paths for a small fixed horizon and evaluate multiple stopping rules:
- stop when success count crosses a threshold;
- stop when failure count crosses a threshold;
- stop at the terminal horizon if no earlier boundary is crossed.

The path probability includes the complete N-step sample path while the test value is evaluated at the first stopping prefix. This makes the exact expectation calculation correct even when the process stops before later (unobserved) suffix outcomes; summing over suffix completions leaves the stopped-prefix probability unchanged.

The test suite also includes an exact negative control: at each prefix n, independently apply a fresh unadjusted level-alpha fixed-horizon Bernoulli test designed for horizon n. It computes the family-wise probability of ever rejecting by enumerating all N-step paths and integrating the independent boundary randomization at each look. This deliberately invalid repeated-testing procedure must exceed nominal alpha, while the induced martingale procedure remains at alpha.

### Matched benchmark comparators

1. **Induced fixed-horizon test:** this reference.
2. **Summable alpha-spending Hoeffding sequence:** current Symthaea noisy-Pareto baseline.
3. **Modern confidence-sequence/e-process controls:** select a method appropriate to the exact bounded-mean or Bernoulli hypothesis and pin its implementation.
4. **Invalid negative control:** fixed-horizon p-values or intervals inspected at every prefix without an optional-stopping correction.

### Metrics

- Exact terminal null size and alternative power.
- Supremum type-I error over enumerated stopping times/rules.
- Optional continuation error after an interim test.
- Average sample number and time-to-decision.
- Early decision loss under predeclared asymmetric terminal losses.
- Family-wise error across candidate/objective tests.
- Full compute and evidence costs.

Use exhaustive finite enumeration first, then independent Monte Carlo campaign clusters. Keep the invalid negative control visible as a diagnostic, never as a valid comparator.

## 7. Next scientific extension: optional continuation

The next step after this reference is not immediately a Pareto confidence interval. First implement optional continuation for two fixed-horizon tests:
- compute the first interim test value as the conditional null expectation of its terminal test;
- use that [0,1]-valued interim value as the conditional significance level for the second test;
- prove/verify that the combined terminal decision has the intended size under the frozen null;
- bind the entire continuation schedule and test identities in one report.

Then study composite bounded-mean tests and multiple objective/candidate control. State the filtration, null model, dependence assumptions and stopping rule explicitly.

## 8. Proposed acceptance gates (not results)

- Terminal randomized test size equals alpha within (10^{-10}) for the finite test fixtures.
- Initial induced value equals terminal null size.
- All prefix values lie in [0,1].
- Maximum null martingale residual below (10^{-10}) on supported fixtures.
- Terminal induced values equal the original terminal test within (10^{-12}).
- Enumerated optional-stopping threshold policies preserve null expectation within (10^{-9}).
- Invalid per-prefix fixed-horizon negative controls demonstrate why correction is needed, without being used as a qualified method.
- No claim of general multi-objective confidence-sequence validity until composite-null and family-wise proofs/tests exist.

## 9. Exact-head validation commands

```bash
cargo fmt --all -- --check
cargo test -p symthaea-architecture-engineering anytime_test_induction -- --nocapture
cargo test -p symthaea-architecture-engineering pareto_noisy_observations -- --nocapture
cargo test -p symthaea-architecture-engineering
cargo clippy -p symthaea-architecture-engineering --all-targets -- -D warnings
```

No compile, format, clippy, CI, benchmark or SOTA pass is claimed.
