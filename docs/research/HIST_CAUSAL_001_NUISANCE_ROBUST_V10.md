# HIST-CAUSAL-001 v0.10: nuisance-robust composite-null calibration

## Why v0.10 exists

v0.9's finite-sample intersection-union test controls false catalogue rejection under known, correct point nuisance parameters and independent exchangeable experimental units. That assumption is too strong for real scientific discovery. Parameters are estimated, and histories/campaigns may contain persistent latent regimes.

v0.10 isolates the first problem: an unknown nuisance parameter in one candidate mechanism.

## Berger–Boos construction

Let the null mechanism have nuisance parameter theta. Use a pilot sample independent of the evaluation sample to construct a confidence set C satisfying:

P_theta(theta in C) >= 1 - beta for every theta in the declared parameter space.

For each fixed theta, let p_theta(D) be a valid p-value for the evaluation data. Define:

p_BB(D) = min(1, sup_{theta in C} p_theta(D) + beta).

On the event theta_true is in C, p_BB is at least p_theta_true. The complement event has probability at most beta. Thus:

P_theta(p_BB <= alpha)
<= P_theta(p_theta_true <= alpha - beta) + P_theta(theta_true not in C)
<= (alpha - beta) + beta
= alpha.

The pilot/evaluation independence makes the conditional argument straightforward. More general confidence sets require a joint-validity argument. A finite nuisance grid is only valid for that grid; it does not cover an unrepresented continuous parameter.

For multiple candidate mechanisms, calculate one nuisance-robust p-value per candidate, then retain v0.9's intersection-union rule: reject the catalogue only if every candidate-specific robust p-value is <= alpha. If the true candidate is in the catalogue, its p-value remains super-uniform and the false catalogue-rejection bound follows.

## Exact finite-grid fixture

This reference uses:
- binomial nuisance grid theta in {1/4, 1/2, 3/4};
- independent pilot n=20;
- evaluation n=20;
- beta=0.01 confidence-set miss allowance;
- alpha=0.05;
- lower-tail evaluation alternative theta=1/8.

The confidence set is obtained by inverting a conservative probability-ordering exact two-sided binomial test. The Berger–Boos p-value is the supremum of lower-tail p-values over the confidence set plus beta, capped at one. An empty set is treated conservatively as p=1.

Exact exhaustive enumeration yields:

| True null theta | Confidence-set coverage | False-rejection probability |
|---:|---:|---:|
| 1/4 | 99.2887% | 2.6573% |
| 1/2 | 99.7423% | 0.9266% |
| 3/4 | 99.2887% | 0.3081% |

All declared nuisance grid points have coverage >=99% and false rejection <=5%. Power against theta=1/8 is 24.8478% for this small pilot/evaluation design. This is deliberately conservative and underpowered; the point is validity, not detection performance.

Six unit tests pass. All probability calculations and the exhaustive type-I enumeration use exact rational arithmetic.

## Persistent-cluster follow-on

This v0.10 file does not establish validity for repeated observations sharing a persistent latent state. Next:
1. define the independent assignment unit (campaign, task family, batch, specimen cluster);
2. keep all repeated reads and intervention carryover within the cluster;
3. calibrate and permute at the assignment unit, not at the participant/read level;
4. compare cluster-level randomization inference, cluster bootstrap and hierarchical models;
5. include persistent latent-regime and cross-cluster leakage generators;
6. score false catalogue rejection per true mechanism and per cluster-generating regime.

The issue #10184 tracks these persistent-cluster tests. An iid bootstrap over repeated reads is not a valid shortcut.

## Boundaries

This exact fixture establishes a nuisance-robust construction only on a three-point parameter grid for a single binomial model. It does not establish continuous-parameter coverage, validity under parameter-grid misspecification, the full six-mechanism catalogue test, persistent-confounder robustness, physical discovery or SOTA performance. The next production implementation remains Rust-first under #10081.
