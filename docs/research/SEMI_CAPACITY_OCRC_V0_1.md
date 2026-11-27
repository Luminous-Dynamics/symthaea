# SEMI-CAPACITY-OCRC-001 — bounded-delay capacity-aware online risk control v0.1

**Issue:** #13979  
**Implementation:** `symthaea-semiconductor::benchmark::capacity_constrained_ocr_control`  
**Base:** streaming evidence adapter / K1 numeric label qualification  
**Status:** Rust reference implementation with authored tests; exact-head validation pending.

## Research source

Bowen Wang and Osvaldo Simeone, *Capacity-Constrained Delayed Feedback for Online Conformal Risk Control*, arXiv:2611.17059v1, 12 November 2026.

- Paper: https://arxiv.org/abs/2611.17059
- Detailed text: https://borowiecki.dev/pdf/2611.17059

The implementation targets the paper's arbitrary bounded-delay case (Assumption 1(a)), Algorithm 1, equations (18)–(29), Theorem 1 and Proposition 1. It does not yet implement the stochastic-delay optimizer/Theorem 2.

## Algorithm contract

At every round t:
1. Freeze the prediction interval from the current threshold r_t.
2. If the set is empty/full, derive its loss exactly and do not track the round.
3. Otherwise consume an externally sampled proxy delay from the declared survival distribution F.
4. Evict tracked rounds iff i + proxy_delay_i < t.
5. Admit the current round iff its sampled proxy delay is nonnegative and capacity remains.
6. Receive feedback only for still-tracked rounds whose actual delay expires at t.
7. Update r using the trivial-set term and inverse proxy-survival weighting for received feedback.

The order is part of the algorithm: prediction → eviction → admission → reception → threshold update.

## Regression interval specialization

The V1 regression score is bounded:
[
S_t(y)=minleft(rac{|y-hat y_t|}{s_t},Bight)in[0,B]
]
where s_t is positive and frozen with a scale-artifact digest. The online set is all y with S_t(y)≤r_t. For nontrivial r in [0,B), the physical-unit interval is [ŷ_t-r_t s_t, ŷ_t+r_t s_t]. For r_t<0 the set is empty; for r_t≥B the set is full.

The loss is binary interval miscoverage, not squared error. The guarantee is not being generalized to unbounded MSE, arbitrary weighted loss, conditional per-wafer coverage or a production process-control objective.

## Proxy-delay survival and risk bound

F(a)=P(proxy_delay≥a), with F(-1)=1 and F(d_max+1)=0. The input vector contains F(0)..F(d_max), must be non-increasing, and must have F(d_max)>0.

Tracking load:
[
ell_F=sum_{a=1}^{d_{max}} F(a)leq K.
]

For arbitrary bounded delays and under the paper's assumptions, Theorem 1 gives:
[
operatorname{RiskGap}(T)leq
rac{B}{eta T}+
rac{K+1}{F(d_{max})T}+
exp[-K h(ell_F/K)],
quad h(x)=x-1-ln x.
]

The Proposition 1 Bernoulli family uses F(0)=1 and F(a)=p for a≥1, with p optimized over 0<p≤K/d_max:
[
p^*=argmin_p
left{rac{K+1}{pT}+exp[-K h(d_{max}p/K)]ight}.
]
The current optimizer is deterministic grid search and retains its returned proxy distribution as an explicit artifact.

When capacity is sufficient to track every pending signal (K≥d_max+1), the reported bound switches to the unlimited-tracking OCRC bound.

## K1 evidence composition

The feedback validator checks:
- exact label ID and report-attested revision;
- measurement event and sample identity;
- full label payload digest and report digest;
- qualified corrected value equality;
- current K1 frontier and eligible/downstream-permitted state;
- label and K1 report availability relative to delivery and prediction times.

A K1-invalid result for a still-tracked round is not treated as a valid OCRC loss. The engine records the failed assumption, does not update r_t, and sets theorem eligibility to ASSUMPTIONS_NOT_MET. The original CA-OCRC theorem is not claimed after this extra qualification failure.

The controller's theorem eligibility output is a software contract flag, not proof that the physical label is authentic or the delay distribution/upper bound is true.

## Important paper-text discrepancy

The paper's Eq. (20) defines eviction as i + proxy_delay_i < t. Its Fig. 1 narrative says a round at t=1 with proxy delay 3 is evicted at t=4, which would require i + proxy_delay_i ≤ t under one-based indexing.

These statements are inconsistent by one round. V1 follows the formal Eq. (20) and tests strict less-than: the t=1, proxy=3 item remains tracked at the start of t=4 and expires at the start of t=5 if it has not already received feedback. The discrepancy is recorded in issue #13979. No attempt is made to make both the equation and figure pass simultaneously.

## Authored tests

The initial test module covers:
1. formal strict-expiry behavior and the paper Fig. 1 off-by-one discrepancy;
2. zero-delay feedback only when the current round is admitted;
3. empty/full interval known losses without tracking;
4. feedback at the strict proxy deadline;
5. invalid K1 feedback does not update the threshold and disables theorem eligibility;
6. revision, payload and corrected-value binding;
7. inverse-probability weight uses F(actual_delay);
8. scalar risk-bound known answer;
9. optimized Bernoulli proxy monotonicity and underload;
10. deterministic replay;
11. delivery after proxy expiry is ignored as untracked.

Tests are authored; exact-head rustfmt, strict Clippy, crate tests and SEMI-BENCH-001 Reference Oracle remain mandatory.

## Next research gate: qualification-aware extension

Issue #13980 separates metrology selection, tracking admission, physical receipt, K1 qualification, availability and actual update use. Its goal is to determine when IPW is valid and when unqualified labels create informative/MNAR censoring that requires robust population-risk bounds.

No inverse-probability correction for qualification failure is claimed by this V1 engine.

## Claim ceiling

This is a faithful software reference for the paper's arbitrary bounded-delay CA-OCRC method on a bounded regression miscoverage loss, conditioned on its declared assumptions. It does not prove that a real fab's delay bound or measurement-qualification process satisfies those assumptions, and it does not authorize wafer disposition, APC/R2R, recipe changes, equipment actions or control.
