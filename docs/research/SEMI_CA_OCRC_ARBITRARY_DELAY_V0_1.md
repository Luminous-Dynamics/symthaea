# SEMI-CA-OCRC-001 — capacity-aware online conformal risk control, arbitrary-delay v0.1

**Issue:** #13499  
**Implementation:** symthaea-semiconductor::benchmark::capacity_aware_ocrcc  
**Base:** fixed-delay evidence-gated ACI #13498 and the validated evidence-to-model-family stack #13465  
**Status:** Rust arbitrary-bounded-delay engine and 12 authored tests; exact-head validation pending.

## Paper and exact algorithm

Wang & Simeone, *Capacity-Constrained Delayed Feedback for Online Conformal Risk Control*, arXiv:2611.17059v1, 12 November 2026.

Primary paper: https://arxiv.org/abs/2611.17059  
Pinned source rendering consulted: https://arxiv.cnoctave.top/pdf/2611.17059

The implementation follows Algorithm 1 and equations (18)–(29) for **arbitrary bounded delays**. It does not approximate CA-OCRC with a conventional fixed-capacity FIFO queue.

Per round, the exact order is:

1. Construct the prediction set from the current risk threshold.
2. Evict tracked round i when i + proxy_delay_i < current_round.
3. For a nontrivial set, independently sample a proxy delay from the declared PMF; trivial full/empty sets use proxy delay -1 and do not consume tracking capacity.
4. Admit the current round only when the proxy delay is nonnegative and post-eviction occupancy is below K.
5. Receive only feedback for still-tracked predictions whose true feedback delay is covered by their proxy delay.
6. Apply the threshold update: trivial known loss plus inverse-probability-weighted received losses.

For arbitrary bounded delays the paper sets d_bar_max = d_max and q_G = 1. Thus the feedback correction is exactly:

r_(t+1) = r_t + eta * phi_t * (L_t - alpha) + eta * sum_(i in O_t) (L_i - alpha) / F(d_i)

where F(d) = Pr[proxy_delay >= d]. The independent proxy delay and its survival function compensate for the random tracking decision. Capacity saturation is accounted for by the theorem's separate saturation term; it is not an extra ad hoc weight.

## Implemented v0.1

- Discrete proxy-delay PMF on {-1, 0, ..., d_max} and exact survival probabilities.
- Deterministic PRNG state sealed into the replay digest.
- Finite tracking set with the paper's strict expiry inequality and post-eviction admission.
- Scalar regression interval [prediction - threshold, prediction + threshold] with bounded score threshold B.
- Trivial full-real-line and empty sets derive loss 0 and 1 by construction, update immediately, and are never tracked.
- Nontrivial intervals are tracked only if proxy delay >= 0 and capacity permits.
- Same-round delay-0 feedback is received at the end of the same round if the prediction was admitted.
- Feedback received while tracked is weighted by 1/F(actual_delay); an expired/unadmitted result is recorded as UNTRACKED and does not update.
- K1 eligibility gate: current K1 frontier, eligible status, downstream permission, exact label ID/revision, measurement-event identity, full payload digest, corrected numeric value, uncertainty, certificate/report digest and clock/availability order.
- UNKNOWN feedback is held while the proxy tracking window remains open; a later K1-qualified receipt may be accepted before expiry. If the window expires with unresolved UNKNOWN feedback, the guarantee state becomes uncertified.
- BLOCKED or identity/digest/timing/value-mismatched feedback never updates the threshold and marks the guarantee state uncertified.
- Immutable prediction, feedback and round receipts; state updates are transactional through a cloned engine.
- A global UnderDeclaredArbitraryBoundedDelayAssumptions vs UncertifiedInvalidOrOutOfModelFeedback state; no guarantee is claimed after invalid feedback is encountered.
- Theorem 1 finite-horizon arbitrary-delay bound helper.
- Proposition 1 Bernoulli proxy-delay family and a one-dimensional optimizer for p*.

## Theorem 1 and Proposition 1 implemented

The paper's arbitrary-delay bound is:

RiskGap(T) <= B/(eta*T) + (K+1)/(F(d_max)*T) + exp(-K*h(ell_F/K))

where h(x)=x-1-ln(x) and ell_F=sum_(a=1)^d_max F(a) <= K.

The optimized Bernoulli proxy design has F(a)=p for a=1,...,d_max, with:

p* = argmin_(0<p<=min(1,K/d_max)) [(K+1)/(p*T) + exp(-K*h(d_max*p/K))]

and the PMF is Pr[proxy=-1]=1-p*, Pr[proxy=d_max]=p*.

The helper refuses a distribution whose tracking load exceeds K and refuses a bound with F(d_max)=0.

## Explicit exclusions

The stochastic-delay model in the paper has an additional q_G(d_bar_max)=Pr[d<=d_bar_max] correction in the threshold update and a different tracking load ell_(F,G)=sum F(a)G(a). **Neither is implemented here.** The stochastic optimal proxy family (equations 31–37), including kappa(a) and theta*, is deferred until an independently checked delay PMF/G-survival contract exists.

The current implementation assumes an externally declared deterministic upper bound d_max; it does not infer an IID physical-metrology delay distribution from observed delays. A fab with unknown/unbounded delays cannot claim the arbitrary-bounded-delay theorem.

## Authored test matrix

Twelve Rust tests cover:
1. exact eviction/admission order and hard capacity;
2. same-round immediate feedback;
3. delayed feedback with the exact inverse survival weight;
4. full/empty set losses without tracking;
5. UNKNOWN hold and later K1-qualified re-presentation within proxy lifetime;
6. BLOCKED feedback no-update and uncertified guarantee state;
7. expired/unadmitted feedback as untracked no-update;
8. duplicate label revision idempotency;
9. PMF survival, tracking load and Assumption 4 validation;
10. optimal arbitrary-delay p* against a dense grid oracle;
11. Theorem 1 bound against a known numeric expression;
12. deterministic replay and capacity invariant.

Exact-head rustfmt, strict Clippy, crate tests and SEMI-BENCH-001 reference workflow are required before pass claims.

## Future stochastic-delay extension

Only after the arbitrary-delay branch passes:
- define a versioned empirical or externally justified delay survival function G;
- distinguish empirical delay diagnostics from the theorem's IID stochastic-delay assumption;
- implement q_G(d_bar_max), stochastic load, Theorem 2 R_(F,G) bound and Proposition 2 optimal proxy family;
- independently verify kappa(a), theta* and grid-search selection;
- test heavy-tailed, multimodal, censored, correlated and regime-switching delays. If IID is not justified, remain on the arbitrary bounded-delay path or mark guarantees unavailable.

## Claim ceiling

This is an algorithmic reproduction of the paper's arbitrary bounded-delay path with additional K1 evidence gates. It is not the stochastic-delay CA-OCRC result, not an empirical semiconductor performance win, not a proof of IID assay delays, and not a fab-control authorization.