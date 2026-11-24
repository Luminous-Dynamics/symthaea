# SEMI-CA-OCRC-001 — arbitrary bounded-delay tracking v1

**Issue:** #13499  
**Implementation:** `symthaea-semiconductor::benchmark::ca_ocrc_tracking`  
**Status:** arbitrary-delay tracking state machine and authored tests; exact-head validation pending.

## Reference algorithm

Wang & Simeone, *Capacity-Constrained Delayed Feedback for Online Conformal Risk Control*, arXiv:2611.17059 (12 November 2026), Algorithm 1.

CA-OCRC uses a random proxy delay to choose which outcomes to retain, and for how long. It is not a simple queue of the latest K measurements.

For target risk `alpha`, learning rate `eta`, capacity `K`, bounded score `L in [0,B]` and maximum real delay `d_max`:

1. Construct the prediction set using the current threshold.
2. Evict tracked issue round i only when `i + proxy_delay_i < t` (deadline is inclusive).
3. If the current set is empty/full, its loss is known: empty -> 1, full -> 0. Do not track it.
4. Otherwise sample proxy delay `dbar_t` from the PMF on {-1,0,...,d_max}. A delay of -1 means no admission. Admit a non-negative delay only when capacity remains.
5. Receive feedback for tracked rounds whose physical outcomes arrive at the current round; remove a received entry.
6. Update threshold:

[
r_{t+1}=r_t
+\eta\,\phi_t(L_t-\alpha)
+\eta\sum_{i\in O_t}
\frac{L_i-\alpha}{q_G(\bar d_{max})F(d_i)}.
]

Here `F(a)=Pr(proxy_delay >= a)`, `q_G(dbar_max)=Pr_G(real_delay <= dbar_max)`, and `phi_t=1` only for full/empty sets. The proxy survival correction `1/F(d_i)` compensates for randomized tracking; `1/q_G` compensates for outcomes whose real delay exceeds the maximum proxy horizon.

## V1 claim boundary: arbitrary bounded delays

This implementation supports the arbitrary-delay assumption only:
- real feedback delays are arbitrary but bounded by known `d_max`;
- proxy horizon equals `d_max`;
- therefore `q_G(d_max)=1`;
- only the `1/F(d_i)` factor is applied;
- a valid proxy PMF must obey the paper's tracking-load constraint
  `ell_F = sum_(a=1)^dmax F(a) <= K`.

The engine enforces the capacity bound and declared delay limit. If feedback is observed later than `d_max`, it records an assumption violation and does not apply the loss update.

The stochastic IID-delay variant is deliberately not implemented. It requires a declared real-delay distribution G, the `q_G` inverse-probability factor, the stochastic tracking-load constraint and the paper's separate optimized F using `kappa(a)`/theta. A caller must not set a guessed q value and label the result stochastic CA-OCRC.

## Implemented state machine

- Each nontrivial issued prediction samples and records a proxy delay; full/empty sets do not consume a tracking slot.
- The tracking map contains at most K entries.
- Expiry is checked before new admission and uses the exact strict inequality.
- Proxy delay zero can only be used for feedback received at the end of the issue round.
- Outcomes are processed after the new round's admission, matching the algorithm order.
- Feedback is sorted deterministically before processing.
- The actual delay is computed from source round and delivery round; it must not exceed the sampled proxy deadline for an entry to remain tracked.
- Full/empty intervals directly update threshold using the known loss and emit an ordinary prediction receipt with no tracking admission.
- A feedback outcome must pass K1/current eligibility, downstream permission, report digest, payload digest, explicit label ID and revision equality, measurement-event identity, numeric value and uncertainty checks.
- UNKNOWN/BLOCKED/stale/mismatched outcomes do not update the threshold and leave a tracked entry available for a corrected K1 re-presentation until the proxy deadline expires.
- Duplicate accepted label ID/revision is idempotent.
- Each round operates on a cloned engine and commits only after all feedback validation, IPW arithmetic and receipts succeed.
- Per-round and prediction receipts bind policy digests, threshold values, proxy decisions, state before/after and feedback contributions.

The PRNG is SplitMix64 solely for deterministic replay/tests. It is not a cryptographic entropy source. Production random tracking would need a suitable independent entropy source with provenance and must not silently reuse a predictable test seed.

## Optimal arbitrary-delay proxy design

For arbitrary bounded delays, the paper's optimal survival structure is:
- `F(-1)=1`;
- `F(0)=1`;
- `F(a)=p*` for a=1,...,d_max.

A representative PMF is delay 0 with probability `1-p*` and delay `d_max` with probability `p*`. The optimal p solves:

[
p^*=\arg\min_{0<p\le K/d_{max}}
\left[
\frac{K+1}{pT}
+\exp\left(-K h\left(\frac{d_{max}p}{K}\right)\right)
\right],
\quad h(x)=x-1-\ln x.
]

The module includes deterministic golden-section minimization and returns the PMF, tracking load and theorem bound. It does not claim the optimal design for stochastic delays.

## Arbitrary-delay finite-horizon bound

The implementation calculates the paper's Theorem 1 bound:

[
RiskGap(T)\le
\frac{B}{\eta T}
+\frac{K+1}{F(d_{max})T}
+\exp\left[-K h\left(\frac{\ell_F}{K}\right)\right].
]

This is an algorithmic bound under the paper's assumptions. It is not a real-fab guarantee. For selected semiconductor metrology, delays may be informative/MNAR, reports may be blocked or revoked, and source eligibility can fail; these conditions must be measured and stated.

## Semiconductor evidence adaptation

The base CA-OCRC paper treats feedback as a loss outcome. This integration adds a separate K1 qualification gate. Only outcomes with a current eligible K1 report and matching measurement event, label ID/revision, payload digest, corrected numeric value and uncertainty can contribute to the threshold update.

The tracking capacity K has a precise software meaning here: maximum number of pending outcomes whose prediction/interval lineage remains retained by the online conformal controller. It is not the number of instruments or the physical assay capacity. A separate benchmark may model physical metrology capacity, but it must not conflate equipment throughput with in-memory outcome tracking.

Untracked outcomes may remain in an offline audit dataset but cannot leak into the online threshold update. Do not use the offline file to “fill in” an update the online tracker missed.

## Authored known-answer tests

The v1 module tests:
1. tracked set never exceeds K;
2. full/empty set updates known loss without tracking;
3. proxy deadline is inclusive and eviction precedes admission;
4. accepted feedback uses inverse proxy-survival weight;
5. feedback after proxy eviction is untracked/no-update;
6. UNKNOWN feedback is held and leaves its entry retained;
7. corrected K1 re-presentation can update before proxy expiry;
8. event mismatch is rejected with no threshold mutation;
9. duplicate accepted label revision is idempotent;
10. capacity denial occurs after proxy delay sampling;
11. fixed-seed replay is deterministic;
12. PMF load greater than K is rejected;
13. optimal arbitrary proxy design obeys the capacity load and produces a finite bound;
14. PMF survival and tracking-load calculations match known values;
15. input threshold must equal the threshold used to construct the interval;
16. rounds and sample IDs are sequential/unique.

These tests are authored development oracles, not yet a passing validation result.

## Next stage: stochastic IID delays

A separate stage must implement:
- a discrete real-delay PMF and survival G;
- exact `q_G(dbar_max)`;
- the second IPW factor `1/q_G`;
- stochastic load `ell_(F,G)=sum F(a)G(a)`;
- the paper's stochastic risk bound and its `Q_(F,G)` term;
- the optimized proxy survival using the paper's `kappa(a)` and theta*.

Do not reuse the arbitrary-delay optimizer for stochastic feedback.

## Claim ceiling

This is an arbitrary-bounded-delay software replay of Algorithm 1, with an extra label-qualification gate. It is not validated against production semiconductor streams. The theorem does not automatically hold for missing-not-at-random physical inspection, unbounded feedback latency, unknown delay distribution, unqualified labels or violated tracking-load assumptions. It does not authenticate measurement sources, prove conditional per-wafer coverage or authorize equipment/process control.
