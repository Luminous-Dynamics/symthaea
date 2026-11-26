# SEMI-CA-OCRC-001 — arbitrary bounded-delay CA-OCRC v0.1

**Canonical research issue:** #13499  
**Related programme/clarifications:** #13479, #13578, #13561, #13645, #13579  
**Implementation:** symthaea-semiconductor::benchmark::ca_ocrc_bounded  
**Status:** Rust engine with unit and independent integration oracle tests authored; exact-head validation pending.

## Primary source and exact algorithm

Wang & Simeone, “Capacity-Constrained Delayed Feedback for Online Conformal Risk Control,” arXiv:2611.17059v1 (12 November 2026): https://arxiv.org/abs/2611.17059

This v0.1 implements only the paper's arbitrary but known-bounded delay case:
- actual feedback delay is declared bounded by d_max;
- proxy delay support is {-1,0,...,d_max};
- q_G(d_max)=1;
- the stochastic-delay/IID-G extension and its second correction factor are explicitly out of scope.

For a nontrivial prediction set at round t:
1. expire tracked i only if i+dbar_i<t;
2. sample an independent proxy delay dbar_t from F;
3. admit iff dbar_t>=0, physical outcome acquisition actually occurred, and capacity remains after eviction;
4. receive qualified tracked outcomes whose actual delay expires at t;
5. remove received entries and update threshold.

A trivial full/empty set has known loss (0 for full, 1 for empty), consumes no tracking capacity, and contributes phi_t(L_t-alpha) immediately.

The exact update is:

[
r_{t+1}=r_t+etaleft[
phi_t(L_t-alpha)+
sum_{iin O_t}rac{L_i-alpha}{F(d_i)}
ight],
]

where F(a)=P(proxy_delay>=a), O_t is the set of tracked qualified outcomes received this round, and q_G=1 because the configured proxy horizon equals the declared arbitrary delay bound.

## Proxy distribution and theorem diagnostics

The policy stores the discrete PMF over delays -1 through d_max. Survival probabilities are derived from the PMF, never supplied independently.

[
F(a)=sum_{j=a}^{d_{max}}P(ar d=j),qquad
ell_F=sum_{a=1}^{d_{max}}F(a).
]

The validator enforces:
- PMF length and nonnegative finite probabilities;
- PMF sum equals one within numerical tolerance;
- positive F(d_max);
- underload ell_F<=K;
- positive capacity and positive declared d_max.

The arbitrary-delay theorem expression is reported only when the caller declares an actual-delay contract and runtime evidence has not invalidated the theorem assumptions:

[
operatorname{RiskGap}(T)le
rac{B}{eta T}
+rac{K+1}{F(d_{max})T}
+exp{-K h(ell_F/K)},
qquad h(x)=x-1-ln x.
]

This field is an assumption-conditional paper bound expression, not a measured fab risk guarantee. The caller-supplied delay-contract digest identifies the declared bound contract; the code cannot independently prove that all physical feedback delays obey it.

The optional optimizer searches the paper's one-dimensional Bernoulli proxy family: P(dbar=-1)=1-p*, P(dbar=d_max)=p*, with p* minimizing the arbitrary-delay objective on its valid interval. The implementation includes a golden-section search; an independent dense-grid oracle cross-checks its output.

## Semiconductor-specific distinction: two capacities and four label states

There are two fundamentally different capacity constraints:

1. **Physical metrology acquisition capacity:** which wafers are actually measured. This determines whether a physical label exists.
2. **CA-OCRC feedback-tracking capacity K:** which existing outcomes the online risk controller retains during their delayed arrival window.

A non-selected wafer has no measurement label. It is not an outcome that was randomly evicted from the CA-OCRC tracking set. The engine therefore records physical selection separately, does not allocate a tracking slot for a non-selected wafer, and invalidates the theorem flag for that run.

The K1 gate is also distinct from proxy expiry:
- current and eligible K1 report: compute the loss from the frozen prediction set and corrected physical value;
- UNKNOWN: hold without threshold update; a later qualified report may be re-presented while the proxy slot remains live, but the run's theorem flag is downgraded because the qualified-only feedback process adds a selection mechanism;
- BLOCKED: no update, remove the invalid outcome from the tracking set, and downgrade theorem applicability;
- identity/revision/payload/event/time mismatch: reject, no update, and record assumption failure;
- a valid outcome arriving after proxy expiry: it is not used online, even if it exists in an offline audit.

The label report must bind the exact label revision, payload digest, event identity and corrected value, and be available by delivery. No unqualified or stale physical value can update r_t.

## Implementation details

- Proxy draws are injected as caller-supplied U_t in [0,1); no implicit global RNG is used.
- Each prediction receipt binds round, event/sample, frozen prediction set, threshold used, global model and interval-method digests, policy digest, draw, sampled proxy delay, admission and occupancy.
- Each feedback receipt binds label ID/revision, payload/report digests, measurement event and corrected value, actual delay, F(d_i), loss and IPW contribution.
- Feedback is processed in stable sample/label/revision order; input vector order is canonicalized for the round digest.
- Each successful step is transactional: a failed validation cannot partially mutate threshold, tracking, accepted revisions or prediction records.
- Exact expiry uses strict <: an outcome on round i+dbar_i is still trackable; eviction occurs before admission on the next round.
- Full and empty sets use immediate known loss and no proxy draw or tracking.
- No recipe/APC/R2R/equipment authority is present.

## Authored test corpus

### Rust unit tests

The module currently includes tests for:
1. PMF-to-proxy draw mapping and survival probabilities;
2. capacity compliance, strict expiry and admission after eviction;
3. proxy delay -1 not admitted;
4. full/empty known-loss immediate threshold updates without tracking;
5. qualified feedback using frozen prediction interval and 1/F(d_i);
6. same-round zero-delay feedback after admission;
7. exact proxy-expiry feedback is accepted before next-round eviction;
8. post-expiry feedback is not used online even when actual delay remains within d_max;
9. simultaneous feedback terms are summed exactly once and are input-order invariant;
10. UNKNOWN K1 held without threshold update and with theorem state downgraded;
11. BLOCKED K1 never consumed and theorem state downgraded;
12. non-selected physical measurement is not treated as tracking eviction;
13. report-attested revision/payload/event/value binding;
14. proxy overload is rejected;
15. underload and paper-bound diagnostics are explicit;
16. deterministic replay of injected proxy draws and feedback sequence.

### Independent integration oracle

A separate, intentionally slow test file reconstructs the tracking set with its own PMF sampler, strict-expiry map, actual-delay calculations and scalar threshold update. It verifies a four-round known-answer stream with zero-delay and delayed feedback, multiple survival weights, tracking occupancy, final thresholds and optimizer output against an independent grid search.

Exact-head rustfmt, strict Clippy, crate tests and SEMI-BENCH-001 Reference Oracle are still required before claiming the authored tests pass.

## Benchmark ladder

Compare on identical synthetic outcome streams and equal tracking capacity:
- unlimited OCRC (K>=d_max+1);
- intermittent OCRC (K=1);
- CA-OCRC general declared F;
- CA-OCRC optimized Bernoulli F*;
- FIFO/oldest-first, random admission and nearest-expiry as explicitly non-theorem baselines;
- fixed-delay tau-ACI as a distinct fixed-delay comparator;
- UP-OCP as a distinct parameter-free threshold/coverage-regret comparator, not a capacity-tracking substitute.

Report:
- empirical risk and |mean(L)-alpha|;
- paper bound expression and slack only under declared assumptions;
- occupancy, saturation, admission, eviction, proxy-expiry and received counts;
- proxy-delay PMF and survival, actual-delay distribution, F(d_i), IPW weights and ESS;
- threshold trajectory, set width/interval score, p50/p95 update latency and state memory;
- selected / not-selected, eligible / unknown / blocked / rejected labels as separate funnels;
- assumption-invalid reasons and the fraction of horizons for which the theorem flag remains intact.

The physical selection and K1 qualification overlays are not part of the CA-OCRC theorem. Report them separately and never apply the paper's risk bound to the selectively qualified subset without additional identification assumptions.

## Stochastic delay is a separate future version

The stochastic-delay extension requires the paper's exact real-delay distribution G, q_G(dbar_max), tracking-load condition and second inverse-probability factor. It must be independently transcribed and tested. Do not estimate G from only proxy-retained/qualified outcomes and then claim the stochastic theorem.

## Claim ceiling

This is an algorithm/replay implementation for arbitrary bounded feedback delays under a declared proxy distribution and explicitly stated assumptions. It is not a physical measurement selection optimizer, does not create labels, does not correct arbitrary MNAR metrology selection, does not prove source authenticity, does not guarantee per-wafer conditional coverage, and does not authorize fab process control.
