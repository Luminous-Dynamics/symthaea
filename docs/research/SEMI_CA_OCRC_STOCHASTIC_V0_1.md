# SEMI-CA-OCRC-STOCHASTIC-001 — IID delayed-feedback risk control v0.1

**Issue:** #13896  
**Base implementation:** #13578 / draft PR #13892  
**Implementation branch:** `feat/semi-ca-ocrc-stochastic-v1`  
**Reference:** Wang & Simeone, *Capacity-Constrained Delayed Feedback for Online Conformal Risk Control*, arXiv:2611.17059 (12 Nov 2026), https://arxiv.org/abs/2611.17059  
**Status:** Section VI stochastic-delay policy, update, theorem bound and proxy optimizer implemented; 25 inherited unit tests + 7 stochastic integration tests authored; exact-head validation pending.

## 1. Why a distinct mode is necessary

The arbitrary-bounded-delay algorithm assumes a known maximum delay (d_{max}) and uses (q_G=1) when the proxy horizon equals the full delay bound. Section VI covers an IID stochastic feedback-delay sequence and permits a shorter proxy-retention horizon (ar d_{max}le d_{max}).

These are distinct estimands and theorem assumptions. The implementation therefore uses an explicit mode:

- `ArbitraryBounded`: (ar d_{max}=d_{max}), no IID (G) profile or delay-law manifest; use the bounded-delay update and Theorem 1/Lemma 2 helper.
- `IidStochastic`: separate (G), (ar d_{max}), a content-addressed delay-law evidence manifest, two-level IPW and Theorem 2/Proposition 2 functions.

The bounded path is unchanged and refuses a stochastic policy. The stochastic path does not reuse the bounded-delay risk bound.

## 2. Delay distributions and policy contracts

The true-delay survival is

[
G(a)=P(d_tge a),quad a=0,ldots,d_{max},
]
with (G(0)=1) and (G(d_{max}+1)=0).

The proxy-retention survival is

[
F(a)=P(ar d_tge a),quad a=0,ldots,ar d_{max},
]
with (F(-1)=1) implicitly.

The proxy horizon must not exceed the declared true-delay support:
[
0le ar d_{max}le d_{max}.
]

The probability the actual feedback arrives within the proxy horizon is

[
q_G(ar d_{max})=P(d_tle ar d_{max})
=1-G(ar d_{max}+1).
]

The policy validator requires:
- a non-increasing finite (G) vector on ([0,d_{max}+1]);
- (G(0)=1) and (G(d_{max}+1)=0);
- positive (q_G);
- (ar d_{max}) is a supported delay point with positive mass (G(ar d_{max})-G(ar d_{max}+1));
- a finite monotone proxy survival vector (F) on ([0,ar d_{max}]), with positive (F(ar d_{max}));
- a valid SHA-256 delay-law evidence-manifest digest for IID mode. The manifest digest is included in every round receipt and the stochastic risk-bound output.

The manifest binds the declared delay law to its data/source window and estimation procedure. It does not prove IID or stationarity by itself.

## 3. Exact stochastic update

For each round, prediction, eviction, admission, feedback reception and threshold update retain the bounded implementation's order. Feedback is IPW-weighted only after the K1 label evidence validates.

The stochastic update is

[
r_{t+1}=r_t+eta,phi_t(L_t-alpha)
+etasum_{iinmathcal O_t}
rac{L_i-alpha}{q_G(ar d_{max})F(d_i)},
]

where (phi_t=1) only for trivial full/empty sets and (mathcal O_t) contains feedback received while its prediction remained tracked.

For actual delay (d_i>ar d_{max}) but (d_ile d_{max}), the feedback is permanently untracked. This is not a delay-support breach; it is the missing-feedback event corrected by (q_G).

For (d_i>d_{max}), the declared IID delay support is breached and the guarantee status must be invalidated.

Each feedback receipt separately records:
- actual delay and proxy delay;
- (F(d_i));
- (q_G(ar d_{max}));
- combined inverse-probability weight (1/(q_GF(d_i)));
- K1 label receipt, label revision and payload digest.

## 4. Theorem 2 diagnostic

The stochastic tracking load is

[
ell_{F,G}=sum_{a=1}^{ar d_{max}}F(a)G(a),
]
and must satisfy (ell_{F,G}le K).

Let

[
aralpha=max(alpha,1-alpha),
]

[
mathbb E[dmid dlear d_{max}]
=sum_{a=0}^{ar d_{max}}
rac{a,[G(a)-G(a+1)]}{q_G},
]

and

[
Q_{F,G}=sum_{a=0}^{ar d_{max}}
rac{G(a)-G(a+1)}{q_G^2F(a)}.
]

The range term implemented from Section VI is

[
R_{F,G}=
aralphaleft(
mathbb E[dmid dlear d_{max}]+2
+rac{Karalpha^2}{2(1-aralpha)}Q_{F,G}
ight).
]

The Theorem 2 bound is

[
operatorname{RiskGap}(T)le
rac{B}{eta T}
+rac{R_{F,G}}{T}
+mathbf 1{ar d_{max}ge K}
expleft[-K hleft(rac{ell_{F,G}}{K}ight)ight],
]
where (h(x)=x-1-ln x).

The stochastic risk-bound output carries the status `DeclaredByCallerNotIndependentlyVerified`: the engine binds the claimed G evidence but does not infer IID from the feedback records that survive proxy censoring. An independent arrival-time audit is necessary to investigate IID assumptions; correlated/queue-driven delays must remain stress-test-only unless their law is established.

The code reports the individual terms—(q_G), (ell_{F,G}), conditional mean delay, (Q_{F,G}), (R_{F,G}), capacity term and final bound—rather than a single opaque number. It refuses (q_G=0), zero proxy survival on a possible feedback age, non-finite terms, or tracking load above (K).

## 5. Proposition 2 proxy optimizer

The optimizer searches (ar d_{max}) only among support points with positive true-delay mass:
[
G(ar d_{max})-G(ar d_{max}+1)>0.
]

When (ar d_{max}<K), the optimum is full proxy retention:
[
F^*(a)=1,quad a=0,ldots,ar d_{max}.
]

When (ar d_{max}ge K):
[
F^*(0)=1,qquad
F^*(a)=min(1,	heta^*sqrt{kappa(a)}),
quad 1le alear d_{max},
]
with
[
kappa(a)=
min_{1le sle a}
max_{ale s'lear d_{max}}
rac{G(s)-G(s'+1)}
{sum_{i=s}^{s'}G(i)}.
]

The implementation computes the finite nested sums deterministically and uses a bracketed golden-section scalar search for (	heta). It enumerates admissible proxy horizons and selects the lowest theorem bound, with deterministic tie-breaking. An independent dense-grid optimizer oracle remains an acceptance requirement.

## 6. Evidence and claim boundary

The stochastic theorem is conditional on:
- IID actual feedback delays with the declared (G);
- independence of proxy randomization from the actual delay sequence;
- bounded score/loss assumptions from the paper;
- valid tracking-load and survival-probability conditions;
- outcomes existing for the rounds considered by the risk estimand.

The K1 gate is an additional engineering requirement. If a physical label is UNKNOWN/BLOCKED or invalid, it does not update the threshold; the run-level status is downgraded from the paper's complete-feedback claim.

The physical-acquisition process is still distinct: a wafer never selected for physical metrology has no acquired label delay to track. Neither (q_G) nor (F(d)) corrects that acquisition bias. Work on selective measurement and population coverage remains in #13579 and #13646. A randomized audit stream or a separately proved propensity estimator is required before making a full-wafer-population risk claim.

Finally, a fab's dispatch queue can generate bursty, correlated and nonstationary label delays. Those are mandatory stress tests, but must not be labelled IID-theorem-certified merely because an empirical histogram resembles (G). An estimated (G) is a conditional plug-in model until its estimation uncertainty and dependence assumptions are separately justified.

## 7. Authored tests and remaining acceptance gates

The stochastic integration tests currently cover:
1. exact (q_GF(d)) update for a known three-point delay law;
2. outcomes after proxy horizon but within true support are not tracked and do not falsely breach (d_{max});
3. the Theorem 2 bound against an independently calculated scalar formula;
4. monotone/supported optimizer output and tracking-load constraint;
5. rejection of (q_G=0) and rejection of stochastic policies by the bounded theorem helper;
6. rejection of an IID profile without a content-addressed delay-law manifest;
7. optimizer comparison against an independent dense-grid theta/horizon reference.

Inherited unit tests cover bounded-mode invariants and the stochastic unit tests will be run in the full semiconductor crate. Exact-head rustfmt, strict Clippy, all unit/integration tests, an independent stochastic known-answer oracle and SEMI-BENCH-001 Reference Oracle validation are still required.

## 8. Claim ceiling

This is a software implementation of the paper's IID-delay theorem under a declared delay-law assumption. It does not prove that real fab delays are IID, stationary or independent of proxy randomization. It does not correct which wafers were physically selected or whether K1 qualification failures are ignorable. Correlated/queue-driven delays are stress-test results, not theorem guarantees. No wafer disposition, process control or equipment authority is added.
