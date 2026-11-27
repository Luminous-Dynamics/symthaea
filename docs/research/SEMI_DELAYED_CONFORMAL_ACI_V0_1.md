# SEMI-DELAYED-ACI-001 — delayed-feedback adaptive conformal control v0.1

**Issue:** #13479  
**Implementation:** symthaea-semiconductor::benchmark::delayed_conformal_aci  
**Status:** Rust controller, immutable interval receipts, K1-gated feedback and fixed-delay phase mode implemented; exact-head validation pending.

## Research basis

### Delayed ACI

El Halabi & Brandt, *Adaptive Conformal Inference Under Delayed Feedback: Coverage Guarantees and a Delay-to-Memory Diagnostic*, arXiv:2609.07251 (7 Sep 2026), define the delayed update

\[
\alpha_{t+\tau} = \alpha_t + \gamma(\alpha - \mathrm{err}_t),
\quad
\mathrm{err}_t=\mathbf{1}\{Y_t\notin\widehat{C}^{\tau}_t(\alpha_t)\}.
\]

Under-coverage (err=1) lowers the miscoverage level, making future sets wider; over-coverage raises it. The authors show the recursion decomposes into \(\tau\) interleaved ACI-like sequences, one per phase modulo \(\tau\). This is why a single scalar alpha updated by arrival order is not the exact fixed-delay algorithm when \(\tau>1\).

### Implementation modes

**FixedDelayPhased** is the reference implementation of the published fixed-horizon recursion:
- delay_steps = \(\tau\) is pinned in the policy;
- separate alpha state is held for each phase sequence modulo \(\tau\);
- each prediction freezes phase, alpha, model digests and interval;
- delivery must occur at the declared stream round prediction_sequence + \(\tau\);
- a phase cannot issue its next same-phase prediction until the prior phase outcome has passed K1 qualification and been consumed;
- the update uses the alpha value recorded on the prediction, not a later global state.

This strict phase barrier is deliberate. An UNKNOWN/BLOCKED label cannot be skipped while still claiming the complete fixed-delay ACI sequence guarantee. A later qualified label may be re-presented; the phase remains blocked until then.

**ArrivalOrder** supports variable delivery delays as a deterministic empirical heuristic:
- each accepted K1-qualified delivery updates the current alpha state in monotonically increasing delivery-time order;
- it is useful for exploratory variable-delay streams;
- it does **not** inherit the paper's fixed-delay interleaving theorem. Do not describe it as a theoretically guaranteed arbitrary-delay ACI algorithm.

### Delay-to-memory diagnostic

The paper proposes \(r=\tau/L\), where \(L\) is the relevant temporal memory scale. We will estimate and report separate memory diagnostics for:
- residual autocorrelation (mean dynamics);
- squared-residual / volatility autocorrelation;
- regime persistence (e.g. Markov transition eigenvalue);
- context-specific residual persistence.

Do not report one universal L: the paper finds the ratio organizes AR(1) behavior more strongly than GARCH(1,1) and Markov switching. The operating point of gamma must be tuned on predeclared validation data only.

## Important statistical boundary: metrology selection

The ACI reference assumes a sequential feedback stream. Semiconductor physical metrology is often actively selected based on VM uncertainty, risk, process context or engineering priorities. That makes the observed/qualified label stream selectively sampled.

Therefore:
- coverage measured only on K1-qualified, selected wafers is **selected-stream coverage**;
- it does not automatically imply coverage over all wafers, tools, recipes or lots;
- the qualification gate prevents bad labels from corrupting calibration, but does not itself correct selection bias;
- population coverage requires a declared selection design and independent random-audit labels, or a method with the relevant selection/FCR guarantee.

CAP (Bao et al., JMLR 2025) is a required comparator for online selective conformal prediction and false coverage-statement rate (FCR) under adaptive picking. We must distinguish selection-conditional coverage, long-run marginal coverage and FCR. They are different estimands.

## Implemented event and receipt contract

The engine does not construct intervals itself. An external calibrated interval builder reads the current ACI alpha and returns a numeric interval; the caller submits the exact alpha, point, bounds, model/calibration digests and clock metadata. The engine verifies that the submitted alpha equals the state for that sequence/phase, then freezes the interval.

When the delayed label arrives, the engine:
1. validates full label-payload digest and report digest;
2. validates K1 frontier, numeric value, revision, measurement-event ID and downstream permission;
3. checks physical-label and K1-report availability against prediction and delivery times;
4. checks fixed-delay stream-round/phase order, or arrival-order monotonic time;
5. computes the miss indicator against the **frozen interval**;
6. updates the appropriate phase/current alpha;
7. emits a digest-sealed receipt with the exact prediction and K1 report references.

UNKNOWN/BLOCKED/stale/mismatched/not-yet-available feedback never updates alpha. Historical intervals are not rewritten. No capacity tracking, feedback eviction or inverse-probability weighting is implemented in v0.1.

## Capacity-constrained extension is deliberately deferred

Wang & Simeone, *Capacity-Constrained Delayed Feedback for Online Conformal Risk Control*, arXiv:2611.17059 (12 Nov 2026), introduce CA-OCRC. It randomly draws a proxy delay, admits/evicts pending feedback under capacity K, and corrects the threshold update using the inverse of both the actual-delay arrival probability and the proxy-delay survival probability.

For nontrivial prediction sets, their update includes

\[
r_{t+1}=r_t+\eta\varphi_t(L_t-\alpha)
+\eta\sum_{t'\in\mathcal{O}_t}
\frac{L_{t'}-\alpha}{q_G(\bar d_{\max})F(d_{t'})},
\]

where \(q_G(\bar d_{\max})=\Pr_G[d_t\leq\bar d_{\max}]\), and \(F(d)\) is the proxy-delay survival probability. Under deterministic delays with \(\bar d_{\max}=d_{\max}\), \(q_G=1\). The algorithm also handles trivial prediction sets without tracking feedback.

We will not implement a superficially similar sampler. The next CA-OCRC implementation gate is:
- transcribe and test the complete Algorithm 1 admission/eviction/reception order;
- freeze F(d), q_G, assumptions and random seeds;
- implement both arbitrary-delay and stochastic-delay variants separately;
- verify the complete IPW update and risk-gap bound;
- test capacity saturation, zero propensity, delayed feedback loss and proxy-delay distribution misspecification;
- compare K=1,2,4,8,16,32 at equal physical-metrology budget.

The paper's optimized proxy-delay design and scaling results apply only under its stated assumptions; do not transfer those theorems to K1-qualified, actively selected fab measurements without a new selection analysis.

## Benchmark matrix

Baselines:
- fixed split conformal;
- rolling residual quantile;
- vanilla immediate-feedback ACI;
- exact tau-DACI with separate modulo-tau alpha states;
- variable-delay arrival-order ACI heuristic;
- online conformal prediction via online optimization (Areces et al., ICML/PMLR 2025);
- CAP selection/FCR control (Bao et al., JMLR 2025);
- CA-OCRC after independent exact reproduction;
- non-exchangeable panel conformal (Tu & Giesecke, arXiv:2605.17705);
- fixed/global interval and no-adaptation baselines.

Streams:
1. AR(1) residual dependence with controlled persistence;
2. GARCH(1,1) volatility persistence;
3. Markov regime switching;
4. abrupt mean and variance shifts;
5. gradual drift and process recipe transitions;
6. chamber/tool-specific and common-reference shifts;
7. heavy tails/outliers and measurement-system errors;
8. constant fixed delay tau=0,1,2,4,8,16;
9. variable and out-of-order arrival-order heuristic streams;
10. K1 UNKNOWN/BLOCKED/revocation and label revision mismatch;
11. selected-only physical metrology, randomized audit and non-random audit;
12. capacity saturation and feedback loss (CA-OCRC phase).

Metrics:
- long-run miscoverage and rolling-window miscoverage;
- per-tool/chamber/recipe coverage and worst-unit coverage;
- selection-conditional coverage and FCR separately;
- interval width, normalized and physical units, interval/Winkler score;
- adaptation recovery time and overshoot after shift;
- alpha phase divergence and oscillation;
- delay-to-memory tau/L by residual process;
- qualified/unknown/blocked/late/lost/tracked labels;
- capacity utilization, pending queue length, feedback retention probability;
- CPU/memory, per-wafer p50/p95 update time, and replay determinism.

## Proposed targets (hypotheses, not achievements)

- 100% no alpha updates from UNKNOWN/BLOCKED/stale/unavailable/mismatched K1 receipts.
- 0 historical prediction interval mutations.
- Exact phase recurrence on fixed-delay known-answer sequences for tau=1,2,3,4,8,16.
- 100% deterministic replay for a frozen prediction/delivery stream.
- Report empirical coverage and finite-sample bounds only under the assumptions actually satisfied.
- Separate selected-stream, randomized-audit population, per-panel and FCR outcomes.
- Capacity mode: exact implementation of the CA-OCRC algorithm and its estimator before claiming its risk bound.

## Claim ceiling

This v0.1 module is a delayed adaptive miscoverage controller around externally produced intervals. It is not a full conformal quantile/score engine, it does not solve adaptive metrology selection bias, it does not implement CA-OCRC, and its variable-delay arrival-order mode has no fixed-delay theorem claim. It does not authenticate physical source data or authorize fab process control.
