# History-dependent projective portfolio mergers for evolving hypotheses

**Date:** 2026-11-01  
**Status:** research construction + exploratory local reference implementation  
**Related issue:** [symthaea#9848](https://github.com/Luminous-Dynamics/symthaea/issues/9848)

## 1. Research gap

The online testing problem has two clocks: hypotheses arrive over time, and evidence for already-active hypotheses continues to update. Recent dynamic e-closure theory gives a coherent framework for stopped-FDR and stronger persistent guarantees, but identifies history-dependent mergers of parallel evolving processes, random predictable arrivals, persistent calibration, and scalable computation as open directions. Pointwise arbitrary-dependence mergers are restricted to the ordinary static e-merger class; to go beyond that class, a merger must use path history or predictable one-step increments.

This project investigates one deliberately narrow class: **self-financing portfolios of coordinate test supermartingales**. It is not a merger for arbitrary e-process objects. Every coordinate must be a nonnegative test supermartingale for its elementary null in the same global filtration used by the portfolio and global stopping rule. An e-process that is not itself a supermartingale, or one valid only in a smaller local filtration, cannot be substituted without a separately proved envelope or filtration-lifting step.

### Prior-art boundary (important)

Vovk and Wang (2024), *Merging sequential e-values via martingales*, already characterize a broad class of history-dependent mergers for **sequentially revealed scalar e-values**. Their martingale-merging representation includes predictable bets based on past scalar e-values; their empirically adaptive method is a required baseline, not a competitor we can ignore. The general fact that predictable self-financing allocations preserve supermartingale validity is also classical. We therefore do **not** claim to invent adaptive martingale merging or the general self-financing argument.

The more specific question here is different: several coordinate test-supermartingales evolve **in parallel**, their one-step returns are observed as a vector under a common global filtration, within-round dependence may be arbitrary, hypothesis rosters grow predictably, and the merged intersection process must be neutral to dormant future coordinates so that it composes with dynamic e-closure. Sequential scalar merging can be an appropriate comparator when each within-round coordinate return remains conditionally valid after earlier same-round returns are revealed. It is not automatically a valid comparator when only the vector-level, pre-round conditional bounds are known.

## 2. Candidate process

Let \((\mathcal F_t)\) be the common global filtration. Let \(I_t\) be a nested active-hypothesis set whose next-step change is known before the next observation. Fix global prior masses

\[
\gamma_i\geq0,\qquad \sum_i\gamma_i\leq1.
\]

For each hypothesis \(i\), let \(M_{i,t}\geq0\) be a test supermartingale under \(H_i\), with the dormant convention \(M_{i,t}=1\) before activation. Fix a finite target intersection \(S\), and write \(A_t^S=S\cap I_t\). For concentration parameter \(\eta>0\), define

\[
c_t^S=1-\sum_{i\in A_t^S}\gamma_i,
\qquad
D_{t-1}^{S,\eta}=c_t^S+\sum_{i\in A_t^S}\gamma_iM_{i,t-1}^{\eta}.
\]

When \(D>0\), choose predictable fractions

\[
p_{0,t}^{S,\eta}=\frac{c_t^S}{D_{t-1}^{S,\eta}},\qquad
p_{i,t}^{S,\eta}=\frac{\gamma_iM_{i,t-1}^{\eta}}{D_{t-1}^{S,\eta}}.
\]

The cash asset has gross return one. If \(M_{i,t-1}>0\), let \(R_{i,t}=M_{i,t}/M_{i,t-1}\). If \(M_{i,t-1}=0\), the corresponding fraction is zero; a nonnegative supermartingale cannot become positive again after hitting zero. Set

\[
V_0^{S,\eta}=1,
\qquad
V_t^{S,\eta}=V_{t-1}^{S,\eta}
\left(p_{0,t}^{S,\eta}+\sum_{i\in A_t^S}p_{i,t}^{S,\eta}R_{i,t}\right).
\]

If the denominator is zero, the portfolio is already at zero wealth and is absorbed there.

### Conditional validity argument

Under \(H_S=\cap_{i\in S}H_i\), every coordinate asset \(M_i\), \(i\in A_t^S\), is a nonnegative supermartingale in the same filtration. Thus

\[
\mathbb E[R_{i,t}\mid\mathcal F_{t-1}]\leq1
\]

whenever its prior value is positive. The cash return equals one. Since all fractions are nonnegative, predictable, and sum to one,

\[
\mathbb E\left[\frac{V_t^{S,\eta}}{V_{t-1}^{S,\eta}}\mid\mathcal F_{t-1}\right]
\leq p_{0,t}^{S,\eta}+\sum_i p_{i,t}^{S,\eta}=1.
\]

Therefore \(V^{S,\eta}\) is a nonnegative test supermartingale under \(H_S\). This is the core proof obligation and depends critically on common-filtration conditional validity and predictable allocation. It is not established by marginal validity alone.

### Projectivity and arrival neutrality

When a future coordinate \(j\) is added to the target set but remains dormant, its evidence is one and it is not in \(A_t^S\). The active set and cash mass are unchanged, so the portfolio paths remain exactly equal through the current horizon.

At activation, \(M_{j,t-1}=1\). The prior mass \(\gamma_j\) moves from cash into the newly active asset at price one: the denominator remains unchanged, all old fractions remain unchanged, and the new asset receives fraction \(\gamma_j/D\). If its first gross return is one, the portfolio wealth is unchanged; otherwise the new evidence affects wealth normally. This produces exact future-extension coherence for dormant extensions.

### Concentration and robust mixture

- \(\eta=1\) is exactly the static globally weighted merger \(1-\sum_{i\in A_t^S}\gamma_i+\sum_{i\in A_t^S}\gamma_iM_{i,t}\).
- \(0<\eta<1\) dampens concentration and favors diversified evidence.
- \(\eta>1\) concentrates more quickly in streams with strong accumulated evidence.

For fixed prior probabilities \(\pi_q\), the mixture

\[
\bar V_t^S=\sum_{q=1}^{Q}\pi_qV_t^{S,\eta_q}
\]

is again a test supermartingale and preserves projectivity. Pathwise,

\[
\log\bar V_t^S\geq\log\pi_q+\log V_t^{S,\eta_q}
\]

for every component \(q\). With a uniform prior over \(Q\) experts, the mixture is at most \(\log Q\) log-wealth units behind the best fixed expert on any path. This is a deterministic regret guarantee, not a guarantee that mixture power exceeds the static merger in every data regime.

### Markov mixture over switching rebalancing policies

The fixed-\(\eta\) mixture cannot follow a regime that changes from one concentration level to another. Let \(\eta_1,\ldots,\eta_Q\) be a finite expert set and let \(P\) be a row-stochastic transition matrix with an ex-ante initial prior \(\pi\). For every deterministic schedule \(s_{1:T}\in[Q]^T\), choosing \(\eta_t=\eta_{s_t}\) before round \(t\) gives a valid predictable portfolio process with one-step factor \(g_t(s_t)\). Put a fixed Markov prior on schedules:

\[
\Pr(s_{1:T})=\pi_{s_1}\prod_{t=2}^{T}P_{s_{t-1},s_t}.
\]

A weighted sum over all such schedule processes is again a test supermartingale. It can be computed without enumerating exponentially many schedules by a forward recursion:

\[
L_1(k)=\pi_k g_1(k),\qquad
L_t(k)=g_t(k)\sum_j L_{t-1}(j)P_{jk},\qquad
\bar V_t=\sum_k L_t(k).
\]

This requires \(O(Q^2)\) operations and \(O(Q)\) state per intersection per round, in addition to the cost of evaluating each \(g_t(k)\). For every schedule with positive prior mass, the pathwise guarantee is

\[
\log\bar V_T\geq\sum_{t=1}^{T}\log g_t(s_t)+\log\pi_{s_1}+\sum_{t=2}^{T}\log P_{s_{t-1},s_t}.
\]

Thus the penalty for tracking a regime-switching schedule is exactly its negative log prior probability. This is a mixture over precommitted predictable policies, not a policy that selects \(\eta_t\) using the current round's outcome. The implementation and benchmark below use a four-state chain with 0.05 switch hazard and uniform transitions between distinct states.

## 3. Connection to dynamic e-closure

For every finite \(S\), the portfolio is an intersection e-process. Its paths agree under neutral padding by future hypotheses. It can therefore be supplied as the intersection-process collection to a dynamic e-closure procedure for stopped-FDR control.

For SupFDR and setwise persistence, a time-monotone certificate is required. The intended extension is to apply a literature-valid adjuster to the running maximum of each intersection process, then re-check coherence. Raw running maxima are not automatically e-processes and must not be used unadjusted.

The full closure still has exponentially many subset constraints in the general case. The present construction changes the evidence merger, not the complexity of exact closure. Any scalable closure search must be treated as a separate proof and benchmark lane.

## 4. Preliminary and fixed-seed benchmark exploration

The reproducible NumPy benchmark uses five coordinate Gaussian test supermartingales over 60 rounds, exponential increment parameter \(\lambda=0.5\), correlation \(\rho\in\{0.2,0.5,0.8\}\), 20,000 streams per cell, and threshold 20 (nominal single-intersection level 0.05). It compares the static \(\eta=1\) merger, a uniform fixed-prior mixture over \(\eta\in\{0.5,1,2,4\}\), and a Markov mixture over the same states with 0.05 switching hazard. The Markov chain transitions uniformly to one of the other three states when it switches. Seeds are fixed by the checked-in seed formula.

The Markov mixture is only a small improvement in some switching settings; it does not dominate the fixed mixture or static method uniformly. The detailed percentages below are **single-intersection crossing diagnostics**, not FDR or SupFDR.

### All hypotheses active at the start

| Correlation | Scenario | Static | Fixed-\\eta mixture | Markov switching mixture |
|---:|---|---:|---:|---:|
| 0.2 | all-null | 3.050% | 3.035% | 3.080% |
| 0.2 | sparse | 84.285% | 83.955% | 84.175% |
| 0.2 | dense | 65.500% | 66.240% | 64.270% |
| 0.2 | switching | 25.415% | 25.685% | 25.690% |
| 0.5 | all-null | 3.450% | 3.450% | 3.455% |
| 0.5 | sparse | 84.425% | 84.990% | 84.900% |
| 0.5 | dense | 55.270% | 54.380% | 53.750% |
| 0.5 | switching | 23.870% | 25.260% | 25.290% |
| 0.8 | all-null | 3.640% | 3.605% | 3.620% |
| 0.8 | sparse | 84.310% | 86.770% | 86.185% |
| 0.8 | dense | 46.295% | 45.625% | 45.605% |
| 0.8 | switching | 21.215% | 24.690% | 24.555% |

### Predictable staggered arrivals

| Correlation | Scenario | Static | Fixed-\\eta mixture | Markov switching mixture |
|---:|---|---:|---:|---:|
| 0.2 | all-null | 1.950% | 2.145% | 2.175% |
| 0.2 | sparse | 6.360% | 5.640% | 5.870% |
| 0.2 | dense | 44.390% | 43.965% | 43.460% |
| 0.2 | switching | 28.700% | 30.525% | 31.025% |
| 0.5 | all-null | 1.995% | 2.090% | 2.110% |
| 0.5 | sparse | 6.105% | 5.520% | 5.715% |
| 0.5 | dense | 40.910% | 40.470% | 40.220% |
| 0.5 | switching | 29.085% | 30.760% | 31.285% |
| 0.8 | all-null | 2.210% | 2.305% | 2.320% |
| 0.8 | sparse | 6.400% | 5.885% | 6.135% |
| 0.8 | dense | 38.130% | 37.990% | 38.065% |
| 0.8 | switching | 28.555% | 30.800% | 31.250% |

### Null crossing confidence intervals

Clopper-Pearson 95% intervals for the 20,000-stream null cells are all below 5% in this particular Gaussian simulation. This is a finite simulation diagnostic, not the proof of validity and not evidence for every dependence structure.

| Arrival regime | \(\rho\) | Static 95% CI | Fixed mixture 95% CI | Markov mixture 95% CI |
|---|---:|---:|---:|---:|
| All active | 0.2 | [2.816, 3.298]% | [2.802, 3.282]% | [2.845, 3.329]% |
| All active | 0.5 | [3.201, 3.712]% | [3.201, 3.712]% | [3.206, 3.717]% |
| All active | 0.8 | [3.385, 3.909]% | [3.351, 3.873]% | [3.365, 3.888]% |
| Staggered | 0.2 | [1.763, 2.151]% | [1.949, 2.355]% | [1.977, 2.387]% |
| Staggered | 0.5 | [1.806, 2.198]% | [1.896, 2.298]% | [1.915, 2.319]% |
| Staggered | 0.8 | [2.011, 2.423]% | [2.101, 2.523]% | [2.116, 2.538]% |

### Interpretation

The clearest pattern is that higher-\(\eta\) or switching allocations can help when signal support changes, especially under staggered arrivals. They can lose under stationary dense alternatives and for a sparse signal that appears only in a late-arriving stream. The fixed-\(\eta\) mixture is a hedge against a fixed unknown concentration regime; the Markov mixture pays an additional prior penalty in exchange for tracking changing regimes. The next benchmark should compare several prespecified switch hazards and a fixed mixture over hazards rather than choose a hazard after inspecting the test data.

These experiments cover one Gaussian factor model and a narrow dimension/horizon. They do not compare against the full Vovk-Wang scalar sequential-merger benchmark, do not provide full-FDR/SupFDR evidence, and are not a formal SOTA comparison.

### Second-level mixture over switching hazards

A further extension removes the need to select a single Markov switch hazard. We now mix five prespecified transition models with hazards \(q\in\{0,0.01,0.05,0.2,0.5\}\), including the no-switch model. Each hazard model is itself a fixed-prior mixture over eta schedules, so the outer fixed convex mixture remains a test supermartingale. The implementation class `MarkovTransitionBank` holds one Markov model per hazard and averages their wealth with a uniform ex-ante prior. Its work is \(O(HQ^2)\) per round and \(O(HQ)\) state for \(H\) hazards and \(Q\) eta states.

A 5,000-stream smoke benchmark (same five hypotheses, 60 rounds, Gaussian factor model) was run for the hazard bank. Switching-signal crossing rates, in percent, were:

| Arrival regime | \(\rho\) | Static | Fixed eta mixture | Markov \(q=0.05\) | Hazard bank |
|---|---:|---:|---:|---:|---:|
| All active | 0.2 | 25.58 | 25.70 | 25.84 | 25.70 |
| All active | 0.5 | 23.98 | 25.36 | 25.52 | 25.44 |
| All active | 0.8 | 20.86 | 24.24 | 24.22 | 23.96 |
| Staggered | 0.2 | 29.48 | 31.18 | 31.76 | 31.90 |
| Staggered | 0.5 | 28.58 | 30.70 | 31.16 | 30.90 |
| Staggered | 0.8 | 29.20 | 31.38 | 31.84 | 31.78 |

This small run is consistent with the hypothesis that adaptation helps on switching signals, but it does not show that a hazard bank uniformly improves on a single hazard: it trails the single \(q=0.05\) model in several cells. Do not infer a SOTA advantage. A higher-stream confirmatory benchmark did not complete within the local execution budget, so only the fixed-hazard 20,000-stream results above are the larger run.

## 5. Benchmark programme

### Baselines, with separate guarantee lanes

1. Static globally weighted projective merger (\(\eta=1\)).
2. Fixed \(\eta\) variants (0.5, 1, 2, 4) and the fixed-prior expert mixture.
3. Existing scalar-history and covariance-preconditioned vector intersection betting.
4. Vovk–Wang (2024) martingale merging functions, including the empirically adaptive sequential scalar merger, when within-round increments remain conditionally valid after earlier same-round observations are revealed.
5. Xu–Fischer–Ramdas (2026) donation e-LOND and online e-LOND in their one-e-value-per-arrival regime; compare only when information timing and assumptions match.
6. LORD, SAFFRON, ADDIS, LOND/e-LOND, alpha spending, and weighted GAI variants in their appropriate one-shot online-p-value/e-value regimes.
7. Valid adjuster-based SupFDR baselines; never compare unadjusted running maxima as if they were valid.

Do not mix stopped-FDR, SupFDR, per-intersection crossing, and single-coordinate type-I error into one metric.

### Workload matrix

- Dimensions: 2, 5, 10, 25, 50; horizons: 100, 1,000, 10,000.
- Alternatives: sparse, dense, heterogeneous, mixed signs, switching support, delayed effects, and abrupt drift.
- Arrivals: immediate, deterministic staggered, predictable history-dependent arrivals, and delayed outcome streams.
- Conditional increments: Gaussian, bounded Rademacher-factor, and heavy-tailed constructions with an explicit valid conditional supermartingale envelope.
- Dependence: independent, static equicorrelation, changing correlation, shared-control, factor, and adversarial cross-stream dependence that preserves each coordinate's conditional null guarantee.
- At least 1,000,000 null streams for confirmatory calibration, with fixed seeds, exact scenario manifests, and 95% binomial confidence intervals.

### Metrics

Report intersection crossing probability; stopped FDR; SupFDR; false-discovery exceedance; average true discoveries; power; time to crossing; worst-regime regret; memory; per-intersection update latency; closure-search latency; numerical stability; and performance under delayed/missing observations.

### Acceptance gates

- Written proof under the stated common-global-filtration and predictable-arrival assumptions.
- Exact \(\eta=1\) parity with the static merger.
- Exact future-horizon padding before activation and activation-at-price-one invariance.
- Independent verifier and negative controls for same-round lookahead and local-filtration leakage.
- Confirmatory error bounds below nominal levels with 95% confidence bounds, and adversarial/misspecification diagnostics.
- End-to-end FDR/SupFDR results before claiming a multiple-testing improvement.

## 6. Current implementation status

The reference implementation is a two-phase Python validation helper, not production code. It exposes `prepare(active_ids)` before `observe(current_values)`, computes allocations in log space, records deterministic receipts, and provides an independent linear replay verifier. Seventeen local unit tests pass for static parity, projective padding, activation neutrality, the two-phase interface, invalid state transitions, fixed-mixture regret, Markov-switching identity and schedule-regret bounds, Markov projectivity, a five-hazard transition bank and its zero-hazard parity, receipt replay and tamper detection, zero absorption, and negative controls for same-round lookahead and local-filtration leakage.

No repository CI or remote workflow result is claimed. The next engineering step is an independently tested Rust implementation and benchmark harness, followed by integration with the existing dynamic closure and closure-search work. The mathematical construction still needs a formal review, particularly the filtration/activation contract and the connection from the coherent intersection processes to the exact stopped-FDR theorem assumptions.

## 7. Literature

- R. de Heide (2026), [Dynamic e-closure for online hypotheses with any-time-valid evidence: closure principles and projective mergers](https://arxiv.org/abs/2608.09927).
- V. Vovk & R. Wang (2024), [Merging sequential e-values via martingales](https://sas.uwaterloo.ca/~wang/papers/2024Vovk-Wang-EJS.pdf).
- Z. Xu, L. Fischer & A. Ramdas (2026), [Improving online FDR procedures via online analogs of e-closure and compound e-values](https://proceedings.mlr.press/v337/xu26a.html).
- Y. Tavyrikov, J. J. Goeman & R. de Heide (2026), [Carefree multiple testing with e-processes](https://doi.org/10.1214/26-EJS2546).
- Y. J. Choe & A. Ramdas (2026), [Combining evidence across filtrations](https://doi.org/10.1093/jrsssb/qkag058).