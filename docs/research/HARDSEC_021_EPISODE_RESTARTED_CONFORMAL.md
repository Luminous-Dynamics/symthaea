# HARDSEC-021 — Episode-Level Evidence and Weighted-Restarted Conformal Detection

**Status:** Rust fixed-episode packer, truth-free HARDSEC-020 context-map integration, weighted-restarted conformal monitor and synthetic late-change fixture authored. No physical PMU detector result or execution-independence proof is claimed.  
**Issue:** [#9551](https://github.com/Luminous-Dynamics/symthaea/issues/9551)  
**Stacked on:** [HARDSEC-020 / PR #9484](https://github.com/Luminous-Dynamics/symthaea/pull/9484), which integrates the independent context assignment audit into HARDSEC-019.

## 1. Paradigm shift: change the statistical unit

The primary problem is not simply that the detector score is noisy. A single execution emits many adjacent PMU windows, and those windows can be strongly correlated. Treating 10,000 windows from one execution as 10,000 independent observations is pseudoreplication.

HARDSEC-021 changes the unit of evidence:

- one fixed-length episode;
- one source execution group;
- one predeclared aggregation function;
- one context assignment covering every constituent window;
- one randomized conformal observation;
- one update to the sequential detector.

The current v1 aggregation is the maximum anomaly score across exactly four contiguous windows. If any window is invalid, stale, unqualified or assigned to an unknown/mixed context, the entire episode abstains. We do not shrink the episode or calculate a maximum over only the surviving windows.

This sacrifices within-execution statistical granularity. Detection delay is measured in independent execution episodes; time spent collecting an episode is reported separately. It is the conservative alternative to inventing independence between adjacent PMU windows.

**Important:** unique execution IDs and separate source trace hashes make the sampling unit auditable, but do not prove that successive executions are independent or exchangeable.

## 2. Research basis: locally restarted conformal mixtures

Bhattacharyya and Ramdas (23 September 2026), [*Change detection with conformal martingales: new optimal constructions, and suboptimality of existing methods*](https://arxiv.org/abs/2609.27179), identify a second weakness in conventional conformal test martingales. A single martingale started at the beginning of a long run can lose detection efficiency when a change occurs late. The paper establishes lower bounds of order Omega(T) for standard PFA-valid conformal test martingales and Omega(sqrt(ARL)) for the studied ARL procedures in their late-changepoint regimes. Their solution starts local martingales at every possible changepoint and combines them with deterministic restart weights.

Their Theorem 2.7 gives a unified construction:
- for PFA control, deterministic weights with sum at most one and threshold 1/alpha;
- for ARL control, bounded weights and threshold gamma, with a separate early-false-alarm inequality.

The paper's Corollary 3.6 and Remark 3.7 show that suitable polynomial or near-harmonic restart weights yield detection delay of order log(T) under its independent-observation, score-separation and betting-family regularity assumptions. It does not provide that theorem for autocorrelated PMU windows.

This Rust implementation uses the near-harmonic schedule
[
w_k=\frac{1}{2k(1+\log k)^2}.
]
For (f(x)=1/[x(1+\log x)^2]), (f) is decreasing and
[
\sum_{k=1}^{\infty}f(k)
\le f(1)+\int_1^{\infty}f(x)\,dx
=1+1=2.
]
Therefore (\sum_k w_k\le1), satisfying the paper's summable-weight condition. Since this schedule is asymptotically proportional to (1/[k(\log k)^2]), its effective boundary has leading order log(T), with a lower-order log-log term. The exact theorem remains conditional on independent/exchangeable episode observations and the required score/betting regularity.

## 3. Randomized online conformal p-values

For each context, the monitor receives one episode score (s_t), after the fixed aggregation. It reverses score direction so high anomaly scores occupy the lower tail:
[
x_t=-s_t.
]
Given previous transformed episode scores and an independently generated (U_t\sim\mathrm{Unif}(0,1)), it computes
[
p_t=\frac{
\#\{i<t:x_i<x_t\}
+U_t\left(1+\#\{i<t:x_i=x_t\}\right)
}{t}.
]
The current observation is included in its own tie block. The randomizer is needed to handle discrete/tied episode scores. Under exchangeability and independent uniform randomizers, the paper's online conformal construction gives i.i.d. uniform p-values under the no-change law.

The randomizer is supplied as an input value and bound to a randomizer-source digest. The tool checks that each value lies strictly in (0,1), but cannot prove that the randomizer is independent of the score or past. The receipt marks this as unverified.

## 4. Weighted-restarted mixture

The finite betting family is preregistered:
[
f_{\theta}(p)=\theta p^{\theta-1},
\qquad
\theta\in\{0.15,0.30,0.50,0.75\},
]
with equal prior weights. Each (f_\theta) integrates to one on (0,1).

For each candidate restart (k), the local mixture is
[
M_{k,t}=\sum_j\pi_j\prod_{r=k}^{t}f_{\theta_j}(p_r).
]
The primary alarm statistic is
[
S_t=\sum_{k=1}^{t}w_kM_{k,t}.
]
For each context, alpha is Bonferroni-allocated as (\alpha_c=\alpha_{total}/K), and the alarm boundary is (1/\alpha_c). The paper's augmented process (S_t+\sum_{k>t}w_k) is a nonnegative martingale with initial value (\sum_k w_k\le1); Ville's inequality therefore gives a per-context PFA bound (\alpha_c). The union bound composes the finite context family.

### Constant-memory recurrence

A naive implementation stores all restart start-times and costs O(T²) memory/work across T episodes. For each betting parameter j, define
[
A_{j,t}=\sum_{k=1}^{t}w_k\prod_{r=k}^{t}f_{\theta_j}(p_r).
]
Then
[
A_{j,t}=f_{\theta_j}(p_t)\left(A_{j,t-1}+w_t\right),
\qquad
S_t=\sum_j\pi_jA_{j,t}.
]
This exact recurrence updates in O(J) work and O(J) memory per context, where J is the number of betting parameters. The executable implements it in log space with log-add-exp to avoid numerical overflow/underflow.

The ordinary fixed-start conformal mixture (M_{1,t}) is also emitted as a diagnostic comparator. Only the weighted-restarted sum is the primary alarm statistic in v1.

## 5. Fail-closed episode packer and HARDSEC-020 integration

Inputs:
- Plan: `schemas/hardsec-episode-restarted-plan-v1.schema.json`
- Episode/window stream: `schemas/hardsec-episode-window-stream-v1.schema.json`
- Benchmark plan: `schemas/hardsec-episode-restarted-benchmark-v1.schema.json`
- Rust executable: `examples/hardsec_episode_restarted_monitor.rs`
- Fixtures: `tests/fixtures/hardsec/episode-restarted-v1/`

The plan pins score, context-assignment and randomizer-source digests separately. It fixes:
- exactly four windows per episode;
- contiguous window indices and source offsets;
- equal window widths inside an episode;
- one unique execution-group ID per episode;
- no duplicate episode IDs, source traces or window IDs;
- strict disjointness of model training, selection and evaluation execution groups;
- maximum-score aggregation and whole-episode abstention on any invalid window;
- fixed context vocabulary and betting family.

The executable consumes the truth-free HARDSEC-020 runtime mapping and joins it by exact source-window sequence and ID. All windows in an accepted episode must map to the same declared context. An unknown, unaccepted or mixed assignment abstains for the complete episode. The truth oracle is not passed to the restarted monitor.

## 6. Synthetic known-answer results

The fixture contains:
- 132 distinct synthetic execution episodes;
- 528 source windows, four per episode;
- 72 compute-context episodes (60 null-like, then 12 increasing high-score episodes);
- 60 memory-context null-like episodes;
- 528 accepted window-context assignments;
- no invalid source windows.

For the deterministic synthetic stream, the compute-context restarted sum crosses at sequence 65, five post-change episodes after the shift begins. The conventional fixed-start CMM also crosses at 65 in this short fixture; the more demanding late-change test below exercises the advantage of local restarts. These are synthetic algorithm-mechanics results, not a detector result or calibrated false-alarm estimate.

A deterministic late-change Rust test uses 1,000 null-like independent episode scores followed by a sustained shift. Under the preregistered power-mixture and deterministic randomizer sequence, the restarted statistic crosses at episode 1,004; the fixed-start mixture crosses at episode 1,025. This is one deterministic stress path, not an expected-delay estimate, Monte Carlo result, or SOTA superiority claim. The benchmark plan requires many independent streams, seeds, changepoint positions and effect sizes before any performance claim.

## 7. Benchmark programme and industry comparison

Compare at matched PFA levels, and report ARL as a separate operating regime:
1. Static threshold.
2. EWMA and Page-Hinkley/CUSUM.
3. HARDSEC-019 DKW-corrected fixed-reference CCTM.
4. Fixed-start online conformal mixture martingale.
5. Weighted-restarted sum with near-harmonic weights (this implementation).
6. Weighted-restarted maximum with near-harmonic weights (next ablation).
7. Pinned official conformal implementation.
8. T-SMADE and other published HPC attack detectors.
9. Intel VTune/perf top-down PMU metrics as an industry profiling/control baseline.

Intel VTune's Top-down Microarchitecture Analysis Method uses PMU events such as cache misses and branch mispredictions to characterize frontend, backend, bad-speculation and retiring behavior. HARDSEC is not a replacement for VTune or a generic profiler. Its distinct target is an auditable security-evidence chain: qualified acquisition, execution-level evidence units, source/context receipts, sequential false-alarm policy and fail-closed handling.

All confirmatory comparisons must hold out executions, program binaries, device configurations, attack families and adaptation families. Include stationary nulls, workload phase shifts, late changepoints at T=10/100/1000, weak/strong attack effects, within-execution autocorrelation, dependent execution streams, shared-latent nulls, context mimicry, unknown regimes, invalid sensors, randomizer reuse and score-dependent randomizer leakage.

Primary measures:
- PFA over the full horizon, with exact confidence bounds over independent streams;
- false alarms per 1,000 independent executions;
- median/p95 delay in execution episodes;
- separate end-to-end latency including episode acquisition time;
- accepted/abstained episode coverage;
- worst-context and worst-attack-family recall;
- source-window overlap and execution reuse;
- CPU time, peak memory and aggregation overhead.

### Aspirational targets (not achieved results)

- PFA no greater than the preregistered 5% under an i.i.d. episode null.
- At matched PFA, at least 25% lower p95 delay than the fixed-start CMM on late changepoints.
- p95 delay at most 20 episodes on the preregistered moderate-effect profile.
- Zero sequential updates from invalid, unassigned, mixed-context or PMU-unqualified episodes.
- Zero duplicate or cross-episode source-window reuse.
- Zero acceptance of variable-length episodes.
- p95 incremental decision overhead below 2 ms per episode on the named reference hardware.

These thresholds are research goals only. A fixed PFA guarantee and a low measured false-alarm rate are different claims; report both.

## 8. Claim ceiling and unresolved assumptions

The paper's PFA theorem assumes independent/exchangeable episode observations under the null and independent uniform randomizers. The episode packer prevents adjacent windows from being counted as separate evidence, but it cannot prove that different executions are independent. A sequence of executions can remain dependent through a shared machine state, workload scheduler, thermal state, cache state, operator process, firmware state or a shared latent environmental factor.

The monitor also cannot prove that abstention is non-informative, that the score artifact truly remained fixed, that the context assignment was exogenous, or that the randomizer is independent. Its receipt therefore records:
- episode-score exchangeability: `assumption_not_verified`;
- cross-execution independence: `not_established`;
- randomizer independence: `assumption_not_verified`;
- context exogeneity: `not_established_from_input_receipts`;
- theorem guarantee: `not_established_from_input_receipts`;
- effective sample size: null.

If episode dependence remains material, use this detector only as an empirical/engineering monitor. Do not label the PFA as formally guaranteed. A future block/mixing approach needs a defensible dependence envelope (for example a validated beta-mixing bound) and its own theorem/claim audit; an estimated autocorrelation or effective sample size does not prove that bound.
