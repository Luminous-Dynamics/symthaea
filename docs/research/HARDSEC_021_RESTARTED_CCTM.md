# HARDSEC-021 — Restarted Fixed-Reference DKW-CCTM

**Status:** Rust implementation and deterministic fixtures authored; no local compile/test pass and no physical PMU benchmark claimed  
**Issue:** [#9593](https://github.com/Luminous-Dynamics/symthaea/issues/9593)  
**Stacked on:** [HARDSEC-020 context assignment PR #9484](https://github.com/Luminous-Dynamics/symthaea/pull/9484) → [HARDSEC-019 fixed-reference CCTM PR #9478](https://github.com/Luminous-Dynamics/symthaea/pull/9478) → [HARDSEC-018 PMU metrology PR #9470](https://github.com/Luminous-Dynamics/symthaea/pull/9470)

## 1. Research hypothesis

HARDSEC-019 and the late-change restart literature solve different weaknesses:

- **Fixed-reference DKW-CCTM:** accounts for finite calibration-reference CDF error without adding test observations to the reference set.
- **Weighted restarts:** prevent evidence accumulated before a late change point from dominating the sequential detector's current state.

Bhattacharyya and Ramdas (2026) show that conventional single-start conformal martingale constructions can have suboptimal late-change delay under their independent-observation model. Their restarted weighted conformal processes obtain minimax-optimal delay rates for their stated PFA/ARL objectives, with near-harmonic weights. See the [paper](https://arxiv.org/abs/2609.27179).

HARDSEC-021 composes the two ideas: start a fresh DKW-corrected CCTM at each accepted observation within each declared context, and combine the local processes using a deterministic summable weight schedule. This is a **derived composition**, not a reproduction of the paper's restarted CMM or the authors' experiments. Its validity is conditional on every local DKW-CCTM process satisfying the same within-context null/reference assumptions.

## 2. Algorithm

### 2.1 Local DKW-CCTM

For context (c), reuse the same immutable reference score sample (R_c) and its DKW confidence width:

[
hat F_{0,c}(x)=\frac{1}{n_c}\sum_{i=1}^{n_c}\mathbf 1\{R_{c,i}\leq x\},
\qquad
\epsilon_c=\sqrt{\frac{\ln(2/\delta_c)}{2n_c}},
\qquad
C_c=\frac{1}{0.5+\sqrt{1+k^2}\epsilon_c}.
]

Each local start (j) has its own predictable ONS state initialized at (eta_{j,j-1}=0), (a_{j,j-1}=1). On the current context score, it computes the paper's DKW-corrected betting factor:

[
b_{j,t}=1+C_c\left[
  \eta_{j,t}(\hat F_{0,c}(X_t)-0.5)
  -\sqrt{\eta_{j,t}^2+k^2}\,\epsilon_c
\right].
]

Its local wealth is the product of those factors since start (j). The ONS update is made **after** the current bet:

[
g'_{j,t}=C_c\left[
 \hat F_{0,c}(X_t)-0.5
 -\frac{\eta_{j,t}}{\sqrt{\eta_{j,t}^2+k^2}}\epsilon_c
\right],
\quad
z_{j,t}=g'_{j,t}/b_{j,t},
\quad
a_{j,t}=a_{j,t-1}+z_{j,t}^2,
]
[
\eta_{j,t+1}=\operatorname{clip}
\left(\eta_{j,t}+\frac{4z_{j,t}}{a_{j,t}},-0.5,0.5\right).
]

Each local start is independent in its **adaptive state**, but shares the same fixed calibration reference and DKW event. We do not spend a new DKW delta for every restart.

### 2.2 Deterministic near-harmonic capital allocation

For context-local start index (j=1,2,\ldots), HARDSEC-021 uses:

[
w_j=\frac{1}{3j\,[\ln(e+j)]^2}.
]

The sum is rigorously bounded below one. Let (f(x)=1/[x\ln(e+x)^2]), which is decreasing. Since (ln(e+x)\geq\ln x) for (x\geq2),

[
\sum_{j=1}^{\infty} f(j)
\leq f(1)+f(2)+\int_2^{\infty}\frac{dx}{x(\ln x)^2}
=f(1)+f(2)+\frac{1}{\ln2}.
]

Therefore:

[
\sum_{j=1}^{\infty}w_j
\leq\frac13\left[
\frac{1}{\ln(e+1)^2}
+\frac{1}{2\ln(e+2)^2}
+\frac{1}{\ln2}
\right]
\approx0.743417<0.744<1.
]

The unallocated capital is retained as a constant reserve. For the set of local processes launched by time (t):

[
E_{c,t}=
1-\sum_{j\leq t}w_j+
\sum_{j\in\mathcal L_{c,t}}w_jM_{c,j,t},
]

where (M_{c,j,t}) is local CCTM wealth and (mathcal L_{c,t}) contains the local starts that remain live. Under the local nonnegative-supermartingale condition, deterministic nonnegative weighting and constant reserve preserve a nonnegative supermartingale with initial capital one. The context alarms when (E_{c,t}\geq1/\alpha_c).

Each context has its own local start clock (only accepted observations for that context advance it), and a separately allocated threshold (alpha_c=\alpha/K). The DKW budget remains (delta_c=\delta/K). The overall bound is conditional on the same fixed-reference DKW and null assumptions as HARDSEC-019.

### 2.3 Bounded lifetime and predictable expiry

To cap memory and update cost, each local process has a preregistered maximum lifetime (L=256) accepted observations in its context. It receives its (L)-th update; at the next context observation its nonnegative contribution is deterministically set to zero. This is a predictable downward kill, not an online reference reset and not a return of spent restart weight to the reserve. Expired weights remain spent.

This bounds the number of simultaneously stored local states by 256 per context and the per-observation ONS updates by 256 per accepted context observation. It can reduce power for effects that require more than 256 observations to accumulate sufficient evidence. The implementation makes this trade-off explicit.

A local process that reaches exactly zero betting wealth is also removed; its assigned weight remains spent.

### 2.4 Mandatory independent workload context

The restarted monitor requires the HARDSEC-020 truth-free runtime context receipt. It verifies the exact stream-window identity, context assignment policy and artifact hashes before updating any local process. Invalid PMU input or an abstained context assignment does not create a local process and does not advance that context's restart clock. However, dropping rows creates a selected-observation subsequence: the guarantee additionally requires the admission/abstention mechanism to be predictable and appropriately exogenous to the current null score, or to have a separately proved conditional validity result. This admission-selection assumption is marked unverified.

The context assignment process remains unsigned and caller-declared; this integration does not authenticate its model or prove exogeneity.

## 3. Implementation

- Rust executable: `examples/hardsec_context_restarted_cctm_monitor.rs`
- Harness feature: `hardsec-context-restarted-cctm-monitor`
- Plan/stream schemas:
  - `schemas/hardsec-context-restarted-cctm-plan-v1.schema.json`
  - `schemas/hardsec-context-restarted-cctm-stream-v1.schema.json`
- Fixtures: `tests/fixtures/hardsec/context-restarted-v1/`
- The source retains the single-start DKW-CCTM as a diagnostic comparator; only the restarted aggregate is used for the primary alarm result.
- The receipt records, per accepted observation, every candidate start's index, weight, age, local log wealth and status (active, zero-wealth-killed, or expiring after the current update); it also emits live-state counts before/after expiry, expired counts, the reserve and aggregate log e-value.
- 27 authored Rust tests cover the DKW and alpha/delta contract, exact stream/context receipts, fail-closed abstention, unit capital bound, deterministic lifetime expiry, per-start receipt fields, and a deterministic late-change stress fixture.

### Fixture known answers

The 26-window synthetic stream has 23 accepted rows and 3 abstentions across compute/memory contexts. Its current deterministic score schedule is designed so:
- the single-start CCTM crosses at global stream sequence 16;
- the restarted aggregate crosses at global sequence 22;
- the compute-context local restart clock, rather than the global sequence clock, determines its weights.

This is an algorithm-mechanics fixture only. It does not establish detection performance, a measured false-alarm rate or superiority.

### Deterministic late-change stress test

The Rust test feeds a fixed xorshift-generated sequence of 10,000 synthetic p-values through both methods, then applies a strong post-change sequence of (p=1). The intended known answer is 50 post-change observations for the restarted aggregate versus 200 for single-start CCTM. The input is deterministic and its null validity is not a theorem; the result is a reproducible unit-test stress case, **not** a Monte Carlo experiment, PMU campaign, confidence interval or SOTA result. The test is not locally executed yet.

## 4. Benchmark programme

The benchmark plan contains 13 comparator arms, 6 candidate architectures, 26 stress scenarios and 6 holdout axes.

### Comparators

- Static threshold, EWMA, CUSUM/Page-Hinkley.
- Fixed-window conformal and growing-reference conformal martingales.
- Fixed-reference DKW-CCTM single-start.
- Fixed-reference DKW-CCTM restarted with near-harmonic weights.
- A telescoping-weight restart ablation (w_j=1/[j(j+1)]).
- The published Bhattacharyya–Ramdas restart construction on its applicable independent-data protocol.
- Authors' source-pinned Python CCTM and paper-equation Rust CCTM.
- Symthaea HDC and liquid temporal score front ends, with the same sequential decision layer.

### Experimental axes

- Early, middle and very late changes ((T=10^3,10^4,10^5,10^6)).
- Multiple effect sizes, duration and attacker adaptation strategies.
- Stationary iid null plus AR/block/long-memory and repeated-latent-score negative controls.
- Benign workload phase transitions, contention and context misassignment.
- Missing/stale/multiplexed sensors, observer effects and PMU acquisition modes.
- Calibration support (n_c), DKW precision, context count, restart lifetime and state-memory limit.
- Leave-one-execution/program/binary/attack-family/adaptation-family/device-configuration/cross-platform holdouts.

### Metrics

Report PFA and ARL as distinct objectives, never as interchangeable thresholds. Also report false alarms per device-hour, any false alarm by fixed horizon, median/p95 delay stratified by change time/effect/context/attack family, attack-effect-conditioned recall, worst-family recall, accepted context coverage, abstention, reference support, CPU cycles, p95/p99 latency, resident memory, energy per device-hour and total acquisition-plus-inference-plus-decision overhead.

Use matched alert operating points and equal information budgets. Do not tune thresholds on held-out data or pool windows across dependent program/execution groups to inflate effective sample size. Report both the count of windows and the count of independently justified calibration/evaluation units.

## 5. Industry comparison and aspirational targets

A September 2026 real-RISC-V HPC study on a single XuanTie C910 reports 99% precision for benign and malicious samples, about 95% malicious classification performance, 2% collection overhead for `perf` sampling at 10 ms cadence, and 98.56% accuracy after instruction-retired normalization in its multi-process setup. It also reports some counters changing by more than 300% under simultaneous `perf` collection and therefore collected counters in isolation. These are useful baselines, but the paper uses 16 benign and 16 malicious programs on one microarchitecture and itself notes limits in cross-platform transfer, temporal modeling and normalization. See [Pou et al., Cybersecurity (2026)](https://doi.org/10.1186/s42400-026-00662-8).

A 2024 SoK warned that published HPC cache-attack detection results often understate accuracy/overhead/detection-speed difficulties and use weak threat models; it also demonstrated a benign-program-modification attack that can evade counter-only monitoring. See [Kosasih et al., ACM AsiaCCS 2024](https://doi.org/10.1145/3634737.3637649).

The targets below are **research aspirations, not achieved results or certification thresholds**:

- ≥20% lower p95 late-change detection delay than the strongest independently reproduced baseline at a matched PFA budget.
- Worst attack-family/context recall gap ≤3 percentage points.
- ≥90% pooled attack-effect recall at the preregistered operating point.
- ≤0.1 benign false alerts per physical device-hour, with exposure hours reported.
- Zero silent-clear outcomes on invalid PMU telemetry or abstained/unseen contexts.
- ≤1% incremental sequential-decision CPU/latency overhead; and measure total PMU acquisition plus scoring plus decision overhead against the 2% published comparator.
- No unbounded state growth: demonstrate the configured maximum restart state count and report its resource cost.

The first competitive step is not to claim higher accuracy on the same dataset. It is to beat the strongest reproduced comparator on **late-shift delay at a matched false-alarm budget**, while holding out programs, executions and devices and accounting for measurement interference and context-assignment coverage.

## 6. Statistical claim ceiling

The restarted sum inherits a PFA-style supermartingale argument only if every local DKW-CCTM is a valid nonnegative supermartingale in its context. This requires:
- iid calibration reference scores within each context;
- iid null test scores within each context, or a separately proved conditional super-uniformity condition;
- a fixed scoring function;
- exogenous context assignment and an appropriate accepted-observation filtration;
- correct reference/evaluation separation.

These assumptions are **not established** by the JSON receipt, caller-declared group IDs, or a digest. In particular, the receipt also requires `accepted_observation_selection_assumption_status=assumption_not_verified`; a sensor-validity or context-confidence gate can itself select a non-iid subset if correlated with the score or attack state. The paper's independent-observation restart result does not imply validity for serially dependent or adversarial PMU telemetry. The dependent-null stress test exists precisely because iid claims cannot be inferred from marginal distributions.

The receipt therefore reports:
- null i.i.d. status: `assumption_not_verified`;
- reference i.i.d. status: `assumption_not_verified`;
- context exogeneity: `not_established_from_input_receipts`;
- accepted-observation/admission exogeneity: `assumption_not_verified`;
- theorem guarantee: `not_established_from_input_receipts`;
- effective sample size: `null`.

PFA and ARL remain separate benchmark objectives. The restarted result is a candidate architecture, not proof of distribution-free hardware attack detection.
