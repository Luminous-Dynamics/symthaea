# HARDSEC-019 — Workload-Conditioned Fixed-Reference Sequential Monitor

**Status:** Rust sequential-monitor candidate, versioned plan/stream schemas and synthetic known-answer fixture authored; no real detector campaign or anytime-valid guarantee claimed  
**Issue:** [#9478](https://github.com/Luminous-Dynamics/symthaea/issues/9478)  
**Stacked on:** [HARDSEC-018 PMU metrology](https://github.com/Luminous-Dynamics/symthaea/pull/9470)

## 1. Research direction

A static anomaly threshold confuses workload changes with attacks. A growing-reference detector can absorb post-change samples into its own null and dilute evidence. The target architecture is a *context-conditioned, fixed-null-reference sequential monitor*:

- identify the current workload regime (compute, memory, I/O, mixed, etc.);
- compare the new score only with the frozen reference for that context;
- accumulate evidence sequentially without resetting after an alarm;
- abstain/escalate on unseen contexts or unqualified telemetry;
- preserve the exact score artifact, reference and stream digests.

This is an evidence-accumulation layer around a detector score. It does not yet implement Symthaea HDC/CfC/LTC feature extraction; `scoring_artifact_sha256` pins the external score-producing artifact so the reference and stream cannot silently use different score generators.

## 2. Current research comparator

Shaer, Bar, Prinster and Romano, *Testing For Distribution Shifts with Conditional Conformal Test Martingales*, ICML 2026 / PMLR 306, identify contamination in growing-reference conformal test martingales: post-shift observations enter the reference set, dilute shift evidence and increase detection delay. Their paper proposes a fixed-reference conditional construction, accounting for finite-reference estimation error, with anytime-valid type-I error control, asymptotic power one and bounded expected delay under its assumptions. [Paper and abstract](https://proceedings.mlr.press/v306/shaer26a.html).

This implementation is **not a reproduction** of that paper's robust conditional construction. It implements a simpler, transparent fixed-context empirical-rank candidate, then explicitly marks conditional validity as unverified. It should be compared to the paper's method only after implementing its actual construction and reproducing its protocol.

A complementary ICLR 2026 line, *Adaptive Conformal Anomaly Detection with Time Series Foundation Models for Signal Monitoring*, studies weighted-quantile conformal scores under distribution shift and resource constraints. This is another comparator for future work, not something claimed by this code. [ICLR 2026 paper record](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f54c9fc57aa6e1c72400cc127917fcf8-Abstract-Conference.html).

## 3. Algorithm

### 3.1 Fixed context references

Each predeclared context (c) owns an immutable calibration score array (R_c). Higher scores mean more anomalous. For a new valid score (s_t) in context (c_t), the monitor computes the conservative upper-tail empirical rank:

[
p_t = \frac{1 + \#\{r \in R_{c_t}: r \geq s_t\}}{|R_{c_t}|+1}.
]

Ties are counted in the upper tail, making this rank conservative under exchangeability. The fixed array is never updated using evaluation observations. Binary search keeps lookup at (O(\log |R_c|)) after one-time sorting.

### 3.2 Mixture betting function

For a predeclared finite mixture of ((\lambda_j,w_j)), where (0<\lambda_j<1), (w_j>0) and (\sum_j w_j=1), the per-observation betting function is:

[
e(p)=\sum_j w_j\lambda_j p^{\lambda_j-1}.
]

For uniform (p), each component has expectation one. The mixture is decreasing in (p), so it has expectation at most one for super-uniform p-values. The implementation computes (\log e(p)) with log-sum-exp and accumulates log wealth:

[
L_t=\sum_{i=1}^{t}\log e(p_i), \qquad
\text{cross when } L_t\geq\log(1/\alpha).
]

Under the *additional* condition that every (p_t) is super-uniform conditional on the complete past and the declared context, the product is a non-negative supermartingale/e-process, and Ville's inequality bounds the probability of ever crossing (1/\alpha) by (\alpha).

### 3.3 The hard validity boundary

A context-specific empirical rank against a fixed exchangeable calibration set can be marginally valid. That alone does **not** establish conditional super-uniformity given the entire history of a temporally dependent PMU stream. Autocorrelation, workload phase persistence, context selection and attacker-induced dependence can violate the condition required by the sequential argument.

Therefore v1 requires `conditional_superuniformity_status=assumption_not_verified`, emits `sequential_false_alarm_guarantee_status=not_established_from_inputs`, and never labels its result a formally guaranteed anytime-valid test. A true guarantee requires a justified conditional construction and assumptions appropriate for the actual PMU sequence.

## 4. Fail-closed state machine

Each window has one of these outcomes:

- `monitoring_no_alarm_yet`: qualified telemetry, known context, evidence updated, boundary not crossed.
- `alarm_crossing`: first threshold crossing; first crossing index recorded.
- `alert_latched`: subsequent valid observations after an alarm; the alarm does not reset.
- `abstain`: invalid/missing/stale/misaligned/multiplexed/untrusted sensor, unqualified PMU measurement, or unseen workload context.
- `alert_latched_observation_abstained`: an alarm is already latched but the current observation cannot update evidence.

An abstention does not update p-values or wealth and is never converted to a benign/clear verdict. No test-time reference update or post-alarm reset is permitted.

## 5. Executable contract

Inputs:

- Plan schema: `schemas/hardsec-context-sequential-plan-v1.schema.json`
- Stream schema: `schemas/hardsec-context-sequential-stream-v1.schema.json`
- Plan fixture: `tests/fixtures/hardsec/context-sequential-v1/plan.json`
- Synthetic stream: `tests/fixtures/hardsec/context-sequential-v1/stream.json`
- Rust executable: `examples/hardsec_context_sequential_monitor.rs`

The plan and stream bind:
- stream and evaluation-group identity;
- immutable per-context references and declared calibration-group IDs;
- the exact external score artifact SHA-256 and score direction;
- predeclared alpha and mixture-betting parameters;
- the required successful HARDSEC-018 PMU qualification result.

The executable rejects reference/evaluation group overlap, cross-context reference-group reuse, duplicate context IDs, insufficient reference score rows, inconsistent score-artifact digests, duplicate windows and non-contiguous sequence indices. It uses exact input digests and an unsigned create-new/no-overwrite receipt.

The fixture has two workload contexts and seven observations. The synthetic monitor crosses its nominal log-wealth boundary at sequence index 4; the last three records exercise stale telemetry, an unseen context and an untrusted sensor. This validates algorithm mechanics only. The fixture is not a detector-performance result or a false-alarm-rate experiment.

Fourteen authored Rust tests cover the known answer and failure modes, including reference-group leakage, duplicate windows, sequence order, mixture weights, insufficient calibration, conservative ties, score artifact binding, invalid/unqualified telemetry and evidence-domain mismatch.

## 6. Run

```bash
cargo test --locked \
  --manifest-path tools/hardsec-adaptive-evasion-audit-harness/Cargo.toml \
  --features hardsec-context-sequential-monitor \
  --example hardsec_context_sequential_monitor

cargo run --locked \
  --manifest-path tools/hardsec-adaptive-evasion-audit-harness/Cargo.toml \
  --features hardsec-context-sequential-monitor \
  --example hardsec_context_sequential_monitor -- \
  --plan tests/fixtures/hardsec/context-sequential-v1/plan.json \
  --stream tests/fixtures/hardsec/context-sequential-v1/stream.json \
  --receipt /tmp/hardsec-019-monitor-receipt.json
```

## 7. Benchmark programme

Use the same score artifact, PMU qualification, outer held-out executions and alert budget for each comparator:

1. Static threshold.
2. EWMA and Page-Hinkley/CUSUM.
3. Fixed-window conformal anomaly scores.
4. Growing-reference conformal test martingales.
5. Fixed-reference conditional conformal test martingales (paper-faithful implementation/reproduction).
6. Context-agnostic fixed-reference betting process.
7. Context-conditioned fixed-reference process (this candidate).
8. Symthaea HDC and liquid temporal score front ends, each evaluated with the same sequential decision layer.

Separate scenarios: stationary null; workload phase shifts; persistent context transitions; autocorrelated/heavy-tailed null streams; benign contention; context mimicry; attack timing/interleaving adaptation; unseen regimes; sensor invalidation; and persistent attack effects. Hold out programs, executions, attack families, adaptation families and physical device configurations.

Primary metrics: false alarms per device-hour on measured benign exposure; probability of any false alarm over a fixed monitoring horizon; median/p95 detection delay; attack-effect-conditioned recall; worst-context and worst-family recall; abstention rate/reason; reference contamination; score calibration; CPU time, memory and end-to-end overhead. Do not compute mixed-class precision/F1 on single-class held-out folds.

### Aspirational targets (not achieved results or standards)

- At a matched operational false-alert budget, reduce p95 detection delay by at least 20% versus the strongest reproduced sequential baseline.
- Maintain worst-context/attack-family recall within 3 percentage points of the pooled result.
- Keep benign false alerts at or below 0.1 per physical device-hour using measured exposure time.
- Reach at least 90% pooled attack-effect recall at the predeclared operating point.
- Zero silent-clear outcomes for invalid sensors or unseen workload contexts.
- Keep incremental decision-layer p95 overhead below 2%, with the score model and PMU acquisition costs separately reported.
- Before claiming an anytime-valid false-alarm guarantee, demonstrate the conditional p-value assumption or implement/reproduce the conditional reference construction; empirical simulations alone do not prove it.

All targets require independent-group support, preregistered thresholds and exact source/run receipts. A sequential guarantee on one stream also does not automatically control error across multiple devices; allocate the family-wise alpha budget across streams before deployment.

## 8. Claim ceiling

The receipt validates fixed-reference monitor mechanics on supplied inputs. It does not prove conditional super-uniformity, temporal independence, false-alarm control, attack-effect detection, SOTA superiority, cross-platform transfer or hardware security. The algorithm is an experimental candidate awaiting benchmark and theorem-level qualification.

## 9. DKW-corrected CCTM implementation

The stronger v1 candidate is implemented separately in `examples/hardsec_context_cctm_monitor.rs` (harness feature `hardsec-context-cctm-monitor`). It follows the published paper's equations (6)–(8) and Algorithms 1–2 rather than treating a fixed-reference ECDF rank as automatically valid.

For context (c), it computes the fixed-reference ECDF value:

[
\hat p_t=\hat F_{0,c}(X_t)=\frac{1}{n_c}\sum_{i=1}^{n_c}\mathbf{1}\{X^0_{i,c}\leq X_t\}.
]

It constructs a DKW band with total reference-confidence budget (delta), allocated as (delta_c=delta/K) across (K) predeclared contexts:

[
\epsilon_c=\sqrt{\frac{\ln(2/\delta_c)}{2n_c}},\qquad
C_c=\frac{1}{0.5+\sqrt{1+k^2}\,\epsilon_c}.
]

The corrected bet is:

[
b_t=1+C_c\left(\eta_t(\hat p_t-0.5)-\sqrt{\eta_t^2+k^2}\,\epsilon_c\right).
]

The ONS update is predictable: use (eta_t) for the current bet, then calculate

[
g'_t=C_c\left(\hat p_t-0.5-\frac{\eta_t}{\sqrt{\eta_t^2+k^2}}\epsilon_c\right),\quad
z_t=\frac{g'_t}{b_t},\quad
a_t=a_{t-1}+z_t^2,
]
[
\eta_{t+1}=\operatorname{clip}\left(\eta_t+\frac{4z_t}{a_t},-0.5,0.5\right).
]

Each context has a separate wealth process with (alpha_c=alpha/K); the DKW failure budget is also divided by context. Under the paper's within-context i.i.d. null/reference assumptions and externally assigned contexts, a union bound composes the finite context family. The output reports the theorem's conditional structure and the corresponding (alpha+delta) unconditional upper-bound implication, but keeps every assumption unverified.

**Scope warning:** the paper's null is an i.i.d. score stream (X_t\sim P_c) within each context, with i.i.d. calibration scores from the same (P_c). It is not a theorem for arbitrary autocorrelated telemetry. PMU windows from a single execution may violate these assumptions badly. The 600 synthetic reference rows are assigned unique synthetic group IDs to test the contract; this does not establish independence in a physical dataset.

### Paper/official-code parity

The paper's HTML v2 Algorithm 2 specifies the ONS step (4z_t/a_t) and equation (8) uses (C=1/(0.5+\sqrt{1+k^2}\epsilon)). The authors' public Python `optimization.py` currently uses a different step coefficient (2/(2-\log 3)), and `utils.py` uses (1/(0.5+(1+k)\epsilon)). This Rust executable follows the published equation/algorithm, not byte-for-byte parity with that Python implementation. A parity/reproduction comparison is required before describing it as a faithful port of the authors' software package. See the [paper](https://arxiv.org/html/2602.13848) and [official implementation](https://github.com/shaersh/cctm).

CCTM fixture:
- Plan: `schemas/hardsec-context-cctm-plan-v1.schema.json`
- Stream: `schemas/hardsec-context-cctm-stream-v1.schema.json`
- Fixtures: `tests/fixtures/hardsec/context-cctm-v1/plan.json` and `stream.json`
- 600 declared reference rows per context; two contexts; 18 total stream records.
- Synthetic sequence crosses the compute-context alpha boundary at stream index 15; stale telemetry, unseen context and untrusted sensor records abstain.
- Sixteen authored Rust tests cover DKW width, (alpha/delta) allocation, ONS bounds, reference-group separation, invalid inputs and the known answer.

Run:

```bash
cargo test --locked \
  --manifest-path tools/hardsec-adaptive-evasion-audit-harness/Cargo.toml \
  --features hardsec-context-cctm-monitor \
  --example hardsec_context_cctm_monitor

cargo run --locked \
  --manifest-path tools/hardsec-adaptive-evasion-audit-harness/Cargo.toml \
  --features hardsec-context-cctm-monitor \
  --example hardsec_context_cctm_monitor -- \
  --plan tests/fixtures/hardsec/context-cctm-v1/plan.json \
  --stream tests/fixtures/hardsec/context-cctm-v1/stream.json \
  --receipt /tmp/hardsec-019-cctm-receipt.json
```

### Dependence negative control

A dedicated Rust test constructs 10,000 equally weighted latent scores (U\in(0,1)), maps each to a null score (X=6U) under the fixture's uniform-score reference grid, and repeats each latent score for 100 consecutive observations. Each time position has the correct marginal null score distribution, but all observations within a stream are perfectly dependent. The fixed reference grid's ECDF is within the declared DKW band for the uniform CDF, isolating the failure of the test-stream i.i.d. assumption in this constructed diagnostic.

A private numerical cross-check of this exact deterministic grid gives 7,983 threshold crossings out of 10,000 latent paths (79.83%) at nominal (alpha=5%). The Rust test asserts a broad 75–85% range to avoid a fragile exact floating-point count. This is a deterministic grid stress test, not a Monte Carlo confidence interval, measured PMU false-alarm probability, or claim about real hardware. Its purpose is to prevent this implementation from being represented as valid for arbitrary dependent telemetry.
