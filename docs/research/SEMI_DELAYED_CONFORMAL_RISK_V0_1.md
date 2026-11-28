# SEMI-VM-ACI-001 — delayed conformal calibration and capacity-aware feedback v0.1

**Issue:** #13479  
**Branch:** `feat/semi-delayed-conformal-risk-v1`  
**Status:** two Rust reference primitives with authored tests; exact-head validation pending.

## Why this is a separate assurance layer

A VM point estimate is not a manufacturing decision. A production-facing prediction also needs a declared interval, a target miscoverage/risk level, a record of the interval that was actually issued, and a policy for how later physical measurements may update calibration.

The new modules separate two mathematical problems:

1. **Delayed ACI (tau-DACI):** how to update the miscoverage level when the outcome for a forecast arrives a fixed number of logical rounds later.
2. **Capacity-aware OCRC (CA-OCRC):** how to retain only a bounded number of pending outcomes while correcting for randomized feedback tracking.

These algorithms do not have interchangeable assumptions. The first uses a fixed delay and phase-interleaved alpha recursions. The second uses bounded arbitrary delays, a randomized proxy-delay policy and inverse-probability weighting.

## 1. Fixed-delay ACI

Research anchor: El Halabi & Brandt, *Adaptive Conformal Inference Under Delayed Feedback: Coverage Guarantees and a Delay-to-Memory Diagnostic*, arXiv:2609.07251 (7 September 2026).

The paper's delayed update is:

`alpha_{t+tau} = alpha_t + gamma (alpha_target - err_t)`

where `err_t = 1` if the frozen interval misses the eventual outcome and 0 otherwise. The delayed recursion decomposes into `tau` interleaved ACI sequences. The implementation therefore stores one alpha state per delay phase; it does not update a single alpha with an ordinary arrival-order rule.

Implemented in `delayed_conformal_aci.rs`:
- explicit fixed-delay logical rounds and phase-specific alpha;
- immutable prediction interval receipts with alpha, lower/upper bounds, model digest, event identity and expected delivery round;
- K1-only label receipt checks, including full payload digest, report-attested revision, measurement-event identity, corrected value and availability ordering;
- byte-replayable prediction/update receipts;
- duplicate revision rejection;
- fail-closed suspension on UNKNOWN/BLOCKED, stale, late, out-of-order, mismatched or unavailable evidence;
- an optional policy requirement for the upstream source-authenticity assertion (not cryptographic verification);
- an AR(1) delay-to-memory diagnostic using either residual level or squared residuals.

The engine receives interval bounds from an upstream prediction-set estimator. It controls the alpha recursion but does not yet implement the paper's rolling conformity-quantile/PSE estimator. The next increment should implement the rolling residual buffer, smoothing/tie policy and quantile construction as a separately tested component.

**Claim boundary:** tau-DACI's fixed-delay recurrence is only applicable when outcomes arrive at the declared fixed horizon and the required outcomes are observed. In this fab evidence system, an unqualified/missing K1 outcome suspends the guarantee. We do not pretend that selectively missing physical labels are ordinary fixed-delay feedback.

## 2. Capacity-aware OCRC

Research anchor: Wang & Simeone, *Capacity-Constrained Delayed Feedback for Online Conformal Risk Control*, arXiv:2611.17059 (12 November 2026).

For a score `S_t(X_t,y) in [0,B]`, the paper constructs `C_t={y:S_t(X_t,y)<=r_t}` and assumes:
- loss is 0 on the full prediction set and 1 on the empty set;
- delays are arbitrary but bounded by known `d_max`;
- the initial threshold is in `[0,B]`;
- proxy tracking load satisfies `ell_F <= K`;
- `F(d_max)>0`.

The CA-OCRC policy samples a proxy delay from a distribution `F`, expires prior tracked rounds whose proxy lease ended, and admits the current round only when capacity permits. For a received outcome with actual delay `d_i`, the threshold update uses:

`r_{t+1} = r_t + eta * phi_t * (L_t-alpha) + eta * sum_{i in O_t} (L_i-alpha)/F(d_i)`

where `phi_t=1` for a trivial empty/full set and 0 otherwise. Under the paper's assumptions, the arbitrary-delay bound is:

[
RiskGap(T) leq \frac{B}{\eta T}
+ \frac{K+1}{F(d_{max})T}
+ \exp\left[-K h\left(\frac{\ell_F}{K}\right)\right],
\qquad h(x)=x-1-\ln x.
]

The optimized worst-case proxy family has mass `1-p` at proxy delay -1 and mass `p` at `d_max`, where `p` minimizes the paper's one-dimensional objective subject to `p <= K/d_max`. The reference implementation includes deterministic grid refinement for this scalar design and computes the published risk-gap bound.

Implemented in `capacity_aware_feedback.rs`:
- bounded-delay policy validation and exact proxy-delay PMF;
- replayable random sampling through a caller-supplied uniform draw;
- expiry and admission under a hard pending-feedback capacity;
- known empty/full set losses applied immediately without tracking;
- nontrivial interval outcomes retained only when sampled and admitted;
- inverse propensity weight `1/F(d)` for a tracked outcome;
- no clipping of large inverse-probability weights: an unsupported numeric weight suspends the guarantee;
- underload condition `ell_F <= K` checked at engine creation;
- deterministic risk-gap bound and optimized proxy-delay PMF;
- K1 evidence gate before a tracked physical outcome can update the threshold.

### Critical distinction: pending-feedback capacity is not metrology capacity

The paper's `K` counts prediction outcomes that can be retained/tracked while feedback is pending. It does **not** count wafers sent to a physical metrology tool and does not allocate SEM, TEM, CD-SEM, ellipsometry or other equipment time.

Physical-measurement selection is a separate policy with cost, turnaround, queue and selection-bias constraints. It must be composed later with #13328/#13336 and evaluated at equal physical-metrology budgets.

### Critical distinction: missing qualification is not random delay

CA-OCRC's correction handles outcomes lost through its declared randomized tracking policy and bounded delay model. A label that is UNKNOWN/BLOCKED because calibration, instrument identity, numeric consistency or provenance failed is not the same as an ordinary randomly untracked feedback event. The module therefore suspends its theorem status when a tracked outcome arrives without usable K1 evidence. It does not apply inverse-probability weighting to unqualified evidence or assert that the paper's risk theorem survives informative metrology missingness.

## 3. K1-qualified finite-sample interval and bounded-score set construction

The first set-construction layer is now in `conformal_interval_builder.rs`.

### ACI interval quantile

For a declared absolute or scale-normalized residual score, the builder:
- accepts only current K1 `EligibleUnderDeclaredPolicy` label reports;
- verifies label/report revision, value, payload and report digest;
- requires label and report availability no later than the new prediction cutoff;
- requires the exact global-model artifact, measurement/measurand-unit contract and score-policy digest to match;
- rejects reused sample IDs, measurement events and label revisions;
- binds each calibration score to both the prediction receipt digest and full calibration-sample digest;
- retains the most recently available K1 outcomes up to the declared window cap;
- computes the finite-sample order statistic `k = ceil((n+1)(1-alpha))`;
- returns `FullSetRequired` when `k>n`, rather than substituting a lower quantile;
- emits an immutable set receipt containing sorted scores, window label IDs, policy/model/measurement identities, quantile rank, interval and SHA-256.

This is a rolling-window empirical conformal interval under a declared calibration-score stream; its coverage claim still depends on the method's exchangeability/online assumptions. It is not automatically conditional coverage under adaptive metrology selection or arbitrary drift.

### CA-OCRC bounded-score set constructor

The same module separately implements `C_t={y:S_t(x,y)<=r_t}` for bounded absolute-residual and bounded scale-normalized-residual contracts:
- `r_t<0` → known empty set;
- `0<=r_t<B` → finite interval;
- `r_t>=B` → known full set.

The set receipt binds the exact bounded score contract, (B), current threshold, model artifact, prediction identity and resulting bounds. CA-OCRC round inputs now bind the active threshold, the score-contract digest and this prediction-set digest; feedback receipts retain that set digest. This is distinct from the ACI quantile builder: ACI chooses its interval from a calibration quantile, while CA-OCRC constructs the current score sublevel set from its online risk threshold.

The bounded-score set constructor and the ACI quantile builder have separate known-answer tests. The independent reference file composes the first with delayed ACI and the bounded sublevel-set builder with CA-OCRC.

## 3. Benchmarks and targets

Run independent, task-specific comparisons:

- fixed split conformal and rolling residual quantile;
- standard ACI with immediate feedback;
- exact tau-DACI with fixed delays;
- online conformal prediction via online optimization (ICML 2025);
- capacity-unlimited OCRC;
- intermittent OCRC;
- exact CA-OCRC bounded-delay proxy tracking;
- CAP only when evaluating adaptive selection and false-coverage-statement rate;
- an explicitly experimental K1-gated/missing-feedback variant, with no inherited theorem claim.

Adversarial residual streams:
- IID Gaussian reference;
- AR(1) residual persistence `phi in {0.1,0.6,0.85,0.95,0.99}`;
- GARCH(1,1) scale persistence;
- Markov switching;
- abrupt mean and variance shifts;
- tool/chamber common-mode shifts;
- variable delays and maximum-delay edge cases;
- capacity saturation and proxy-lease eviction;
- unqualified K1 reports, retroactive calibration revocation and numeric label tampering;
- selected-only metrology with informative missingness.

Measure, separately:
- empirical coverage at nominal 90%, 95% and 99%;
- long-run risk gap and ratio to the theorem bound (only when assumptions hold);
- worst rolling-window and worst tool/chamber coverage;
- interval width in normalized and physical units;
- interval score/Winkler score;
- delay-to-memory ratio `tau/L` and residual dependence class;
- pending-feedback occupancy, tracked fraction, saturation count and inverse weights;
- K1 eligible/unknown/blocked counts and guarantee suspensions;
- update latency, replay determinism and state size.

Software acceptance targets:
- 100% no alpha/threshold update from UNKNOWN/BLOCKED/stale/mismatched/unavailable labels;
- 0 historical interval mutation;
- byte-identical replay under a frozen event stream and uniform-draw stream;
- pending tracker count never exceeds K;
- exact inverse-probability weights (never silently clipped);
- risk bound implementation matches known-answer calculations;
- no theorem claim after violating the stated delay, score, capacity, qualification or feedback assumptions.

Predictive superiority targets should be set only after a faithful reproduction of the selected baselines. Do not pool D1/D2 normalized metrics or compare different physical measurands in one scalar leaderboard.

## 4. Industry/SOTA position

- GRAPE (arXiv:2611.00806): streaming feature/context adaptation; our delayed label/evidence gate is a complementary reliability dimension.
- Online Conformal Prediction via Online Optimization (ICML 2025): important adversarial/stochastic online coverage baseline; its authors report evaluation across 15+ datasets.
- CAP (JMLR 2025): selection-conditional coverage/FCR baseline when adaptive sampling selects which wafers receive intervals.
- NIST OGP-CMP (IEEE TSM 2025): industrially relevant uncertainty-aware dynamic metrology sampling comparator.
- Delayed ACI (September 2026): fixed-horizon persistence-aware feedback benchmark.
- CA-OCRC (November 2026): bounded arbitrary delay and limited pending-feedback tracker benchmark.

These are not one leaderboard. The winning research direction is a composition of prediction efficiency, selective coverage, delayed feedback, capacity-aware tracking and evidence qualification—with each guarantee attached to its own assumptions.

## New selection-conditional extension

Issue [#14169](https://github.com/Luminous-Dynamics/symthaea/issues/14169) tracks CAP-style selection-conditional conformal calibration under active metrology. It explicitly distinguishes a wafer being selected, physically measured, K1-qualified and retained by the feedback controller. Selected-wafer FCR/coverage and full-fab population coverage are separate outputs; deterministic zero-propensity strata remain UNKNOWN without a randomized audit path.

## Claim ceiling

These are software/statistical reference implementations. They do not authenticate metrology issuers, establish physical traceability, prove conditional per-wafer coverage, demonstrate fab performance or authorize wafer disposition, model retraining, APC/R2R, recipe updates or equipment control.
