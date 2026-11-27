# SDCB finite-sample distinguishability of correlated sensor processes v1

**Status:** Research-only analytic numerical reference. The machine-readable challenge is SDCB_GAUSSIAN_PROCESS_SEPARATION_V1.json. The standalone Rust reference is scripts/research/sdcb_gaussian_process_separation.rs and its harness is scripts/research/run_sdcb_gaussian_process_separation_tests.sh. The source fixture is a stationary scalar OU process followed by a first-order low-pass sensor. No device has been calibrated, no QUANT data have been analyzed, and no SOTA result is claimed.

## 1. Research shift: identifiability is a budgeted property of the observation protocol

The exact pole-swap result in #14002 established **structural non-identifiability**: distinct physical/sensor assignments can induce the same complete passive single-channel Gaussian output law. More data of the same kind cannot distinguish exactly equivalent mechanisms.

A different failure occurs when two mechanisms are not exactly equivalent, but their finite noisy observations are too similar to support a reliable decision. This is **practical indistinguishability**.

These must not be collapsed:

| Case | Mathematics | Correct interpretation |
|---|---|---|
| Exact pole swap | Identical complete mean/covariance law | Structurally non-identifiable under the declared protocol |
| Near pole swap | Distinct covariance kernels, tiny finite-sample divergence | Structurally distinguishable, practically unresolved at the declared budget |
| Sensor-inlet stimulus | Different sensor response rates under the frozen model | Discriminating intervention, conditional on its calibration/traceability contract |
| Unknown third mechanism | Candidate family may not contain the data-generating mechanism | Model adequacy unresolved; do not force a candidate label |

A large sample of the same passive observations may resolve a near-equivalence, but cannot resolve an exact equivalence. Conversely, a carefully chosen sensor-inlet stimulus can be worth more than thousands of redundant passive samples.

## 2. M0 finite covariance-process contract

For a frozen sampling design D=(t_1,...,t_n), model the entire observation vector as

\[
Y_D \sim \mathcal N(m_\theta(D), \Sigma_\theta(D)),
\qquad
\Sigma_\theta(D)=K_\theta(D)+R_D.
\]

Here K is the full process-plus-instrument covariance matrix, and R is the declared readout/acquisition-noise covariance. This is one joint Gaussian vector. Its time samples and autocovariance lags are not independent scalar replicates.

For the zero-mean positive-definite case, the directional divergence is

\[
D_{\mathrm{KL}}(A\Vert B)=
\frac12\left[
\operatorname{tr}(\Sigma_B^{-1}\Sigma_A)-n+
\log\frac{\det\Sigma_B}{\det\Sigma_A}
\right].
\]

Report both directions. The implementation must use Cholesky factorization and triangular solves, not a materialized matrix inverse. With \(\Sigma_A=L_A L_A^\top\) and \(\Sigma_B=L_B L_B^\top\),

\[
\operatorname{tr}(\Sigma_B^{-1}\Sigma_A)
=\lVert L_B^{-1}L_A\rVert_F^2.
\]

This gives an efficient, auditable computation for the frozen n<=64 benchmark.

### Conservative finite-sample interpretation

Pinsker's inequality provides a total-variation upper bound from either direction:

\[
\operatorname{TV}(P_A,P_B)
\le \sqrt{\frac{D_{\mathrm{KL}}(A\Vert B)}{2}},
\qquad
\operatorname{TV}(P_A,P_B)
\le \sqrt{\frac{D_{\mathrm{KL}}(B\Vert A)}{2}}.
\]

Both are valid, so their minimum is also a valid upper bound. For equal prior probabilities, the optimal binary classification accuracy is (1+TV)/2. Substituting the TV upper bound yields an **accuracy ceiling**, not an achieved classifier score or a lower bound on error from a specific algorithm.

A small KL is not an exact power calculation. M0 reports directional KL, the Pinsker bound, and the resulting equal-prior accuracy ceiling only. A later exact operating-characteristic track may evaluate the finite-sample likelihood-ratio statistic as a Gaussian quadratic form and compare it against calibrated profile-likelihood or simulation-based tests. It must not relabel KL as power.

### Structural equivalence remains a different test

Structural equivalence requires equality of the complete declared mean and covariance functions over the admissible observation protocol. Equality at a finite set of lags, a tiny finite-design KL, or failure to reject a hypothesis is not proof of structural equivalence.

The finite-sample KL belongs to one exact sampling design, response/noise parameterization and sample budget. Bind those identities into every result.

## 3. Frozen near-pole-swap challenge

World A reuses the exact #14002 pole-swap World A:

- Physical decay lambda_A=0.25.
- Sensor response rate mu_A=4.
- Physical stationary variance sigma²_x,A=9.

World B is a nearby but non-equivalent alternative:

- Physical decay lambda_B=4.04.
- Sensor response rate mu_B=0.25.
- Physical variance is selected to preserve the same zero-lag output variance q, where q=mu*sigma²_x/(lambda+mu).
- The frozen variance is 145.35529411764705882352941176470588235294117647059.

The output kernels are not equal. At lag h=0.5, the high-precision values are:

- A: 7.89719444223072776693323753225
- B: 7.89423872892642929696050463627

The difference is about -0.0029557133. This is a near-equivalence, not an exact alias.

### Frozen observation protocol

- Samples at t_i=i*0.5, starting at i=0.
- Nested sample prefixes n in {8,16,32,64}.
- Independent additive Gaussian readout noise, variance 0.1 per sample.
- Full covariance matrix assembled from the response autocovariance at every pairwise time lag plus 0.1I.
- One synthetic unit of acquisition cost per sample.

The 60-digit reference KL values and Pinsker bounds are in the JSON fixture. At n=32:

\[
D_{\mathrm{KL}}(A\Vert B)=0.00025683365293439291035\ldots
\]

\[
D_{\mathrm{KL}}(B\Vert A)=0.00025805588272451917037\ldots
\]

\[
\operatorname{TV}(P_A,P_B)\le0.011332114827656683\ldots
\]

\[
\operatorname{Acc}^{*}_{1/2}\le0.505666057413828342\ldots
\]

Thus, even an optimal classifier using the complete 32-dimensional observation vector cannot exceed this conservative equal-prior accuracy ceiling under this bound. This is a statement about the frozen synthetic distribution and design—not a general statement about field sensors.

The bound grows as the nested sample budget increases, as expected. More passive observations gradually help, but the gain is small over these budgets because the observations are temporally correlated and the response kernels are close.

### Positive disambiguation

At t=0.5, a traceable unit step applied directly at the sensor inlet yields normalized responses:

- World A, mu=4: 0.8646647167633873...
- World B, mu=0.25: 0.1175030974154046...

This is a strong discriminating control within the declared two-pole model. A physical experiment must additionally model stimulus uncertainty, timing, inlet conditions, readout noise and reference-chain ancestry. The noiseless values are not a device validation.

## 4. Numerical implementation boundary

M0 supports finite positive-definite covariance matrices only.

- Build the full covariance matrix directly from the declared kernel and readout covariance.
- Use Cholesky, triangular solves and log determinants.
- Never explicitly invert a matrix in the production path.
- Never add jitter, diagonal loading, PSD projection, or silent covariance repair.
- If factorization fails, return an explicit failure/unsupported disposition and preserve the supplied matrix.
- Bind time ordering, units, model parameters, covariance/noise identities, numerical precision, algorithm version, resource budget and source digest.
- Report both directional KL values, TV upper bound, equal-prior accuracy upper bound, matrix dimension, condition diagnostics, factorization residual, latency and peak memory.

Singular Gaussian measures require a separate support-aware contract. A pseudodeterminant alone is not enough: support inclusion and absolute continuity matter. Until that is implemented, singular covariance inputs remain unsupported rather than being regularized.

The Rust source contains a dependency-free reference and ten authored tests. A local environment probe confirmed that rustc and cargo are unavailable here, so the tests are not executed. The frozen numerical values were independently calculated using 60-digit mpmath arithmetic in the authoring runtime. That is not Rust compilation, CI, formal verification, or physical validation.

## 5. How this composes with existing Luminous work

| Existing owner | Reuse | Boundary |
|---|---|---|
| #14002 / #14003 | Exact OU + first-order response equations and pole-swap negative/positive controls | Does not establish physical calibration |
| #14038 / #14041 | Mechanism identifiability versus decision sufficiency | Does not compute finite-sample covariance discrimination |
| #12454 / #12457 | Fixed finite expected-mean separation under independent known-variance Gaussian replicates | Not a correlated Gaussian-process oracle |
| #11091 | Local rank versus global discrete ambiguity | Candidate-geometry screen, not process-covariance power |
| #11262 | Open-world misspecification and representative/informative experiment design | No duplicate general planner here |
| #8628 | Distinguish physical-law drift from measurement-process drift | Existing diagnostic-action owner |
| #13567 | Exact finite-horizon robust/Bayesian policy value | Adaptive experiment policy owner |
| #13739 | Existing Symthaea experiment-selector comparisons | Selector tournament owner |
| #14000 | Probability enclosures and conservative action-value intervals | Numerical decision owner |
| #13805 | Independently verified receipts and atomic belief transitions | Evidence authority |

This M0 reference is specifically the missing covariance-process path. It does not widen #12454's independent-replicate assumptions or introduce another planner.

## 6. Benchmark tournament and SOTA comparison

The field already has mature approaches to structural identifiability, practical identifiability, optimal design and active learning. We must compete with them, not claim to have invented them.

### Structural and practical identifiability

Compare:
- Symbolic structural analysis with StructuralIdentifiability.jl / SIAN-style methods where the model class is supported.
- Local Fisher information, Jacobian rank, singular values and Cramér–Rao diagnostics.
- Profile likelihood and multi-start/global alternative screening.
- Explicit symmetry/equivalence-class enumeration.

A 2026 open tutorial emphasizes that structural identifiability concerns ideal observations, while practical identifiability concerns finite/noisy/discrete data; full local rank does not prove global uniqueness. It also documents a pinned, reproducible symbolic-identifiability workflow. SDCB should use such tools as comparators and focus its own contribution on connecting those results to actual time-windowed sensor response, finite covariance discrimination and terminal decisions.

### Experiment design under misspecification

Compare:
- D- and A-optimal designs / FIM-based selection.
- Bayesian EIG/BOED.
- Max-min pairwise KL or Chernoff discrimination.
- Profile-likelihood-targeted control design.
- R-IDeA: representative, informative and de-amplifying acquisition.
- Robust sequential experiment design with worst-case error criteria.
- Exact #13567 policy values on small worlds.

The 2026 AISTATS R-IDeA paper explicitly identifies that informative designs can be unrepresentative under covariate shift, and adds representativeness and error de-amplification to informativeness. The 2026 ICML robust sequential design paper studies model misspecification and worst-case mean-squared-error bounds. SDCB's tournament must include these methods in their applicable domains; it must not compare against EIG alone and then claim robust superiority.

### AI-designed experiments and autonomous science

The 2026 Nature review Designing physics experiments with artificial intelligence frames the frontier as the joint design of expressive hardware/configuration spaces, fast reliable simulators, computable scientific objectives and AI search over discrete/continuous design choices. This suggests a longer-term SDCB direction beyond choosing the next scalar observation: jointly design sensor placement, sampling interval, response calibration, stimulus waveform and reference allocation, with decision loss and safety constraints.

Autonomous Science Bench 0.1 evaluates complete agent/model/tool/harness configurations on six pilot laboratory tasks, including choosing experiments, reading results, adapting and respecting lab constraints. Its page reports GPT-6 Astra Codex at 72.2% and Claude Opus 5.5 Claude Code at 66.7%, while noting only three trials per task and that small score gaps are not meaningful. Use it as an external workflow benchmark, not a metrology benchmark; an SDCB-specific adapter must preserve scientific measurement and evidence qualification.

### Current sensor-calibration industry baseline

A Sensors paper published 6 November 2026 evaluates five multimodal devices across two Brussels sites, with only 9–68 days of data per device. It compares conventional regression and tree methods with DANN transfer, and uses cross-sensor residuals to flag injected gradual drift. Its own results show transfer can lose to the uncalibrated sensor for some pollutants; differences in mean transfer ratios are not statistically significant; and cross-sensor residual monitoring detects inconsistency rather than isolating the faulty sensor. The drift cases are injected, not observed natural ageing.

This creates a realistic comparison target:
- Calibration accuracy alone is insufficient.
- Transfer ratio alone can hide high absolute error.
- Reference-free inconsistency flags do not establish fault attribution.
- The system should preserve common-cause ambiguity and select a genuinely discriminating reference/calibration action when the decision warrants its cost.

## 7. SDCB target metrics

Report distinct outcomes; no composite score may compensate for false identification or invalid evidence.

| Metric | Proposed target |
|---|---:|
| Exact pole-swap false unique assignments | 0 |
| Exact pole-swap equivalence classification | 100% |
| Directional KL reference containment / reproduction | 100% frozen fixtures |
| False practical-identifiability claims on unresolved budgets | 0 |
| Silent covariance repair | 0 |
| Invalid/singular covariance silently accepted | 0 |
| Decision-invariant unnecessary positive-cost probes | 0 |
| False independent confirmation | 0 |
| Invalid/stale receipt accepted | 0 |
| Policy regret on exact small worlds | <= 2% |
| Decision-loss improvement per equal reference cost | >= 10%, only after powered paired held-out evaluation |

Also report:
- cost-to-identification versus cost-to-decision;
- sample cost to a predeclared false-assignment/power operating point;
- directional KL and symmetric divergence, with exact definition;
- Pinsker looseness versus calibrated finite-sample test power;
- abstention/unsupported rate and selective risk;
- device, manufacturer, site, pollutant, acquisition-window and temporal-regime performance;
- calibration drift detection delay and false alarms;
- absolute target error, not only transfer ratios;
- full failure denominators, missing cohorts and negative controls.

The 10% decision-loss improvement target is a hypothesis, not a result. Require preregistered primary endpoints, paired independent clusters, confidence intervals excluding zero, equal resource budgets and external reproduction.

## 8. Next paradigm shifts

1. **Identifiability budget surfaces:** for each mechanism pair and candidate observation protocol, map acquisition cost against structural equivalence, finite-sample distinguishability and terminal decision value. Do not confuse these three surfaces.
2. **Decision-directed metrology:** stop collecting information when all plausible mechanisms imply the same decision; acquire a reference only when it changes the decision or its safety margin.
3. **Open-world challenge by construction:** compose with #11262 and #8628. Hold out sensor-response families, shared-reference faults, time-varying kernels and physical-law changes. If none of the candidates explains the evidence, retain OTHER_UNRESOLVED.
4. **Co-design the observation system:** use AI search over sensor location, time resolution, excitation waveform, reference chain and budget. Evaluate hardware feasibility and observability before ranking candidate configurations.
5. **Evidence-carrying scientific decisions:** every conclusion binds the observation operator, covariance model, reference ancestry, numerical method, uncertainty enclosure and independently verified receipt. A mathematically valid likelihood is not itself evidence authentication.

## 9. Implementation sequence

1. Freeze and independently parse the near-swap JSON known-answer fixture.
2. Run the standalone Rust Cholesky/KL reference and resolve every authored test on an exact source head.
3. Add a separately checked finite-sample likelihood-ratio power calculation; do not infer power from KL.
4. Add a time-varying sensor-response and common-reference covariance family, reusing #13805 and #8628 ownership.
5. Add an out-of-family challenger with explicit OTHER_UNRESOLVED handling, reusing #11262.
6. Connect decision-value outputs to #13567 and existing selectors through #13739.
7. Reproduce the QUANT/Kok–Tabandeh empirical baselines before adding any SDCB superiority claim.
8. Run controlled physical sensor-inlet step/frequency tests with synchronized independent reference, documented sample-line delay and calibration traceability.

## 10. References

- Alsammani (2026), A Tutorial on Symbolic Structural Identifiability Analysis of ODE Models in Julia, Bulletin of Mathematical Biology. https://doi.org/10.1007/s11538-026-01779-0
- Tang, Sloman & Kaski (AISTATS 2026), Representative, Informative, and De-Amplifying: Requirements for Robust Bayesian Active Learning under Model Misspecification. https://proceedings.mlr.press/v300/tang26d.html
- Wen et al. (ICML 2026), Robust Sequential Experimental Design for A/B Testing. https://proceedings.mlr.press/v306/wen26i.html
- Klimesch et al. (Nature 2026), Designing physics experiments with artificial intelligence. https://doi.org/10.1038/s41586-026-10898-6
- Sivakumar et al. (Sensors, 6 November 2026), Data-Driven Calibration and Reference-Free Drift Detection for Multi-Modal Sensor Systems. https://doi.org/10.3390/s26217016
- Autonomous Science Bench 0.1 (2026). https://asb.open-ssi.org/about

## 11. Claim boundary

This is an analytic finite-sample Gaussian-process reference and benchmark specification. It does not demonstrate physical sensor calibration, metrological traceability, real-data QUANT performance, production scientific-agent capability, SOTA performance, or superiority over existing methods.


## 12. Metrology standards and edition control (checked 27 November 2026)

The standards lane must pin the edition and the procedure actually executed.

- The BIPM/JCGM publication page currently lists GUMsuite:2026, JCGM 100:2008/Amd.1:2026 on nonlinearity in measurement models, and JCGM GUM-5:2026 examples. JCGM 101:2008, Monte Carlo propagation, remains listed. See https://www.bipm.org/en/committees/jc/jcgm/publications/.
- CEN/TS 17660-1:2021 for gaseous pollutant sensor systems is listed as active by the national standards registry; another catalogue records a review-enquiry stage with completion on 14 October 2026. Continue edition/successor tracking rather than assuming the 2021 text will remain the only applicable edition. See https://app.nbn.be/data/r/platform/frontend/detail?p40_id=199863&p_lang=en and https://www.iteh.us/catalog/standards/cen/5bdb236e-95a3-4b5b-ba7f-62ab08cd21f8/cen-ts-17660-1-2021.
- PD CEN/TS 17660-2:2024 is listed as current for particulate-matter sensor performance. Its scope is individual low-cost sensor systems at fixed sites; it does not guarantee performance at other locations, different meteorological regimes, or long-term stability without ongoing quality-control strategies. It explicitly does not apply to sensor systems as part of a network, although it includes informative material on network nodes. See https://knowledge.bsigroup.com/products/air-quality-performance-evaluation-of-air-quality-sensor-systems-particulate-matter-in-ambient-air.

**Architecture consequence:** SDCB should preserve a distinct device-conformance lane and a distinct network-decision-assurance lane. The first follows the licensed, version-pinned standard procedure. The second evaluates cross-device covariance, shared references/common causes, site/time transfer, calibration lineage, drift and terminal decision loss. Passing one lane does not imply passing the other. A standards edition being listed or tracked is not evidence that SDCB conforms to it; no conformance test is claimed here.


## 13. Fixed-budget sampling-interval sensitivity

The fixture now includes a second design comparison with a fixed 32 samples and readout-noise variance 0.1, while the sample interval varies. A 60-digit reference gives:

| Sample interval | Observation horizon | KL(A || B) |
|---:|---:|---:|
| 0.1 | 3.1 | 0.000159870651 |
| 0.25 | 7.75 | 0.000237458040 |
| 0.5 | 15.5 | 0.000256833653 |
| 1.0 | 31.0 | 0.000098327854 |
| 2.0 | 62.0 | 0.000015743738 |
| 4.0 | 124.0 | 0.000001879240 |

The 0.5 interval has the largest KL among these six candidates in either direction. This illustrates a nontrivial co-design trade-off: samples that are too dense spend the fixed sample budget over too short an observation horizon, while samples that are too sparse miss the response dynamics. The result is only a finite-grid known answer—not a continuous optimum or a universal sampling rule.

A real design tournament must preserve the entire resource vector. At a minimum, compare sample count, elapsed horizon, reference-instrument time, energy and compute. A design with greater KL but longer delay or greater lifecycle cost is not automatically preferable. The exact policy/decision value still belongs to #13567, and a learned design policy must be compared at equal feasible budgets.
