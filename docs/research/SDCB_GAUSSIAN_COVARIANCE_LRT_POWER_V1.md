# SDCB Gaussian covariance likelihood-ratio power v1

**Status:** preliminary numerical research reference; not Rust-compiled, CI-qualified, or physically validated. The machine-readable companion is *docs/research/SDCB_GAUSSIAN_COVARIANCE_LRT_POWER_V1.json*.

## 1. Why this increment matters

The parent covariance-process benchmark (#14065) reports directional Gaussian KL divergence and conservative Pinsker bounds. Those are useful distribution-separation measures, but neither is the operating power of a specified scientific test.

This increment evaluates the finite-sample Neyman–Pearson likelihood-ratio test for the frozen near-pole-swap pair. It turns the statement “these finite observations are difficult to distinguish” into a directly interpretable operating characteristic: at a declared 5% false-positive rate, what is the probability of detecting World A when World A is true?

This is a narrow fixed-design reference. It does not estimate unknown covariance parameters, validate arbitrary model families, or provide optional-stopping guarantees.

## 2. Mathematical contract

For a frozen design with n observation times, compare two fixed, known, zero-mean Gaussian laws:

\[
H_A: Y \sim \mathcal N(0,\Sigma_A), \qquad
H_B: Y \sim \mathcal N(0,\Sigma_B).
\]

Both covariance matrices must be symmetric positive definite. The log-likelihood ratio is

\[
\ell(y)=\log\frac{f_A(y)}{f_B(y)}
=\frac12\log\frac{\det\Sigma_B}{\det\Sigma_A}
+\frac12 y^\top(\Sigma_B^{-1}-\Sigma_A^{-1})y.
\]

The production implementation should evaluate the expression using Cholesky factors and triangular solves, not materialize matrix inverses. For \(\Sigma_A=L_A L_A^\top\) and \(\Sigma_B=L_B L_B^\top\), define

\[
c=\frac12(\log|\Sigma_B|-\log|\Sigma_A|).
\]

Under A and B respectively, with \(Z\sim\mathcal N(0,I)\),

\[
\ell(Y_A)=c+\frac12 Z^\top G_A Z,\quad
G_A=L_A^\top\Sigma_B^{-1}L_A-I,
\]

\[
\ell(Y_B)=c+\frac12 Z^\top G_B Z,\quad
G_B=I-L_B^\top\Sigma_A^{-1}L_B.
\]

The symmetric matrices \(G_A,G_B\) can be diagonalized. Their eigenvalues \(\gamma_i\) give central, potentially indefinite Gaussian quadratic forms

\[
Q=\sum_i\gamma_i Z_i^2.
\]

The distribution of Q is evaluated through characteristic-function inversion (the Imhof family of methods). The statistical distribution is the exact finite-dimensional distribution under the declared Gaussian model; the numerical integration is approximate and must report its tolerance, convergence status, evaluation count and error diagnostics. The current pilot does **not** carry a certified global quadrature error bound.

### 5% operating point

For the test \(H_0=B\) against \(H_1=A\), choose \(\tau_{0.05}\) so that

\[
P_B(\ell(Y)\ge \tau_{0.05})=0.05.
\]

Then report both achieved size and power:

\[
\mathrm{size}=P_B(\ell(Y)\ge\tau_{0.05}),\qquad
\mathrm{power}=P_A(\ell(Y)\ge\tau_{0.05}).
\]

For a continuous LRT statistic, strict versus non-strict inequality is immaterial. Exact aliases create an atom and require explicit randomization at the threshold.

### Equal-prior optimal classification

At equal priors, the likelihood-ratio rule chooses A for \(\ell>0\), B for \(\ell<0\), and splits ties 50/50. Its accuracy is

\[
\mathrm{Acc}^*=\frac12\left[
P_A(\ell>0)+\frac12P_A(\ell=0)
+P_B(\ell<0)+\frac12P_B(\ell=0)
\right]
=\frac{1+\mathrm{TV}(P_A,P_B)}{2}.
\]

This is an achieved optimal accuracy for the stated simple-hypothesis experiment, not a Pinsker bound and not the accuracy of a learned classifier.

**Exact-alias control:** under the #14002 pole swap, the two complete observation laws are identical. The log-likelihood ratio is constant (zero under exact arithmetic). A deterministic rule that accepts A whenever \(\ell\ge0\) would produce a meaningless tie artifact. The randomized size-\(\alpha\) test must have power \(\alpha\), and the equal-prior classifier that splits ties must have accuracy 0.5. Any apparent advantage on this control is a test implementation failure.

## 3. Frozen near-swap result

The reference reuses #14065 without changing its worlds or observation protocol:

- World A: physical decay \(\lambda=0.25\), sensor rate \(\mu=4\), process stationary variance \(9\).
- World B: physical decay \(\lambda=4.04\), sensor rate \(\mu=0.25\), variance adjusted to preserve the same zero-lag output variance.
- Samples: \(t_i=i\cdot0.5\), with nested budgets \(n\in\{8,16,32,64\}\).
- Independent additive Gaussian readout-noise variance: \(0.1\).
- Full joint covariance is retained; time samples are not flattened into independent replicates.

The table compares finite-sample LRT operating characteristics with the earlier conservative Pinsker accuracy ceiling.

| Samples | LRT accuracy (equal prior) | Pinsker accuracy ceiling | LRT power at 5% size |
|---:|---:|---:|---:|
| 8 | 50.2018% | 50.2626% | 5.0703% |
| 16 | 50.3068% | 50.3911% | 5.1225% |
| 32 | 50.4486% | 50.5666% | 5.1964% |
| 64 | 50.6447% | 50.8107% | 5.3011% |

For 32 samples, the log-likelihood threshold is approximately \(0.032736059434691\). The preliminary numerical reference gives 5% size and 5.1964% power. An exploratory three-million-draw check gave 5.0162% size and 5.2019% power, with Wilson 95% intervals of [4.9916%, 5.0410%] and [5.1769%, 5.2271%] respectively. The seed and PRNG identity were not frozen for this exploratory check, so it is a sanity check, not a reproducible qualification artifact.

**Interpretation:** this particular passive design is practically uninformative against this near-swap. Increasing the observation budget from 8 to 64 samples only lifts power from about 5.07% to 5.30% at a 5% false-positive rate. This is not a claim about all OU processes or all sensor arrangements. It is a result about the frozen pair, noise model and acquisition protocol.

## 4. Sampling interval is a joint density/horizon choice

At n=32 and readout-noise variance 0.1, the candidate interval sweep gives:

| Interval | Horizon | KL(A || B) | LRT equal-prior accuracy | LRT power at 5% size |
|---:|---:|---:|---:|---:|
| 0.1 | 3.1 | 0.00015987 | 50.3501% | 5.1417% |
| 0.25 | 7.75 | 0.00023746 | 50.4312% | 5.1896% |
| 0.5 | 15.5 | 0.00025683 | 50.4486% | 5.1964% |
| 1.0 | 31 | 0.00009833 | 50.2765% | 5.1171% |
| 2.0 | 62 | 0.00001574 | 50.1105% | 5.0466% |
| 4.0 | 124 | 0.00000188 | 50.0381% | 5.0169% |

Interval 0.5 is best **only among these six candidates**. A real acquisition-design comparison must not optimize KL, power, elapsed time, energy and cost as if they were one interchangeable scalar. Publish a Pareto frontier and let the task's decision loss determine which trade-offs matter.

## 5. Paradigm shift: from identifiability scores to a scientific decision envelope

SDCB should treat scientific uncertainty as four related but independent axes:

1. **Structural equivalence:** do distinct mechanisms produce the same complete observation law under the declared measurement protocol? The output is an equivalence class, not a forced unique mechanism.
2. **Finite-budget distinguishability:** under the finite acquisition, noise, covariance and instrument response, what are the attainable size, power, coverage and separation bounds?
3. **Model adequacy:** could the truth lie outside the declared mechanism or measurement family? If so, return an explicit open-world disposition such as OTHER_UNRESOLVED, not the least-wrong named model.
4. **Decision sufficiency:** do all mechanisms still compatible with the evidence imply the same optimal action? When yes, more mechanism information may have zero decision value.

Do not collapse these axes into one confidence score. In particular, low KL is not structural equivalence; failure to reject is not model adequacy; high EIG is not decision value; and a decision that is invariant over an equivalence class does not mean the mechanism has been identified.

## 6. How SDCB can outperform current practice

The goal is not to claim invention of structural identifiability, Fisher information, optimal design or robust Bayesian acquisition. Those have mature literatures and useful public comparators:

- Symbolic ODE identifiability: SIAN and StructuralIdentifiability.jl. SDCB should benchmark against exact structural analyses and carry the full instrument/acquisition operator into the observation contract.
- Local and practical identifiability: Fisher information, profile likelihood and likelihood-ratio tests. SDCB should compare local approximations against global finite-sample distribution separation and abstention behaviour.
- Robust design under misspecification: R-IDeA explicitly adds representativeness and error de-amplification to informativeness. Treat it as a tournament arm, not an opponent to dismiss.
- Robust sequential design: compare the actual SDCB policy under identical context, observation, cost and model uncertainty budgets against robust sequential-design baselines.
- Autonomous laboratory agents: use AS-Bench for end-to-end lab execution and task completion, but do not conflate an agent's task score with the scientific validity of its measurement model or conclusion.
- Sensor calibration and air-quality standards: CEN/TS 17660-1 and -2 provide device-level performance frameworks. Keep device conformity, metrological traceability, network-level decision assurance and long-term stability as separate claims. GUM/JCGM uncertainty propagation and conformity guidance should be reflected in the measurement contract.

SDCB's potential differentiator is the auditable composition of these disciplines across the entire measurement path, rather than an isolated curve-fitting metric.

## 7. Research and engineering architecture

Compose with existing repository owners; do not add a competing planner or evidence ledger.

| Layer | Responsibility | Existing owner |
|---|---|---|
| Exact structural equivalence | Full mean/covariance observation law and pole-swap control | #14002 |
| Finite covariance separation | Joint Gaussian KL and Pinsker references | #14065 |
| Fixed-design LRT power | This child issue: exact simple-hypothesis size/power | #14110 |
| Open-world adequacy | Support omission, hidden mechanisms and OTHER_UNRESOLVED | #11197 / #11262 |
| Power under adaptive discovery | Alternative support and policy-induced detection limits | #12375 |
| Observation-model stress | AR, heavy tails, heteroskedasticity, regime shift and contamination | #12540 |
| Common-cause/reference evidence | Shared reference, per-device state and reference-chain context | #13805 / #13951 |
| Sequential decision oracle | Non-myopic policy and exact small-world decision regret | #13567 |
| Selector tournament | Existing proposal methods under matched budgets | #13739 |
| Evidence authority | Independently verified receipt and atomic state transition | #13805 |

A future observation contract should bind, at minimum: physical system and candidate mechanism identities; instrument transfer/response model; calibration and reference-chain ancestry; readout-noise covariance; sample times and ordering; excitation or intervention and its uncertainty; units; acquisition budget; and exact model/code versions.

A separation result should be a typed, immutable statistical receipt. It can contribute to experiment evaluation, but it must not authorize the physical probe, mutate belief state or claim source traceability. Those remain responsibilities of the existing owners.

## 8. Benchmark targets

Targets below are **proposed gates**, not achieved results.

| Track | Target |
|---|---|
| Structural equivalence | Zero false unique-mechanism promotions on exact-equivalence controls |
| LRT / practical power | Empirical size inside preregistered binomial intervals; report power by alternative family and minimum detectable effect |
| Material alternatives | Aim for at least 80% power at declared \(\alpha\) and acquisition budget; otherwise label UNDERPOWERED and publish the power/cost frontier |
| Open-world adequacy | Zero false closed-catalogue promotions on sealed out-of-family controls; OTHER_UNRESOLVED preserved |
| Decision quality | At most 2% regret versus the exact small-world policy oracle |
| Real systems | At least 10% lower terminal decision loss per equal reference cost versus the strongest matched baseline, with paired uncertainty intervals |
| Metrology | No conflation of device conformance, reference traceability, network validity or drift stability |
| Numerical reliability | Frozen known answers contained by independently checked numerical enclosures; no silent jitter or covariance repair |

Use held-out devices, families, sites and time periods. Preregister sample allocation, alternative envelopes, thresholds, primary endpoints and missing-data rules. Report every seed, failure, abstention, invalid input and unsupported case. Compare against Fisher/profile-likelihood, D-optimal, EIG, robust acquisition and fixed/flexible sampling under matched observation, elapsed-time, compute, energy and reference-material budgets.

The 80% power and 10% decision-loss improvements are target values, not claims of current superiority. An underpowered design is a useful result when it is honestly reported.

## 9. Next execution sequence

1. Implement the Gaussian quadratic-form CDF with a documented Imhof/Davies numerical contract; compare at least two independently structured routines and retain convergence/error diagnostics.
2. Reproduce this fixture in Rust only after resolving the dependency boundary. The active default-branch symthaea-statistics crate is v0.1.0 and currently exports only descriptive/distribution/inference/regression modules. The richer v1.x Cholesky/eigensolver/QMC material found during this audit is in versioned snapshots and combined patch files, not an active production API. Either promote and qualify the required primitives through that separate package work, or justify a bounded research-only implementation; do not assume an unlanded API or silently duplicate a production numerical substrate.
3. Add exact-alias randomized tie controls, permutation invariance, nested-observation monotonicity, readout-noise stress, and invalid/SPD failure cases.
4. Add a frozen seed/PRNG manifest for Monte Carlo cross-checks; distinguish statistical simulation uncertainty from quadrature error.
5. Extend to uncertain response and shared calibration/reference processes by consuming #13805/#13951 rather than duplicating their authority.
6. Add out-of-family hidden mechanisms and model-adequacy scoring through #11197/#11262/#12375.
7. Compare experiment selectors against #13567's exact oracle and the strongest robust-design baselines at equal resource-vector budgets.
8. Only then move to QUANT/raw-data reproduction and controlled sensor-inlet experiments. Synthetic performance must not be called physical qualification.

## 10. Validation status

- Existing #14065 covariance fixture and KL references are reused unchanged.
- A preliminary authoring-runtime Imhof calculation and a separate quadrature implementation were compared; a three-million-draw n=32 simulation was used as a sanity check.
- The exploratory Monte Carlo seed/PRNG identity is not yet frozen.
- A certified global quadrature error bound has not been established.
- Rust implementation is not yet authored for this LRT extension; Rust compilation, tests and Clippy are not run.
- Hosted CI, QUANT data analysis, physical experiments, traceability and external superiority are not claimed.

### External comparators / standards to track

- StructuralIdentifiability.jl: https://github.com/SciML/StructuralIdentifiability.jl
- SIAN: https://github.com/pogudingleb/SIAN
- R-IDeA (AISTATS 2026): https://proceedings.mlr.press/v300/tang26a.html
- Robust Sequential Experimental Design for A/B Testing (ICML 2026): https://proceedings.mlr.press/v306/chu26a.html
- Designing Physics Experiments with Artificial Intelligence (Nature, 2026): https://www.nature.com/articles/s41586-026-09640-3
- BIPM/JCGM publications: https://www.bipm.org/en/committees/jc/jcgm/publications/
- CEN/TS 17660-1:2021 and CEN/TS 17660-2:2024: device-level ambient-air sensor performance; not network-level scientific decision assurance.

