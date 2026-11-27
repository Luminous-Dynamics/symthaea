# SDCB dynamic covariance belief and temporal measurement model v1

**Status:** research design plus one exact covariance primitive. The Rust OU window-kernel reference and its authored tests are not compiled or executed in this environment. No dataset tournament or physical measurement is claimed.

## 1. Why this increment is necessary

The current stack has several distinct and useful references:

- #13793: finite two-stage decisions with per-device binary Markov drift;
- #13804: a shared binary common-cause Markov state coupled to device-local states;
- #13805: evidence-gated joint scientific/common/device belief transitions, but no dynamic state propagation inside that ledger;
- #13848 and #13881: continuous Gaussian common causes, then full prior and residual covariance;
- #13882: one common subset selected by finite-scenario maximin covariance risk.

This proposal does not add another binary drift model or another static covariance-scenario optimizer. It connects the continuous covariance model to elapsed time, unequal measurement windows, and uncertainty about the dynamics/covariance parameters themselves.

The key shift is:

**A measurement is not just a value and a noise variance. It is a time-windowed observation of an evolving physical process, acquired through a changing measurement system whose own dynamics are uncertain.**

## 2. State and causal-factor contract

Keep the following state roles distinct:

| State | Meaning | Example |
|---|---|---|
| Scientific quantity | Target parameter or mechanism | reaction rate, concentration response |
| Physical process | Time-varying measurand field | changing gas concentration or flow |
| Shared measurement cause | Correlated error affecting several channels | reference drift, common power, common environment, common firmware |
| Reference-chain state | Uncertainty inherited from calibration ancestry | working standard, transfer standard, master |
| Device-local state | Bias/drift unique to a device | sensor ageing or local offset |
| Residual acquisition noise | Error remaining after explicit causes | electronics / quantization residual |

Correlation in the physical measurand is not automatically correlation in sensor error. A common residual correlation is not, by itself, proof of a shared physical cause. If a factor is represented explicitly as a latent state, do not count the same factor again inside residual covariance.

## 3. Dynamic state-space model

For a bounded linear-Gaussian reference, let the state include the static scientific parameter and time-varying physical, shared, reference-chain and device-local coordinates:

\[
X(t) = [\theta,\ q(t),\ z(t),\ r(t),\ d_1(t),\ldots,d_D(t)]^\top.
\]

Under covariance/dynamics model \(s\), use a continuous-time linear process:

\[
dX(t) = A_s X(t)\,dt + G_s\,dW(t),\qquad
Q_{c,s}=G_sG_s^\top.
\]

For elapsed time \(\Delta\):

\[
F_s(\Delta)=e^{A_s\Delta},\qquad
Q_s(\Delta)=\int_0^\Delta e^{A_s u}Q_{c,s}e^{A_s^\top u}\,du,
\]

\[
X(t+\Delta)\mid X(t),s
\sim \mathcal N(F_s(\Delta)X(t),Q_s(\Delta)).
\]

The static scientific parameter is represented by a zero-drift state coordinate unless the scientific model explicitly says it evolves. The shared and local measurement factors may have different time constants and cross-covariances. A vector process is required when two factors are dynamically coupled.

### 3.1 Scalar OU exact primitive

The first executable primitive is the stationary scalar OU factor:

\[
\operatorname{Cov}(z(t),z(u)) =
\sigma_z^2 e^{-|t-u|/\tau}.
\]

For measurement windows \(I_i=[a_i,b_i]\), \(I_j=[a_j,b_j]\), with durations \(T_i,T_j\), define

\[
\Psi_\tau(v)=\tau^2\left(e^{-|v|/\tau}+|v|/\tau\right).
\]

The covariance of the two rectangular-window averages is

\[
K_\tau(I_i,I_j)=
\frac{\sigma_z^2}{T_iT_j}
\left[
\Psi_\tau(b_i-a_j)-\Psi_\tau(a_i-a_j)
-\Psi_\tau(b_i-b_j)+\Psi_\tau(a_i-b_j)
\right].
\]

For an identical window of duration \(T\), this reduces to

\[
\operatorname{Var}(\bar z_T)=
\frac{2\sigma_z^2}{T^2}
\left[\tau T-\tau^2(1-e^{-T/\tau})\right].
\]

For disjoint ordered windows with gap \(g\), durations \(T_i,T_j\), the integral has the stable factored form

\[
K_\tau(I_i,I_j)=
\frac{\sigma_z^2\tau^2}{T_iT_j}
e^{-g/\tau}(1-e^{-T_i/\tau})(1-e^{-T_j/\tau}).
\]

The implementation evaluates the small-\(T/\tau\) remainder with a series expansion and uses the factored form for disjoint windows. This avoids the worst cancellation in the naive four-term expression. Times are seconds relative to a declared epoch; callers must not pass large absolute epoch values as f64 when sub-second precision matters.

For multiple independent OU factors, sum each factor's kernel multiplied by the two observation loadings. A correlated vector OU factor requires its full cross-covariance kernel and is a subsequent extension.

### 3.2 Unequal averaging windows

An action is bound to its actual acquisition interval and integration kernel. A fast mobile sensor and a slow reference may not measure the same temporal average even when their records share a timestamp.

For a general state covariance kernel \(K_X(t,u)\) and observation loading \(c_i\), the process contribution is

\[
V_{ij}^{\text{process}} =
\frac{1}{T_iT_j}
\int_{I_i}\int_{I_j}
c_i^\top K_X(t,u)c_j\,du\,dt.
\]

A shared continuous-time white-noise contribution with spectral density \(q_w\) and rectangular averaging windows is

\[
V_{ij}^{\text{white}} =
q_w\frac{|I_i\cap I_j|}{T_iT_j}.
\]

The final observation covariance is the sum of explicitly represented process/factor contributions and residual acquisition covariance. Distinct independent device electronics do not receive a shared white-noise term merely because the devices are measured at the same time.

## 4. Uncertainty about the covariance model

Keep **aleatoric state uncertainty** and **epistemic covariance-model uncertainty** separate.

For v1, let a finite model index \(s\in\mathcal S\) define the dynamics and covariance parameters:

\[
\phi_s=(A_s,Q_{c,s},R_s,\Sigma_{0,s},\text{factor loadings}).
\]

The model index may represent alternative OU time constants, process variances, cross-factor structures, residual covariance models, or a documented regime. Each model has its own valid state-space filter and predictive likelihood.

If and only if model probabilities have a defensible prior and validation protocol, update them using the predictive likelihood:

\[
w_{s,t} =
\frac{w_{s,t-1}\,p(y_t\mid y_{<t},a_{\le t},s)}
{\sum_r w_{r,t-1}\,p(y_t\mid y_{<t},a_{\le t},r)}.
\]

For a linear-Gaussian filter the innovation contribution is

\[
p(y_t\mid y_{<t},s)=
\mathcal N(\nu_{t,s};0,S_{t,s}).
\]

The model-averaged scientific variance must include both within-model and between-model uncertainty:

\[
\operatorname{Var}(\theta\mid D)=
\sum_s w_s\left[
P_{\theta\theta,s}
+(\mu_{\theta,s}-\bar\mu_\theta)^2
\right],\qquad
\bar\mu_\theta=\sum_s w_s\mu_{\theta,s}.
\]

Reporting only \(\sum_s w_s P_{\theta\theta,s}\) silently omits model uncertainty and is a required negative-control failure.

Do not conflate Bayesian model averaging with maximin robustness. Maintain two separately labelled outputs:

1. **Bayesian mixture:** posterior model weights, mixture posterior, predictive score and expected decision value. Weights require a defensible probabilistic model and calibration.
2. **Robust support set:** the declared plausible models and the worst-case loss over that set. The set has no probability interpretation unless one is separately supplied.

Do not remove a model from the robust support set because its posterior weight becomes small unless a preregistered, independently validated exclusion rule permits it. A finite support set is not a guarantee outside that set.

### 4.1 Constructing plausible covariance models

The next estimator should combine:

- check-standard and reference-comparison residuals;
- device and reference-chain ancestry;
- synchronized environmental, power, firmware and acquisition metadata;
- chronological residual autocorrelation and cross-channel covariance;
- regime/change-point evidence;
- uncertainty in the fitted OU time constants and innovation covariance.

Use chronological or blocked resampling that preserves autocorrelation, not IID row bootstrap. Preserve every fitted model's training period, data digest, factor graph, estimator version, uncertainty region and held-out predictive score. A covariance matrix repaired to positive-definite form must carry the repair method, magnitude and reason; v1 exact oracles should continue to reject invalid matrices rather than silently adding jitter.

## 5. Adaptive experiment planning

An action must declare more than a sensor ID:

- device and reference-chain identities;
- start/end time and averaging kernel;
- location and environmental context;
- target/factor loading;
- cost, duration, energy and pathwise resource constraints;
- calibration epoch and measurement-assurance evidence version;
- output resolution or categorical outcome definition;
- whether the action changes the physical process or instrument state.

A sequential policy maps the observed history to the next action. It must propagate the common and local states across the actual elapsed time before scoring the next observation. A calibration/reset action is an intervention with an explicit state-transition and exclusion model, not a magical return to a known state.

For a bounded two-stage oracle:

\[
\pi^*=\arg\min_{\pi\in\Pi_B}\max_{s\in\mathcal S}
\mathbb E_s[L(\text{terminal action},\theta)].
\]

The pathwise budget is checked on every possible branch. Expected cost is reported separately. Target variance, terminal decision value, model discrimination and expected information gain remain separately reported objectives; do not collapse them into one opaque scalar.

The first adaptive implementation may use genuinely quantized instrument outcomes or finite categorical observation alphabets. If continuous readings are discretized only for tractability, the bin width and approximation error must be reported, and the result must not be described as an exact continuous-observation policy.

## 6. Evidence and authority boundary

The design process may recommend an action; it must not authorize or execute a physical intervention. Preserve the existing sequence:

**proposal → authority check → physical execution → observation receipt → independent receipt verification → measurement-assurance check → atomic belief transition → independent replay.**

Keep these claims distinct:

- receipt authenticity and custody;
- reference-chain traceability and uncertainty contribution;
- calibration/model adequacy;
- scientific identifiability and causal validity.

A valid hash does not prove a sensor reading is true. A calibrated instrument does not make every result traceable. A model's posterior does not prove the causal assumptions used to construct it.

The #13805 ledger should eventually consume a dynamic-state prediction and a verified receipt as separate inputs. Expired or stale common-cause evidence must not be applied to a newer state version. A failed receipt must produce zero mutation of the scientific posterior, common-cause state, device states, model weights, evidence versions and budgets.

## 7. Implementation sequence

| Increment | Deliverable | Exact/independent evaluator |
|---|---|---|
| DC0 | Scalar stationary OU window covariance (this PR) | Closed-form same-window, overlap, duplicate and separated-window controls |
| DC1 | Vector OU/state-space propagation with full cross-factor covariance | Matrix-exponential and Van Loan covariance fixtures |
| DC2 | Unequal/overlapping integration kernels for mobile-reference observations | Independent high-precision quadrature and analytic OU controls |
| DC3 | Finite covariance/dynamics model posterior plus robust support set | Known-answer mixture weights, total-variance decomposition, maximin |
| DC4 | Dynamic two-stage policy that advances common/local state by real elapsed time | Exact finite small-instance evaluator; pathwise budget |
| DC5 | Dynamic EGBT prediction/receipt transition adapter | Atomicity, stale-version, replay and invalid-receipt controls |
| DC6 | HDC/CfC approximate experiment policy | Exact-oracle regret and held-out mechanism/device/time splits |

The scalar OU primitive is deliberately narrower than the full programme. It creates a testable temporal covariance foundation without claiming that the complete dynamic adaptive planner already exists.

## 8. Falsification and non-claims

The proposed architecture should be rejected or narrowed if:

- a correct OU/Kalman or Gaussian-process baseline matches or exceeds SDCB on all held-out, equal-cost tasks;
- mixture uncertainty does not improve calibration or decision quality over a well-calibrated plug-in model;
- the covariance-model weights are not calibrated on held-out chronological data;
- the uncertainty set excludes plausible held-out regimes;
- adaptive gains disappear against a matched fixed policy once all methods receive the same observation resolution and action budget;
- the measured common-cause structure is actually process covariance in the measurand;
- the benchmark leaks future reference data into model fitting or action selection.

No claim is made here of continuous distributionally robust optimality, universal calibration validity, physical causal discovery, SOTA performance, autonomous laboratory authority, or physical validation.
