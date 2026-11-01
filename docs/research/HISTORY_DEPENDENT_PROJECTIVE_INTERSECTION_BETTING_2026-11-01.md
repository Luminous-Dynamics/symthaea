# History-dependent projective intersection betting under structured dependence

**Status:** model-specific research prototype and candidate construction. No arbitrary-dependence, novelty, or power-improvement claim is made. The conditional Gaussian model is an explicit assumption that must be validated in an application.

## 1. Why open a fourth track?

The global affine projective merger solves predictable arrivals under arbitrary cross-coordinate dependence by assigning fixed, globally summable coordinate weights. That restriction is structural within the paper's pointwise arbitrary-dependence merger class. On a countably infinite open-ended universe, a nontrivial symmetric pointwise merger is impossible; useful affine constructions must encode pre-data asymmetry.

The dynamic e-closure paper explicitly identifies direct intersection betting, structured-dependence mergers, history-dependent mergers, and scalable computation as future research. This track asks whether a valid joint conditional model can support richer intersection-specific betting strategies without paying the same coordinate-wise summability penalty.

This is not a replacement for #9695. It is a model-specific alternative whose power, robustness, and calibration cost must be compared against the arbitrary-dependence approach.

## 2. Shared-control Gaussian model

At global update t, suppose the currently active experiments provide a shared control C_t and arm outcomes X_{i,t}. Define standardized differences

\[
Z_{i,t}=(X_{i,t}-C_t)/\sqrt{2}.
\]

For every finite active subset A of size m, assume the conditional law given the common global filtration F_{t-1} is multivariate Gaussian:

\[
Z_{A,t}\mid\mathcal F_{t-1}
\sim \mathcal N(\theta_{A,t},\Sigma_m),
\qquad
\Sigma_m=\tfrac12 I_m+\tfrac12\mathbf1_m\mathbf1_m^\top.
\]

Thus each coordinate has conditional variance one and each pair has conditional covariance one-half. This is the shared-control dependence structure obtained, for example, when independent unit-variance arm noise is differenced against one independent unit-variance common control and standardized by sqrt(2).

The intersection null for S is

\[
H_S:\quad \theta_{i,t}\leq0
\quad\text{for every }i\in S\cap I_t
\text{ and every global update }t.
\]

The conditional mean and covariance model are assumed known in this first prototype. Unknown nuisance parameters, estimated covariance, heavy tails, non-Gaussian shocks, and model uncertainty are explicitly out of scope until robust conditional-mgf bounds are derived.

For a subset A of size m,

\[
\mathbf1_m^\top\Sigma_m\mathbf1_m=\frac{m(m+1)}2.
\]

## 3. A predictable intersection-specific betting factor

For each fixed finite intersection S, at time t let

\[
A_{S,t}=S\cap I_t.
\]

Choose a scalar \(\lambda_{S,t}\geq0\) that is measurable with respect to \(\mathcal F_{t-1}\), using only past observations belonging to A_{S,t}. Define

\[
X_{S,t}=
\exp\left(
\lambda_{S,t}\sum_{i\in A_{S,t}}Z_{i,t}
-\frac{\lambda_{S,t}^2}{2}
  \mathbf1^\top\Sigma_{|A_{S,t}|}\mathbf1
\right).
\]

When A_{S,t} is empty, set X_{S,t}=1. Under the declared Gaussian model,

\[
\begin{aligned}
\mathbb E[X_{S,t}\mid\mathcal F_{t-1}]
&=\exp\left(
\lambda_{S,t}\sum_{i\in A_{S,t}}\theta_{i,t}
\right)\\
&\leq1
\qquad(P\in H_S),
\end{aligned}
\]

because lambda is nonnegative and every active mean in the intersection null is nonpositive. Consequently,

\[
E_t^S=\prod_{u=1}^t X_{S,u}
\]

is a nonnegative test supermartingale under H_S in the common global filtration, hence an e-process.

The betting scale can adapt to past data. The implementation uses a bounded historical-mean policy:

\[
\lambda_{S,t}=
\min\{\lambda_{\max},
\lambda_0+\gamma(\bar Z_{S,t-1})_+\}.
\]

The current observation Z_{.,t} is not used to select lambda_{S,t}. This is a proof requirement, not a coding-style preference.

The implementation stores log wealth to avoid direct product overflow/underflow.

## 4. Predictable arrivals and exact projectivity

Assume active sets are nested:

\[
I_0\subseteq I_1\subseteq\cdots
\]

and each arrival decision at t is measurable with respect to F_{t-1}. Hypotheses not yet active have neutral factor one.

For any fixed S and any t, the current active intersection is S intersect I_t. At every earlier u<=t,

\[
(S\cap I_t)\cap I_u=S\cap I_u.
\]

Because both the predictable betting rule and the observation history are restricted to this active subset, every historical factor is identical. Therefore the entire historical path is identical:

\[
(E_u^S)_{u\leq t}
=
(E_u^{S\cap I_t})_{u\leq t}.
\]

This is exact neutral-padding projectivity, not merely equality of the current value.

The new implementation contains:
- a nested predictable-arrival fixture driven only by the preceding update;
- an optional arrival-policy receipt that is independently replayed from observations through t-1;
- an intersection-specific predictable lambda policy;
- one-step log-factor receipts and pre-update prefix digests;
- a separate verifier that reconstructs each factor and checks every historical projective equality.

The arrival-policy verifier can establish consistency with its declared deterministic rule. It cannot prove that the real system exposed every relevant covariate or that external timestamps are trustworthy. Those require signed source events and a trusted global sequencer.

## 5. What this does—and does not—escape

This construction does not claim validity under arbitrary dependence. It replaces the arbitrary-dependence assumption with a known conditional Gaussian joint model and a common global filtration. It therefore may avoid the fixed global coordinate-weight budget by constructing an intersection process directly from its joint conditional law.

The intersection-specific lambda policies can differ across S. They remain valid because each policy is predictable and each factor is conditionally bounded under its own intersection null. The projective guarantee follows because future dormant coordinates are absent from both the current active subset and the historical betting policy.

There is no free lunch: dependence modelling, covariance calibration, and filtration completeness become part of the statistical trust boundary. A misspecified joint law can invalidate every downstream guarantee.

A robustness extension should replace the known covariance penalty with an upper bound V_{S,t} satisfying

\[
\operatorname{Var}\left(
\sum_{i\in A_{S,t}} Z_{i,t}\mid\mathcal F_{t-1}
\right)
\leq V_{S,t}
\]

uniformly over the declared nuisance set, and use

\[
X_{S,t}=\exp\left(
\lambda_{S,t}\sum_i Z_{i,t}
-\tfrac12\lambda_{S,t}^2 V_{S,t}
\right).
\]

This is conservative but remains conditionally valid under a Gaussian or sub-Gaussian conditional-mgf upper bound. Deriving the tightest projectively consistent V_{S,t} is a separate theorem/optimization problem.

## 6. Filtration boundary

Every factor above is valid in one declared global filtration. If a component e-process is valid only in a local/coarser filtration, it cannot simply be averaged with evidence from a finer filtration. Choe and Ramdas (2026) establish adjust-then-combine methods that lift evidence across filtrations and explain the logarithmic recovery cost that can arise when validity is recovered in a finer filtration.

This track should benchmark:
1. direct construction in the global filtration;
2. local-filtration evidence lifted by a valid adjuster and then combined;
3. the cost in power and time-to-certification of the lift.

A digest cannot establish filtration measurability or repair leakage from same-round evidence.

## 7. Stopped-FDR and SupFDR

For a finite active family I_t, use the intersection-specific e-processes in the local closure constraints:

\[
\operatorname{FDP}_{S}(R_t)
\leq \alpha E_t^S
\qquad
\text{for every nonempty }S\subseteq I_t.
\]

Together with predictable nested activation, exact projectivity, and the valid global filtration, the dynamic e-closure argument yields stopped-FDR control.

Ordinary e-process validity does not automatically give SupFDR. For SupFDR, apply one valid running-maximum adjuster to every intersection path, with the required integral budget, and use the adjusted certificates in the all-time local closure constraints. Because the entire historical path is projective, the common adjuster preserves projectivity.

The affine O(n) separator/MILP in #9695 does not carry over automatically. These joint log-e-processes are intersection-specific and nonlinear. This prototype explicitly enumerates all nonempty finite intersections as a small-family reference oracle. A new separator must be derived for this family rather than reusing the affine formula.

### Preliminary component-e-process experiment

A local, independent Monte Carlo implementation of the checked-in benchmark design was run with 5,000 streams per cell, 60 global updates, five possible hypotheses, alpha=0.05, a predictable arrival rule (add the next hypothesis when the previous active-set mean exceeds zero), and a common alternative mean of 0.2.

Three methods were compared:
- **Joint nominal:** history-dependent shared-control betting using an assumed equicorrelation upper bound rho=0.5.
- **Joint robust:** the same betting rule using the conservative upper bound rho=0.8.
- **Global affine:** fixed coordinate e-processes with globally summable geometric weights; valid under arbitrary cross-coordinate dependence.

Empirical first-crossing rates by the end of the horizon:

| Actual rho | Mean | Joint nominal | Joint robust | Global affine |
|---:|---:|---:|---:|---:|
| 0.2 | 0.0 (null) | 0.30% | 0.08% | 0.26% |
| 0.2 | 0.2 | 28.46% | 13.14% | 14.06% |
| 0.5 | 0.0 (null) | 2.00% | 0.78% | 0.34% |
| 0.5 | 0.2 | 35.40% | 21.86% | 15.66% |
| 0.8 | 0.0 (null) | 4.30%* | 2.34% | 0.62% |
| 0.8 | 0.2 | 40.42%* | 28.40% | 16.22% |

*At actual rho=0.8, the nominal joint method's rho=0.5 bound is invalid. Its empirical 4.30% false-crossing rate is not evidence of validity. The robust rho=0.8 bound is the valid model-based comparator for that scenario. The affine method makes no joint-covariance assumption.

The observed alternative crossing-time medians among crossings were:
- rho=0.2: 43 updates (joint nominal), 45 (joint robust), 50 (affine);
- rho=0.5: 39, 40, and 49;
- rho=0.8: 34, 35, and 48, with nominal marked invalid.

This first result suggests a plausible power advantage from correctly specified joint betting, and quantifies the calibration penalty from a conservative covariance bound. It is **not** a full online multiple-testing comparison: it tests one intersection e-process crossing event, not a closed rejection set, FDR, or SupFDR. It does not compare against optimized AWAIRE or online-e-closure procedures. It is a preliminary Monte Carlo diagnostic; the conditional-mgf proof remains the validity argument.

The local experiment used an independent implementation of the same generator and factor equations. The checked-in runner \`validation/benchmark_history_dependent_projective_betting.py\` and its tests have been authored but have not yet been executed from the repository checkout. Reproduction command:

    PYTHONPATH=validation python validation/benchmark_history_dependent_projective_betting.py --replicates 5000 --horizon 60 --n-hypotheses 5 --seed 20261101 --output joint-vs-affine.json

## 8. Negative controls and benchmark design

The exact tests cover:
- shared-control covariance and sum-variance identities;
- conditional Gaussian one-step expectation under the composite intersection null;
- rejection of mean vectors outside the null cone;
- rejection of negative/nonfinite betting scales;
- current-round evidence not affecting the current betting scale;
- nested arrivals generated from prior observations only;
- neutral-padding and historical projectivity;
- independent replay of factors, prefixes, and arrival policy;
- tampered lambda receipt;
- closure checks for every finite active intersection;
- nonnested arrival and nonfinite observation rejection.

Next statistical comparisons must include:
- independent coordinates, where joint betting should have little advantage;
- shared-control Gaussian dependence, where the construction may exploit covariance;
- block/common-shock dependence and covariance misspecification;
- robust covariance upper bounds;
- sparse, dense, and heterogeneous alternatives;
- early, late, bursty, and data-dependent predictable arrivals;
- delayed/missing observations and asynchronous streams.

Compare against #9538/#9539 global affine projective mergers, #9648 finite-envelope arrivals, #9684 countable prior-weighted closure, #9695 countable global-projective closure, AWAIRE-style predictable mixtures, online e-closure/compound e-values, and Carefree multiple testing. Guarantee classes must match: report stopped-FDR and SupFDR separately.

Report intersection log-wealth, calibration/robustness penalty, time-to-certification, power, true actionable discoveries, false escalations, precision@B, recall@B, utility regret, capacity use, p50/p95 decision latency, memory, and verifier cost. Use 5,000 exploratory streams followed by 50,000 frozen confirmatory streams with preregistered seeds and null generators. Simulation is diagnostic and cannot substitute for the conditional-mgf proof.

## 9. Novelty boundary and references

The shared-control Gaussian factor is a standard exponential-supermartingale construction under a specified conditional Gaussian law. The projective arrival argument uses neutral padding and predictable betting. The combination may be a useful specialization of established joint e-process and dynamic-closure principles; it is not claimed as a new general e-merger or e-closure theorem.

Relevant work:
- de Heide (2026), *Dynamic e-closure for online hypotheses with any-time-valid evidence: closure principles and projective mergers*, https://arxiv.org/abs/2608.09927
- Choe and Ramdas (2026), *Combining Evidence Across Filtrations*, JRSS-B 88(4), 1492–1514, https://doi.org/10.1093/jrsssb/qkag058. Evidence in a coarser filtration cannot simply be averaged with evidence in a finer filtration; adjust-then-combine can recover validity, with a logarithmic cost in general.
- Xu, Fischer, and Ramdas (2026), *Improving online FDR procedures via online analogs of e-closure and compound e-values*, https://proceedings.mlr.press/v337/xu26a.html
- Tavyrikov, Goeman, and de Heide (2026), *Carefree multiple testing with e-processes*, https://doi.org/10.1214/26-EJS2546

No power/SOTA claim is justified until the direct joint betting construction is matched against the affine and online-closure baselines under the same data-generating process, filtration, FDR/SupFDR guarantee, and operational objective.
