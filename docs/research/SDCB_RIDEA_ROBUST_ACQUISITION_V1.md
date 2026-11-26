# Representative, Informative and De-Amplifying Acquisition for SDCB (R-IDeA) v1

**Status:** research protocol and standard-library-only formula reference
**Parent:** Robust and Pareto-Aware Scientific Decision Design v1
**Implementation:** [scripts/research/sdcb_ridea.rs](../../scripts/research/sdcb_ridea.rs)
**Schema:** [SDCB_RIDEA_EPISODE_V1.schema.json](SDCB_RIDEA_EPISODE_V1.schema.json)
**Example:** [SDCB_RIDEA_EPISODE_V1.example.json](SDCB_RIDEA_EPISODE_V1.example.json)
**Expected result:** [SDCB_RIDEA_EPISODE_V1.expected.json](SDCB_RIDEA_EPISODE_V1.expected.json)

## 1. Why R-IDeA belongs in SDCB

SDCB now distinguishes identifiability, decision value, finite-scenario robustness and Pareto trade-offs. One remaining failure mode is active-data distribution shift under model misspecification.

Ordinary Bayesian optimal experimental design (BOED) selects designs by expected information gain (EIG). But a model can be wrong: omitted variables, an under-expressive hypothesis family, heteroskedasticity, or an incorrect noise model can make the most informative designs unrepresentative of the deployment/test distribution. In that case the acquisition policy can over-sample unusual, high-disagreement regions and worsen generalization.

Tang, Sloman & Kaski (AISTATS 2026) formalize a third interaction in generalization error beyond misspecification bias and estimation bias: their signed cross-term captures whether estimation error reinforces or counteracts misspecification. They propose R-I (representative + informative) and R-IDeA (representative + informative + de-amplifying). The paper reports gains over informativeness-only BOED on its polynomial regression, source-localization and pharmacokinetic experiments, while also showing that hyperparameter choice matters and that no single setting dominates every condition.

SDCB should include R-I and R-IDeA as explicit external baselines. Our current finite-scenario RPSD method is complementary: it reasons about a declared ambiguity set and worst-case decision value, while R-IDeA targets generalization robustness under a train/test design-distribution shift and proxy-estimated error de-amplification.

## 2. Published acquisition equations

Let h_(t-1) be the selected design history and d_test the target/test design distribution. Tang et al. define a representative-information criterion:

\[ R\text{-}I(\xi_t)=EIG(\xi_t)\left(1-\lambda\frac{MMD(h_{t-1}\cup\{\xi_t\},d_{test})}{MMD(h_{t-1},d_{test})}\right). \]

The reference implementation uses empirical squared MMD with the Gaussian/RBF kernel described in the paper's numerical appendix, then takes its square root for the MMD ratio:

\[ k(x,y)=\exp\!\left(-\frac{\|x-y\|^2}{2\sigma^2}\right), \]

\[ MMD^2(P,Q)=\frac{1}{m^2}\sum_{i,i'}k(p_i,p_{i'})+\frac{1}{n^2}\sum_{j,j'}k(q_j,q_{j'})-\frac{2}{mn}\sum_{i,j}k(p_i,q_j). \]

R-IDeA multiplies R-I by a sigmoid de-amplification weight:

\[ DeA(\xi_t)=\operatorname{sigmoid}\!\left(\frac{|\hat f^{(n)}(\xi_t)-g(\xi_t)|-\tau}{\kappa}\right), \]

\[ R\text{-}IDeA(\xi_t)=R\text{-}I(\xi_t)\cdot DeA(\xi_t). \]

Here \hat f^(n) is the current fitted predictor and g is a proxy intended to approximate the best-in-class predictor while retaining disagreement with the current model. The original paper trains g with a data-fit term plus a hinge penalty that encourages a minimum separation from \hat f^(n). It does not assume access to the true data-generating function during acquisition.

## 3. What the reference implementation covers

The Rust file implements the paper's scoring layer for a finite candidate pool:

- biased empirical MMD² with an RBF kernel and caller-supplied bandwidth;
- MMD before/after adding the candidate to selected-design history;
- R-I robust ratio with caller-supplied EIG and lambda;
- DeA sigmoid from supplied current-model and proxy predictions;
- R-IDeA score and an explicitly separate score-per-cost extension;
- deterministic candidate ranking, tie-breaking, and abstention if all scores are non-positive;
- fail-closed validation for malformed samples, dimensions, non-finite predictions, invalid scales, duplicate IDs and zero baseline MMD.

The reference does **not** train g, infer EIG, construct a posterior, or assess downstream decision value. EIG is supplied by a separate method. The proxy is also supplied by an upstream model. This keeps the equation reference small and makes the missing pieces explicit.

### Important boundaries

1. **R-IDeA is not EVSI.** EIG measures information about a parameter/hypothesis. EVSI measures expected improvement in a declared decision. Report them separately; a high R-IDeA score does not prove an experiment changes the best action.
2. **R-IDeA is not a finite ambiguity-set maximin guarantee.** RPSD evaluates the worst scenario in a declared finite set. R-IDeA targets generalization under design-distribution shift and model misspecification using a proxy model.
3. **R-IDeA does not certify truth.** The proxy g can be wrong; the reference only calculates the prescribed acquisition score.
4. **MMD denominator.** If baseline MMD(history, target) is zero, the ratio is undefined. The reference returns `ZeroBaselineMmd`; it does not silently fall back to EIG.
5. **Kernel and hyperparameters.** RBF bandwidth, representativeness weight lambda, threshold tau and sigmoid scale kappa affect rankings. The code exposes them and does not claim a universally correct setting. It restricts lambda to [0,1] as a conservative reference configuration; this is an implementation guard, not a claim that the paper's equation mathematically requires that interval.
6. **Cost.** The original R-I/R-IDeA formulas do not divide by cost. The reference additionally reports R-IDeA per cost unit as a separate project ranking view; raw R-IDeA remains available.

## 4. Frozen counterexample

Target design samples are `{0, 1, 2}` and the existing history is `{0, 2}` in one dimension, with RBF sigma 1.0. Baseline MMD is 0.1984961019.

Two candidate designs:

| Candidate | EIG | Design | Cost | |f-hat − g| |
|---|---:|---:|---:|---:|
| representative/informative/de-amplifying | 0.8 | 1 | 1 | 1.5 |
| high-EIG extrapolation | 1.2 | 4 | 1 | 0.1 |

With lambda = 0.5, tau = 0.5 and kappa = 0.5:

- Candidate 1 exactly closes the empirical MMD gap for this fixture: MMD after = 0; R-I = 0.8; DeA = 0.880797; R-IDeA = 0.704638.
- Candidate 2 increases MMD to 0.468779; its robust ratio is −0.180826; R-I = −0.216991; DeA = 0.310026; R-IDeA = −0.067273.

The first candidate wins despite the second having 50% higher raw EIG. This is a hand-checkable illustration of the published acquisition factors, not a scientific benchmark result or evidence that our proxy accurately identifies de-amplifying regions.

## 5. Benchmark comparisons and what to aim for

Use the original R-IDeA release/code as the direct baseline wherever possible. Do not claim an improvement by changing its metric or data split.

| Comparator | What it measures | SDCB extension |
|---|---|---|
| Random / space-filling | Representativeness without targeted information | Establish floor and target coverage |
| Traditional BOED / EIG | Informativeness | Quantify generalization failures from over-concentrated designs |
| R-I | EIG + MMD representativeness | Add decision regret and evidence receipts |
| R-IDeA | EIG + representativeness + proxy de-amplification | Add task-relative identifiability, EVSI and evidence-gated updates |
| RPSD maximin | Robust value across explicit scenarios | Compare with R-IDeA under a common misspecification suite; keep different assumptions clear |
| DSSI EVSI / exact non-myopic | Decision value and adaptive policy regret | Ensure informative data also changes decisions when it should |
| Pareto / MOBO | Multi-objective trade-offs | Preserve hard safety floors and cost/resource Pareto frontiers |

Report at least:

- held-out test-distribution MSE or domain-appropriate predictive loss;
- empirical MMD and density-ratio/covariate-shift diagnostics;
- EIG, R-I, DeA and R-IDeA score distributions;
- decision regret and EVSI as separate axes;
- worst-regime and OOD performance;
- sample efficiency, experiment cost, latency, compute and energy;
- calibration and uncertainty coverage;
- hard-constraint violation rate and all evidence-integrity failures.

Project targets (not published standards or achieved results):

- Reproduce the published R-I/R-IDeA formula and small analytic fixtures exactly.
- Under frozen misspecification families, beat EIG-only and R-I on held-out predictive generalization without material loss in well-specified settings.
- Maintain decision regret no worse than the best matched EVSI policy within a preregistered non-inferiority margin.
- Keep hard safety-constraint violations at zero on frozen controls.
- Evaluate lambda/tau/kappa and RBF bandwidth on validation only; freeze before hidden test.
- Use paired seeds and task/family-level uncertainty intervals; do not treat adjacent design points as independent samples.
- Independently reproduce on a second implementation before making external performance claims.

## 6. New research opportunity: decision-relevant R-IDeA

A promising future hypothesis is to evaluate R-IDeA and EVSI jointly rather than multiply them without justification:

- R-IDeA addresses whether a selected experiment will produce representative, informative and de-amplifying training data.
- EVSI addresses whether its possible outcomes can improve a declared action under the posterior.
- Robustness addresses whether those benefits survive plausible alternative models.

Treat these as orthogonal criteria first and publish their joint Pareto frontier. Only propose a combined scalar acquisition after an independent study shows that a specific composition improves held-out decision regret and predictive generalization across pre-registered tasks.

## 7. References

1. Tang, R., Sloman, S. J. & Kaski, S. (2026). Representative, Informative, and De-Amplifying: Requirements for Robust Bayesian Active Learning under Model Misspecification. AISTATS 2026, PMLR 300, 3016–3024. https://proceedings.mlr.press/v300/tang26d.html
2. Paper PDF and exact acquisition equations: https://openreview.net/pdf/13def9e0a72acfb9ce90afddd501d3585625477f.pdf
3. Code referenced by the paper: https://github.com/TrbingWY/robust-boed
4. Aqeeli, Leelawat & Shorthouse (2026). Novelty-aware evolutionary Bayesian optimisation for multi-objective discovery science. Digital Discovery 5, 2684–2700. https://doi.org/10.1039/D6DD00134C

## 8. Claim boundary

The Rust source contains 16 authored unit tests, but it has not been compiled or executed in this environment. Its proxy predictions and EIG are caller-supplied. It does not train models, run an upstream R-IDeA reproduction, test Symthaea HDC/CfC, authenticate physical observations, or demonstrate SOTA performance.