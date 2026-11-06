# SPECTRA Source–Target Transport and Sampling-Bias Qualification v0.1

**Status:** research specification; no target-population guarantee implemented.  
**Issue:** #10687.  
**Related work:** #10618 scenario-bank assurance; #10639 joint resource allocation; #10641 joint data/simulation allocation; #10601 locked allocator comparison.

## 1. Problem statement

The current SPECTRA ranking certificate is conditional on a frozen scenario bank and declared scenario weights. That is appropriate for the existing finite-bank decision problem. It does not automatically establish that the bank or its weights represent a deployment population.

Let \(P_S\) be the **source distribution** of sampled contexts and \(P_T\) the **target distribution** for which a design decision is intended. For action \(a\), define the target estimand

\[
\mu_T(a)=\mathbb{E}_{X\sim P_T}[f(a,X)].
\]

A precise Monte Carlo bound conditional on the source distribution can support the wrong target decision when \(P_S\ne P_T\). Support mismatch can make target transport impossible, not merely noisy.

## 2. Keep the evidence objects separate

SPECTRA must represent the following as separate, versioned objects.

1. **Source distribution:** where observed or simulated input contexts were sampled, with acquisition/sampling design and propensity evidence.
2. **Target population:** the declared population or deployment context mix and its versioned definition.
3. **Target prevalence:** empirical or externally measured population frequencies. These may change as representative data improve.
4. **Normative importance:** ethical, policy or mission-level weights. These do not automatically change when observed prevalence changes.
5. **Weight/transport uncertainty:** the set of plausible ways to transport source observations to the target population.
6. **Monte Carlo score uncertainty:** conditional uncertainty in simulated score estimates.
7. **Scenario support/coverage:** evidence that target-relevant regions are represented in the observed and simulation banks.
8. **Model adequacy:** evidence that the measurement/simulator is adequate over the declared scope.

Never infer normative importance from prevalence, or prevalence from adaptive acquisition frequency.

## 3. Candidate transport formulations

Do not choose one formulation until the estimand and available evidence are explicit.

### 3.1 Known density ratio

If the Radon–Nikodym derivative exists and is known,

\[
w(x)=\frac{dP_T}{dP_S}(x),\qquad
\mu_T(a)=\mathbb{E}_{P_S}[w(X)f(a,X)].
\]

A finite-sample bound requires a declared bound or tail condition on \(w\), not merely a point estimate. When \(0\le w\le W\) and \(0\le f\le1\), the weighted observations are bounded by \(W\), but the exact confidence construction must still handle any estimated normalization or propensity.

### 3.2 Moment-constrained target ambiguity

When target moments are known but the target density is not, study an ambiguity set of density ratios:

\[
\mathcal W(m)=\{w:\;0\le w\le W,\;
\mathbb{E}_{P_S}[w]=1,\;
\mathbb{E}_{P_S}[w g(X)]\in\mathcal M\}.
\]

Here \(g(X)\) is a frozen vector of context features and \(\mathcal M\) is an externally justified moment region, not a normatively chosen weight vector.

For each action, define robust target utility through an optimization over \(\mathcal W\), for example

\[
\underline\mu_T(a)=\inf_{w\in\mathcal W}
\mathbb E_{P_S}[w f(a,X)].
\]

The corresponding upper bound uses a supremum. Empirical finite-sample versions need a joint confidence region for the observed score/context moments and the target moment inputs. Plug-in LP solutions alone are not confidence bounds.

### 3.3 Sensitivity-bounded transport

If a full density ratio is not identified, consider an explicitly declared sensitivity class (bounded likelihood ratio, f-divergence / KL radius, or total-variation radius). Report the sensitivity parameter and its source. Do not select it after observing which value preserves the preferred action.

### 3.4 Adaptive context selection / logged propensities

If contexts are actively sampled, the source sampling design must retain conditional inclusion probabilities or another valid design description. Unknown or zero-probability regions cannot be repaired by post-hoc importance weighting. The transport estimator and its confidence construction must be valid under the actual adaptive selection protocol.

## 4. Non-negotiable support and overlap gate

If the target distribution has mass in regions outside source support, density-ratio transport is not identified from source data alone.

Required diagnostics include:
- target/source support overlap by declared context stratum;
- minimum estimated or certified inclusion probability;
- maximum weight / effective sample size;
- proportion of target mass with no source support;
- sensitivity to truncation/clipping;
- target moment feasibility and uncertainty;
- hidden target strata and rare-context stress tests.

Weight clipping changes the estimand or introduces bias. Report that bias/sensitivity explicitly. Do not silently clip extreme weights and continue to call the output an unbiased target estimate.

No support, no certificate: output Blocked or NotEvaluable unless an additional target-sampling or sensitivity argument covers the missing region.

## 5. Composing uncertainty without double counting

The target decision may depend on both sampled-context uncertainty and simulation uncertainty. The composition needs its own proof.

Possible architectures to investigate:
- a single joint confidence region over target moments, source empirical moments and conditional simulation scores;
- a conservative, preregistered alpha split between context transport and Monte Carlo score bounds;
- robust optimization over a confidence region for the target distribution and simultaneous action-score intervals.

Do not multiply marginal confidence levels as if they were independent without a valid dependence argument. Do not append simulation results from heterogeneous input generator versions to the existing IID per-cell stream. Every stage/input snapshot requires provenance and a qualified estimator.

The receipt must bind target definition, source sampling design, propensity model/version, target moment/sensitivity set, support diagnostics, input data version, score generator, action/scenario scopes and confidence allocation.

## 6. Benchmark matrix

Construct known-truth target/source distributions with:
- source = target control;
- covariate shift with full overlap;
- high weight variance but full support;
- near-positivity / near-zero inclusion probabilities;
- exact support holes and target-only rare strata;
- unknown target weights with known first/second moments;
- selection bias from adaptive context acquisition;
- unobserved confounding / missing-not-at-random negative controls;
- input generator changes across collection stages;
- changing target prevalence while normative importance stays fixed.

Compare:
- no correction / source-only ranking;
- known-ratio importance weighting;
- self-normalized importance weighting;
- clipping plus explicit sensitivity analysis;
- moment-constrained ambiguity-set robust bounds;
- target-context acquisition / source–target joint allocation;
- hidden-target oracle, reported only as an unattainable reference.

Relevant matched-assumption literature:
- Cheshire & Clémençon (AAAI 2026), Best Arm Identification with Biased Contexts, which studies adaptive selection bias with known target moments and unknown weighting function.
- Yang, Tan & Cheung (UAI 2025), Best Arm Identification with Possibly Biased Offline Data, including an impossibility result without a prior bias bound.
- Wang & Zhou (2025), simultaneous input-data and simulation budget allocation for evolving input distributions.

## 7. Metrics and proposed acceptance criteria

Report per regime:
- target robust regret and probability of correct target selection;
- false DecisionSupportedForDeclaredRegime rate;
- coverage of target utility / target weight ambiguity bounds;
- target-support hole mass and false acceptance under support-negative controls;
- bias, interval width and effective sample size;
- context-data collection cost, simulation cost and total cost-to-certificate;
- sensitivity curves over the declared transport ambiguity radius;
- divergence between empirical prevalence and frozen normative weights.

Mandatory negative controls:
- zero support for a target stratum must never pass a source-only transport gate;
- absent bias bound must not yield a target-population certificate;
- unknown sampling propensity must not be silently treated as uniform;
- prevalence changes must not mutate normative importance weights;
- incompatible generator versions must not be appended into one IID confidence sequence.

## 8. Claim boundary

This specification does not establish target-population certification. It identifies the estimands, assumptions, failure modes and benchmark programme. No production target-transport gate should be implemented until the finite-sample uncertainty composition and support conditions have been independently reviewed.

## References

- Cheshire, J. & Clémençon, S. (2026). Best Arm Identification with Biased Contexts. AAAI-26, 20536–20543. https://doi.org/10.1609/aaai.v40i25.39189
- Yang, L., Tan, V. Y. F. & Cheung, W. C. (2025). Best Arm Identification with Possibly Biased Offline Data. UAI 2025, PMLR 286:4715–4730. https://proceedings.mlr.press/v286/yang25b.html
- Wang, Y. & Zhou, E. (2025). Data-Driven Ranking and Selection With Simultaneous Input Data Collection and Simulation. Naval Research Logistics 72(8):1170–1183. https://doi.org/10.1002/nav.22275
