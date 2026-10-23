# EXD-001 — Model-Specific Extra-Dimensional Physics Tests

**Status:** research contract / proposed  
**Tracker:** [Symthaea #7854](https://github.com/Luminous-Dynamics/symthaea/issues/7854)  
**Version:** 0.1-draft  
**Evidence class:** specification only; no physical model has been fitted or qualified by this document.

## 1. Objective

Build a falsifiable evaluation program for hypotheses involving additional physical spacetime dimensions. The unit of evaluation is **a named physical model plus a declared experiment and observable**, not a generic “higher-dimensional signature.”

The core question is:

> Given a model with explicit assumptions, parameter bounds and predicted measurements, can a frozen analysis distinguish it from the strongest conventional/null explanations on independent observations, while achieving calibrated false-positive rates and correct uncertainty coverage?

This is a distinct task from:
- recognizing a mathematical state space with four or more coordinates;
- discovering an unusual radio/astronomy candidate;
- detecting information-bearing or adaptive behavior;
- establishing a living or intelligent entity.

None of those is a substitute for model-specific physical evidence.

## 2. Paradigm shift: a model must risk being wrong

Replace “find patterns that look higher-dimensional” with a versioned **observable contract**:

```text
named model + assumptions + parameter domain + existing constraints
  -> predicted observables with units and uncertainty
  -> experimental / instrument response
  -> preregistered nulls and competing models
  -> blind injections + holdout scoring + nuisance controls
  -> calibrated support / exclusion / unresolved disposition
  -> independent replication or external analysis
```

Every claim-bearing model must make a prediction before the target holdout is opened. A flexible model that fits the data better in-sample is not sufficient. A model whose effect lies outside the experiment's sensitivity envelope cannot be “excluded” by a null result, and a parameter region not excluded is not positively supported.

## 3. Compose existing project responsibilities

- **Symthaea:** candidate ranking, explicit hypothesis comparison, uncertainty analysis, and follow-up experiment prioritization. Its mathematical N-D competence work remains a separate evidence class.
- **Symtropy:** declared forward models, experiment/sensor response simulation, deterministic numerical replay, and nuisance perturbation studies within a documented validity envelope.
- **Mycelix:** reuse existing scientific evidence typing, investigation capsules, and capture/provenance receipts. Preserve data/model/evaluator identity; do not treat shared evidence storage as a physics verdict.
- **OPEN-OBS-001 #7835:** handles unusual-observation candidate discovery and candidate-vs-control evaluation. A candidate from that lane enters EXD-001 only if a named model predicts a discriminating observable.
- **HYPERSPACE-001 #2023** and wave-discovery [PR #7282](https://github.com/Luminous-Dynamics/symthaea/pull/7282): remain mathematical/synthetic competence work, not physical detection.

Do not create a second general anomaly detector, evidence ledger, state-space framework, or universal dimensionality score.

## 4. Initial model families and observables

These are candidate experimental tracks, not claims that any effect has been observed.

### 4.1 Large flat extra dimensions (ADD-like)

Potential observable contracts may include:
- deviations from the inverse-square gravitational law at short distances;
- collider distributions involving missing transverse momentum / graviton-emission signatures;
- other published observables only where the model-to-observable transfer calculation and experiment response are documented.

Bind the number and geometry of dimensions, compactification scale, coupling assumptions, parameter ranges and exact published constraints. Do not combine tabletop and collider limits as if they were the same measurement.

### 4.2 Warped extra dimensions (Randall–Sundrum-like)

Possible observables include predicted Kaluza–Klein resonances or continuum effects in named invariant-mass / scattering channels. Bind the model variant, coupling assumptions, width, decay channels, selection cuts, detector response and validity range. A generic spectral bump is not a Randall–Sundrum signal unless the predicted relationships and competing explanations are tested.

### 4.3 Compactified Kaluza–Klein families

Version the exact compactification, resulting spectrum, field content, couplings and selection rules. “KK-like” is not one model and cannot serve as a post-hoc label for any excess.

### 4.4 Astrophysical and cosmological constraints

Use only model-specific channels with published prediction/transfer calculations. Record astrophysical priors and environment assumptions separately from laboratory/collider evidence. Uncertain astrophysical inputs must broaden the inference or reduce applicability; they must not be fixed after the result is seen.

### 4.5 Unspecified higher-dimensional entity

An unspecified “higher-dimensional being” hypothesis is **NotTestable / Underspecified** until it provides:
- a mechanism connecting the proposed entity to measurable observables;
- predicted signal forms and parameter ranges;
- expected coupling / interaction with the instruments and environment;
- a falsifiable difference from conventional sources and artifacts.

Do not upgrade it from NotTestable because a general anomaly model flags an observation.

## 5. Versioned observable contract

Each model/experiment profile should include, at minimum:

1. **Identity:** immutable model ID, version, source references, authoring date, code revision, profile digest and declared applicability.
2. **Theory:** equations / parameterization, units, priors or allowed parameter bounds, assumptions, approximations and excluded regimes.
3. **Observable:** named raw or derived measurements, units, expected distribution, uncertainty, parameter dependence and predicted effect sizes.
4. **Response model:** detector acceptance/efficiency, calibration, resolution, background process, selection cuts and systematic nuisance parameters.
5. **Nulls:** Standard Model / conventional reference, detector artifact models, relevant alternative new-physics models and the rationale for which nulls are included.
6. **Sensitivity envelope:** declared detectable effect/parameter regions, minimum useful power, out-of-domain rules and how low sensitivity is reported.
7. **Preregistration:** primary statistic, secondary analyses, thresholds, scans, multiple-testing correction, holdout identity, exclusions and stop rules.
8. **Evidence lineage:** source-data identity, provenance group, calibration and response versions, model binary/code digest, fit output, evaluator version and replication links.

An independent checker must be able to re-run the declared response and metric calculation from the frozen profile and data, or identify exactly which missing artifact prevents reproduction.

## 6. Evaluation campaign

### Stage A — mathematical and response correctness

- Compare the forward model against analytic/reference solutions where tractable.
- Verify units, conservation laws, boundary conditions, limiting cases and numerical convergence.
- Record solver non-convergence and out-of-domain outputs; never silently substitute plausible defaults.
- Validate detector/experiment response on calibration data independent of the claim-bearing holdout.

### Stage B — synthetic null calibration

- Generate null data from the preregistered conventional and known-artifact models.
- Include calibration shifts, background mismodelling, finite sample size, selection effects, and nuisance-parameter extremes.
- Measure false-positive rate with uncertainty bounds after the **entire declared scan**, including look-elsewhere / multiple-testing correction.
- Test interval coverage and parameter recovery across seeded truth points.

### Stage C — injection and sensitivity

- Inject synthetic predicted effects over the declared parameter range, including boundary cases and effect sizes below sensitivity.
- Report detection probability / power by effect size and parameter region.
- Publish the smallest effect reliably recoverable by the exact profile, with confidence intervals.
- Verify that effects outside sensitivity are labelled “not tested / insufficient sensitivity,” not “excluded.”

### Stage D — held-out real data

- Freeze data splits, model/profile version, priors, primary metric and exclusions before revealing the holdout.
- Compare the named model against the strongest relevant nulls and alternatives under identical acquisition, calibration and selection conditions.
- Preserve every failed fit, null result, exclusion and non-convergence.
- Treat independent experiments as independent only where shared systematics, selection, data and model assumptions justify it.
- Seek external analysis or independent replication for any positive claim.

## 7. Required metrics

Report per model/experiment and parameter region, never only as a single composite score:

| Metric | What it establishes |
|---|---|
| Null false-positive rate and confidence bound | Whether the full search procedure overclaims under declared nulls |
| Injection recovery / power by effect size | Where the experiment can actually detect the predicted effect |
| Confidence/credible interval coverage | Whether uncertainty statements are calibrated under tested conditions |
| Held-out predictive score / likelihood ratio | Whether predictions outperform named alternatives out of sample |
| Prior and nuisance sensitivity | Whether support depends strongly on reasonable prior/model choices |
| Parameter bias / interval width | Whether estimates recover injected truth and with what precision |
| Multiple-testing correction | Whether significance survives the entire preregistered search |
| Model-selection stability | Whether nearby parameterizations or alternative baselines reverse the conclusion |
| Reproducibility | Whether an independent implementation can reproduce predictions and statistics |
| Cost / latency / compute | Operational expense of evaluating the declared sensitivity range |

Use experiment-specific alpha, power and coverage targets selected before holdout analysis and grounded in experimental convention or explicit error-cost analysis. Do not impose an arbitrary universal sensitivity percentage across gravity tests, collider searches and astrophysical observations.

## 8. Outcomes must remain distinct

Use a versioned disposition such as:
- **Supported within declared model/profile:** preregistered evidence favors the model over named alternatives on the independent data and survives controls.
- **Constrained / excluded in a parameter region:** the experiment demonstrably had adequate power there and the result passes the declared exclusion calculation.
- **Not excluded:** current data do not rule out the model; no positive evidence is implied.
- **Insufficient sensitivity:** the declared experiment could not adjudicate that region.
- **Inconclusive / model-nondiscriminating:** available observations do not distinguish the model from competitors.
- **Rejected under declared profile:** predictions fail the preregistered test in the applicability domain.
- **NotTestable / Underspecified:** observable or falsifiable mechanism is not sufficiently defined.

“Anomaly candidate,” “model fit improved,” “not excluded,” “extra-dimensional model supported,” and “intelligence supported” must never be synonyms.

## 9. Frozen negative controls

- Known conventional nulls are recovered with calibrated type-I error within uncertainty bounds.
- Detector/calibration shifts and background mismodelling cause visible changes; they do not silently become extra-dimensional support.
- A synthetic R4 escape path, HDC embedding dimension, visualization, Φ-like metric or network topology never counts as a physical observable.
- Adding unconstrained parameters cannot win solely by fitting in-sample data better.
- Post-hoc plot-window changes, parameter bounds, priors, thresholds, data exclusions or new primary statistics invalidate confirmatory status.
- A null result outside sensitivity cannot be reported as exclusion.
- Multiple analyses of the same underlying acquisition are not independent replications.
- A physics model pass alone cannot imply life, agency or a being.

## 10. Research basis

- **Particle Data Group, Review of Particle Physics 2025, “Extra Dimensions”:** describes model-dependent constraints from sub-millimetre gravity experiments, collider searches and astrophysical/cosmological considerations. Start from the current published review, then add newer experiment-specific sources only for the chosen profile: https://pdg.lbl.gov/2025/reviews/rpp2025-rev-extra-dimensions.pdf
- **NASA UAP Independent Study (2023):** emphasizes calibration, metadata and multiple well-calibrated observations; unusual data require reliable instrument and context records: https://science.nasa.gov/uap/
- **Pardo et al. (2025), Breakthrough Listen anomaly detection:** large-scale candidate ranking did not yield a candidate that survived basic scrutiny. It motivates the separation of triage and confirmation, but is not an extra-dimensional-physics experiment: https://arxiv.org/abs/2505.03927

## 11. Claim ceiling

A qualified EXD-001 run establishes only that the specified model/experiment pair has the reported calibration, sensitivity, null-control, predictive and reproduction properties for the exact profile and parameter envelope. Synthetic success establishes algorithm/model behavior under those assumptions.

Physical extra dimensions require model-specific, independent, reproducible observational support. A confirmed extra-dimensional model would still not establish intelligent occupants, their location or their nature; agency and “being” claims need their own discriminating observables and evidence.
