# CROP-INTEL-001 — Active experiment selection

**Status:** research design with a finite-hypothesis expected-information-gain reference in Rust; not compiled or executed.

## Motivation

A conventional anomaly detector returns a score. A research system should instead recommend the next observation that could distinguish competing explanations.

The action is not always "collect another image". It may be:
- obtain a calibrated image from a different viewing angle;
- map crop-flow direction along a boundary;
- measure crop-stage and soil-moisture covariates;
- locate an independently documented creation timeline;
- inspect matched plant specimens under blinded labels;
- run a controlled mechanical-flattening experiment;
- decline to sample because permission, power or custody requirements are unmet.

## 1. Keep hypothesis dimensions separate

Do not make one mutually exclusive list called "origin". Maintain distinct hypothesis variables for:

- **authorship:** documented human, probable human, unknown, conflicting;
- **physical mechanism:** mechanical, wind/rain/lodging, other specified physical model, mixed, unresolved;
- **plant response:** absent, present, inconclusive, not sampled;
- **information structure:** observed, regular, code-like, payload, not assessable;
- **evidence process:** provenance/rights/custody/calibration quality.

A case can be human-made and still have interesting geometry. A code-like pattern does not settle authorship. A measured biological difference does not prove an unusual energy source.

## 2. Candidate action model

For each proposed action a, define:
- predicted observation distribution p(y | H_i, a, D) under each live hypothesis H_i and current evidence D;
- time, money, permission and operational cost C(a);
- expected sample size and measurement precision;
- safety, privacy, rights and landowner constraints;
- smallest scientifically relevant effect and statistical power;
- probability the action will return unusable or missing evidence.

Retain a separate model-uncertainty hypothesis so that the system does not force the true explanation to be one of the currently enumerated models.

## 3. Expected information gain

For a finite hypothesis set, a first reference criterion is expected reduction in posterior entropy:

`EIG(a) = H(H | D) - E_{y \sim p(y|a,D)}[H(H | D,y,a)]`

Equivalently, this is the expected Kullback–Leibler divergence between the updated and current posterior over hypotheses, under the current predictive model.

This quantity is only as reliable as the models and priors supplied to it. Therefore report:
- sensitivity across plausible prior ranges;
- prior predictive checks;
- model-misspecification / unknown-model mass;
- predictive interval width and calibration;
- the chance of a null/inconclusive observation;
- cost and sample power separately.

Do not collapse all of these into a single "best experiment" score without retaining the component vector.

## 3A. Goal-oriented design, identifiability and model discrepancy

Parameter information is not always the scientific objective. A parameter posterior can become narrower while the downstream prediction or mechanism decision remains unchanged. Define the quantity of interest (QoI) first: for example, the held-out stem-deflection profile, the probability of root versus stem failure under a specified load, or the predictive difference between two physical mechanism families.

Compare actions by expected change in the predictive distribution of the QoI or by expected discrimination among the live mechanism hypotheses—not only by entropy reduction over all model parameters. Record both parameter-identifiability diagnostics (profile likelihood, local sensitivity/Fisher-information rank, posterior correlations) and goal-oriented predictive uncertainty. If two parameters are not separately identifiable but their combination predicts the QoI accurately, preserve that distinction.

Add an explicit model-discrepancy term. Sensor noise, biological variability and structural model error are different quantities; do not inflate one to conceal another. Use a model-inadequacy hypothesis when all registered beam models fail posterior-predictive checks. For the piecewise beam reference, test whether the predicted curve residuals show systematic curvature, hysteresis, rate dependence or load-direction effects before adding complexity.

A staged fidelity ladder is preferable to jumping directly to an expensive simulator:
1. analytical uniform beam;
2. piecewise-EI beam with calibrated root springs;
3. nonlinear/finite-element beam with measured cross-sections and material behaviour;
4. root–soil interaction and canopy coupling;
5. instrumented controlled-plot data.

Every fidelity upgrade must demonstrate better held-out predictive accuracy or uncertainty calibration at a matched data and compute budget. More parameters alone are not progress.

This is consistent with recent work on goal-oriented Bayesian design for nonlinear models, which targets information about predictive quantities of interest rather than parameters alone ([Zhong et al., SIAM/ASA JUQ, 2026](https://doi.org/10.1137/24M1649344)), and new work on experimental designs for practical parameter identifiability and model discrimination ([Mathematical Biosciences, 2026](https://doi.org/10.1016/j.mbs.2026.109710)).

## 3B. Rust reference implementation

The dependency-free lab exposes `expected_information_gain_bits(priors, conditional_outcome_probabilities)`. It calculates the mutual information between the discrete mechanism hypothesis and the outcome of one proposed action:

`I(H;Y|a) = H(H) - Σ_y P(y|a) H(H|y,a)`

The function validates that the prior and every conditional row are finite, nonnegative probability distributions that sum to one, that there is one row per hypothesis, and that each row uses the same outcome ordering. It rejects malformed matrices rather than silently normalizing them. Nine known-answer / invalid-input tests are authored.

The utility assumes that the supplied predictive distributions are already defensible. It does not estimate them, include cost, model measurement error, or enforce safety/permission gates. Compute information gain for eligible actions only after hard gates; retain cost, expected power, missingness, and model-discrepancy sensitivity as separate outputs. A model-unknown hypothesis should be included only with a defensible predictive distribution. EIG must be stress-tested against plausible priors and alternative predictive models before it guides real experiment selection.

## 4. Cost and feasibility

A practical decision record should contain:
- raw EIG and units (bits or nats);
- expected and worst-case cost;
- estimated power at the smallest effect of interest;
- predicted probability of a result that changes the leading hypothesis;
- expected time to result;
- permission/rights status;
- physical safety and ethical review;
- whether the action is reversible and non-destructive;
- required custody / calibration procedures.

An infeasible or unauthorized experiment is not made acceptable by high EIG. Enforce permission, safety and rights as hard constraints before ranking.

## 4A. Mechanical-model experiment ladder

For plant mechanics, compare candidate actions that identify parameters and distinguish model structures:
- multiple loading locations to separate a uniform EI profile from a rootward/upper-stem stiffness gradient;
- bending in two orthogonal directions to identify cross-section anisotropy;
- direct base translation and base rotation measurements to estimate root translational and rotational compliance;
- repeat loading/unloading cycles to detect hysteresis and departures from linear elasticity;
- controlled loading rates to quantify rate effects;
- cross-section sampling along the internode to test whether an EI profile is supported by direct measurements.

A single tip-force/tip-displacement observation cannot generally identify EI profile and root stiffness independently. The experiment registry must state this identifiability limit, and the model must not report unique parameters when multiple parameter sets predict the same response.

## 5. Worked example: centre-relative crop-flow discrimination

Suppose two viable models predict a similar plan-view crop boundary:
- H1: an outward radial or sectoral mechanical flattening process with a declared entry/exit path;
- H2: a specified wind-driven lodging model with a direction field that varies with terrain and wind exposure.

The formation outline alone may not discriminate them. A calibrated sample of individual stem-axis directions at multiple radial distances can estimate:
- local nearest-neighbour directional alignment;
- centre-relative radial alignment (outward/inward);
- centre-relative tangential alignment (clockwise/counter-clockwise);
- bend magnitude and breakage distribution.

Before sampling:
1. survey the field centre and boundary with independent spatial checkpoints;
2. freeze transect locations and the coordinate-frame/angular sign convention;
3. register the number of plants and primary directional outcomes;
4. calculate power with within-transect and within-formation clustering;
5. blind the measurement and laboratory assessors;
6. predeclare how uncertainty in the centre propagates into radial/tangential scores;
7. specify predicted field maps and falsification outcomes under each model.

A single centre-relative scalar is insufficient when the field is multimodal or the centre is poorly determined. Preserve all per-plant vectors and report directional distributions by zone.

### Related boundary-orientation example

Suppose two currently viable mechanisms predict similar plan-view geometry:
- H1: a mechanical flattening process with a particular entry/exit path;
- H2: a specified wind-driven lodging model with a spatially varying flow field.

The outline alone may not discriminate them. A prospective transect of stem bend azimuths, breakage, and standing/flattened transitions across the boundary could produce markedly different predictive distributions.

Before sampling:
1. register the transect locations and number of plants;
2. calculate required sample size from the smallest effect and within-field clustering;
3. define collection and blinding;
4. record weather, crop stage, cultivar, terrain and moisture;
5. state which predicted patterns would falsify each mechanism.

If neither hypothesis predicts a distinguishable result under the achievable measurement precision, Symthaea should say so and recommend no such experiment.

## 6. Sequential updating safeguards

- Freeze the next measurement and scoring rule before observing the result.
- Preserve original raw data and instrument metadata.
- Do not reuse controls as held-out validation.
- Do not repeatedly inspect the same stream and stop when a preferred result appears unless a qualified sequential procedure is used.
- Retain contradictory observations and failed predictions.
- Update separate causal dimensions independently.
- Keep the prior, likelihood and posterior versions in the evidence receipt.
- When all declared models have poor predictive fit, raise model inadequacy rather than force posterior mass onto the least-wrong hypothesis.

## 7. Evaluation benchmark

Compare experiment-selection policies under matched simulation worlds where the generator is hidden from the policy:
- random safe measurement;
- fixed expert-designed protocol;
- maximum posterior entropy reduction;
- expected discrimination between leading hypotheses;
- cost-normalized information gain;
- power-constrained information gain;
- oracle action with access to the generating model (upper bound).

Measure:
- information gained about the true mechanism;
- number of actions and cost to reach a predeclared evidence threshold;
- false discovery / false-confident rate;
- calibration and posterior predictive coverage;
- frequency of uninformative or underpowered actions;
- harm/permission/rights constraint violations (must be zero);
- abstention when no affordable, adequately powered action exists.

Keep the oracle upper bound separate from deployable policies. The policy must not be given the generating model or hidden ground truth. Include model-discrepancy scenarios where all candidate models are imperfect; the safe policy should preserve an unknown-model disposition instead of confidently selecting the least-wrong hypothesis.

## 8. Integration path

1. Begin with finite, transparent predictive models and a deterministic action registry.
2. Reuse Symthaea's existing open-world campaign and falsification/power work only after checking exact source paths, APIs and assumptions; do not duplicate their statistical engines.
3. Store action proposals as hypotheses, never as authorization to act.
4. Require a human/landowner and appropriate ethics/rights gate before real sampling.
5. Record the action selection rationale and pre-observation predictions before execution.
6. Recompute on held-out observations and preserve replayable evidence.

**Claim ceiling:** success on synthetic worlds demonstrates policy performance for those declared generators only. It does not validate a physical crop-formation mechanism or justify an unauthorized real-world experiment.
