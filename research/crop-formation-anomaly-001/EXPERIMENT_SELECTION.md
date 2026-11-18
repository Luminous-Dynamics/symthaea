# CROP-INTEL-001 — Active experiment selection

**Status:** research design; no experiment policy implemented.

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

## 6. Sequential updating safeguards

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

Keep the oracle upper bound separate from deployable policies. The policy must not be given the generating model or hidden ground truth.

## 8. Integration path

1. Begin with finite, transparent predictive models and a deterministic action registry.
2. Reuse Symthaea's existing open-world campaign and falsification/power work only after checking exact source paths, APIs and assumptions; do not duplicate their statistical engines.
3. Store action proposals as hypotheses, never as authorization to act.
4. Require a human/landowner and appropriate ethics/rights gate before real sampling.
5. Record the action selection rationale and pre-observation predictions before execution.
6. Recompute on held-out observations and preserve replayable evidence.

**Claim ceiling:** success on synthetic worlds demonstrates policy performance for those declared generators only. It does not validate a physical crop-formation mechanism or justify an unauthorized real-world experiment.
