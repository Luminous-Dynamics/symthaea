# HIST-CAUSAL-001 v0.6: open-world model adequacy and abstention

## Motivation

v0.5 tests classification when the true process belongs to one of six declared candidate models. Scientific discovery must also detect that the candidate catalogue is incomplete. Otherwise the agent can confidently choose the least-wrong familiar explanation.

v0.6 adds an evaluator-only population mixture between primary-sensor artefact and actuator non-compliance. The latent component is redrawn for each independent probe trial/experimental unit, so this is a heterogeneous population mixture—not a claim that one specimen changes mechanism between probes.

## Exact design

For every probe p, the out-of-family distribution is:

P_ood(y|p) = 1/2 P_sensor-artefact(y|p) + 1/2 P_actuator-noncompliance(y|p)

All probabilities are exact rationals inherited from the v0.5 recipe. The solver sees the same six in-family candidates plus two additional permitted outputs: model_misspecified and abstain. It never receives the mixture components, mixture weight, target or selected parameters.

For each candidate model, compute the average total-variation distance over all six probe contexts:

D(m) = (1/6) sum_p TV(P_ood(.|p), P_m(.|p))

This measures exact signature separation for the declared finite model family and probe menu. It is not a finite-sample power result or an empirical detector accuracy.

## v0.6 development result

Across 100 public development seeds:
- all out-of-family probe distributions normalize exactly;
- the out-of-family signature differs from each of the six candidate signatures for every seed;
- minimum nearest-candidate average TV = 689/6480 (approximately 0.1063);
- mean nearest-candidate average TV = 28060229/246758400 (approximately 0.1137);
- 8 additional unit tests pass.

The closest candidate can vary by seed; the correct benchmark outcome is open-world misspecification detection or abstention, not forced selection of that nearest candidate.

## Important limitations

- The mixture model resamples its latent component per independent experimental unit. It is not a persistent hidden world across sequential probes.
- The six candidate distributions are evaluated at each recipe's generated parameter point. This is a finite signature comparison; it does not prove separation from every parameterized extension of every candidate family.
- Average TV separation does not guarantee easy finite-sample detection. A later layer must compare power, calibration and false misspecification rates across sample sizes and priors.
- An adaptive solver may not inspect evaluator keys. Only the public agent view is passed to the solver.

## Paradigm shift

A benchmark should reward knowing when its own model family is inadequate. This requires:
1. a closed-world lane, where a correct candidate exists;
2. an open-world lane, where no candidate is correct;
3. non-identifiable lanes, where several candidates remain observationally equivalent;
4. calibrated abstention and a falsification certificate;
5. held-out mechanism compositions and private confirmation seeds.

This complements Symtropy's joint physical/sensor, robust model-indexed and challenge-generation work rather than implementing a new experiment planner.

## Claim boundary

v0.6 establishes exact signature separation for a finite synthetic population mixture across six declared probe contexts. It does not demonstrate finite-sample open-world detection, physical mechanism discovery, cross-system historical influence or SOTA superiority.
