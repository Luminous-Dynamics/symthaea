# MUS-DOE-001 — Cost-aware instrument experiment design v0.1

Status: proposed protocol and synthetic policy fixtures only. No physical result, model performance, selected hardware design or measured experiment recommendation is asserted.

Parent: https://github.com/Luminous-Dynamics/symthaea/issues/8085
Instrument engineering: https://github.com/Luminous-Dynamics/symthaea/issues/8036
Manufacturing/lifecycle comparison: https://github.com/Luminous-Dynamics/symtropy/issues/1994
Measurement/digital-thread owner: https://github.com/Luminous-Dynamics/symthaea/issues/3695

## Paradigm shift

Choose the next experiment for its **decision-relevant information per unit cost**, not only predicted improvement. In a small instrument program, experiments are expensive: a new build may require parts, fabrication, assembly, setup, calibrated acquisition, tuning and scarce musician time. A cost-aware strategy should decide whether the next run is intended to discriminate competing models, reduce uncertainty near a hard constraint, measure repeatability, or verify an expected improvement.

This is an experiment-selection problem, not permission for a model to validate itself. The hypotheses, admissible candidate set, acquisition policy/version, baseline selection policies, validation partitions, outcome metrics, exclusions, and stopping rule must be frozen before outcomes are examined. Safety, basic playability, and measurement validity are hard gates outside any learned acquisition policy.

## Leakage-resistant evaluation

Split by independent groups rather than randomly splitting correlated waveform frames. Relevant group axes may include physical unit, manufacturing batch, builder, material lot, room/environment and measurement session.

- Fit/training groups estimate model parameters.
- Calibration groups are used only for uncertainty calibration or threshold choices declared before testing.
- Locked test groups are held until the model/policy/metric are frozen; their outcomes may not guide selection, tuning, stopping or calibration.

If one instrument contributes 1,000 note windows, those are not 1,000 independent physical builds. Windows from that unit must not appear in both fit and locked test when the claim concerns generalization to unseen units. If there are too few distinct units or batches to create independent partitions, mark physical generalization as unassessed instead of hiding the limitation.

The schema/corpus here are **synthetic fixtures**. Group labels are fake identifiers and are not observations or measurement evidence.

## Policies to compare fairly

Compare fixed-grid or preregistered design-of-experiments, seeded random selection, uncertainty-first selection, model disagreement, cost-aware information gain/expected value of information, and predicted-mean exploitation. Candidate selection occurs only among experiments that pass hard admissibility constraints. Run every policy on the same candidate space and budget, and repeat over a frozen seed set.

Information gain is meaningful only relative to a specified uncertain quantity or decision. A raw entropy-like score is not automatically a calibrated probability or evidence that the next experiment will help.

## Benchmark outputs

Report separate outcomes:
- held-out error by independent physical unit/build batch/environment;
- prediction interval coverage and width for any claimed predictive intervals;
- false-feasible rate against hard engineering constraints;
- improvement in a declared decision criterion or reduction in a declared uncertainty per unit experiment cost;
- experiment count, elapsed time and material/setup cost until a frozen stop condition;
- performance against fixed-grid, random, uncertainty-first, model-disagreement and predicted-mean baselines;
- repeatability and between-unit/build variation;
- sensitivity to model misspecification, heteroscedastic noise, drift, missingness and measurement uncertainty;
- complete trial denominators, protocol deviations and excluded-run reasons.

No universal scalar “experiment intelligence” score should conceal a high false-feasible rate or badly miscalibrated prediction intervals.

## Proposed gates

**Stage 0 — synthetic policy integrity**
- 100% of group-overlap, locked-test-contamination, inadmissible-candidate, seedless-random and post-hoc-policy negative controls rejected.
- Identical canonical inputs, selection policy version and seed yield identical selected experiment sequences and reports.
- No locked-test outcome is consumed by fitting, calibration, candidate selection or stopping rules.
- All declared grouping axes are checked; missing group identity yields incomplete/unassessed rather than implicit independence.

**Stage 1 — synthetic policy tournament**
- Proposed target: at least 100 fixed seeds per synthetic response/noise family.
- Equal candidate space and experiment budget for every policy.
- Include known truth, model misspecification, heteroscedastic noise, batch effects, drift and failed/missing experiments.
- Publish distributions and uncertainty, not only the best seed.
- Active selection is useful only if it improves the predeclared decision or information metric versus baselines without degrading held-out calibration or constraint handling beyond preregistered bounds.

**Stage 2 — measurement-method pilot**
- After a real fixture and calibrated acquisition chain exist, target at least 10 repeats per condition to estimate within-setup repeatability.
- Freeze excitation, mounting, sensor geometry/gain, sample rates, room/environment, run order and analysis.
- Repeats of one unit do not establish between-unit or between-builder reproducibility.

**Stage 3 — held-out physical build validation**
- Target at least three independent units and, when feasible, three batches/builders; these are pilot goals, not generally sufficient sample sizes for all claims.
- Size confirmatory studies from pilot variance and preregistered precision/power.
- Report applicability limits and residuals by relevant condition; no random-window split can substitute for held-out builds.

## Research references

- ASME, [V&V 10 — Verification and Validation in Computational Solid Mechanics (2019, reaffirmed 2025)](https://www.asme.org/codes-standards/find-codes-standards/standard-for-verification-and-validation-in-computational-solid-mechanics), a framework for model credibility.
- ASME, [VVUQ 10.2 — Role of Uncertainty Quantification in V&V](https://www.asme.org/codes-standards/find-codes-standards/the-role-of-uncertainty-quantification-in-verification-and-validation-of-computational-solid-mechanics-models), which describes uncertainty in simulation and experimental evidence, including experiments intended to validate predictive capability.
- Siska (2026), [A Guide to Bayesian Optimization in Bioprocess Engineering](https://doi.org/10.1002/bit.70129), a current review of sequential experimentation for noisy or costly systems. This is a candidate technique, not evidence of superiority for instrument engineering.
- NIST, [Digital Thread for Manufacturing](https://www.nist.gov/programs-projects/digital-thread-manufacturing), relevant to exact design/build/measurement identity and interoperability.

## Nonclaims

This work does not establish that a cost-aware or Bayesian policy is superior to simple baselines, that any predictive model is validated, or that any physical instrument has been improved. Stage 0 establishes only synthetic split/selection-policy integrity; later performance claims require equal-budget comparisons, locked test groups, and source-bound physical evidence.
