# Symtropy × Symthaea: physics-history lab integration v0.1

**Status:** experimental planning component; not connected to a live Symtropy scene yet.

## Purpose

Symtropy's Physics Experiment Atlas supplies deterministic experiment/simulation models. Symthaea's experiment-design crate ranks declared candidate interventions before they are run. The point is to choose interventions that distinguish rival physical explanations—not to produce plausible-looking scenes or use an AI confidence score as a substitute for uncertainty.

- Symtropy or a separately reviewed physics model supplies predictions for each candidate hypothesis and observable.
- Measurement/model uncertainty and relative resource cost are explicit inputs; neither is invented from Symthaea's subjective confidence.
- Every observable carries a unit. Each hypothesis prediction carries a canonical model descriptor + SHA-256 digest and a separately digested uncertainty descriptor.
- The planner recomputes descriptor digests and rejects mismatches before scoring.
- This crate validates the design contract and ranks candidates using a prior-free, maximin standardized-separation heuristic.
- The simulation or laboratory produces observations later. An independent scorer compares those observations to frozen predictions.

The first cross-project fixture is Galileo's ideal inclined-plane question: constant acceleration and constant velocity are matched at a two-second observation, so that observation cannot discriminate between the models. Under the fixture's declared idealized equations, a four-second displacement differs by about 19.61 m and ranks ahead of the matched-time candidate.

This is a pedagogical synthetic demonstration, not a reconstruction of Galileo's historic apparatus or his source data.

## CLI contract

Run the bundled example:

    cargo run -p symthaea-experiment-design -- --demo

Rank a JSON design provided on stdin:

    cargo run -p symthaea-experiment-design < design.json

The strict physics-experiment-design-v0.1 input includes:
- a target question;
- at least two uniquely named rival hypotheses;
- candidate experiments with unique IDs, positive relative cost units and non-negative duration;
- at least one observable per candidate, with a non-empty unit;
- exactly one finite predicted mean and positive finite predictive standard uncertainty for every hypothesis/observable pair;
- a non-empty model descriptor and uncertainty descriptor per prediction, each carrying a lowercase 64-character SHA-256 digest matching the exact descriptor UTF-8 bytes.

Missing hypotheses, duplicate IDs, unknown hypotheses, empty prediction sets, missing units, altered descriptors with stale digests, invalid uncertainty, non-finite scores, and invalid resource costs fail closed with typed errors. Unknown JSON fields are rejected to surface schema drift instead of silently ignoring it.

The digests verify only that the supplied descriptor text matches its fingerprint. They do **not** attest that an executable produced the prediction, that the source code matches the descriptor, or that the uncertainty is calibrated. Those claims require higher-level evidence receipts containing source revision, run configuration, input/output digests and calibration provenance.

The output is a deterministic ranked JSON receipt. It identifies the hypothesis pair that is hardest to distinguish for each candidate, reports standardized separation, cost-adjusted score and measurement count, and discloses limitations. Input-order changes should not alter scores or ranking.

## What the score means—and does not

For hypothesis pair a,b and observable j, the declared normalized distance is

    d_j(a,b) = abs(mu_a,j - mu_b,j) / sqrt(sigma_a,j^2 + sigma_b,j^2).

Assuming conditional independence across observables, the multi-observable distance is sqrt(sum_j d_j^2). A candidate's robust separation is the minimum distance across all hypothesis pairs. The displayed screening score divides this minimum by caller-supplied relative resource cost.

This is **not** Bayesian expected information gain, a Bayes factor, a posterior probability, or a guarantee that an experiment will be informative in practice. The method intentionally avoids priors; it optimizes against the pair that is hardest to separate. If the two most confusable hypotheses make identical predictions, the score is zero even if all other models are easy to separate.

### Required qualifications before physical use

1. **Uncertainty calibration.** Predictive uncertainty must come from instrument specifications, residual calibration, or a pinned reference—not a language-model confidence field.
2. **Dependence.** The current score treats observables as conditionally independent. Correlated channels require a covariance-aware score; do not duplicate correlated channels to inflate a score.
3. **Parameter nuisance.** Sweep unknown nuisance parameters or marginalize/profile over them before trusting a fixed mean.
4. **Model discrepancy.** Include a declared discrepancy term where even the best model is not a complete description of reality.
5. **Cost realism.** Cost units are user-defined relative weights; report real apparatus time/cost separately and freeze the scale before ranking.
6. **Pre-registration.** Freeze candidate predictions, expected uncertainty, falsification conditions and data-exclusion rules before collecting new data.
7. **Independent adjudication.** Symtropy's reference solver or a laboratory analysis scores the observation. The planner cannot validate its own proposed predictions.

## Benchmark ladder

| Gate | Benchmark | Proposed acceptance |
|---|---|---|
| Contract integrity | missing/duplicate hypotheses and observations; wrong schema; missing unit; altered model/uncertainty descriptor without digest update; NaN/infinite mean or uncertainty; zero cost | 100% of malformed fixtures rejected |
| Determinism | same input repeated; permutations of candidate and prediction order | exact same rank, metrics and hypothesis pair |
| Discriminating power | toy rival models equal at one control value and different at another | select the non-degenerate intervention; identical predictions score exactly zero |
| Robustness | increase predictive uncertainty, change relative cost, and make the hardest pair overlap | score must fall or not improve spuriously; ranking recomputed from declared inputs |
| Calibration | predicted intervals scored on held-out simulator and laboratory cases | evaluate interval coverage and standardized residuals; no universal threshold before data characteristics are known |
| External validity | compare to a Bayesian likelihood-based design on models with calibrated predictive distributions | report rank agreement, decision regret and compute cost; keep separate metrics, don't declare either oracle by definition |

## Planned connection to the experiment atlas

1. Map each Symtropy model version + parameters to named hypotheses and versioned observable predictions.
2. Carry model digest, units, solver/tolerance, initial conditions, nuisance settings, calibration source, and uncertainty provenance in the plan receipt.
3. Have Symthaea propose candidate interventions and predict the ranking before seeing held-out outputs.
4. Execute candidates in Symtropy's physical-state engine; preserve raw generated/recorded data independently of the scene.
5. Compare candidate ranking and predictions to the held-out oracle. Record negative results, resource use and regressions.
6. Add covariance-aware discrimination and a true likelihood/EIG option only with validated distributions, not by relabeling this heuristic.

## Shared Galileo handoff fixture

The JSON input in `fixtures/galileo-discrimination-v0.1.json` is deliberately also stored in Symtropy at `tools/physics-experiment-atlas/fixtures/galileo-discrimination-v0.1.json`. It is a transport fixture, not generated observation data. Both copies identify the same model equations and uncertainty assumptions using SHA-256 digests of canonical descriptor strings. The digest check catches accidental descriptor edits that were not accompanied by a digest update, but it is not proof that code executed the described equation.

## Relationship to existing experiment design

Symthaea's existing spark-engine::optimal_experiment is useful but its current documented model uses signature-level discrete outcome classes for a lattice-confinement-fusion question. It remains appropriate for that domain. This new module does not silently reinterpret that specialized score as a general physical measurement model and does not modify the fusion program.

## Reference context

- Millikan's primary 1913 oil-drop paper is available through CaltechAUTHORS: https://authors.library.caltech.edu/records/sfc90-k2g07
- NIST's historical constants account documents both Millikan's oil-drop result and the importance of viscosity correction: https://physics.nist.gov/cuu/Constants/historical1.html
- UNSW Physics derives the Foucault-pendulum latitude dependence and rotating-frame sign convention: https://newt.phys.unsw.edu.au/~jw/pendulumdetails.html

References are historical/model context, not validation of this planner. The runnable Galileo fixture is synthetic.
