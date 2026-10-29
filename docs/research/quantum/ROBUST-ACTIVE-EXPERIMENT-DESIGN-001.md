# QARCH-006: robust experiment selection under model misspecification

**Status:** implementation draft; analytic fixtures authored, Rust tests not yet executed  
**Code:** \`crates/domains/symthaea-quantum-comp/src/robust_experiment.rs\`  
**Reference fixture:** \`docs/research/quantum/benchmarks/QARCH-006-ROBUST-EXPERIMENTS.json\`  
**Related:** QARCH-003 utility crossover; QARCH-005 decision-focused active experiment selection; Symtropy QSIM-001 physical-context co-simulation.

## Why this layer is needed

QARCH-005 ranks an experiment by expected value of sample information (EVSI) per measurement cost. That is decision-focused, but it still relies on the prior and likelihood model used to calculate EVSI.

This is a material failure mode, not a theoretical footnote. Callahan and Catanach's AISTATS 2026 paper, *On the Misinformation in a Statistical Experiment*, shows that under model and inference misspecification, experiments ranked as most informative can amplify bias and produce confident but wrong inference. Tang, Sloman, and Kaski's AISTATS 2026 paper studies robust active learning under misspecification and motivates representative, informative, and de-amplifying choices. These papers do not establish this module's performance; they motivate a specific falsifiable failure mode and benchmark.

References:
- [Callahan & Catanach (AISTATS 2026), On the Misinformation in a Statistical Experiment](https://proceedings.mlr.press/v300/callahan26a.html)
- [Tang, Sloman & Kaski (AISTATS 2026), Representative, Informative, and De-Amplifying](https://proceedings.mlr.press/v300/tang26d.html)
- [Go & Isaac (UAI 2022), Robust expected information gain using ambiguity sets](https://proceedings.mlr.press/v180/go22a.html)
- [Google Quantum AI (July 2026), continuous learning from QEC error events](https://research.google/blog/towards-a-quantum-computer-that-learns-from-its-errors/)
- [Open QBench](https://quantum.psnc.pl/en/openqbench/) for interoperable benchmark execution across quantum-classical workflow layers.

## Proposed paradigm

Treat the experiment model itself as uncertain. Instead of trusting one nominal hypothesis-prior and observation-likelihood model, make a finite, named ambiguity set of plausible alternatives. Examples include:

- nominal optical transmission and a higher-loss alternative;
- alternate detector-noise calibration models;
- source distinguishability estimates with different uncertainty envelopes;
- Symtropy layout scenarios passed through separately justified geometry-to-optics adapters;
- alternative priors over physical failure mechanisms;
- held-out model-discrepancy patterns.

For every candidate experiment, calculate EVSI/cost under each model. Report the nominal value, minimum, maximum, mean, and spread. Rank by minimum EVSI/cost when the declared policy is minimax decision robustness. Keep the nominal ranking alongside the robust ranking so that the conservatism and disagreements are visible.

This is an explicit finite-ensemble method, not a guarantee against unrepresented models and not a claim that minimax is universally optimal. A badly constructed ambiguity set can make a robust score misleading. Ensemble definition and held-out validation remain first-class evidence.

## Contract

All scenarios in one ensemble must share:
- hypothesis labels and order;
- action labels and order;
- action-utility matrix and utility unit;
- measurement-cost unit;
- candidate experiment IDs, outcome labels and their order, and measurement costs.

Priors and likelihood matrices may differ to represent uncertainty. Each model is independently validated by QARCH-005's probability checks. The current implementation intentionally keeps the utility matrix fixed: changing utilities changes the decision semantics rather than just the likelihood model, and should be modeled as a separate policy question.

The ranking is:
\[
R(e)=\min_{m\in\mathcal{M}}\frac{\operatorname{EVSI}(e;m)}{c(e)}
\]
where \(\mathcal{M}\) is the set of explicitly declared plausible models and \(c(e)\) is the shared measurement cost. It also reports nominal, best-case, mean, and spread. Stable lexical ID tie-breaking makes output order reproducible.

A zero worst-case score is not silently promoted. It signals that at least one declared model says the experiment has no decision value. Depending on the application, the correct response may be to select a diagnostic experiment, acquire calibration data, or abstain until the uncertainty is narrowed.

## Hand-checkable counterexample

The frozen JSON fixture has two hypotheses with prior \([0.5,0.5]\) and two actions that each score one utility point when the corresponding hypothesis is true.

- Experiment A is perfect under the nominal likelihood model, yielding EVSI/cost 0.5. Under an equally plausible alternative model it is uninformative, yielding 0.0.
- Experiment B yields EVSI/cost 0.3 under both models.

Nominal EVSI ranking chooses A (0.5 > 0.3). Worst-case ranking chooses B (0.3 > 0.0). This is the intended behavior for the declared minimax objective. It is a synthetic model-checking fixture, not a quantum device result.

## Integration with Symthaea, Symtropy, and QARCH

### Symtropy QSIM-001

Symtropy can produce deterministic, replayable physical-context and perturbation scenarios. Those scenarios become alternative likelihood models only through an explicit, validated geometry-to-physics adapter. A simulated placement shift is not a measured probability distribution.

### Symthaea physics / decoherence

Physics models can propose hypothesis families, invariants, and alternative noise explanations. Their predictions must be calibrated to the hardware regime before being treated as empirical likelihoods. Missing model scope or parameters means incomplete evidence, not a successful prediction.

### QARCH-001 and QARCH-002

Photonic route/primitive envelopes and bounded adaptive-fusion transitions may vary across the ensemble where assumptions vary. Do not mix architecture-incompatible quantities or treat abstract transition probabilities as bench-measured values.

### QARCH-003 and QARCH-005

QARCH-003 defines the downstream quantum-versus-classical quality/resource gate. QARCH-005 computes EVSI for a declared decision model. QARCH-006 asks whether the selected experiment remains worthwhile across explicit plausible models. The output should feed experiment scheduling, not autonomously control a physical backend.

## Benchmark plan

### Core contract tests

- The nominal winner in the frozen fixture collapses under the alternative model; robust ranking instead chooses the candidate with higher worst-case EVSI/cost.
- The robust score equals the minimum of the scenario scores; mean, maximum, and spread agree with independently calculated values.
- Candidate order changes do not change rankings beyond documented lexical tie-breaks.
- Duplicate scenario IDs, zero or multiple nominal scenarios, mismatched hypothesis/action/utility semantics, mismatched units, missing candidates, altered outcomes, or differing costs fail closed.
- Invalid probabilities, NaN/Inf, dimension mismatch, and score overflow are rejected by the underlying validated model.
- Zero worst-case EVSI remains zero and never receives a favorable finite bonus from an unavailable model.

### Model-misspecification comparison

On frozen finite decision problems with a known simulated ground-truth model, compare:

1. nominal EVSI/cost (QARCH-005);
2. robust worst-case EVSI/cost (this module);
3. expected EVSI/cost under a declared ensemble distribution, if such weights are justified;
4. entropy/information-gain-only selection;
5. random selection;
6. fixed sweeps;
7. robust Bayesian experimental-design baselines when problem assumptions match.

For each policy, report decision regret, correct-decision rate, false-confident-decision rate, experiments-to-decision, measurement cost, nominal performance, worst-case performance, and how often the policy abstains. Include scenario-wise results, repeated seeds where stochasticity exists, and uncertainty intervals.

### Held-out physical calibration

Use Symtropy to generate controlled perturbations only within the validity of its physical models. Reserve real measured regimes or devices as held-out conditions when data are available. Evaluate:
- predictive residuals by regime;
- probability calibration and proper scoring rules;
- failure to cover actual measurement outcomes;
- sensitivity of ranking to prior/likelihood perturbation;
- whether robust selection improves decision quality, not merely average information gain.

A finite-ensemble result cannot establish robustness beyond the tested ambiguity set. Physical claims require calibrated adapters and hardware observations.

## Proposed targets

- 100% agreement with analytic reference fixtures.
- Zero false accepts for all invalid model/contract controls.
- Zero false-robust-winner outcomes on the frozen deterministic test set.
- On preregistered model-mismatch tests, lower worst-case decision regret than nominal EVSI selection, while reporting any reduction in nominal performance.
- On held-out measured scenarios, decision quality and calibration that are no worse than the strongest applicable baseline, with the test set and metrics frozen before execution.

These are proposed targets, not achieved results or an industry standard.

## Current implementation boundary

The Rust module implements finite-ensemble ranking by worst-case EVSI/cost, preserves nominal and scenario-level scores, and rejects mismatched candidate contracts and decision semantics. Five unit tests are authored. The JSON counterexample is recorded as a reviewable fixture. Static checks verify module registration, public exports, source layout, and fixture parsing. Rust tests have not yet been executed in this environment.

The module does not generate physical models, calibrate likelihoods, learn from hardware observations, execute a backend, or demonstrate improved performance on industry benchmarks.
