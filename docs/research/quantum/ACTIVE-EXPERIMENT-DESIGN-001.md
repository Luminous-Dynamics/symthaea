# ACTIVE-EXPERIMENT-DESIGN-001
## Decision-focused experiment selection for quantum architecture research

**Status:** implementation draft; analytical fixtures authored, not yet executed in Rust  
**Code:** \`crates/domains/symthaea-quantum-comp/src/active_experiment.rs\`  
**Related:** QARCH-001 photonic architecture; QARCH-002 bounded adaptive fusion; QARCH-003 utility crossover; QARCH-004 workload discovery; Symtropy QSIM-001 physical-context co-simulation.

## Core idea

Select the next experiment by **expected improvement in the downstream decision per unit measurement cost**, not by maximizing information gain in the abstract.

A measurement can sharply distinguish two hypotheses without changing the architecture, experiment plan, or quantum-versus-classical choice. That measurement has information value but no decision value for the decision currently under review. Conversely, a less informative measurement can be valuable when it might reverse a costly design choice.

This module ranks candidates in a finite, discrete decision model. It is not a physical predictor, Bayesian optimizer for continuous parameters, reinforcement-learning agent, or live hardware controller. The likelihoods and action utilities must come from a declared and appropriate model.

## Mathematical contract

Let hypotheses \(h\) have prior probabilities \(p(h)\), candidate actions \(a\) have utility \(U(a,h)\), and candidate experiments \(e\) produce outcomes \(y\) with likelihood \(P(y\mid h,e)\).

The current best expected utility is

\[
V_0 = \max_a \sum_h p(h) U(a,h).
\]

For a particular outcome, the posterior is proportional to \(p(h)P(y\mid h,e)\). Expected utility after seeing the outcome, before it has been observed, is

\[
V_e = \sum_y P(y\mid e)\max_a
\sum_h P(h\mid y,e) U(a,h).
\]

The expected value of sample information is

\[
\operatorname{EVSI}(e)=V_e-V_0.
\]

Rank by EVSI divided by the declared measurement cost. The code computes the outcome-weighted expected utility without dividing by small outcome probabilities, and reports expected information gain in bits as a diagnostic. A negative EVSI larger than the declared roundoff tolerance is a model/calculation error, not a negative-value experiment.

The units matter: utilities must represent one predeclared objective and costs must share one declared measurement-cost unit. Keep time, energy, monetary cost, scientific error, and other incomparable objectives separate unless an independently justified decision model defines a common utility.

## How this integrates the Luminous stack

### Symtropy — scenarios, not probability by fiat

Use QSIM-001 to generate repeatable physical-context perturbations: placement shifts, frame changes, supported mechanical dynamics, and other scenarios where Symtropy's model is applicable. A scenario output becomes a likelihood only after a physical adapter defines the mapping and supports it with equations, calibration, or measured data. Geometry by itself does not imply photon-loss or phase-error probabilities.

### Symthaea physics — priors and hypotheses with declared scope

The physics and decoherence modules can motivate the competing hypotheses, invariants, observables, and possible model-discrepancy families. The existence of a model is not evidence that its predictions are calibrated for a particular device. Priors, likelihoods, validity domains, and physical units must be captured in a workload/experiment manifest.

### QARCH-001 and QARCH-002 — consequences of a measurement

The photonic path and component envelope and the bounded stateful-fusion estimate can supply consequences for candidate architectures, subject to their assumptions. Do not treat an abstract transition probability as a lab measurement.

### QARCH-003 — decision utility

QARCH-003 defines the end-to-end cost and quality gate for quantum versus classical candidates. QARCH-005's action utility should be derived from that decision objective (or a separately declared calibration objective), not an ad hoc aggregate score. Run separate rankings for non-comparable objectives.

### Real observations update the model

When physical observations arrive, retain the instrument configuration, calibration revision, raw-artifact digest, uncertainty, timestamp, and evidence class. Do not relabel a simulated outcome as measured. Compare prior predictions to held-out measurements and track whether uncertainty intervals remain calibrated.

## Research context and novelty boundary

Active measurement selection is not new and should not be presented as such. A 2025 NIST-linked study on adaptive quantum sensing demonstrated cost- and parameter-aware designs for NV-center sensing and reported an almost five-fold improvement in magnetic-field sensitivity in its studied setting ([NIST publication record](https://www.nist.gov/publications/weighing-unequal-parameter-importance-and-measurement-expense-adaptive-quantum-sensing)). A July 2026 Nature experiment learned from quantum-error-correction events to continually adapt control and demonstrated a 3.5-fold improvement in logical stability against injected drift on the studied processor ([Nature, 2026](https://www.nature.com/articles/s41586-026-10759-2)). Active learning has also been studied for entangling-gate calibration using surrogate-based optimization ([2026 preprint](https://arxiv.org/abs/2607.00284)).

The proposed contribution is not to claim a new class of Bayesian experiment design. It is to connect decision-focused experiment ranking to the specific Luminous research loop: Symtropy's reproducible physical scenarios, Symthaea's physics-informed hypotheses, QARCH's architecture and utility estimates, and a reproducible evidence manifest spanning simulation and hardware observations.

This remains a hypothesis until tested against meaningful baselines and held-out, preferably calibrated scenarios.

## First benchmark: finite decision landscapes

### Required baselines

1. **Random candidate:** uniform selection among available experiments, with repeated randomized trials.
2. **Maximum information gain:** select the candidate with the greatest expected entropy reduction.
3. **Fixed sweep:** run a predetermined sequence independent of current evidence.
4. **Cost-aware generic selection:** when assumptions and candidate spaces match, compare against a standard cost-aware Bayesian experimental-design/optimization method.
5. **Decision-focused EVSI/cost:** the implementation in this module.

Use the same initial prior, experiment budget, stopping rule, action utilities, outcome-generation process, and data per trial. Freeze test cases before comparing planners. For stochastic simulations, pair seeds where practical.

### Outcomes to report

- Decision regret relative to the best action with the simulated ground-truth hypothesis known.
- Number and cost of experiments until the action is correct or a preregistered confidence/utility threshold is met.
- False-decision rate and probability of switching away from the best action.
- Total measurement cost and wall time; later include measured energy if available.
- Calibration of model likelihoods and performance under prior shift / out-of-distribution scenarios.
- EVSI and information-gain ranking disagreement: this is a key diagnostic, not an error by itself.
- Per-scenario and aggregate distributions with uncertainty; do not report only the best seed.

### Exact test targets

- The perfect two-hypothesis fixture has prior expected utility 0.5, EVSI 0.5, and expected information gain of 1 bit.
- An experiment that perfectly distinguishes hypotheses but cannot change the best action has 1 bit of information gain and zero EVSI.
- For two perfect experiments with the same modeled outcomes but costs 1 and 10, the cheaper experiment ranks first by EVSI/cost.
- An uninformative experiment has zero EVSI and zero information gain.
- Outcome marginal probabilities and every posterior distribution sum to one within the declared numerical tolerance.
- Malformed probabilities, duplicate labels, dimension mismatch, non-positive/non-finite measurement costs, non-finite utility, and overflow fail closed.
- Candidate order cannot change the score; equal score ties use a stable lexical experiment-ID order.
- Mutation controls that inject probability leakage or incorrect outcome likelihoods must be rejected rather than silently normalized.

Targets above are deterministic implementation-contract gates. They are not evidence of improvement over an industry experiment-design baseline.

## Higher-value research question

The testable longer-term hypothesis is:

> For a fixed measurement budget in a calibrated photonic architecture study, a planner ranked by EVSI per cost will reach the correct architecture choice or the preregistered “do not decide yet” outcome with lower regret and fewer expensive measurements than random, fixed-sweep, and information-gain-only selection.

Do not force an architecture decision when evidence is insufficient. Include an explicit abstain/defer action in the utility matrix when premature selection is costly.

## Staged acceptance criteria

1. **Implementation correctness:** analytic EVSI fixtures agree; malformed distributions are rejected; negative controls fail closed.
2. **Decision simulation:** on frozen finite scenarios, compare regret and experiments-to-decision with the baselines above. Report all scenarios, not just favorable ones.
3. **Model validity:** test likelihood predictions on held-out physical scenarios and measured outcomes. Report calibration, residuals, and out-of-distribution failures.
4. **Hardware evaluation:** only after model and experimental protocols are sound, run approved hardware calibration/sensing experiments and record raw outcomes, instrument cost, and configuration.
5. **Utility claim:** claim measured decision improvement only when all models and observations meet their evidence requirements and an independent reproduction confirms the result.

## Current implementation boundary

The dependency-free module represents discrete hypotheses, prior probabilities, actions, action utilities, candidate outcomes, measurement cost, EVSI, information gain, and outcome probabilities. Twelve Rust unit tests are authored for analytic fixtures and fail-closed validation. At the time of writing, the Rust tests have not been executed in the current environment. The analytic reference cases are also recorded in `docs/research/quantum/benchmarks/ACTIVE-EXPERIMENT-DESIGN-001.json`; the JSON is currently a reviewable fixture artifact, not yet consumed by a Rust loader. The module does not ingest Symtropy manifests directly yet, predict a physical likelihood, control hardware, or demonstrate improved measurement efficiency.
