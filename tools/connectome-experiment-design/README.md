# Connectome experiment design: finite-hypothesis EIG

A standalone, dependency-free Rust 2024 reference crate for ranking candidate experiments by expected information gain (EIG) over a finite set of competing hypotheses.

This crate is a small mathematical utility, **not** a neural simulator, causal estimator, continuous control-input optimizer, or biological model. The caller must provide the hypotheses, prior probabilities, conditional outcome likelihoods, and experiment costs. It does not infer or calibrate any of them. In particular, it complements rather than reproduces recent control-theoretic work on designing continuous perturbation waveforms for system identification: this implementation ranks a finite list of discrete candidate experiments under caller-supplied hypothesis likelihoods.

## Run

From the repository root:

```sh
cargo test --locked --manifest-path tools/connectome-experiment-design/Cargo.toml
```

The crate is an isolated Cargo workspace with no third-party dependencies and must not be added to the production workspace dependency graph accidentally.

## Mathematical contract

For a candidate experiment e, hypothesis H, and possible outcome Y:

```
EIG(e) = H(H) - E_Y[H(H | Y, e)] = I(H; Y | e)
```

All entropies and information gains are measured in bits. The crate returns both raw EIG and EIG divided by caller-defined positive cost.

The likelihood matrix is indexed as:

- outer row: hypothesis h;
- inner column: outcome y;
- cell: P(Y = y | H = h, e).

The prior vector must sum to one, and every hypothesis likelihood row must independently sum to one. Invalid probability mass is rejected; the crate never silently renormalizes malformed inputs.

## API

- `evaluate_candidate(priors, candidate)`: calculates prior entropy, expected posterior entropy, EIG, and EIG/cost.
- `outcome_probability(priors, candidate, outcome_index)`: calculates the marginal probability of an outcome.
- `posterior_for_observed_outcome(...)`: applies Bayes' rule after an outcome is observed.
- `rank_candidates(...)`: deterministic ranking by EIG/cost, then raw EIG, then ascending experiment ID.
- `selection_regret(...)`: measures opportunity regret against the best candidate in the supplied pool under the EIG/cost objective.

## Synthetic known-answer cases

The unit tests cover:

- uninformative experiments (0 bits);
- perfect discrimination (1 bit for a uniform binary prior);
- a noisy 90%/10% binary discriminator (0.5310044064107188 bits);
- perfect discrimination with an asymmetric prior (0.7219280948873623 bits);
- Bayes posterior updates;
- a cost-aware ranking that selects a lower-raw-EIG but more efficient experiment;
- deterministic tie-breaking and selection regret;
- invalid priors, invalid likelihood mass, NaN, zero cost, duplicate outcomes/IDs, impossible outcomes, and missing selections.

The analytical values are independently recorded in `fixtures/known-answers.json`.

## Scientific limitations and safe use

1. EIG is conditional on the candidate hypothesis set, prior, likelihood model, outcome discretization, and cost definition. Misspecified likelihoods can confidently select the wrong experiment.
2. EIG per cost is an information-efficiency objective, not expected behavioural benefit or downstream decision utility. Track decision regret separately.
3. A high EIG score does not establish that a hypothesis is biologically true.
4. Likelihoods should be calibrated on training/validation data and frozen before held-out outcomes are revealed.
5. Compare the policy against random, degree, centrality, largest-predicted-response, and expert-prior baselines under equal experiment budgets.
6. Prospective utility requires running the selected experiment and comparing the realized outcome to the preregistered predictions.

The fixture evidence class is `synthetic_known_answer` and claim ceiling is `SYNTHETIC_ONLY`. No empirical experiment ranking or biological result is claimed.


## Open-world EIG over a declared likelihood ambiguity set

Closed-world EIG can select an experiment that very efficiently distinguishes two incorrect candidate mechanisms. The open-world extension requires the hypothesis list to include a positive-prior `OTHER_OR_UNRESOLVED` state and evaluates the candidate across a finite, explicit set of plausible likelihood scenarios. It ranks experiments by the minimum EIG per unit cost across those declared scenarios, while reporting the best-case value and every scenario score.

This is a lower/upper envelope over the caller-supplied finite ambiguity set only. It does not guarantee robustness to mechanisms outside that set, automatically estimate the unknown-model prior, or determine whether the candidate set is adequate. The OTHER prior and likelihood scenarios must be calibrated from independent/held-out mechanism families or prespecified sensitivity assumptions. A broad ambiguity set may correctly drive worst-case EIG to zero; the selector should then report that lack of guaranteed information rather than promote the optimistic scenario.

The unresolved-state prior is never silently renormalized away. The Rust API rejects a missing or zero-prior OTHER state, duplicate/empty scenario IDs, malformed likelihood rows and invalid probabilities. Scenario ordering does not affect output. The versioned API report is `connectome-open-world-eig-v1`.

Input contract: `schemas/open-world-input.schema.json`. Output contract: `schemas/open-world-output.schema.json`. Known-answer example: `fixtures/open-world-robust-eig.example.json`; analytic results: `fixtures/open-world-robust-eig-known-answers.json`.

### Worked synthetic example

With priors `[0.45, 0.45, 0.10]`, the brittle experiment has an optimistic 0.9-bit EIG when the named hypotheses are perfectly separated, but a second plausible shared-misspecification scenario yields 0 bits. The robust candidate yields 0.106838 bits in its least informative declared scenario and 0.250265 bits in its best. At equal cost, the robust candidate ranks first by worst-case EIG. These are analytic synthetic fixtures, not biological likelihood estimates or a real experiment recommendation.

### What this still does not solve

A declared ambiguity set can still omit the true mechanism. A separate held-out model-adequacy gate must test predictive performance on unseen mechanisms/animals/contexts, calibrate abstention, and decide when no named model is adequate. See [SCI-CONNECTOMICS-046 (#9796)](https://github.com/Luminous-Dynamics/symthaea/issues/9796).

## Held-out open-world model-adequacy gate

A finite likelihood ambiguity set is not enough: it may still omit a mechanism. The adequacy reference therefore calibrates one **mechanism-family-level nonconformity score** in two disjoint cohorts:

- In-scope calibration: exactly one preregistered aggregate score per independent in-scope mechanism family.
- Omitted-mechanism calibration: exactly one aggregate score per independent omitted mechanism family.

For each family, aggregate its animal/session recordings with the same frozen predictive score, trajectory aggregation, observation model, and normalization. The held-out test score must use that same family-level aggregation. Frames, neurons, trials, and repeated animal/session recordings within one family are not independent calibration units; do not count them as additional calibration families.

For target false-adequacy level alpha, with n independent omitted mechanism families, the reference selects:

\`k = floor(alpha * (n + 1))\`

and uses the k-th smallest omitted-family score as the threshold. A held-out mechanism family is called ADEQUATE_UNDER_CALIBRATION only when its aggregate score is strictly less than that threshold. Strict inequality handles ties conservatively. Under exchangeability with the omitted-family calibration population, the false-adequacy probability is bounded by k/(n+1). This is a family-level guarantee only; it does not certify arbitrary unknown mechanisms or shifts not represented in the calibration population.

The gate also computes a one-sided Wilson lower bound on in-scope family acceptance. The calibrator is usable only if this lower bound meets the preregistered minimum-acceptance floor. If k=0 or the in-scope floor fails, every held-out decision is UNRESOLVED.

**Sample-size floor:** at alpha=0.05, fewer than 19 independent omitted mechanism families yields k=0, so no nontrivial threshold exists. Nineteen is only a coarse mathematical minimum, not a general recommendation; useful calibration will generally require substantially more independent families and a defensible exchangeability argument.

A high score means the declared candidate set plus its observation model is inadequate or shifted under this calibration. It does not identify which mechanism is missing. Registration errors, optical artefacts, latent common input, extrasynaptic signalling, and context/observation-model shift can all cause the same signal.

Input: \`schemas/adequacy-calibration-input.schema.json\`. Output: \`schemas/adequacy-calibration-output.schema.json\` (versioned \`AdequacyCalibrationReport\`). Synthetic fixtures: \`fixtures/adequacy-calibration.example.json\` and \`fixtures/adequacy-calibration-known-answers.json\`.

This is a first reference implementation, not a biological adequacy certificate. Its mechanism-family sampling frame, score construction, exchangeability scope, and train/calibration/test separation must be reviewed and frozen before any biological test.


## Goal-oriented target-query EIG

Global mechanism EIG answers: "Which experiment best distinguishes the full candidate mechanism identity?" That is not always the scientific question. A scientist may instead ask whether an intervention increases a target output, whether an effect persists, whether a dose threshold is crossed, or whether an action is worth taking.

A \`TargetCausalQuery\` maps every candidate hypothesis—including \`OTHER_OR_UNRESOLVED\`—to an explicit answer class. For each outcome, the implementation marginalizes hypotheses within each answer class and calculates:

\`I(Q;Y | e) = H(Q) - E_Y[H(Q | Y,e)]\`

Experiments are then ranked by the worst-case target-query information gain per unit cost across the caller-declared likelihood scenarios. Global mechanism EIG remains available as a separate comparator. The query's hypothesis IDs must exactly match candidate ordering; missing or empty answer labels fail closed.

### Why the objective changes the selected experiment

The analytic fixture uses priors \`[0.3, 0.3, 0.3, 0.1]\` for two mechanisms that both imply a positive target effect, one no-effect mechanism, and \`OTHER_OR_UNRESOLVED\`.

- \`mechanism-discriminator\` perfectly distinguishes the two positive mechanisms. It earns 0.600 bits of global mechanism EIG, but **0 bits about the target positive/negative/unresolved answer**.
- \`outcome-discriminator\` distinguishes positive from negative outcomes and retains an unresolved state. It earns 0.435946 bits of global EIG and **0.435946 bits about the target query**.

At equal cost, global EIG prefers the first experiment; the target-query objective correctly prefers the second for this declared question. The numbers are analytic synthetic fixture values, not biological likelihood calibration or experiment advice.

Input: \`schemas/target-query-input.schema.json\`. Output: \`schemas/target-query-output.schema.json\`. Fixtures: \`fixtures/target-query-eig.example.json\` and \`fixtures/target-query-eig-known-answers.json\`.

### Boundary

Query entropy is still not decision utility. A decision-focused planner needs an explicit downstream action set and utility matrix and should calculate expected value of sample information. It must not call target-query EIG "expected treatment benefit". Sequential dynamic regimes additionally require an assignment/availability history, positivity/identifiability contract, and excursion-effect estimand; see #9831.


## Decision-focused expected value of sample information (EVSI)

Target-query EIG asks how much an experiment teaches about a declared causal answer. It is still not a decision objective. A downstream decision requires an explicit action set and utility for each action under each candidate hypothesis, including OTHER_OR_UNRESOLVED.

The \`DecisionProblem\` API takes:
- ordered hypothesis IDs, exactly matching the open-world candidate;
- one or more downstream actions;
- a finite utility for each action × hypothesis pair.

For each likelihood scenario, the implementation computes current best expected utility, expected post-experiment utility after selecting the best action for each observed outcome, and their difference (EVSI). Candidates are ranked by worst-case EVSI per unit cost over the declared finite likelihood ambiguity set. It includes a no-action/control alternative when the caller declares one.

### Synthetic known-answer example

Priors are [0.3, 0.3, 0.3, 0.1] over two mechanisms that imply a positive functional-output effect, one no-effect mechanism and OTHER_OR_UNRESOLVED. The utility matrix assigns utility 2 to intervention under either positive mechanism, -2 under no-effect and -1 under unresolved; no-intervention has utility 0 in all cases. Current best expected utility is 0.5.

- The mechanism-discriminating experiment separates the two positive mechanisms, but both imply the same optimal action. EVSI = 0.
- The outcome-discriminating experiment distinguishes positive from negative evidence. After response, intervene; after no response, do not intervene. Expected post-experiment utility is 0.97, so EVSI = 0.47 at unit cost.

The EVSI objective therefore ranks the outcome-discriminating experiment above the mechanism discriminator. This is a decision-specific result under the supplied synthetic utility matrix and likelihoods, not an empirical treatment recommendation.

Input: \`schemas/decision-value-input.schema.json\`. Output: \`schemas/decision-value-output.schema.json\`. Fixtures: \`fixtures/decision-value.example.json\` and \`fixtures/decision-value-known-answers.json\`.

### Boundary

EVSI values depend on the utility matrix, target population and downstream actions. They are not the expected benefit of administering the experimental perturbation, and they do not justify unsafe actions. Experimental costs and biological safety must remain separate hard constraints. The worst-case EVSI guarantee is only over the caller-specified finite ambiguity set; the model-adequacy gate and held-out mechanism calibration remain required.


## Threshold-hit acquisition (Probability-of-Hit style)

Some scientific searches care about discovering interventions likely to cross a preregistered meaningful threshold, not identifying the complete mechanism and not necessarily optimizing a downstream utility matrix. The threshold-hit objective declares:
- a numeric outcome value for every candidate outcome label;
- a threshold and direction (at least / at most);
- the exact outcome ordering for every candidate.

The selector calculates the prior-predictive probability of a threshold-crossing outcome under each declared likelihood scenario, reports best/worst probability and probability per cost, and ranks by worst-case hit probability per cost. This is a transparent discrete Probability-of-Hit-style acquisition, not a claim to reproduce the full continuous Bayesian-optimization method from Rubbi et al.

### Synthetic known-answer

With priors [0.3, 0.3, 0.3, 0.1] and a functional-output hit threshold of at least 0.8:
- The mechanism discriminator has hit probability 0.50.
- The hit-focused discriminator has hit probability 0.56 and ranks first at equal cost.

This objective is distinct from global EIG, target-query EIG and EVSI. It seeks the chance of crossing a specified outcome threshold; it does not say whether crossing is safe, useful, causally identified, or the best downstream decision. Safety and feasibility remain hard gates.

Input: \`schemas/hit-probability-input.schema.json\`. Output: \`schemas/hit-probability-output.schema.json\`. Fixtures: \`fixtures/hit-probability.example.json\` and \`fixtures/hit-probability-known-answers.json\`.

Related comparator: Rubbi et al., Many Needles in a Haystack (ICML 2026), https://proceedings.mlr.press/v306/rubbi26a.html.
