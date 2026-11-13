# Battery design decision-value benchmark v0.1

Status: finite known-truth decision evaluator and synthetic fixture. No measured battery performance or physical experiment advantage is claimed.

## Why this benchmark exists

The inverse-design seed PR #11672 has a transparent information-gain-per-resource heuristic. That is a useful first baseline, but raw information gain is not engineering value. The next battery design experiment should be selected because its possible results can change an admissible design or qualification decision.

A high-entropy observation can be irrelevant to the design choice. A narrow uncertainty at a hard mission or manufacturing boundary can dominate the next decision. This benchmark makes those differences measurable.

## Reused owners

- Symthaea BATTERY-INVERSE-DESIGN #11671 owns mission-conditioned candidate design and finite experiment proposals.
- Symthaea SE-OPT-BENCH #7803 owns the cross-domain decision-value benchmark contract.
- Symthaea POWERTRAIN-DISCOVERY #10143 owns reusable multi-objective optimization and decision-aware experiment-selection comparisons.
- MAT-013 #4344, MAT-PSPP #6224 and MAT-UQ #6225 own physical lineage, process/structure/property semantics and calibration.
- Symthaea BATTERY-DESIGN-LIFETIME-TRANSFER #11686 owns unseen-design lifetime transfer.
- Symtropy EV-BATTERY-CLOSED-LOOP-QUAL #4848 owns downstream design-to-mission evidence closure.

This benchmark consumes those authorities. It is not a new generic optimizer or uncertainty engine.

## Two distinct modes

### Probability-bearing EVSI

For an admitted probability model, expected value of sample information is:

EVSI = current optimal expected loss - expected optimal posterior loss after observing the experiment.

The model, validity domain, calibration profile and evidence receipt are explicit. The selected decision must remain feasible in every world with non-zero posterior support.

The fixture probabilities are synthetic known-truth frequencies by construction. They do not represent a learned or empirically calibrated battery model.

### Distribution-free minimax regret

When uncertainty is a set of possible worlds, the evaluator does not invent probabilities. It calculates current worst-case decision regret and the worst post-observation regret across every possible experiment outcome.

Robust regret reduction = current worst-case regret - worst post-observation regret.

The two outputs are not numerically interchangeable. EVSI is not computed from interval endpoints.

## Benchmark challenge

The two-world fixture defines:
- DESIGN-A: loss 0 in WORLD-A, loss 10 in WORLD-B;
- DESIGN-B: the reverse;
- ABSTAIN: loss 6 in both worlds.

The prior is 0.5 / 0.5. Two experiments are compared:

1. HIGH-ENTROPY-IRRELEVANT: outcome entropy is one bit, but the outcome distribution is identical in both worlds. Expected decision value is zero.
2. DECISION-BOUNDARY: outcomes are 90% / 10% conditional on the true world. Expected posterior decision loss is 1, so gross EVSI is 4 before the declared experiment cost.

This is a deliberately tiny known-answer challenge, not a claim about real chemistry.

## Fair comparison tournament

Use the same candidate designs, world support, initial evidence, hidden outcomes, and resource budgets for:
- seeded random;
- space-filling / factorial design where applicable;
- maximum uncertainty/entropy;
- the current #11672 information-gain heuristic;
- generic information gain under explicit probability assumptions;
- multi-objective BO / expected hypervolume improvement where appropriate;
- EVSI / expected regret reduction under admitted probabilities;
- minimax regret / robust feasibility reduction under sets;
- fixed expert/DOE baseline;
- STOP / NO_EXPERIMENT.

Report decision regret, hard-gate false-feasible rate, feasible-design discovery, time/material/compute/cost to decision, calibration and subgroup coverage, OOD abstention, failed/censored/missing outcome accounting, and the fraction of tests that change the selected design. Keep them separate.

The outcome-status counts in the current oracle refer to declared world/outcome possibilities, not observed physical trials. Physical attempted/failed-test ledgers remain owned by the evidence and experiment execution systems.

## Proposed gates — targets, not achievements

- 100% of EVSI calculations bind a probability model, calibration receipt, validity domain and admitted evidence.
- Zero probability derivation from interval endpoints or uncalibrated surrogate variance.
- Zero hard-gate violations offset by utility or experiment value.
- 100% of declared possible, failed, censored, invalid and missing experiment outcome classes remain accounted.
- Deterministic fixtures agree with the independent exhaustive decision oracle.
- On held-out decision-relevant cases, the decision-aware policy should reduce normalized decision regret relative to random and information-only selection at equal experiment cost.
- On no-information cases, the evaluator should select STOP / NO_EXPERIMENT.

A future physical superiority claim requires prospective blind candidate builds and independently reviewed complete-cell outcomes.

## Claim ceiling

A software PASS demonstrates only decision arithmetic and protocol correctness on the finite test worlds. It does not validate real battery priors, experimental likelihoods, material synthesizability, physical cell performance, process causality, cycle life, safety, manufacturing yield or commercial economics.
