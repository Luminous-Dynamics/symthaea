# OLI-XD-13: targeted/untargeted measurement portfolio and discovery reserve

**Status:** synthetic known-truth development fixture; not a spacecraft payload recommendation  
**Owner:** Symthaea OLI experiment-selection adapter  
**Related:** [OLI-EXPERIMENT-001 #11446](https://github.com/Luminous-Dynamics/symthaea/issues/11446)  
**Source comparator:** Davila et al., *The Abiotic Background: Challenges and Opportunities in the Search for Evidence of Life Beyond Earth*, Astrobiology 26(11), first published 15 September 2026, DOI [10.1177/15311074261484864](https://doi.org/10.1177/15311074261484864)

## Research premise

The 2026 interdisciplinary workshop report describes a fundamental mission-design tension: targeted measurements are easier to trace to registered hypotheses and can offer high sensitivity; untargeted measurements preserve discovery potential in unfamiliar chemical space but may increase contamination/interpretation burden; contextual measurements constrain the abiotic baseline; all must fit spacecraft mass, power, data, reliability, operations and mission-traceability constraints. The report explicitly asks how much measurement flexibility should be held for unexpected findings. <Cite refs={["turn845638view0"]}/>

OLI should therefore optimize a **portfolio** of complementary capabilities rather than always selecting a single next experiment.

## Machine-readable case

The finite development fixture is `benchmarks/oli/measurement-portfolio-option-value-v0.json`.

It defines four synthetic hidden worlds:
- registered target biosignature;
- abiotic organic background;
- unfamiliar biological signature outside the targeted library;
- unfamiliar abiotic process.

Five actions are available: targeted chirality, untargeted organic profiling, independent contextual geophysics, mandatory blank/calibration bundle, and reserved post-acquisition discovery reanalysis.

The action probabilities, utility values and resource costs are entirely synthetic. They are intended to test optimizer behavior and exact-oracle parity, not estimate Enceladus or any other ocean world.

## New design variable: discovery option value

The fixture reserves 2 of 10 data-storage units (20%) for reanalysis or follow-up capability that remains uncommitted until a novel signal is found. The reserve is not counted as a measurement already performed.

The benchmark sweeps reserve fractions from 0 to 30% and reports:
- novel-signature recall and reanalysis success;
- incremental expected utility at fixed total resource budget;
- lost targeted sensitivity from reserving capacity;
- false-positive and false-negative changes;
- marginal value and opportunity cost of reserve.

No universal reserve percentage is prescribed. An acceptable optimizer must expose the Pareto frontier and the dependence on the assumed frequency/value of unexpected signatures.

## Non-tradable constraints

The optimizer must always preserve the mandatory blank/calibration bundle and fit mass, peak power, operation time, storage and sample-consumption budgets. Contamination control, rights, safety, planetary protection and human authorization are hard constraints; they cannot be offset by higher nominal information gain.

## Common-cause dependence

The targeted and untargeted channels share a sample-preparation group. Their outcomes cannot be treated as conditionally independent merely because they are different instrument functions. Contextual geophysics has a separate measurement group. The benchmark requires a dependence-aware joint likelihood and an ablation against a naive-independence model.

## Baselines

Under the same action budget and hard constraints, compare:
1. all-targeted;
2. all-untargeted;
3. fixed expert mixed portfolio;
4. nominal-EIG greedy;
5. cost-normalized greedy;
6. robust/minimax portfolio;
7. explicit discovery-reserve portfolio;
8. exact subset oracle.

For this five-action fixture the exact oracle can enumerate all 32 subsets, reject infeasible combinations and calculate the feasible Pareto set. The exact oracle is a small-case grader, not a scalable mission planner.

## Required metrics

Report separately:
- expected and robust lower-bound utility;
- cost-to-resolution and exact-oracle optimality gap;
- targeted-signal sensitivity;
- novel-signature recall;
- abiotic false-positive rate and false-absence rate;
- calibration error;
- common-cause penalty;
- discovery reserve fraction;
- mass, power, data, time and sample use;
- feasibility and safety violations (target exactly zero).

Never collapse these into one scalar leaderboard score without publishing all component metrics and the trade-off curve.

## Adversarial cases

- The targeted-only portfolio misses both unfamiliar signature families.
- The untargeted portfolio is more exposed to abiotic false positives when context/calibration are omitted.
- The two sample-prep channels are mistakenly credited as independent evidence.
- A nominal-EIG optimizer consumes the entire budget on known hypotheses and leaves no discovery reserve.
- Removing contextual geophysics increases ambiguity between biological and abiotic worlds.
- A novel signature is recognized only after acquisition; the archive and reserved reanalysis capability determine whether it can be investigated.

## Claim ceiling

This fixture creates a reproducible development benchmark for budgeted portfolio selection and discovery option value. It does not validate a mission design, estimate real ocean-world chemistry, authorize instrument selection, or establish life.
