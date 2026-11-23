# SC-BUDGETED-SELECTION-001 — fixed-budget candidate-selection benchmark

**Tracking issue:** [Symthaea #13107](https://github.com/Luminous-Dynamics/symthaea/issues/13107)  
**Parent programme:** [SC-DISCOVERY-TOURNAMENT-001 #12634](https://github.com/Luminous-Dynamics/symthaea/issues/12634)  
**Implementation branch:** `research/sc-budgeted-selection-v0.1`

## Why this benchmark exists

A retrospective regression score does not answer the operational question: with a fixed follow-up budget, which candidate policy delivers the most independently credible outcomes?

BEE-NET reports 0.87 K MAE for its (T_c)/Eliashberg predictions relative to DFT-based Allen–Dynes targets. Its multi-stage workflow screened more than 1.3 million structures to 741 dynamically and thermodynamically stable compounds with DFT (T_c>5) K, reporting 86% final precision after expensive (alpha^2F(omega)) computations and two synthesized candidate compositions. These are meaningful computational and workflow results, but the 0.87 K figure is not an experimental-(T_c) error. The paper also says its ambient-pressure conventional electron–phonon training is not directly transferable to unconventional nickelates or high-pressure hydrides without track-specific data and models:
https://www.nature.com/articles/s41524-026-01964-8

A 2026 review argues that random splits can leak doped variants or structurally similar material families across train/test and inflate apparent generalization. It recommends family / prototype disjoint splits and explicit OOD and uncertainty evaluation:
https://pmc.ncbi.nlm.nih.gov/articles/PMC13261481/

The benchmark therefore separates two questions:
1. Is a property prediction numerically accurate for a well-defined target?
2. Does the model allocate scarce DFT / synthesis / characterization effort better than simple alternatives?

The first is a model-validation task. The second is a decision-quality task. They must not be conflated.

## Current executable contract

- Schema: `data/schemas/sc_budgeted_selection_v0_1.schema.json`
- Synthetic fixture: `tools/superconductivity-discovery-harness/fixtures/budgeted_selection_smoke_v0_1.json`
- Rust auditor: `tools/superconductivity-discovery-harness/src/bin/budgeted_selection_audit.rs`

The v0.1 cohort is deliberately restricted to one target track:
- mechanism: conventional electron–phonon;
- target: binary superconductor classification;
- evidence class: synthetic.

This is not a BEE-NET reproduction and not a (T_c)-regression benchmark. Experimental measured (T_c), DFT Allen–Dynes (T_c), Eliashberg calculations, synthesis success and bulk confirmation require separate target tracks and separate cohorts.

## Split and candidate rules

The auditor enforces train/evaluation disjointness by:
- family ID;
- structure prototype ID;
- reduced formula;
- source paper ID.

Candidate IDs and nominal material IDs must be unique across both pools.

The evaluation candidates carry:
- known synthetic truth for scoring only;
- positive integer follow-up cost;
- model probability or explicit abstention;
- domain status and abstention reason;
- chemistry heuristic score;
- deterministic random rank.

An in-domain scored candidate must have a finite probability in ([0,1]) and must not be marked abstained. An out-of-domain candidate must have null probability, a declared abstention reason and `abstained=true`. The model policy cannot select OOD or abstained candidates.

## Compared policies

All policies receive the same evaluation pool and same total cost budget:
1. Model probability per cost unit, with OOD abstention.
2. Chemistry heuristic score per cost unit.
3. Deterministic random order.
4. Exact synthetic-truth oracle, which defines a ceiling over this generated fixture only.

The model's and baseline's selected outcomes are compared by selected count, true positives, precision, recall, true positives per cost unit, enrichment versus random and regret against the oracle.

Calibration is evaluated only on non-abstained candidates:
- Brier score;
- five-bin expected calibration error;
- classification error at a 0.5 probability threshold;
- explicit coverage / abstention count.

This avoids a method appearing calibrated merely because it abstains on nearly all difficult cases.

## Synthetic adversarial case

The fixture contains four held-out families and intentionally includes overconfident false positives, lower-scoring positives, an OOD candidate and heterogeneous costs. Under the five-unit budget, its authored expected result is:
- model policy: 2 true positives;
- exact synthetic oracle: 3 true positives;
- model regret fraction: one third;
- enrichment over deterministic random: 2x.

Those are fixture arithmetic checks only. They are not experimental discovery results, measured model performance or expected real-world improvement.

## Next phases

1. Reproduce the admitted BEE-NET baseline on its exact dataset, split, checkpoint, target and compute environment before adapting it.
2. Reproduce a simple chemistry baseline and a family/prototype-disjoint conventional ML baseline.
3. Add leave-family-out, leave-prototype-out, source-paper-held-out, chemistry-held-out and time-forward cohorts.
4. Add probability calibration, selective risk and OOD detection based only on training/validation groups.
5. Compare candidate selection at identical cost budgets and report paired uncertainty intervals.
6. Freeze prospective candidates before DFT confirmation, literature lookup or experimental follow-up.
7. Add independently confirmed bulk/synthesis outcomes as separate labels; do not use DFT values as a substitute.

## Claim ceiling

A passing synthetic fixture validates evaluator semantics and arithmetic only. It does not reproduce BEE-NET, establish extrapolative model quality, validate experimental (T_c), discover a new superconductor or authorize candidate procurement/synthesis.