# AtomFab: Pareto discovery, not one-number optimization

**Date:** 2026-10-20  
**Status:** Research protocol and executable metric proposal; no scientific performance claim.

## Paradigm shift

AtomFab must not collapse discovery count, best validated material objective, and cost into one hand-tuned scalar. A weighted score can hide a policy that discovers more structures but produces worse minima, or a policy that finds a good result only by spending more. The new Rust decision-frontier kernel instead reports nondominated policies in three declared dimensions:

1. **Discovery breadth:** maximize unique structures that passed the frozen reference validator.
2. **Validated objective:** optimize the best objective among those discoveries in its declared direction and unit; the preregistered tolerance defines a meaningful improvement threshold, while no-worse comparisons remain direction-correct to keep dominance transitive.
3. **Total cost:** minimize integer fixed-resolution cost units, with an explicit unit.

Wall time and the number of reference calculations remain visible in each candidate record but are not silently converted into extra weighted objectives. A benchmark may add a distinct time-efficiency analysis when the hardware/environment is fixed. This preserves interpretability instead of inventing a universal scalar.

## Non-negotiable feasibility boundary

A candidate with any false-feasible acceptance or without an explicit hard-constraint assessment is marked **ineligible**, but remains visible in the report. Safety cannot be traded against a better objective or lower cost. A candidate with zero discoveries must have no best validated objective; a non-finite objective or a free/zero-cost outcome is invalid. Each candidate has a comparison-contract digest and evidence-bundle digest.

Frontiers may be compared only within one matched seed and exact comparison contract. Across seeds, report paired campaign distributions, effect sizes and bootstrap intervals instead of pooling rows into a pseudo-single campaign. The output is deterministic under input reordering.

## Why this matters for supported nanoparticles

Xu et al. (2026) specifically distinguish energy accuracy from structural exploration for Cu nanoparticles on Al2O3 surfaces. Their study reports that MatterSim-v1.0.0-1M can discover lower-energy configurations in some cases despite larger binding-energy deviations, while the domain-specific DP-UniAlCu model was roughly two orders of magnitude faster than the universal models in the reported MD comparison. This motivates two separate scientific scoreboards and explicit cost accounting; it does not imply one policy or model wins universally.

Primary source: [Xu et al., *npj Computational Materials* (2026), DOI 10.1038/s41524-026-02043-8](https://doi.org/10.1038/s41524-026-02043-8).

The reference benchmark uses supported nanoparticles, including Cu on alumina, and describes the UniAlCu dataset as available from its related publication. A separate publicly listed Zenodo dataset (DOI 10.5281/zenodo.15865800) contains training/validation structures and global minima for the related Cu/Al2O3 sintering work; **do not assume it is byte-identical to the 2026 benchmark's complete evaluation set without checking the publication's exact data manifest**. [Zenodo record](https://zenodo.org/records/15865800).

For a first benchmark, freeze task-specific structure identities, relaxation settings, reference methods, deduplication tolerances and split-by-parent-system rules before evaluating policies. Reproduce the published protocol for the chosen systems before altering the search.

## Benchmark gates

- Require unique policy IDs and identical seed, comparison-contract digest, objective definition/direction/tolerance, and cost unit.
- Mark unsafe/unassessed plans ineligible; never remove them from reporting.
- Compare policies on the Pareto frontier of validated discoveries, validated objective, and total cost.
- Preserve every per-seed run; report median paired deltas, bootstrap intervals, target-reaching acceleration only when both policies reach the same frozen target, plus all failed runs.
- Report energy/force error on matched structures separately from discovery metrics on each policy's generated structures.
- Compare against a pinned domain-specific baseline (DP-UniAlCu where access and terms permit), pinned universal baselines (e.g., MACE-OMAT and MatterSim versions compatible with the task), randomized/local search, and the same potential with Symthaea's selection logic disabled.
- Do not claim state-of-the-art superiority until the exact task, source data, checkpoint, calculation settings, evaluator, compute budget, and baseline results are independently reproduced.


## Published ranking parity: RBO and Kendall tau-b

The 2026 paper reports both rank-biased overlap (RBO, with persistence p=0.94) and Kendall's tau for model-vs-DFT energy rankings over the same DFT-refined candidate set. AtomFab's existing normalized finite-prefix RBO-style metric intentionally implements a different finite-prefix contract, so it must not be labelled as the paper's exact RBO result. A new pure Rust module, `published_ranking.rs`, separately computes finite-list extrapolated RBO and tie-aware Kendall tau-b on the identical canonical candidate universe. Exact RBO ties break by canonical structure ID; Kendall tau-b uses the frozen energy tie tolerance. Discovery-set Jaccard remains separate: the ranking metrics are undefined for different candidate sets, not a substitute for measuring unique discoveries. [Xu et al. (2026), Methods and Table 2](https://doi.org/10.1038/s41524-026-02043-8).

Artifacts: `crates/domains/symthaea-atomfab/src/published_ranking.rs` and `schemas/published-ranking-metrics-v1.schema.json`. Hand-calculated tests cover identical rankings (RBO=1, tau-b=1), reversed rankings, energy ties, mismatched candidate IDs, duplicates, and non-finite energies. Exact-head CI still needs to validate this new module before it is counted as passed.

## Implementation

- Rust module: `decision_frontier.rs`
- Illustrative candidate fixture: `fixtures/decision-frontier-candidates-v1.json`
- Candidate schema: `schemas/decision-frontier-candidates-v1.schema.json`
- Report schema: `schemas/decision-frontier-report-v1.schema.json`

The fixture values are synthetic test inputs, not experimental or benchmark results. This module supplies the evaluation machinery; it does not run ML interatomic potentials or perform DFT.
