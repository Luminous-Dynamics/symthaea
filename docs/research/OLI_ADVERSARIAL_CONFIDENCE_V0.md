# OLI adversarial confidence-surface harness v0

**Status:** first executable red-team primitive; not a published replication  
**Owner:** Symthaea / `symthaea-alife`  
**Related:** [OLI-OOD-ATTACK-001 #11451](https://github.com/Luminous-Dynamics/symthaea/issues/11451)

## Purpose

The adversarial confidence harness searches for discrete candidates that a classifier scores highly. It makes the central distinction explicit:

> Model confidence is the attack objective, not ground truth.

The harness has no replication, life, intelligence, or consciousness oracle. The benchmark adapter must evaluate returned candidates with a separate known-truth evaluator after the search and preserve both results.

## First implementation

Module: `crates/domains/symthaea-alife/src/adversarial_confidence.rs`

- Inference-only `ConfidenceModel` trait.
- Fixed-length discrete candidate representation (`Vec<u8>`).
- Single-site mutation over an explicit finite alphabet.
- Deterministic, seeded XorShift64 proposal generator.
- Greedy confidence ascent: accept only strict score improvements.
- Random-walk control: accept every proposal but retain the best candidate separately from the final candidate.
- Strict score validation: confidence must be finite and in [0, 1].
- Exact per-query trace, candidate, confidence, acceptance, and best-so-far flag.
- No external random-number dependency and no access to ground-truth labels from the search API.

This is deliberately a benchmark primitive, not an Avida simulator, classifier, or result reproduction. The model is injected by a caller; no classifier is trained or evaluated by this module.

## Mapping to Gupta & Adami (2026)

Gupta & Adami, *Can AI Detect Life? Lessons from Artificial Life*, arXiv:2604.11915v2 (22 June 2026), describes:
- 26-symbol Avida instruction alphabet and length-9 sequences;
- 36,171 viable sequences among (26^9) possible strings;
- an 80/10/10 split of known positives, with sampled negatives to produce balanced splits;
- reference MLP test accuracy of 99.97%;
- 30 seeds × 52 starts = 1,560 attack runs;
- single-site greedy confidence ascent and alternative search controls;
- mean confidence reaching 1.0 by the reported query count 150;
- no actual replicator recovered among the reference MLP attack endpoints.

Paper: https://arxiv.org/abs/2604.11915

The paper explicitly limits the inference: Avida sequence-space spoofing shows an OOD confidence vulnerability in a digital-polymer task; it is not a direct spoofing experiment against py-GC-MS spectra or actual extraterrestrial samples. The paper calls for that separate follow-up.

### Query accounting

This API counts every model call, including the initial candidate score. The paper plots the initial candidate at query 0 and describes a 300-mutation proposal budget. A faithful adapter should therefore report both:
- `proposal_queries = model_queries - 1`;
- `model_evaluations = model_queries`.

For 300 proposed mutations plus the initial baseline score, use `max_model_queries = 301`. Do not compare curves until the paper's original counting convention has been reconciled with the reproduced source implementation.

### What is not yet reproduced

- The exact Avida 9-mer viable-sequence corpus and functional oracle.
- The reference MLP architecture, training split, training protocol, and baseline metrics.
- The 30 × 52 run matrix and reported confidence trajectories.
- Random initialization protocol and exact proposal-selection semantics.
- Stochastic hill climbing, tournament selection, truncation selection, CNN, transformer, SVM, random forest, XGBoost, and logistic-regression comparisons.

These remain work under #11451. A toy model in a unit test verifies the attack API only; it is not evidence of a reproduced scientific result.


## Avida ground-truth identity trap: circular code rotations

The underlying complete-map paper (C G & Adami, 2021) explicitly notes a distinction between 30,547 replicator representatives when circular code rotations are treated as equivalent and 36,171 raw length-9 sequences when rotation variants are counted. The 2026 adversarial study uses 36,171 unique lower-case length-9 strings in its classification split.

Sources:
- 2021 genotype-phenotype map: https://arxiv.org/abs/2107.03522
- 2026 adversarial study: https://arxiv.org/abs/2604.11915

The reproduction adapter must not silently canonicalize away this distinction. Preserve both:
- the raw length-9 sequence identifier used by the classifier and its split;
- a circular-rotation canonical identifier for functional equivalence analysis, if the Avida execution semantics and source oracle confirm that equivalence.

Ground-truth validation must use the original colony-forming criterion, not merely string membership or a model label. Report both raw-sequence and rotation-aware distances when analyzing spoof endpoints. This is a benchmark reproducibility requirement, not an assumption that all rotations are functionally equivalent in every Avida configuration.


## Published attack baselines to reproduce

The arXiv v2 paper gives a more complete reference table than the headline q=150 result. For the reference MLP, mean *best-seen* confidence was:

| Model query | Random start | Uniform start |
|---:|---:|---:|
| 0 | 0.0000 | 0.1375 |
| 25 | 0.3203 | 0.3409 |
| 50 | 0.8266 | 0.7685 |
| 75 | 0.9708 | 0.9532 |
| 100 | 0.9972 | 0.9973 |
| 125 | 0.9998 | 0.9997 |
| 150 | 1.0000 | 1.0000 |
| 175 | 1.0000 | 1.0000 |
| 200 | 1.0000 | 1.0000 |

At a fixed budget of 300 model queries, the paper's search-policy ablation reported:

| Search policy | Uniform-start mean best confidence | Random-start mean best confidence | True replicator recovery |
|---|---:|---:|---:|
| Greedy hill climb | 100.00% | 100.00% | 0% |
| Random walk | 28.70% | 19.30% | 0% |
| Stochastic hill climb | 47.00% | 43.09% | 0% |
| Tournament selection | 75.80% | 87.27% | 0% |
| Truncation selection | 98.32% | 99.26% | 0% |

The classifier ablation is also material. At 300 queries, logistic regression, SVM, and transformer reached 100% mean best-seen confidence on both start types; XGBoost reached 96.64% on both. Random forest was less susceptible (40.76% uniform / 74.53% random), but its best-seen endpoint was still a true replicator in only 12.69% / 26.03% of runs. CNN reached 76.55% / 100% confidence, with true-replicator endpoint rates 0.51% / 1.03%. This is why OLI should report confidence and oracle-verified function separately for every architecture and attack.

Source: Gupta & Adami, arXiv:2604.11915v2, Tables 2–5 and supplementary attack discussion.

## Search policies now implemented in the draft harness

The Rust module now contains five query-budgeted policies:

1. **Greedy ascent:** accept only strict confidence improvements.
2. **Random walk:** accept every single-site mutation; keep best-seen separately.
3. **Stochastic hill climb:** use a fixed, explicit Metropolis temperature to sometimes accept lower-confidence moves.
4. **Tournament population:** choose parents through configurable k-way tournaments; use elitist top-N survival.
5. **Truncation population:** choose parents uniformly from a configurable top fraction; use the same elitist top-N survival.

The two population methods accept an ordered initial population. The model-call budget includes evaluation of the initial population. With a 30-member population and 300 total model calls, this implementation evaluates the initial population plus nine offspring populations. The paper's "30 members / 10 generations / 300 queries" accounting must be checked against its implementation details before a source-faithful comparison is claimed.

The current tournament and truncation implementations are **explicit paper-inspired variants**, not asserted to be exact reproductions of the authors' code. The stochastic temperature, tournament size, truncation fraction, parent/survivor semantics, and initial-population construction are all recorded configuration and must be matched to source details before a benchmark replication is claimed.

## Metrics required for every run

- Mean and maximum **best-seen** confidence at fixed proposal/query checkpoints.
- Terminal-state confidence, reported separately from best-seen confidence.
- Ground-truth replicator recovery using the independent colony-forming oracle.
- Query count to thresholds (0.5, 0.9, 0.99, 0.999).
- Raw Hamming distance to nearest viable 9-mer; rotation-aware distance reported separately if the source oracle confirms circular equivalence.
- Distinct endpoint count, repeated-attractor frequency and cross-seed variance.
- Per-start-type and per-model-family results.
- False-negative behavior and in-domain recall under any defense.
- Runtime and candidate/model-evaluation cost.

The paper's confidence curves are best-seen metrics. Reporting only final-state confidence would be a different statistic and must not be substituted.


## Minimum next benchmark increment

1. Acquire or reconstruct the 9-mer ground-truth set from a source with reproducible provenance and permission to use it. The paper points to the complete genotype–phenotype map, but exact artifact/version and license must be confirmed before ingestion.
2. Reproduce the train/validation/test split law and classifier before running the attack.
3. Pin the candidate encoding, optimizer proposal law, random seeds, model calls, and functional oracle.
4. Run the exact reported reference configuration and publish a result table with confidence at queries 0/25/50/75/100/125/150/200/300.
5. Report actual replicator endpoints separately from maximum predicted confidence.
6. Compare the same attack budget against the simple baselines and defenses in #11451.
7. Keep all model and dataset digests, seeds, source commits, and outputs in a replay manifest.

## Tests included

The module tests:
- a toy confidence attractor that is independently known not to be the functional positive class;
- random walk best-so-far versus final-state separation;
- fixed-seed trace reproducibility;
- explicit query-budget accounting;
- rejection of NaN, infinity, and out-of-range scores;
- invalid candidate and configuration rejection.

These are unit-level structural checks only. The local environment for this authoring pass did not have Rust/Cargo installed, and no exact-head CI pass is claimed here.

## Claim ceiling

This code establishes that Symthaea contains a deterministic, query-budgeted discrete confidence-search primitive with explicit separation between model score and functional truth, subject to code review and test execution. It does not establish resistance to OOD spoofing, reproduce Gupta & Adami, detect natural life, or qualify any scientific instrument.
