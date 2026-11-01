# NATS-Bench-MCU v2 strict Rust import and replay contract

Tracking issue: #9411  
Dataset DOI: 10.5281/zenodo.21644470  
Author repository: https://github.com/ankilab/NATS-Bench-MCU  
Pinned mirror commit: c92adc0ec112bdee6c73358cc52f3e026367b5a0

## Source identity

The authors' corrected publication CSVs are tracked under export_corrected/. The importer pins SHA-256 of the exact UTF-8 CSV contents and the repository commit.

| File | SHA-256 |
|---|---|
| hw_metrics.csv | ae318c606d6095ec31ac7383c20e330a6b1e850c8430f6ba20974e00c5d4f693 |
| accuracies.csv | 948fa57fa643dbf749dc394f9f38ab20639737de1a2054d5a72e1be15dd76fae |
| failures.csv | 88fbc6c4a79f44c2a6c33cdfb5a47eca3cc3a038d88b136cda7b39303413565f |
| coverage.csv | f0b6eed8771757d16f4fc0d961251aff4ce850c05f676488ca9e18e306b7da47 |
| int8_rank_correlation.csv | ae764d9267d3f530a9e5396bdee618baed88a6fc65af859ac81cce74427a5596 |
| int8_rank_summary.csv | b968e3cc98cdb3433263faf9612c41f83ee779eafffb8ff13b0f34802d38d5ad |
| nas_baselines_summary.csv | 4fad709bb1b6566ee09a780db841d26b3c640564099f7926022ca52a94136fd3 |
| README.md | cac028ee0c345d544b31a7b796622956912294281b1a9cc0f15449ade330c5cb |

These SHA-256 digests were computed from the exact author-repository file contents at the pinned commit. Git blob SHA-1 values are separate identifiers, not substitutes for SHA-256.

## Import invariants

- Exact CSV headers and column order.
- RFC-4180 quoted-field handling, including doubled quotes, embedded commas and JSON memory-category fields.
- Unique measured hardware and accuracy candidate IDs.
- One-to-one accuracy join for all measured candidates.
- Failure IDs disjoint from hardware IDs.
- Complete 0..15,624 candidate partition: 14,281 measured + 1,344 infeasible.
- v2 failure label strictly infeasible; no silent v1 taxonomy recoding.
- Successful INT8 model sizes no greater than 746 KB and successful ROM no greater than 1,024 KB.
- Failed INT8 flatbuffer sizes must exceed the pinned 746 KB feasibility ceiling.
- Positive finite latency/current/power/energy/ROM/RAM fields; bounded accuracy percentages; embedded memory-category JSON must be non-empty and finite.
- Coverage pipeline stage counts and percentages must reconcile to the frozen population.
- The import receipt binds the exact source manifest, file hashes, measured/infeasible counts, summary statistics, board summaries and two-dimensional Pareto counts.

The CSV parser is implemented without a new third-party dependency. The crate uses its existing serde, serde_json and sha2 dependencies.

## Observed corrected-export diagnostics

The author mirror parsed to 14,281 unique hardware rows, 14,281 unique accuracy rows, 1,344 unique infeasible rows and exactly 15,625 unique candidate IDs. Accuracy coverage was 100%; failure rows did not overlap successful measurements.

- Mean latency 3.0881 s; median 3.1818 s; p90 5.2265 s; p99 6.4165 s.
- Mean power 11.7167 mW; coefficient of variation 1.732%.
- Mean energy 36.2086 mJ; median 36.7582 mJ; p90 61.3546 mJ; p99 74.6842 mJ.
- Median ROM 633.5986 KB; max ROM 1,021.8877 KB; max INT8 flatbuffer 744.3047 KB.
- RAM is 366.2861 KB throughout this export; tensor arena allocation is 350 KB.
- Pearson correlation of latency and energy is 0.99961.
- Derived two-objective CIFAR-10 Pareto fronts: 28 for latency, 27 for energy and 21 for ROM. These are maximize-accuracy/minimize-one-cost counts, not a composite leaderboard.
- Three nRF5340-DK board IDs exist in the export, but each architecture has one board assignment, not same-architecture replication on all three. The data alone cannot establish per-architecture fleet robustness.

### Publisher baseline summary

| Algorithm | Final accuracy mean | Gap to oracle | Median evaluations to oracle | Wasted eval fraction |
|---|---:|---:|---:|---:|
| Random Search | 92.834% | 0.416 pp | 169 | 54.4% |
| Regularized Evolution | 92.810% | 0.440 pp | 110 | 33.2% |
| Local Search | 92.989% | 0.261 pp | 133 | 37.8% |

The rank export reports FP32-vs-INT8 Spearman 0.999 and top-10 overlap 9/10, while NATS accuracy vs INT8 rank reports Spearman 0.9027 and top-10 overlap 4/10. Search ranking based on a proxy must therefore be tested against the objective actually optimized.

## Proposed Symthaea replay target

On the same TSS candidate universe, task, data split, fidelity, budget and stopping rule, target:
- CIFAR-10 oracle gap no worse than 0.261 percentage points;
- median evaluations at least 25% below the 110-evaluation Regularized Evolution reference (target <=82.5);
- wasted evaluations below the 33.2% Evolution baseline;
- no increase in false-feasible deployment outcomes;
- report wall time, compute, build, instrument and total vector cost separately.

These are future acceptance targets, not achieved results. The publisher baseline rows are parsed and preserved but are not re-run by this importer.

## Evidence boundary

This is an import/replay substrate only. It does not retrain the models, reproduce the original optimization runs, compile/flash firmware or perform new physical measurements. The hardware values were acquired by the dataset authors; Symthaea validates source hashes, structure, joins and derived summary calculations. New algorithmic performance claims require an executable matched-budget replay and exact-head reproduction.
