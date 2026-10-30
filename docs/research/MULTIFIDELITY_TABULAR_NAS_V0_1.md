# Multi-Fidelity Tabular NAS Firewall v0.1

**Status:** Rust source and synthetic tests authored; not compiled, executed, or qualified.  
**Tracking:** [#9169](https://github.com/Luminous-Dynamics/symthaea/issues/9169), dependent on [#9164](https://github.com/Luminous-Dynamics/symthaea/pull/9164).  
**Implementation:** `crates/domains/symthaea-architecture-engineering/src/multi_fidelity_tabular_nas.rs`.

## 1. Why add a second evaluator

The query-firewalled tabular NAS evaluator in `tabular_nas_firewall.rs` models ZAPS-style search: a candidate receives a final target label only when a unique architecture query is made. This provides a clear query-budget contract, but it does not fully represent early-stopping or multi-fidelity methods.

Two relevant 2026 methods make the gap concrete:

- **ZAPS** (Touayouch et al., arXiv:2609.14184, September 2026) uses zero-cost proxy rankings and active acquisition under distinct architecture-evaluation budgets. At 200 NAS-Bench-201 evaluations, it reports P@100 of 52.3% on CIFAR-10 and 65.8% on CIFAR-100 (means over 200 seeds), alongside normalized regret. Its offline ProxyFit calibration uses full-search-space proxy-accuracy correlations; that protocol must remain separate from a strict no-oracle calibration protocol. [Paper](https://arxiv.org/abs/2609.14184).
- **CoRA-NAS** (Yang et al., arXiv:2609.11884, September 2026) combines coarse rankings from zero-cost proxies with a small number of early-training anchor runs and residual refinement. Its reported CoRA-Refine Spearman correlations are 0.946 on NAS-Bench-201, 0.715 on NAS-Bench-101, 0.786 on TransNAS-Bench-101 and 0.894 on NATS-SSS. The paper reports roughly 1% prefix-training budgets in its benchmark experiments, compares against random+early-stop and successive halving, and includes an important counterexample: on real DARTS, early-curve refinement improved rank correlation but increased compute and did not produce a better Pareto point than faster search approaches. [Paper](https://arxiv.org/abs/2609.11884).

This motivates a common evaluator that separates *candidate exposure*, *final confirmation* and *resource use*. A candidate touched at 5 epochs is not a model confirmed at 200 epochs.

### What CoRA-NAS contributes to the benchmark design

The CoRA-NAS preprint reports CoRA-Refine Spearman correlations of 0.946 (NAS-Bench-201), 0.715 (NAS-Bench-101), 0.786 (TransNAS-Bench-101) and 0.894 (NATS-SSS). The paper uses a stratified anchor set and early training curves; on NATS-SSS it reports a 2,800 epoch-equivalent budget (about 1% of the full 270,000 epoch-equivalent search-space training budget). Its result tables also show that coarse proxy rankings can already be very strong in capacity-dominated spaces, so residual refinement adds little in some conditions.

The real-DARTS discussion is an especially important counterexample: CoRA-Refine improves rank correlation within a shared candidate pool, but incurs additional training compute and does not dominate faster gradient/training-free approaches on the overall accuracy/compute frontier. Our benchmark therefore must report final quality and resource Pareto points, not only proxy-vs-label Spearman correlation.

Reference: [Yang et al., CoRA-NAS, arXiv:2609.11884](https://arxiv.org/abs/2609.11884). These are the preprint's reported results, not Symthaea measurements.

## 2. Information-flow and query contract

The policy receives:
- architecture IDs, topology encodings and the declared zero-cost proxy features;
- a narrow query capability allowing one candidate to be evaluated at a declared fidelity rung;
- the target observation for that candidate at that rung, only after a successful query.

The full candidate-by-fidelity target table, final-fidelity oracle ranking, best final metric and evaluator are private. The benchmark is consumed by the one-shot policy invocation. The capability does not expose unqueried target observations or oracle top-K sets.

A candidate can progress only monotonically through the declared fidelity ladder. Re-querying the same rung or requesting a lower rung is rejected and recorded, without revealing a value or charging cost. Unknown candidate IDs, invalid fidelity IDs, final-query-budget exhaustion and resource-budget exhaustion are also recorded as failed attempts.

This is an API boundary for trusted policy code, not an operating-system sandbox. Production comparisons against untrusted or hostile methods need process/container isolation, with the evaluator as the only process that stores the complete target table.

## 3. Fidelity ladder and honest cost accounting

A manifest declares an ordered ladder with strictly increasing per-candidate costs, such as:

| Fidelity | Example training prefix | Cost units from clean start |
|---|---:|---:|
| F0 | 5 epochs | 5 |
| F1 | 15 epochs | 15 |
| F2 | full 200 epochs | 200 |

The values above are an illustrative ladder, not measured NAS-Bench-201 training costs.

The evaluator requires the caller to select one of two cost models before a run.

### ResumablePrefix

Use when F1 actually resumes the same F0 training run, and F2 resumes F1. Promoting from F0 to F1 costs (15 - 5 = 10) additional units; promoting from F1 to F2 costs (200 - 15 = 185). The total charge for one candidate trained through all rungs is 200 units.

### IndependentRuns

Use when a higher-fidelity result is produced by a fresh run, rather than resuming the existing checkpoint. The same promotions cost 5 + 15 + 200 = 220 units. Charging only the cumulative-cost difference in this situation would undercount compute.

The manifest also declares the resource unit (for example, epoch-equivalents, GPU-seconds or CPU-seconds). Do not compare arbitrary synthetic units against measured GPU-hours. For multi-hardware work, store actual measured resource receipts and keep wall-clock, device time, energy, memory and monetary cost as separate fields.

Every successful intermediate or final observation is charged. Failed attempts do not reveal values and do not charge a training cost in this first contract; the production runner must add actual failed-run and retry costs from execution receipts before using this evaluator for an industry comparison. This is an explicit limitation and a future integration requirement.

Two independent hard caps apply:
1. the maximum number of distinct architectures confirmed at final fidelity;
2. the total resource-unit budget across all fidelity evaluations.

## 4. Two kinds of top-K capture

The evaluator reports two separate families of metrics.

**Touched at any fidelity** counts an architecture as touched if the policy successfully observed it at any declared rung. This is useful for measuring search-space exploration and broad evidence acquisition. It is not a final-quality achievement metric.

**Confirmed at final fidelity** counts only architectures that actually reached the last rung. This is the appropriate population for ZAPS-style P@1/P@10/P@100 and best-found final-quality regret.

An early observation, an architecture prediction or a recommendation that was never queried at final fidelity receives no final-quality credit. The report separately returns:
- unique candidates touched;
- number of intermediate observations;
- number of final-fidelity confirmations;
- P@1/P@10/P@100 for touched candidates;
- P@1/P@10/P@100 for final-confirmed candidates;
- best final metric, final simple regret and normalized regret percentage;
- final-query budget and total resource-budget utilization;
- proxy feature-bank computation time;
- every query attempt, fidelity, cost and outcome digest.

A multi-fidelity method cannot inflate its final P@100 by touching many candidates at a cheap rung. Conversely, a method that spends heavily on a small anchor set can demonstrate useful final-confirmed quality without pretending that its early anchors are final models.

## 5. Benchmark comparison protocol

Use the same evaluator, task metric, canonical candidate IDs, final-fidelity definition and strict no-oracle information boundary for every method.

Required methods:
1. full-fidelity random search (ZAPS-style unique architecture query baseline);
2. ZAPS paper-protocol reproduction and a separate strict no-oracle variant;
3. CoRA-Rank and CoRA-Refine, with the exact anchor sampling, early training prefixes, proxy/curve feature construction and residual learner pinned;
4. R-MF-NAS with the paper's trajectory archive, first-improvement/escape procedure and successive-halving injection protocol pinned;
5. HiP-BO (published online 16 October 2026), with its relevance-weighted task-prior GP and pretrained learning-curve predictor, when code/data are available;
6. Successive Halving / Hyperband with pinned implementation and bracket/schedule;
7. BOHB or DEHB, or a pinned TPE multi-fidelity implementation where dependencies and fidelity interface are reproducible;
8. random + early stopping;
9. Symthaea proxy-only, Symthaea anchor/curve-informed and Symthaea audit-budgeted promotion variants.

### Equal-cost and equal-query comparisons

Report two distinct comparison tracks:

**Equal final-query budget.** Fix the number of full-fidelity architecture confirmations (for example 50, 100, 200, 400). This isolates search quality per final confirmation and permits comparison with ZAPS's published P@K values.

**Equal total-resource budget.** Fix the total cumulative training/proxy cost, including anchor training, repeated proxy computation, restarts, failed runs and retries. This is the primary fair comparison between methods that use different fidelity schedules. Also report actual wall-clock, accelerator time, energy and memory. A method must not get free early curves or free proxy recalculation.

For every seed, retain:
- best final-confirmed metric versus cumulative resource cost;
- final P@1/P@10/P@100;
- touched P@1/P@10/P@100;
- raw and normalized final regret;
- intermediate/final evaluation counts;
- resource cost per fidelity and per candidate;
- proxy computation time;
- all failed, invalid, duplicated and retried operations;
- query/receipt/canonical import digests.

Run at least 100 paired seeds for the confirmatory study or justify sample size with a preregistered power analysis. Report paired effects, 95% uncertainty intervals, effect sizes, per-landscape outcomes and worst-case task regressions. Do not report only global rank correlation: CoRA's own discussion illustrates that a correlation gain may not produce a better deployment/resource Pareto point.

## 6. Research thresholds

Initial engineering targets, not observed Symthaea results:

- At equal total cost, improve final-confirmed P@100 and normalized regret against the strongest pinned baseline.
- Preserve or improve final P@10 and P@1; do not trade away the best candidate while only improving broad top-100 touch coverage.
- Report the anytime frontier of best final-confirmed quality versus total resource spend.
- Achieve a positive paired effect with a 95% interval excluding zero on at least one held-out search space; no more than 5% regression on a preregistered critical task.
- Replicate on a second search space and a new challenge split without recalibrating against its target labels.
- Report the resource Pareto frontier (quality, GPU/CPU time, energy, memory) rather than calling a single scalar “SOTA”.

## 7. Authored Rust primitive

The initial module implements:
- strict fidelity ladder/resource-unit validation;
- complete architecture × fidelity target-table validation;
- explicit resumable-prefix and independent-restart charging;
- monotonic per-candidate promotion;
- independent hard caps for final confirmations and total resources;
- query attempt receipts with per-rung cost and observation hashes;
- distinct touched and final-confirmed candidate sets;
- evaluator-only touched/final P@K and direction-aware final regret;
- strict canonical JSON import with unknown-field rejection and order-invariant content digest.

Seventeen Rust unit tests are authored for resumable versus restart costs, touched versus final capture, independent caps, monotonicity, malformed/incomplete target tables, fidelity ladder validity, receipts, metric direction, canonical imports, empty queries and the preflighted Successive Halving schedule. They have not been compiled or executed.

### Runnable Successive Halving comparator

The multi-fidelity evaluator now exposes `run_successive_halving(initial_candidate_ids, rung_populations)`. The caller supplies an initial candidate pool (for example, the trajectory planner's promotion plan) and one strictly decreasing population count for each fidelity rung. At each rung, the runner queries all candidates in the current pool, ranks them by the observed metric using the declared maximize/minimize direction, and advances only the top population to the next rung. Ties are broken by architecture ID.

The schedule is preflighted *before any query is made*: initial IDs must be known and unique; the rung count must match the fidelity ladder; populations must be nonzero and strictly decrease; final confirmations must fit the hard query cap; and the exact resource cost must fit the hard resource budget. After execution, the receipt cost must reconcile exactly with the preflight estimate or the run is rejected.

Worked example with the illustrative ladder F0=5, F1=15, F2=200 epoch-equivalent units and a population schedule 8 → 4 → 2:

| Cost model | F0 | F1 promotion | F2 promotion | Total |
|---|---:|---:|---:|---:|
| Resumable prefix | 8 × 5 = 40 | 4 × (15−5) = 40 | 2 × (200−15) = 370 | **450** |
| Independent restarts | 8 × 5 = 40 | 4 × 15 = 60 | 2 × 200 = 400 | **500** |

This is a reproducible Successive Halving baseline, not Hyperband (multiple brackets), BOHB, DEHB or TPE. It selects on observed intermediate target metrics, and its query trace contains those observations' digests. It is not allowed to inspect unqueried fidelity labels.

The complete synthetic search chain can now be compared directly:
1. random or ranked initial candidate pool;
2. target-free proxy consensus and/or trajectory-preserving promotion shortlist;
3. the same preflighted Successive Halving schedule and resource budget;
4. final-confirmed P@K and normalized regret from the same evaluator.

This isolates whether the *initial pool construction* improves downstream multi-fidelity selection, rather than comparing different evaluators.

## 8. Trajectory-preserving promotion (R-MF-NAS-inspired)

A proxy-ranked top-k shortlist can discard strong architectures encountered earlier in a local-search trajectory. R-MF-NAS (Phan & Luong, ACM TELO, published 29 June 2026) documents this failure: high-performing candidates can appear early or midway through proxy-guided local search, while the terminal proxy elite may be weak. Its R-MF-NAS selector therefore adds first-improvement neighbors and 2-opt escape candidates to the successive-halving pool instead of selecting only the highest terminal proxy scores. In its reported 500-run results, the worst proxy variant on NB201-CIFAR100 improves from a mean ± SD of 69.96 ± 3.45 for MF-NAS to 71.46 ± 1.05 for R-MF-NAS; for NBASR-TIMIT the cross-proxy SD falls from 19.01 to 0.42. These are published results, not Symthaea measurements. [ACM paper](https://doi.org/10.1145/3774940).

The new Rust primitive is `nasbench201_trajectory.rs` and its tracking issue is [#9191](https://github.com/Luminous-Dynamics/symthaea/issues/9191). It implements a reproducible, bounded, target-free approximation of this candidate-retention idea over NAS-Bench-201's six-operation topology:
- seeded first-improvement 1-opt local search driven by one declared proxy;
- local-optimum recording and bounded random 2-opt escapes;
- preservation of starts, first-improvement candidates, local optima, escape candidates and all visited IDs;
- feature-catalog digest over architecture IDs, six-op topology and selected proxy values;
- archive digest over the frozen config, catalog digest, all candidate lists and ordered event trace;
- final promotion planner with explicit proxy-elite, trajectory and topology-diversity quotas; remaining slots are labelled fallback, never silently counted as trajectory discoveries.

The archive's target-metric ID is explicit and must differ from the proxy ID. The planner verifies that it receives the same feature catalog and archive content that generated the trajectory, rejects unknown candidates and rejects quota overcommit. It uses no target labels. The resulting IDs can be passed into the multi-fidelity query oracle to receive actual fidelity observations and final-confirmation credit.

This is **not an R-MF-NAS reproduction**: it does not implement the paper's complete Search Trajectory Network diagnostics, candidate injection details or exact benchmark protocol. It is an independently testable archive/promotion primitive inspired by the published finding. Nine Rust tests are authored for deterministic replay, first-improvement behavior, exact 2-opt escape distance, visit budgets, fixed-size promotion, distinct quota accounting, digest binding and invalid inputs. They have not been compiled or executed.

## 9. Target-free multi-proxy consensus prior (CoRA-inspired)

CoRA-NAS (Yang et al., arXiv:2609.11884, September 2026) uses per-proxy percentile ranks, a leave-one-out Spearman agreement gate (threshold -0.3; minimum three retained proxies), and a sum of log-percentile ranks as a composite prior. Its paper is explicit that agreement does not prove accuracy: the gate catches clear sign-flips but does not identify all weak-positive proxy dilution. Its attempted training-dynamics B-signal features did not improve the residual model in the reported ablation. These limitations should remain visible in the benchmark.

The new Rust module is `proxy_consensus.rs`, tracked by [#9192](https://github.com/Luminous-Dynamics/symthaea/issues/9192). It adds:
- declared maximize/minimize direction for every proxy;
- average midranks for tied proxy values;
- leave-one-out Spearman agreement against the mean rank of other proxies;
- rejection of proxies below the anti-correlation threshold, with a deterministic minimum-retained fallback;
- a distinct `undefined_agreement` diagnostic for constant/degenerate proxies;
- CoRA-style `sum(ln(rank/N))` aggregation and final rank-percentile prior;
- per-proxy gate diagnostics, exact source benchmark digest, disjoint evaluation/calibration split IDs and a calibration-decision digest;
- content digests binding proxy values, runtimes, topology, configuration and all consensus outputs;
- `apply_consensus_prior`, which verifies the consensus result against the exact input catalog before adding a derived proxy feature for trajectory search.

The module rejects non-finite scores, missing/negative proxy runtime, malformed NAS-Bench-201 topology, target-metric columns even when not explicitly selected, and use of the target metric ID as a proxy. It accepts no target labels or accuracy correlations. A constant proxy has undefined correlation and is not treated as passing the gate; it is retained only when the configured minimum-proxy fallback requires it.

The full target-free chain is now composable in the tests:

`declared static proxy features -> leave-one-out consensus prior -> first-improvement/2-opt trajectory archive -> quota-controlled promotion plan -> multi-fidelity query firewall`

The composite prior is still only a hypothesis for allocating training resources. The pipeline tests verify composition both up to promotion planning and through the multi-fidelity Successive Halving query oracle, without exposing target labels to consensus or trajectory generation; this does not establish that the composite strategy improves NAS performance. Sixteen Rust tests are authored for anti-correlation rejection, midranks, maximize/minimize direction, fallback diagnostics, undefined agreement, split/digest checks, target-leak rejection in both value and runtime columns, malformed/duplicate topology, tamper detection and end-to-end composition. They have not been compiled or executed.

## 10. Metered proxy acquisition (end-to-end compute fairness)

The evaluator previously received the entire precomputed proxy feature table and reported its full feature-bank runtime. That does not establish how much proxy compute a specific search policy actually requested. Charging every method the same complete bank is conservative but can hide the distinction between a random baseline, a single-proxy policy and a consensus policy that needs many proxies.

The new Rust module is `proxy_acquisition.rs`, tracked by [#9211](https://github.com/Luminous-Dynamics/symthaea/issues/9211). It introduces a second capability boundary before target training:
- policy receives architecture IDs and topology encodings only;
- one `query(architecture_id, proxy_id)` reveals one proxy score and charges its measured per-pair runtime;
- duplicate pair queries, unknown architecture/proxy IDs, pair-query budget exhaustion and compute-time budget exhaustion reveal nothing and are recorded;
- independent limits exist for unique architecture/proxy pairs and total proxy seconds;
- receipt binds source artifact, disjoint calibration/evaluation splits, source proxy-catalog digest, every query attempt, every observation digest, cumulative proxy time and both budgets;
- a materializer creates an `ArchitectureFeatures` table only when every candidate has every declared proxy successfully queried; partial data is rejected rather than filled with hidden source values or sentinels;
- the materializer verifies receipt metadata, receipt digest, per-observation digest, cumulative/remaining time and exact proxy schema before producing the table.

Thirteen Rust tests are authored for candidate metadata isolation, runtime charging, duplicate pair rejection, independent budgets, unknown IDs, deterministic receipts, exact feature schema, target-leak rejection, provenance, topology uniqueness, budget limits, complete table materialization and tamper rejection. They have not been compiled or executed.

This capability boundary is still for trusted in-process policy code; a closure could capture the original feature table externally. For adversarial policy evaluation, the proxy oracle must run in a separate process/container and be the only process able to access source proxy values. The receipt makes the intended capability interface testable; it is not a security sandbox by itself.

### Correct total-cost comparison

Every method must now report at least two separate ledgers:
1. **Proxy acquisition:** number of unique architecture/proxy pairs and measured proxy seconds, with the source feature-bank digest.
2. **Target training:** fidelity requests and cumulative training-resource units, with resumable-prefix versus independent-restart cost model.

The outer execution harness must later add process startup, optimizer CPU/GPU time, failed evaluations, retries, timeouts, energy, memory and accelerator utilization. Until those execution receipts exist, call the combined numbers *proxy seconds + synthetic/declared training units*, not total compute or end-to-end cost.

For equal-budget experiments, run random, single-best-proxy, consensus, trajectory and combined policies through the same proxy acquisition ledger and the same multi-fidelity query oracle. Compare (a) equal proxy seconds, (b) equal training resource and (c) equal combined declared resource as separate tracks. This prevents proxy-heavy methods from receiving free evidence and prevents target-query count from being mistaken for compute cost.

## 11. External benchmark and data limitations

The initial Rust test fixture is synthetic; it does not reproduce NAS-Bench-201, CoRA-NAS or R-MF-NAS. A one-time curve importer has now been authored at `scripts/research/import_naslib_nb201_learning_curves.py`. It merges a pinned SuiteZero canonical feature import with the official NASLib NAS-Bench-201 per-task learning-curve pickle.

The observed NASLib API structure is:
- `nb201_data[architecture_string][dataset_key]["eval_acc1es"]`;
- CIFAR-10 maps to `cifar10-valid`, CIFAR-100 to `cifar100`;
- the curve is indexed by epoch (NASLib's query API returns the first 50 elements for `full_lc=True, epoch=50`);
- the importer uses one-based milestones (default 5, 15, 200) and maps them to zero-based curve indices (4, 14, 199).

The SuiteZero importer uses six-integer tuple-string architecture IDs, while the NASLib learning-curve pickle uses graph-string keys. The curve importer now reproduces the official NASLib `convert_op_indices_to_str` mapping exactly (edge order `(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)`; op IDs 0..4 map to `skip_connect, none, nor_conv_3x3, nor_conv_1x1, avg_pool_3x3`). A pinned unit test checks the conversion against a concrete NASLib example. Directly comparing the two key formats would have rejected valid data; this mismatch is fixed in the importer. The CIFAR-10 pickle is documented upstream as `nb201_cifar10_full_training.pickle`, Google Drive ID `1sh8pEhdrgZ97-VFBVL94rI36gedExVgJ`; NASLib's Colab artifact log reports 117 MB. The importer requires a caller-supplied expected raw SHA-256 and checks it **before** unpickling. Pickle deserialization can execute code: only use the official artifact after verifying the expected digest from a trusted, independent source. The importer validates exact architecture-set equality against SuiteZero, validates complete curves, checks the final curve point against SuiteZero `val_accuracy` within a declared tolerance, and emits the multi-fidelity canonical schema. It records the raw curve digest and SuiteZero canonical-import digest in a composite source provenance digest.

Nine standard-library Python tests are authored in `scripts/research/test_import_naslib_nb201_learning_curves.py` for epoch mapping, endpoint agreement, missing/extra architectures, missing/short curves, non-finite values, schedule/budget guards and pre-unpickle SHA verification. They have not been executed. The real 117 MB artifact has not been downloaded or hash-pinned, so importer validation remains synthetic until that step is completed. The 4.7 GB checkpoint or 226 GB trained-weight archive is not a prerequisite for this initial curve table.

ZAPS and CoRA published values are reference targets from their papers, not results from this branch. No real NAS artifact has been loaded and no reproduction has been run.

## 12. Claim ceiling

This module makes multi-fidelity resource use auditable. It does not implement ZAPS, CoRA, Hyperband, BOHB, DEHB or an optimizer. Passing its synthetic tests would qualify only the evaluator contract, not NAS search quality, transfer or SOTA. Exact-head Rust tests and pinned real benchmark imports are required before making any performance claim.
