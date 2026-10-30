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
4. Successive Halving / Hyperband with pinned implementation and bracket/schedule;
5. BOHB or DEHB, or a pinned TPE multi-fidelity implementation where dependencies and fidelity interface are reproducible;
6. random + early stopping;
7. Symthaea proxy-only, Symthaea anchor/curve-informed and Symthaea audit-budgeted promotion variants.

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

Fifteen Rust unit tests are authored for resumable versus restart costs, touched versus final capture, independent caps, monotonicity, malformed/incomplete target tables, fidelity ladder validity, receipts, metric direction, canonical imports and empty queries. They have not been compiled or executed.

## 8. External benchmark and data limitations

The initial Rust test fixture is synthetic; it does not reproduce NAS-Bench-201 or CoRA-NAS. A one-time curve importer has now been authored at `scripts/research/import_naslib_nb201_learning_curves.py`. It merges a pinned SuiteZero canonical feature import with the official NASLib NAS-Bench-201 per-task learning-curve pickle.

The observed NASLib API structure is:
- `nb201_data[architecture_string][dataset_key]["eval_acc1es"]`;
- CIFAR-10 maps to `cifar10-valid`, CIFAR-100 to `cifar100`;
- the curve is indexed by epoch (NASLib's query API returns the first 50 elements for `full_lc=True, epoch=50`);
- the importer uses one-based milestones (default 5, 15, 200) and maps them to zero-based curve indices (4, 14, 199).

The CIFAR-10 pickle is documented upstream as `nb201_cifar10_full_training.pickle`, Google Drive ID `1sh8pEhdrgZ97-VFBVL94rI36gedExVgJ`; NASLib's Colab artifact log reports 117 MB. The importer requires a caller-supplied expected raw SHA-256 and checks it **before** unpickling. Pickle deserialization can execute code: only use the official artifact after verifying the expected digest from a trusted, independent source. The importer validates exact architecture-set equality against SuiteZero, validates complete curves, checks the final curve point against SuiteZero `val_accuracy` within a declared tolerance, and emits the multi-fidelity canonical schema. It records the raw curve digest and SuiteZero canonical-import digest in a composite source provenance digest.

Eight standard-library Python tests are authored in `scripts/research/test_import_naslib_nb201_learning_curves.py` for epoch mapping, endpoint agreement, missing/extra architectures, missing/short curves, non-finite values, schedule/budget guards and pre-unpickle SHA verification. They have not been executed. The real 117 MB artifact has not been downloaded or hash-pinned, so importer validation remains synthetic until that step is completed. The 4.7 GB checkpoint or 226 GB trained-weight archive is not a prerequisite for this initial curve table.

ZAPS and CoRA published values are reference targets from their papers, not results from this branch. No real NAS artifact has been loaded and no reproduction has been run.

## 9. Claim ceiling

This module makes multi-fidelity resource use auditable. It does not implement ZAPS, CoRA, Hyperband, BOHB, DEHB or an optimizer. Passing its synthetic tests would qualify only the evaluator contract, not NAS search quality, transfer or SOTA. Exact-head Rust tests and pinned real benchmark imports are required before making any performance claim.
