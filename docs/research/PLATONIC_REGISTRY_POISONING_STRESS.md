# Registry Enrollment Poisoning and Fixed-Budget Replacement Benchmark

**Status:** executable synthetic benchmark authored; exact-head validation pending  
**Implementation:** `crates/core/symthaea-core/examples/platonic_registry_poisoning_stress.rs`  
**Runner:** `.github/workflows/levin-pattern-experiments.yml`  
**Follow-up:** [Issue #8317 — fixed-budget replacement and temporal poisoning](https://github.com/Luminous-Dynamics/symthaea/issues/8317)

## Research question

Can a hostile enrollment change retrieval outcomes by adding attacker-labelled near-duplicates—or, more severely, replacing a genuine prototype while holding memory capacity constant? Do similarity and margin gates reject the attack, accept the attacker identity, or simply move errors into abstentions? Does the result change with query noise?

The two attack modes must not be pooled as equivalent:

- **Insertion:** keeps every genuine entry and appends attacker-labelled copies/near-duplicates, increasing the entry and vector-byte count.
- **Replacement:** removes each targeted genuine entry and inserts an attacker-labelled copy/near-duplicate. Entry count and estimated vector-payload bytes remain constant. Queries for those underlying genuine classes are scored separately as targeted queries, making class loss visible.

This is an intentionally controlled synthetic registry-state attack. It is not a model-weight poisoning study, a physical memory bit-flip demonstration, an authenticated registry implementation, or a prevalence estimate for real-world attacks.

## Executable protocol

- Vector dimension: 1,024 bipolar coordinates.
- Registry: 16 genuine classes.
- Seeds: 17, 29, and 41.
- Attack budgets: 0, 1, 2, 4, and 8 attacker entries.
- Prototype corruption for the attacker entry: 0%, 2%, or 5%.
- Query corruption: 0%, 10%, or 20%.
- Query corpus: 8 samples per genuine class for each query-noise level. The query vectors are generated once per seed and reused across all insertion and replacement profiles, giving paired comparisons.
- Attack targets are unique within a declared budget of up to eight classes, selected by a deterministic permutation.
- Fixed diagnostic acceptance rule: best similarity at least 0.25 and top-two margin at least 0.05. This rule is neither calibrated nor backed by a finite-sample risk guarantee. No threshold is fitted on the reported results.

The zero-entry baseline runs with zero attacker corruption only. In replacement mode, the genuine entry for every targeted class is removed; attacker labels are added in its place. The bank therefore remains exactly 16 entries in replacement mode, while insertion mode uses 16 plus the attacker count.

## Measurements

Rows are separated by seed, attack mode, attack-entry count, attack corruption, and query corruption.

- **Overall top-1 accuracy**, coverage, accepted errors and selective risk.
- **Targeted-class metrics:** top-1 correctness, accepted coverage and false-rejection rate on queries whose genuine class has been attacked.
- **Retained-class metrics:** top-1 correctness, accepted coverage and false-rejection rate for classes whose genuine entry remains present.
- **Targeted attack success:** frequency with which the attacker-labelled entry for the true target is the top-ranked label, both over all targeted queries and conditional on targeted queries accepted by the diagnostic rule.
- **Attacker prediction rate:** fraction of all genuine queries whose top-ranked entry is attacker-labelled.
- **Resource cost:** entry count and estimated vector payload bytes, registry build time, p50/p95 latency around exact ranking, exact-scan wall time, and aggregate single-thread QPS.

A zero-accepted denominator emits `NA`, not a fabricated zero risk. Payload estimates count only `entry_count × dimension × sizeof(i8)`; labels, allocation overhead, indexes, process RSS, allocator capacity and runtime are excluded. Per-query p50/p95 measures the `rank` call. Scan-loop wall time and QPS also include per-query timing calls and metric bookkeeping, while excluding query generation and registry construction. Do not compare these synthetic timings with ANN-database throughput without matching hardware, corpus, batching, warm-up, ground truth, memory accounting and timing boundaries.

The three-seed, eight-query-per-class corpus is smoke-scale. It is intended to reveal deterministic failure modes and validate the measurement contract, not to substantiate stable population rates. Wilson or other binomial intervals would be misleading if they ignore repeated queries sharing a prototype; larger studies must use uncertainty estimates appropriate to the class/prototype sampling design.

## Falsifiable negative controls

1. **Insertion exact clone:** both original and attacker entries score 1.0 against the clean target vector. The lexical tie-break favours the attacker label, but the top-two margin is zero and the fixed diagnostic abstains.
2. **Replacement exact clone:** the legitimate target entry has been removed, so the attacker copy is the unique best match. It can score 1.0 with a large margin and pass the same acceptance rule. This shows why rejection based on ambiguous duplicate candidates is not enough when registry replacement is possible.
3. **Clean registry:** with zero attack entries, all prototype labels remain genuine and every clean prototype query retrieves its expected label.
4. **Constant replacement budget:** replacement with 1/2/4/8 entries must always retain the same bank-entry count and estimated vector payload as the clean baseline.
5. **Paired query stream:** attack modes and budgets are evaluated against the same query vectors for each seed/noise stratum. A change in outcome is not caused by generating a different test set.

The essential boundary is architectural: similarity ranks candidates; a distinct authorization/provenance path determines which identities are allowed to enter or replace registry records. Neither high similarity nor a successful statistical certificate authenticates a registry mutation.

## Paradigm shifts and next work

Move from append-only prototype storage toward an **auditable registry state machine** with explicit enrollment authority, provenance, conflict/quarantine states, revocation, monotonic audit events, replay protection, and rollback that cannot silently resurrect revoked identities. Candidate retrieval can remain fast, but identity ownership and enrollment should be controlled separately.

The immediate follow-up is [Issue #8317](https://github.com/Luminous-Dynamics/symthaea/issues/8317), covering benign multi-template controls, temporal enrollment/revoke/replay sequences and exact-byte-budget replacement. Later attack arms should include repeated low-rate poisoning, multiple labels for the same vector, label collisions, replayed enrollment receipts, prototype bit corruption, and a registry-held-out split. Each defence must be compared on the identical attack/query corpus and at matched coverage or a declared coverage budget.

## Benchmark targets (targets, not achieved results)

- **Replacement capacity:** entry count and vector payload equal to the clean baseline for every replacement attack budget.
- **Security utility:** reduce accepted targeted attack success at matched coverage or a declared coverage budget. Abstaining on every query is not a winning mitigation.
- **Failure visibility:** report attack success, targeted and retained class coverage, selective risk, false rejection, and resources separately.
- **Uncertainty quality:** at least five independent seeds and 1,000 or more targeted queries per operating point before making stable rate claims; uncertainty must reflect clustering by identity/prototype.
- **Equal-workload comparison:** exact same queries, seed, precision, vector dimension, threshold policy and hardware across prototype/HDC variants and conventional nearest-prototype or classifier baselines.
- **Measured runtime:** include total RSS and index/label costs where feasible, in addition to estimated vector payload and build/query timing.
- **Provenance contract:** independently test valid enrollment, unauthorized insertion, unauthorized replacement, conflicting labels, replay, revocation, and rollback. Similarity metrics do not count as a provenance control.

## Industry comparison and research context

- **HDC classification benchmarks:** Vergés et al. (2025) propose consistent evaluation across seven established HDC application datasets and 121 UCI classification tasks, making them relevant public-task targets for later comparison. They also emphasize the importance of comparing HDC approaches with other ML classifiers: https://doi.org/10.1007/s10462-025-11181-2
- **Memory integrity:** *HyperAttack* studies targeted bit flips in HDC associative memory rather than registry entry insertion/replacement; its threat model differs, but it is an essential adversarial comparator for a broader memory-integrity suite: https://doi.org/10.1109/DAC56929.2023.10247811
- **Accelerator security:** a 2025 HDC side-channel study reports class-hypervector recovery from voltage traces on an FPGA accelerator. This exposes a different physical leakage surface and should motivate a separate threat model, not be mixed into this software-only experiment: https://doi.org/10.1109/ITC-Asia67627.2025.00019
- **Distribution shift and uncertainty:** a 2026 microcontroller HDC study evaluates uncertainty and conformal prediction across partial and mixed sensor shifts and reports coverage, set size, OOD detection and device resources. These are useful evaluation dimensions for Symthaea's later non-synthetic workload tranche, but its results do not transfer automatically to registry poisoning: https://doi.org/10.1038/s41598-026-57375-8

## Claim ceiling

A passing exact-head run would establish only that the executable and tests produce the declared deterministic synthetic measurements. It would not establish secure identity, authenticated provenance, cryptographic guarantees, robustness to unseen attacks or shifts, real-world prevalence, deployment safety, or superiority to SOTA. Those require separately specified evidence.
