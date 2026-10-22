# Registry Enrollment Poisoning Stress Benchmark

**Status:** executable synthetic benchmark authored; exact-head validation pending  
**Implementation:** \`crates/core/symthaea-core/examples/platonic_registry_poisoning_stress.rs\`  
**Runner:** \`.github/workflows/levin-pattern-experiments.yml\`

## Research question

How quickly does prototype retrieval degrade when an attacker can enroll a wrong-labelled duplicate or near-duplicate? Do similarity and margin gates reject the attack, or merely convert some poisoning events into abstentions? Does the answer change as query noise increases?

This fills a gap between the open-set memory-growth stress test and identity-evidence API contract. The open-set test studies unknown observations; this one studies hostile changes *inside the registry*. It is deliberately a controlled insertion attack—not a model-weight poisoning study, authenticated-registry security proof, or estimate of real-world attack prevalence.

## Protocol

- 1,024-dimensional bipolar prototype vectors; 16 genuine classes.
- Three deterministic seeds: 17, 29, and 41.
- Poison entry counts: 0, 1, 2, 4, and 8. The zero-poison case is the clean baseline; other cases add attacker-labelled entries rather than silently replacing legitimate identities.
- Each malicious entry is a copy of a genuine prototype with 0%, 2%, or 5% coordinate corruption, but carries a separate \`attacker-XX-YY\` label. Attack targets are distinct within each declared budget.
- Query populations contain 8 observations per genuine class at 0%, 10%, and 20% corruption. The same deterministic query corpus is reused across attack profiles for paired comparisons.
- A fixed score threshold (0.25) and top-two margin threshold (0.05) provide a visible selective-decision diagnostic. They are not calibrated and have no finite-sample risk guarantee.

No threshold is tuned against this result. The test corpus is small and synthetic by design; the values should expose mechanisms and counterexamples, not support broad statistical or deployment claims.

## Output and metrics

Each row is stratified by seed, poison-entry count, poison corruption, and query corruption. The runner records:

- **Top-1 accuracy** over genuine queries, including predictions captured by attacker labels as incorrect.
- **Coverage** and **selective risk** under the fixed score-plus-margin diagnostic; zero accepted results produce \`NA\` risk, not zero.
- **Attacker prediction rate** across all genuine queries.
- **Targeted attack success rate** among queries whose genuine class has a poison entry, plus the conditional attacker success rate among accepted targeted queries.
- **Entry count** and estimated vector payload bytes, excluding labels, runtime overhead, allocator capacity, and indexes.
- **p50/p95 exact-scan latency**, scan-loop wall time, and aggregate single-thread query throughput. These timings include ranking only and must not be compared with ANN-database throughput absent matched hardware, corpus, batching, warm-up, memory accounting, and ground-truth protocol.
- Registry-build time in a comment record. This is kept separate from query/scan timing.

Rates stay separated by seed and noise level so aggregate averages cannot hide an attack that is highly effective in a particular regime. The 3-seed, 8-query-per-class smoke setup is not enough for tight confidence intervals; it is meant to establish a reproducible measurement contract and regression controls.

## Crucial negative controls

1. **Exact clone:** an attacker entry that is bit-identical to an enrolled prototype is observationally indistinguishable under vector similarity. The current deterministic tie break favours the lexical \`attacker-\` label over \`id-\`, so an exact copy captures clean-prototype queries for its target. The unit test preserves this adversarial counterexample.
2. **Clean registry:** with zero poison entries, all stored labels remain genuine and every clean prototype query returns its own identity.
3. **Paired query stream:** all poisoning profiles receive the same generated observations for a given seed/noise condition; an apparent difference cannot be attributed to drawing an easier query set for one profile.
4. **Memory accounting:** insertion attacks increase memory. Results must not be presented as equal-memory comparisons with the clean baseline; a future replacement-attack arm should hold total entry count fixed and report missing-class effects separately.

An exact duplicate cannot be made safe by a better similarity threshold: the two entries have the same vector. The architectural control is to separate similarity candidate generation from authority to register an identity, then require independently authenticated enrollment/provenance evidence. The benchmark measures the consequences when that trust boundary is absent; it does not implement authentication.

## Paradigm shift and next experiments

Treat enrollment as a security-sensitive state transition rather than an append-only vector write. Candidate ranking can be high-performance and approximate, but identity assignment should depend on a distinct, auditable authorization path. A robust memory should also support quarantine, provenance-aware conflict resolution, registry rollback, and monotonic revocation without silently relabelling an identity.

Next comparisons should add (a) a fixed-entry-count replacement attack, (b) repeated low-rate attacks over time, (c) label collisions and replayed enrollment receipts, (d) independently authenticated vs unauthenticated enrollment events, (e) a registry-held-out evaluation, and (f) paired comparisons with a faithful class-wise selective-risk baseline. Any defence must be evaluated against the same attack corpus and include benign near-duplicate controls.

## Proposed gates (targets, not achieved results)

- Exact-clone counterexample remains detected by a regression test.
- Every attack profile reports attack success, accepted attack success, ordinary accuracy, coverage, selective risk, and memory cost independently.
- A proposed mitigation must reduce accepted targeted attack success at matched or explicitly budgeted coverage; a result that only raises abstention is not automatically an improvement.
- Use at least 5 seeds and 1,000 or more targeted queries per operating point before estimating stable rates; report uncertainty with a sampling unit appropriate to repeated prototypes and classes.
- Benchmark replacement attacks at equal entry and byte budgets, insertion attacks at reported extra memory, and all methods on identical query streams.
- Compare against conventional nearest-neighbour/prototype baselines and a verified provenance-aware contract; keep identity assurance claims separate from geometric classification.

These gates are targets. This benchmark has no deployment-level statistical certificate.

## Research context

- A 2023 study on HDC-based data poisoning in federated learning describes a distinct attack surface involving adversarial samples around hyperdimensional class representations: <https://www.sciencedirect.com/science/article/pii/S0957417423016949>. That setting is not the same as registry-entry insertion here; it motivates testing HDC-specific poisoning rather than assuming the representation itself prevents it.
- A 2025 review of HDC/VSA classification stresses cross-task comparisons across established HDC datasets and UCI tasks, including conventional classifier baselines: <https://doi.org/10.1007/s10462-025-11181-2>.
- A 2026 HDC uncertainty-aware human-activity-recognition study combines geometry-derived uncertainty with conformal prediction and evaluates sensor/distribution shifts while reporting on-device resources: <https://doi.org/10.1038/s41598-026-57375-8>. It motivates explicit robustness and resource measurements, but does not certify this synthetic registry experiment.
- The selective-classification-under-distribution-shift literature shows that calibration and rejection behavior under a changed query distribution cannot simply be assumed from in-distribution results: <https://pmc.ncbi.nlm.nih.gov/articles/PMC12470254/>.

## Claim ceiling

The experiment can establish a deterministic, synthetic poisoning stress result for the exact source revision. It cannot establish robust identity, authenticated provenance, statistical guarantees under unknown distributions, real-world attack prevalence, superiority to SOTA, or safe deployment.
