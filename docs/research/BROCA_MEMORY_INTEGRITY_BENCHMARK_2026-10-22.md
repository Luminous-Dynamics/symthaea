# Liquid Broca Memory Integrity Benchmark (MIRE)

**Status:** preregistration proposal; no benchmark has been executed  
**Date:** 2026-10-22  
**Scope:** controlled memory behavior for Symthaea Liquid Broca and its recurrent/HDC memory alternatives  
**Related work:** [architecture bake-off protocol](BROCA_ARCHITECTURE_BAKEOFF_PROTOCOL_2026-10-22.md), [#8308 training/evaluation validity](https://github.com/Luminous-Dynamics/symthaea/issues/8308), [#8330 bake-off protocol PR](https://github.com/Luminous-Dynamics/symthaea/pull/8330), [#8364 LoRA update isolation](https://github.com/Luminous-Dynamics/symthaea/issues/8364)

This document proposes a separate benchmark for **memory integrity over time**. It complements ordinary retrieval and long-context QA. It does not replace them and does not claim results for Liquid Broca.

## 1. Research question

When information is written, updated, contradicted, made irrelevant, or explicitly deleted, does the system retrieve the right version, reject stale or unsupported records, and avoid contaminating unrelated memories—at a measured compute and storage budget?

The core hypothesis is that useful memory is not equivalent to maximum recall. A good system should retain valid information, revise it when warranted, ignore irrelevant interference, abstain when evidence is insufficient, and honor the declared deletion boundary.

The motivating literature includes:

- **ICF-Bench (ICLR 2026)** studies selective forgetting of interference in multi-turn context and reports that stronger memory alone does not guarantee stronger forgetting: https://proceedings.iclr.cc/paper_files/paper/2026/hash/b13d00a62d438856cfe6fbd13b6b2cb8-Abstract-Conference.html
- **From Recall to Forgetting (Findings of ACL 2026)** studies long-term personalized-agent memory and reports problems with stale memories and reconciling changes: https://aclanthology.org/2026.findings-acl.1337/
- **Minerva (ICML 2025)** provides programmable tests for searching, recalling, editing, matching, comparing, and maintaining state in context memory: https://proceedings.mlr.press/v267/xia25c.html
- **HELMET (ICLR 2025)** motivates a task-diverse suite rather than treating one synthetic retrieval score as a reliable proxy for downstream usefulness: https://proceedings.iclr.cc/paper_files/paper/2025/hash/f5332c8273d02729730a9c24dec2135e-Abstract-Conference.html

These findings motivate tests; they are not evidence that any one benchmark design transfers directly to Broca.

## 2. Benchmark record model

Generate each episode from a versioned, deterministic generator. A memory event contains:

- entity_id: randomly generated, split-specific entity identifier;
- attribute_key: stable key from a controlled schema plus held-out combinations;
- value: random value or relation, not answer text embedded in the key;
- version and event_time: explicit ordering ground truth;
- source_reliability: known generator-side evidence quality;
- operation: write, update, contradict, expire, scope-change, or delete;
- scope: episode-local namespace to detect cross-episode leakage;
- query_policy: whether answer, abstention, or a qualified/conflict-aware answer is correct.

The evaluator—not the model—retains the ground-truth event log. Split-specific identifiers and values must be generated after model/checkpoint selection. Do not train on the test episode's keys, answers, ordering, or deletion markers. Publish generator version, seeds, fixture digest, and a digest of each split.

Keep these concepts distinct in reports:

1. **Context recall:** the system can find a record still present in a provided context.
2. **Persistent-memory behavior:** information is written to state and later queried without replaying the full original context.
3. **Logical forgetting:** the system's supported retrieval surface no longer returns information marked deleted or invalidated.
4. **Privacy erasure:** a stronger property involving all copies, logs, checkpoints, caches, and adversarial extraction. This benchmark alone cannot establish privacy erasure or certify secure deletion.

## 3. Task families

Each family must include multiple templates, entity/value permutations, query paraphrases, distractor levels, and held-out combinations. Score per family and difficulty; never report only one pooled average.

### A. Stable fact retention under interference

Write key/value facts, then insert unrelated records and near-collision records before querying. Vary event count, key similarity, query delay, and the position of the original write.

Suggested scales: 16, 64, 256, 1,024, and 4,096 events, limited by the supported model and budget. Report effective capacity rather than assuming every setting will fit.

Measure exact retrieval, wrong-value retrieval, abstention, performance by lag, and per-event storage/runtime. Separate exact-key lookups from semantically similar distractors.

### B. Update and conflict resolution

Present a sequence such as office = A, later office = B, then distractors that repeat A. Ask for the current value, the historical value at an explicit time, or both.

Vary update distance, number of stale repeats, source reliability, timestamps, and whether an update is explicit or inferred. The expected answer is determined by the task's declared policy—not a blanket rule that the latest string always wins.

Measure current-version accuracy, historical-query accuracy, stale-answer rate, conflict-detection rate, and correct abstention when the evidence does not resolve a conflict.

### C. Selective forgetting and deletion

After writing facts, issue an explicit delete or invalidation event, then query the target through the ordinary retrieval interface, paraphrases, association chains, and plausible neighboring keys. Also query unaffected facts in the same episode.

Use two separate regimes:

- **Logical invalidation:** a tombstone or policy blocks retrieval of the record. Report compliance of the retrieval path, including whether lower layers can still expose the item.
- **Physical-state erasure:** only test this when the implementation claims to erase state bytes, cache entries, or fast weights. Inspect the relevant storage artifacts and state digests; do not infer physical erasure from a failed natural-language query.

Measure target deletion leakage, time-to-invalidation, and collateral forgetting of unaffected records. A deleted target becoming unretrievable is not sufficient if unrelated state is destroyed, nor does a tombstone prove that raw bytes have been securely erased.

### D. Associative interference and key collisions

Create records with similar entities, overlapping attributes, shared values, reversed relations, and distractor records containing one matching field but a different binding. Include adversarially salient irrelevant records and records from another episode namespace.

Measure binding swaps, cross-entity contamination, wrong-episode retrieval, and false confidence. An answer with the right value bound to the wrong entity is a failure.

### E. Unknowns, ambiguity, and evidence quality

Ask about unseen keys, underdetermined values, inconsistent sources, expired records, and questions whose answer requires a relation absent from memory. Provide known source-quality metadata only when the tested system is meant to consume it.

Measure abstention precision/recall, unsupported-answer rate, calibration where meaningful, and whether a justified update changes confidence appropriately. A system should not earn a correct-answer point by guessing a value that is merely plausible.

### F. Compositional and multi-hop memory

Store separate relations, then ask a query requiring two or three joins. Add near-neighbor facts so a model cannot pass by matching one salient word. Include versioned relations and deletion of one edge in a chain.

Measure exact relation-chain accuracy and error attribution: missing record, wrong binding, stale edge, invalid inference, or unsupported completion.

## 4. Baselines and architecture arms

Use identical generated episodes, split manifests, and query budgets where possible.

| Arm | System | Purpose |
|---|---|---|
| A | Mamba language substrate with no explicit persistent memory | Recurrent-only baseline |
| B | Current latent thought projection / recurrent state | Measures what compression into latent state provides |
| C | B + explicit HDC-keyed associative memory | Isolates explicit retrieval contribution |
| D | Exact bounded key-value store with declared update/delete policy | Upper bound for structured tasks, not a language-model competitor |
| E | Simple recency/version-aware memory with the same storage budget | Tests whether HDC retrieval adds value beyond a simple policy |
| F | Full-context or conventional retrieval-augmented baseline, where feasible | Strong reference for context-replay approaches |
| G | B + HDC memory + selective controller | Tests whether adaptive retrieval is worth its overhead |

Include an oracle only as a diagnostic upper bound and label it clearly. It must not be included as a competitive system score. Keep a record of all hyperparameter tuning and do not select settings on the final test set.

For architecture claims, distinguish equal-parameter, equal-storage, equal-token, and equal-latency comparisons. These are different questions. Report the achieved budget rather than describing a comparison as “matched” when the costs differ materially.

## 5. Metrics and definitions

Report task counts and exact numerators/denominators alongside percentages.

- **Valid recall:** proportion of queries answered with the correct currently valid record/value.
- **Stale-answer rate:** proportion of update/conflict queries that return a superseded value as current.
- **Binding-error rate:** proportion with a correct value attached to the wrong entity, key, or relation.
- **Deletion leakage:** proportion of post-invalidation queries that expose the invalidated record through the tested interface.
- **Collateral forgetting:** proportion of unaffected records that become unavailable or incorrect after a local update/delete.
- **Cross-episode leakage:** proportion of queries answered using another episode's records.
- **Abstention quality:** precision and recall for cases where the declared policy requires abstention or conflict reporting.
- **Compositional accuracy:** exact correctness of all required edges in the queried relation chain.
- **Capacity/retention curve:** each key metric by event count, lag, distractor count, update distance, and query type.
- **Efficiency:** total write/query/update/delete latency (p50 and p95), model tokens consumed, retrieval calls, bytes per event, measured peak RSS/VRAM, and energy only when reliable measurement is available.

For latency, report cold and warm paths, and isolate retrieval-only time from end-to-end answer time. Report theoretical/raw payload bytes separately from measured memory; they are not interchangeable.

## 6. Preregistered experiment protocol

1. Freeze the generator version, scoring rules, system configurations, and thresholds before test evaluation.
2. Use development data to debug; use separate validation data for tuning; keep the test split locked until configurations are frozen.
3. Start with a deterministic unit-scale generator and exact key-value baseline. Verify that scoring catches stale values, wrong-entity bindings, invalidated records, and cross-episode contamination.
4. Evaluate at least five independent generator seeds when randomization is present, with at least 500 query episodes per primary task cell where practical. If hardware cost requires fewer cases, preregister the reduced design and report its power/precision limitations before looking at outcomes.
5. Use paired bootstrap confidence intervals across the same query instances for system comparisons. Report seed-level results, every task family, all exclusions and failures, and the aggregate. No best-seed-only reporting.
6. Apply matched-budget comparisons and report quality/latency/storage Pareto fronts. Record time spent building state, not only answer-generation latency.
7. Repeat critical cells with unrelated/random memory insertion, shuffled values, wrong-namespace records, and no-memory controls. These negative controls should make inappropriate retrieval fail visibly.
8. Do not mix data from different seeds or episodes through persistent state unless cross-episode learning is the explicit treatment. Reset caches and adapters between episodes and test that reset contract.
9. Use exact source-head SHA, model/checkpoint/tokenizer hashes, feature flags, data digests, hardware/runtime details, command, exit status, and logs in the run receipt.

## 7. Proposed acceptance gates

These are practical starting targets—not results and not universal standards. Freeze or revise them **before** test runs begin.

- **Stable retrieval:** at least 95% exact retrieval at 1,024 events on the preregistered exact-key cell, with performance reported separately under semantic interference.
- **Update correctness:** at least 95% current-version accuracy on clean update sequences and no more than 1% stale-answer rate on the designated stale-distractor cell.
- **Deletion behavior:** logical-invalidation leakage no greater than the preregistered negative-control rate plus one percentage point, with a paired 95% confidence interval; report physical erasure separately only if directly inspected.
- **Collateral damage:** no more than one percentage point loss in retrieval accuracy on unaffected records after a local update or deletion.
- **Architecture value:** HDC/controller features must improve the preregistered interference/update/deletion composite by at least 3 percentage points over the strongest non-oracle simple baseline at the declared budget, with the paired 95% confidence interval excluding zero; otherwise treat the extra mechanism as unproven for this workload.
- **No language regression:** any selected memory configuration must remain within the preregistered non-inferiority margin on the independent held-out language objective. A suggested first margin is 1% relative NLL degradation, subject to freezing before runs.
- **Efficiency claim:** state actual storage, write/update/query work, p50/p95 latency, and end-to-end budget. An accuracy improvement purchased with unreported extra tokens or unlimited memory is not an efficiency win.
- **Task validity:** missing splits, corrupt/mismatched checkpoints, unsupported operations, skipped runs, NaNs, or absent receipts produce NOT RUN / invalid, never a numeric success entry.

A failure to pass a proposed threshold is useful evidence: simplify the design, narrow its claimed operating region, or improve the mechanism rather than changing thresholds after results are visible.

## 8. Paradigm-level follow-up experiments

If this protocol reveals a reliable bottleneck, prioritize these targeted interventions instead of adding more independent cognitive components:

1. **Evidence-addressable memory:** represent each entry with a stable key, value, provenance/version, validity status, and bounded retrieval score. Test retrieval and update policy as separate factors.
2. **Selective controller:** invoke retrieval or adaptation only when calibrated uncertainty or an explicit conflict signal predicts benefit. Compare against always-retrieve and never-retrieve controls.
3. **Reversible state transition:** make each write/update/delete a testable operation with before/after digests and a replayable event log. Keep integrity receipts separate from claims about semantic correctness.
4. **Budget-aware retention:** evict or compress entries based on measured downstream utility while preserving versioned and deletion-critical metadata. Compare against FIFO, recency, frequency, and random eviction at the same byte budget.
5. **Counterfactual test generation:** every passing case should have nearby mutations—change the entity, swap a relation, reverse update order, insert a stale repeat, or delete one needed edge—so superficial pattern matches are exposed.

These are hypotheses for follow-on implementation, not assertions about existing Broca behavior.

## 9. Result receipt

Each run should emit a machine-readable receipt (JSON or equivalent) with:

- schema version and benchmark protocol revision;
- repository URL, exact commit SHA, branch, dirty-state and patch digest;
- model and checkpoint hashes; tokenizer/config/features; adapter reset policy;
- generator version, seed, split digest, query IDs and per-query outcomes;
- all metric definitions, counts, exclusions, confidence intervals and failure reasons;
- memory byte accounting, RSS/VRAM, write/query/update/delete latency distributions, token/FLOP accounting where available;
- hardware, operating system, compiler/runtime, precision, batch size and command;
- exit status, test/run status (PASS, FAIL, NOT RUN, or INVALID), and log digest.

The receipt must make it possible for an independent reviewer to establish exactly which implementation and data produced each reported number.

## 10. Decision rule

Do not select a memory design using only a weighted score. First eliminate configurations that fail correctness, deletion-policy, state-isolation, or language non-inferiority gates. Then compare the surviving systems on the quality/storage/latency Pareto frontier for each workload.

The intended result is not proof that HDC memory or liquid state is inherently superior. It is a reproducible map of when persistent associative state helps, when a simple versioned store is enough, when full-context retrieval is better, and where extra cognitive machinery adds cost without capability.
