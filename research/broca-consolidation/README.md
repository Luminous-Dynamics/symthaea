# Broca governed-consolidation reference harness

This dependency-free Rust harness is the first executable slice for [issue #8445](https://github.com/Luminous-Dynamics/symthaea/issues/8445). It creates deterministic synthetic evidence profiles, separates policy-visible evidence features from evaluator-only ground truth, scores candidate consolidation actions with an explicit utility/storage-cost table, and emits per-candidate JSONL receipts.

**It is not the published ConsolidationBench, is not an evaluation of Liquid Broca, and does not establish generalization or SOTA.** The oracle is constructed from the harness's own synthetic utility contract. Results are valid only as tests of policies on this generator.

## Run

With Rust installed:

```sh
cargo test --manifest-path research/broca-consolidation/Cargo.toml
cargo run --manifest-path research/broca-consolidation/Cargo.toml --release -- \
  --source-sha "$(git rev-parse HEAD)" \
  --seed 17 --candidates 120 \
  > consolidation-reference.jsonl
```

The exact 40-character source commit is required. The harness validates its oracle and negative controls before emitting the completion record or any result-shaped rows.

## Decision levels

- `Raw`: retain the source evidence.
- `Provisional`: retain as a low-confidence candidate, not as a durable rule.
- `Summary`: compress a repeated evidence cluster.
- `Skill`: promote a repeatedly successful procedure.
- `Rule`: promote a strongly supported cross-episode invariant.
- `Retire`: do not retain as an active memory.

## Separation of features and labels

A policy receives only `ObservedFeatures`: repeat count, distinct episodes, contradictions, recency, correction seen, independent successes and provenance diversity. Evaluator-only `GroundTruth` is a separate struct; the policy API does not accept it. The evaluator uses hidden candidate kind only after the policy picks its action. This structural separation reduces accidental leakage; it is not a proof against every form of benchmark leakage, so external probes and held-out real traces are still required.

## Policies and controls

- `always_raw`: retain everything as raw evidence.
- `repeat_threshold_summary`: summarize candidates after the observed repeat-count threshold, otherwise retain raw.
- `dedup_only`: summarize repeated candidates, otherwise retain raw.
- `feature_threshold`: a simple observable-feature baseline.
- `typed_lifecycle`: a baseline with separate summary, skill and rule promotions.
- `retire_everything`: intentionally degenerate anti-control. It should show zero load-bearing coverage and maximum destructive forgetting.
- `promote_everything`: intentionally faulty mutation control. It should trigger false high-level promotion on poison/obsolete items.
- `oracle_diagnostic`: computed separately by the evaluator, never exposed to a policy; an upper bound on this synthetic generator only.

Mutation controls are not competitive systems. A successful self-check means the evaluator recognizes their intended failure—not that the harness proves all possible consolidation bugs are detected.

## Utility and score definition

For each candidate the evaluator selects the action that maximizes:

`net_utility = task_reward(candidate_kind, action) - storage_cost(action)`

Storage costs are explicit units: Raw=5, Provisional=3, Summary=2, Skill=3, Rule=2, Retire=0. Task rewards are synthetic and fixed in source, separately defined by candidate kind and action. Regret is `oracle_net_utility - chosen_net_utility`. Because this is an oracle-by-construction generator, it provides a transparent diagnostic, not a proxy guaranteed to predict real transfer.

Per-policy summaries report mean decision regret, total net utility, storage units, coverage of load-bearing candidates, false high-level promotion of poison/obsolete candidates, and destructive forgetting. This prevents a policy from appearing successful by retiring almost everything. No policy is selected by a weighted average that can hide critical governance failure.

## Output and limitations

Output is deterministic JSONL with:
- one `run` receipt containing source SHA, seed, candidate count, action/utility version, and non-cryptographic fixture fingerprint;
- per-candidate decision records and a separate oracle diagnostic;
- per-policy summaries and explicit anti-control results.

The FNV-1a fixture fingerprint is non-cryptographic and is not a dataset-integrity proof. Replace it with the repository's canonical cryptographic digest when integrating with production benchmark tooling.

This slice does not test real user conversations, semantic retrieval, HDC embeddings, asynchronous consolidation, physical erasure, reversal logs, poisoning beyond its fixed controls, or cross-benchmark transfer. Those belong in subsequent phases. The published [ConsolidationBench paper](https://arxiv.org/abs/2609.33013) remains a separate external work; do not claim reproduction without reproducing its specification and evaluation.