# Broca governed-consolidation reference harness

This dependency-free Rust harness is the first executable slice for [issue #8445](https://github.com/Luminous-Dynamics/symthaea/issues/8445). It creates deterministic synthetic evidence profiles with seed-controlled noise in observable features, separates policy-visible evidence features from evaluator-only ground truth, scores candidate consolidation actions with an explicit utility/storage-cost table, and emits per-candidate JSONL receipts. Protocol `GMCv2` changes observable evidence features across seeds rather than merely reshuffling a fixed set of profiles.

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

The exact 40-character source commit is required. The harness validates its oracle and negative controls before emitting the completion record or any result-shaped rows. Candidate-policy quality is reported, not used to invalidate a run: a poor result must remain observable rather than being silently suppressed.

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
- one `GMCv2` `run` receipt containing source SHA, seed, candidate count, feature-noise version, and non-cryptographic fixture fingerprint;
- per-candidate decisions containing the observed feature vector and evaluator-only class label, plus a separate oracle diagnostic;
- per-policy summaries with one-decision-per-candidate completeness checks and explicit anti-control results.

The run-validity gate checks the oracle, complete candidate-ID coverage, and whether the anti-controls are detected. It deliberately does **not** require the candidate policy to beat a baseline: a bad result is a valid result and must remain reportable.

Seeded feature noise perturbs repeat count, distinct episodes, contradictions, recency, independent successes, provenance diversity and occasionally the observed correction signal. Policy-visible features can therefore differ within one ground-truth class; the generator remains synthetic and its class-conditioned templates are still hand-specified. This is a useful stress test, not realistic distributional coverage.

The seeded feature perturbation now varies repeat count, distinct episodes, contradiction count, recency, independent successes, provenance diversity, and occasionally the observed correction signal. Class-conditioned feature templates remain hand-specified, so this is a controlled robustness generator, not realistic distributional coverage.

The FNV-1a fixture fingerprint is non-cryptographic and is not a dataset-integrity proof. Replace it with the repository's canonical cryptographic digest when integrating with production benchmark tooling.

This slice does not test real user conversations, semantic retrieval, HDC embeddings, asynchronous consolidation, physical erasure, reversal logs, poisoning beyond its fixed controls, or cross-benchmark transfer. Those belong in subsequent phases. The published [ConsolidationBench paper](https://arxiv.org/abs/2609.33013) remains a separate external work; do not claim reproduction without reproducing its specification and evaluation.

## Budget-matched consolidation frontier (GMC-BUDGET-FRONTIER-V1)

The companion binary models an additional question: **does consolidation help once every policy has the same memory-token budget, including an unconstrained regime where raw evidence fits?** This extends the synthetic instrument; it does not reproduce external results and does not claim real task accuracy.

Run it from the exact checked-out commit with Rust installed:

```sh
cargo test --manifest-path research/broca-consolidation/Cargo.toml
bash -n scripts/run_broca_budget_frontier.sh scripts/check_broca_budget_frontier.sh scripts/test_broca_budget_frontier.sh
bash scripts/test_broca_budget_frontier.sh
bash scripts/run_broca_budget_frontier.sh
```

The run emits a JSONL header and one row per policy and budget. It sweeps abstract token-capacity budgets 32, 64, 128, 256, 512, 1024, 2048, 4096 and 8192. With the default 120-candidate generator (8–40 raw tokens each), the 8192-token arm is guaranteed to fit all raw evidence, giving an unconstrained-retention comparison rather than testing only the tight-budget regime. Candidate counts are bounded at 200: each candidate costs at most 40 raw tokens, so even the maximum supported workload needs at most 8000 tokens and the 8192-token raw-fit upper arm is guaranteed. This also bounds the oracle dynamic program's time and memory. Candidate IDs, latent classes, raw evidence lengths and future-use counts are held fixed across seeds; only policy-visible feature noise changes. Cross-seed comparisons therefore test sensitivity to noisy observations on the same toy task set, not independent task generalization. Within each seed, all policies share the exact same candidate stream.

Arms are no-memory/retire-all, always-raw, always-summary, fixed-interval summary, dedup-and-merge, observable-feature threshold, budget-adaptive, promote-everything negative control, and a separate oracle dynamic-programming diagnostic. The oracle sees synthetic ground truth only and is never a candidate policy. Every non-oracle policy receives only observed counts, provenance diversity, recency, corrections and contradictions, and a capacity-pressure bit.

Each output reports a bounded synthetic query score under the synthetic reward table (not empirical accuracy), task utility, net utility after an explicit token storage penalty, oracle regret, load-bearing coverage, unsafe/stale items retained and promoted, compression ratio, and utility per token. All budget units, query utilities, and token costs are toy definitions; they are not model tokens billed by a provider, product memory bytes, or measured task outcomes. The oracle is an upper bound for this fixture, not evidence that a learnable policy can attain it. The runner writes a JSONL output, stderr log and sidecar receipt with source SHA, command, seed, machine/toolchain metadata, exit status, and SHA-256 digests; it refuses to overwrite artifacts from an earlier run. It invokes the independent `scripts/check_broca_budget_frontier.sh` verifier before reporting success. The verifier requires exactly 82 JSONL records, all 9×9 policy/budget pairs, receipt and stderr/output digests, metadata consistency, nonnegative oracle regret with zero regret for the oracle, budget adherence, a true raw-fit upper-budget control, and the faulty-promotion negative control. `scripts/test_broca_budget_frontier.sh` generates synthetic protocol-shaped fixtures to test positive acceptance and rejection of tampering, missing rows and oracle-regret violations; these are verifier tests, not consolidation outcomes. The CLI requires `--source-sha` to equal `git rev-parse HEAD` and rejects tracked dirty-worktree changes; an untracked output file is permitted.

### Why this extension matters

Recent relevant research supports three hard requirements for our experiments, while not proving any specific Broca design wins:

- [Retain or Consolidate? (arXiv:2607.17545)](https://arxiv.org/abs/2607.17545) reports a budget-dependent crossover: consolidation can improve answer accuracy under tight evidence budgets, while raw retention can be preferable when evidence fits. Therefore the protocol includes both tight and unconstrained budgets and reports each budget, not a single average.
- [The Epistemics of Agent Memory (arXiv:2609.33013)](https://arxiv.org/abs/2609.33013) reports a negative external-validity result: its synthetic consolidation-quality score did not predict transfer accuracy across the studied real benchmark cells. This is a warning not to mistake an oracle-by-construction score for real generalization.
- [Are Online Skill and Memory Modules Always Worth Their Tokens? (EMNLP 2026)](https://www.servicenow.com/research/publication/sina-hajimiri-are-emnlp2026.html) reports that augmentation benefits may disappear against a token-matched actor baseline. Every future real-task comparison must therefore account for memory's write/search/consolidation overhead and compare it with equal-budget alternatives.

This synthetic frontier is a source-level instrument check. It cannot establish those papers' hypotheses, reproduce their numbers, or answer whether Broca improves a real agent. It is useful because the full trade-off is explicit and the benchmark can fail visibly before any expensive model run.
