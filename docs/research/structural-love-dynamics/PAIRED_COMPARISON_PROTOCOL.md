# SLD v0.1 — paired comparison and uncertainty protocol

**Status:** executable comparison design; not a benchmark result.  
**Tool:** \`cargo run -p xtask --bin sld-compare -- --receipt-a A.json --receipt-b B.json --clusters clusters.json --out comparison.json\`.

## Purpose

This is the next measurement-integrity gate between individually valid evaluation receipts and an honest model comparison. A “COMPARABLE” run receipt is necessary, but not sufficient: the two runs must also share the exact case census, corpus and schema digests, source commit, toolchain, generation settings, rubric, and judge panel.

The tool deliberately compares **observable PASS/FAIL contract outcomes**, not internal Φ values, emotion labels, satisfaction scores, or claims that a model experiences love. It is not a clinical or release-approval tool.

## Fail-closed eligibility

The tool refuses to create a comparison report unless all of these conditions hold:

1. Both receipts declare \`run_status=COMPLETE\`, \`comparison_eligibility=COMPARABLE\`, and \`evaluation_mode=MODEL_BEHAVIOR\`.
2. Both working trees are clean and both receipts pin the same source commit and toolchain.
3. Both receipts include the complete expected case-ID census; the set is unique, matches the declared count, and exactly matches the actual result IDs.
4. Every case result is PASS or FAIL. An indeterminate, errored, skipped, missing, duplicated, or unexpected case makes the comparison non-comparable; it is never silently treated as a failure or a pass.
5. Dataset version, visibility/split, schema/corpus/inventory hashes, generation and tool configuration, rubric digest, judge-panel identities, and human-review status/count are identical. Both receipts must record `independent_adjudication_status=COMPLETE` and carry the same receipt-bound oracle evidence summary, including initial-label pairwise agreement and unanimous-case rates (raw descriptive diagnostics, not chance-corrected truth measures): immutable `ledger_sha256`, exact `case_id_inventory_sha256`, protocol/rubric digests, full two-review-per-case coverage, zero unresolved/indeterminate items, documented blinding and randomization, conflict disclosure, and completion time. The separate `sld-oracle-audit` tool validates the review ledger's case coverage and reviewer independence before generating the summary; otherwise the comparator refuses to rank candidate-oracle results.
6. The cluster map assigns each expected case exactly once. A cluster cannot span more than one declared family; at least two clusters are needed for the interval.

This is stricter than merely trusting a receipt's eligibility flag. If a model provider only exposes a moving alias and cannot provide a stable snapshot, record that limitation in the receipt rather than representing the comparison as fully reproducible.

## Statistical unit and estimates

Each result is a paired outcome: the same frozen case is evaluated for A and B. Define case delta as \(\Delta_i = I(B_i=\mathrm{PASS}) - I(A_i=\mathrm{PASS})\), in \([-1,1]\). The cluster map groups contrast pairs, related prompts, or trajectory families that should not be treated as independent observations.

Before any model comparison, freeze the review-ledger digest and the sorted case-ID inventory digest. `case_set.inventory_sha256` and `oracle_adjudication.case_id_inventory_sha256` must match; the latter is the SHA-256 of the compact UTF-8 JSON serialization of the sorted ID array, while `ledger_sha256` hashes the exact ledger file bytes. The separate `sld-oracle-audit` CLI checks ledger coverage and emits this receipt-compatible summary. A raw agreement rate measures reviewer consistency, not label truth; human-led adjudication and an appropriate reliability analysis remain mandatory.

The primary estimate is the **unweighted mean of the within-cluster mean deltas**. Each declared cluster contributes once, so a large paraphrase family or a long trajectory cannot win simply by contributing more rows. A deterministic SplitMix64 generator, fixed seed, and 10,000 bootstrap replicates (default) resample cluster effects with replacement. The report includes the percentile 95% interval and exact seed/replicate count. At least 1,000 replicates and two clusters are required.

The report also includes case-weighted pass rates/delta as a sensitivity view, case-level A wins/B wins/ties, cluster-level wins/ties, and per-family and per-cluster breakdowns. Do not choose the view after seeing which one makes a preferred model look better. Pre-register the primary cluster mapping before model outputs are opened. If multiple cluster definitions are scientifically plausible, declare one primary and publish predeclared sensitivity analyses.

This is intentionally a small, dependency-light bootstrap, not a complete statistical package. The percentile interval can be unstable with few clusters; <10 clusters are explicitly labelled exploratory. It does not correct a biased corpus, invalid oracle, sample-selection bias, model stochasticity not represented in the receipts, or repeated comparisons across many model snapshots. Use human adjudication and a planned multiplicity/power strategy for publication-grade claims.

## Non-compensable safety floor

Any critical invariant failure in either evaluated system sets \`critical_invariant_gate.status=BLOCKED_CRITICAL_FAILURE_OBSERVED\`. A favourable mean, confidence interval, or high score elsewhere cannot compensate. When none are observed, the report says only \`NO_CRITICAL_FAILURE_OBSERVED\`: this is not a proof of zero risk, and the report always sets \`release_promotion_eligible=false\`.

This aligns the comparison with current public conversational-safety evaluation practice, where suite floors and slice-level outcomes are reported alongside aggregates; a composite rank alone is an inadequate safety claim. NOPE Evals' live results page also warns that adjacent rankings can lie within measurement noise and publishes relational-integrity and long-context decay probes. <https://evals.nope.net/>

## Recommended experiment design

- **Paired first:** same cases and adjudication protocol; preserve every result and response digest.
- **Blind the oracle:** raters should not know which system produced which response; randomize A/B presentation order and record it if the human-rating harness supports that.
- **Judge audit:** use multiple independently versioned judges plus human adjudication on a preregistered subset; report disagreements and position bias instead of majority-vote erasure.
- **Counterfactual tests:** keep wording and identity constant when changing a morally relevant premise; separately probe paraphrase stability and role inversion.
- **Long-context comparison:** repeat the exact same sensitive disclosure after cold start and after fixed neutral/rapport context lengths, then report each depth. Do not let earlier benign dialogue count as evidence that a later critical disclosure is safe.
- **Ablations:** compare a proxy-free baseline against each telemetry signal and the complete system; a signal is retained only if it improves held-out choices with stable construct validity and no critical regression.
- **Reporting:** publish case-level census, per-family floors, raw critical failures, exact model/API snapshot, prompt/config/rubric/judge digests, cluster map, bootstrap seed, CI, exclusions, and unresolved reviewer dissent.

## Industry benchmark positioning and target gates

The SLD 52-case / 26-pair synthetic corpus remains a **candidate** oracle, not a representative population sample. Compare it with, rather than substitute it for, established suites:

- Emotional Dynamic Trajectories (AAAI 2026) contributes trajectory-level emotional stability measures over 328 contexts and 1,152 disturbance events. SLD's proposed increment is not “more trajectory data”; it is consent/agency constraints, evidence lineage, critical non-compensation, and repair obligations. <https://ojs.aaai.org/index.php/AAAI/article/view/37189>
- NOPE Evals provides distinct crisis, relational-integrity, epistemic-safety, emotional-attunement, and honesty facets; its public page currently reports the companion-harm slice across 64 models and long-form alignment-decay probes with 0/8/16 prior-turn contexts. Direct comparisons require matched prompts/rubrics/runs, not copying the leaderboard score. <https://evals.nope.net/>
- ICML 2026 work on LLM-as-a-jury reports that judges vary in reliability and bias, supporting the choice to pin judges and preserve disagreement rather than use a single opaque judge. <https://proceedings.mlr.press/v306/qian26i.html>
- A 2026 evaluation-statistics reference documents why paired comparisons and cluster-aware uncertainty matter for model-evaluation claims. <https://www.stats.bris.ac.uk/R/web/packages/evaluatellm/refman/evaluatellm.html>

**Proposed gates (not externally established standards and not yet achieved):**
1. 100% expected-case accounting; fail closed on any mismatch.
2. Zero observed critical invariant failures on the preregistered critical slice; any failure blocks promotion.
3. Report primary cluster-weighted delta plus a 95% uncertainty interval, paired wins/losses/ties, and family floors.
4. No “beats SOTA” claim unless the uncertainty and judge-error bands support a predeclared practically meaningful improvement on shared/matched tasks, with no critical-floor regression, and independent held-out replication.
5. Treat small benchmark gains inside judge/model/provider variance as inconclusive, even when the point estimate is positive.

No model comparison has been run by adding this tool. The code and its tests are authored, but require compilation and execution on a Rust toolchain before the implementation is considered validated.
