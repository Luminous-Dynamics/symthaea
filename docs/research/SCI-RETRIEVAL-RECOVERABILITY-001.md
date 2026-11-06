# SCI-RETRIEVAL-RECOVERABILITY-001 — model-free benchmark recoverability and agency credit

Issue: Symthaea #10690
Parent: SCI-RETRIEVAL-CAUSAL-001 / #8907
Code lineage: research/sci-retrieval-recoverability-001

## Why this work exists

End-to-end agent success is not one capability. It mixes evidence reachability, retrieval, reading, evidence use, inference, and action selection. A task whose required source is unreachable under the declared search surface is not a fair test of the reader; a task whose gold source is already present but misread is not a retrieval failure.

The 2026 REALM study argues that agentic retrieval performance depends on benchmark recoverability and the load-bearing agent component. In the evaluated corpora, headline agent gain ranged from -0.21 to +0.30 in each corpus's native metric. On reformulation-recoverable corpora, a choice-free fixed loop recovered 65–96% of the gain typically credited to agency under two stronger readers. These results are study-specific, but they motivate a mandatory one-shot / fixed-loop / full-agent comparison.

Retrieval Myopia (EMNLP Findings 2026) reports up to 46.2% wrong answers when the gold-bearing document was already present in retrieved results. A clean-format re-read of the same documents recovered up to 48.9% of those cases without weight updates. Thus reader-format intervention belongs beside gold-evidence injection, not hidden inside a retrieval score.

RDQA (EMNLP 2026) adds source-grounded document, video and audio retrieval. It reports 24.0% adjusted accuracy for its best generic agent and 41.1% for a process-aware baseline, with Parametric Leakage Rate up to 96% in studied settings. These are not interchangeable with text-only QA leaderboard scores.

References:
- REALM 2026: https://aclanthology.org/2026.realm-1.47/
- Retrieval Myopia: https://aclanthology.org/2026.findings-emnlp.1713/
- RDQA: https://aclanthology.org/2026.emnlp-main.1855/
- Search-Time Contamination: https://aclanthology.org/2026.findings-emnlp.760/
- Agentic-R retriever: https://aclanthology.org/2026.findings-acl.785/

## Existing work composed

- #8907 / PR #8929 freezes counterfactual conditions before the run.
- #8950 / PR #8960 owns post-run trial outcome receipts, denominators and orthogonal metrics.
- #8865 / #8882 / #8885 / #8906 own retrieval exposure and contamination evidence.
- #10644 owns anchor-preserving temporal retrieval and stage-separated evaluation.
- #10671 interprets Mycelix EPI-013R receipts as candidate-only execution observations.
- Mycelix #5875/#2779 own the canonical tool receipt shape.

This work only classifies benchmark-relative reachability before outcomes are observed.

## Recoverability classes

For each required, pinned evidence unit, a finite graph binds the initial state to the exact artifact node. Edges represent permitted operations. Every edge has a positive cost in one frozen budget unit.

| Class | Operational definition |
|---|---|
| DirectlyReachable | Route uses first-pass retrieval only |
| ReformulationRecoverable | Route requires query reformulation and a post-reformulation retrieval, but no navigation |
| NavigationRecoverable | Route requires a structural operation such as open, follow link/TOC, pagination or in-document find |
| UnreachableWithinBudget | No permitted route fits the frozen graph and budget |
| NotEvaluable | Search surface is not frozen/observed or artifact availability is unknown/unavailable |

A direct route takes precedence over a reformulation or navigation route; reformulation takes precedence over navigation. The report includes the deterministic shortest route witness and route cost. These witnesses and gold artifact references are evaluator-side only.

The classification is relative to a particular finite corpus snapshot, search surface, route policy, and budget. UnreachableWithinBudget is not an open-world claim that the source is absent. A partial/unknown surface yields NotEvaluable, not an unreachable label.

## Executable contract

Implementation lives in research/science-bench-refresh/src/recoverability.rs. The typed manifest contains:
- pinned corpus, search-surface and route-policy SHA-256 digests;
- start node, finite positive route budget and unit;
- bounded nodes and typed operation edges;
- required evidence-unit IDs, artifact digests, source-family refs and availability state;
- a domain-separated commitment over the typed manifest fields.

CLI:

~~~sh
cargo run --locked --manifest-path research/science-bench-refresh/Cargo.toml -- \\
  commit-recoverability-manifest recoverability-fields.json

cargo run --locked --manifest-path research/science-bench-refresh/Cargo.toml -- \\
  audit-recoverability-manifest recoverability-manifest.json
~~~

The CLI is offline. It never opens URLs, calls a model, or executes a retrieval operation. The evaluator must independently anchor the manifest before any run. The output says external_anchor_required=true.

## Agency-credit experiment

Under equal budgets and fixed corpus/index/model/prompt/tool policy:
- A0 one-shot RAG;
- A1 choice-free deterministic iterative/reformulation loop;
- A2 full open-ended agent;
- A3 exact gold-evidence injection for reader/extractor isolation;
- A4 clean-format re-presentation of the exact same evidence bytes;
- A5 gold-evidence withheld, measuring abstention and parametric leakage.

Report A2−A0 total gain separately from A1−A0 fixed iteration and A2−A1 incremental agency. Report A4−A3 as a reader-format effect. Do not attribute improvements from fixed iteration or re-presentation to open-ended agency.

## Synthetic contract controls

The initial synthetic fixture covers:
- first-pass/direct retrieval;
- query-reformulation-only retrieval;
- navigation via open → paginate → in-document find;
- no route within budget;
- unknown artifact availability;
- duplicate artifact mirrors with a shared source family.

Metamorphic checks verify that reducing budget never improves reachability, adding a concrete first-pass edge can improve only the target unit, partial search surfaces yield only NotEvaluable, and a tampered manifest commitment yields no route labels. A model-supplied judgment field is rejected as an unknown property.

## Benchmark strategy and target comparisons

Once the synthetic contract and independent oracle pass:
1. use the REALM recoverability diagnostic before measuring agent gains;
2. run BM25, dense and hybrid retrieval, multi-query/fixed-iteration RAG, reranking, and an Agentic-R-style retriever under matched models and budgets;
3. compare one-shot, fixed-loop, and full-agent arms with the same action/token/time/tool ceilings;
4. use TEMPO-compatible temporal retrieval slices for time-coverage, AfrIFact-compatible cross-language retrieval where the data terms permit, and RDQA-compatible source-grounded multimodal slices only when licensed/pinned;
5. use gold-evidence reader tests and clean-format re-presentation to measure reading failure independently;
6. preserve search-time contamination canaries from #8906 and report all contaminated/blocked/failed/invalid/not-started trials in the assigned denominator.

Metrics remain separate: Recall@k, evidence-family recall, path completion and route cost; gold-conditional extraction and evidence-use effect; citation/span validity; abstention quality and parametric leakage; paired incremental agency effect; latency/tokens/tool calls/cost. No single score may compensate for a reachability, grounding or integrity failure.

## Initial qualification gates (targets, not results)

- 100% agreement between independent Rust and Python implementations on the synthetic corpus.
- 100% monotonicity on the frozen budget/route mutation suite.
- 0 model-dependent recoverability labels.
- 0 evaluator route witnesses/gold labels in policy-visible task input.
- 100% of evidence digests preserved by re-presentation controls.
- A2 is credited with an agency-specific gain only when A2−A1 is positive on preregistered recoverability strata at matched budgets, after paired uncertainty and family-wise correction.
- Baseline-only pilot and power analysis precede choosing any minimum empirical effect size.

## Claim ceiling

A passing synthetic graph contract establishes only deterministic classification of its frozen graph and budget. It does not prove live retrieval accuracy, open-world search completeness, source truth/authenticity, model evidence use, benchmark independence, language parity, or SOTA. No model-performance result is claimed by this implementation.
