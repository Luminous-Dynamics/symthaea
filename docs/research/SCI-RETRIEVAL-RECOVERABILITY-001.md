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

The classification is relative to a particular finite corpus snapshot, search surface, route policy, and budget. UnreachableWithinBudget is not an open-world claim that the source is absent. Reachability is asymmetric: a concrete pinned route is positive evidence even when the observed surface is partial; a missing route is not evidence of unreachability unless the declared finite surface and route inventory are frozen and complete. Thus a partial surface can preserve DirectlyReachable, ReformulationRecoverable, or NavigationRecoverable witnesses, but an item with no observed route becomes NotEvaluable. An entirely unknown surface makes every item NotEvaluable.

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

Metamorphic checks verify that reducing budget never improves reachability, adding a concrete first-pass edge can improve only the target unit, partial surfaces preserve positive route witnesses while blocking negative claims, unknown surfaces yield only NotEvaluable, and a tampered manifest commitment yields no route labels. An independent Python reference is compared against Rust for every classification and shortest-route witness. A model-supplied judgment field is rejected as an unknown property.

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


## New paradigm: evidence-asymmetric reachability

This evaluator uses a proof asymmetry that is easy to miss in ordinary retrieval metrics:

- One valid, pinned route witness proves that the benchmark's declared search surface permits access to that exact evidence unit.
- Failure to find a route proves nothing unless the benchmark has enumerated the complete route space for its exact declared finite surface and operation policy.
- Unknown or incomplete route inventories can therefore establish positive reachability, but cannot establish negative reachability.
- Even a complete graph only supports `UnreachableWithinBudget` inside its committed snapshot and budget; it says nothing about the open web or the world.

This is especially important for scientific and multilingual retrieval: partial API access, unpinned index state, rights restrictions, provider outages, or missing source versions must not be laundered into evidence absence.

## Research comparisons

The current literature gives us concrete evaluation priorities:

- REALM 2026 recommends a model-free recoverability diagnostic and a one-shot / fixed-loop / full-agent ablation ladder. In its studied corpora, headline gain varied from -0.21 to +0.30 by recoverability regime, and fixed iteration recovered 65–96% of the apparent agency gain on reformulation-recoverable corpora. These are paper-specific findings, not Symthaea results: https://aclanthology.org/2026.realm-1.47/
- Retrieval Myopia found up to 46.2% of examples answered incorrectly despite gold evidence already being present in retrieved results; clean-format re-presentation recovered up to 48.9% of those cases without weight updates. We therefore measure reader-format gain separately from retrieval and agentic gain: https://aclanthology.org/2026.findings-emnlp.1713/
- RDQA reports 24.0% adjusted accuracy for its best generic agent against 41.1% for a process-aware baseline across document, video and audio tasks, with parametric leakage as high as 96% in the studied conditions. This argues for source-grounded, modality-aware controls rather than answer-only scoring: https://aclanthology.org/2026.emnlp-main.1855/
- TEMPO reports 32.0 nDCG@10 and 71.4% Temporal Coverage@10 for the best evaluated system in its 16-system comparison, showing that topical relevance can coexist with missing required periods: https://aclanthology.org/2026.emnlp-main.291/
- AfrIFact covers ten African languages plus English and reports persistent cross-lingual retrieval weakness for leading embeddings. We should evaluate South African language/document slices directly rather than extrapolate parity from English: https://aclanthology.org/2026.emnlp-main.133/
- Agentic-R reports improvements over strong baselines on seven single-hop and multi-hop QA benchmarks using a retriever trained around multi-turn utility. It is a competitive retrieval baseline to include, not proof that agentic retrieval is universally superior: https://aclanthology.org/2026.findings-acl.785/

The comparison ladder is therefore BM25, dense multilingual retrieval, hybrid retrieval, standard reranking, fixed multi-query/iterative RAG, Agentic-R-style retrievers, and the full agent. All must use matched corpus snapshots, models, tool/action budgets, retrieval depths, evidence-token ceilings, cost and latency accounting. The primary claim should be incremental value over the fixed-loop baseline, not merely a win over one-shot RAG.
