# MACE LongMemEval-V2 Adapter V1

Status: standalone Rust adapter contract and strict schema authored; dedicated validation is pending. The official LongMemEval-V2 repository and evaluator remain unchanged.

## Official interface and privacy boundary

LongMemEval-V2 memory backends implement insert(trajectory) and query(question_text, optional_question_image). The harness also gives the backend only a random run-local query_invocation_id through thread-local query context. It does not expose the dataset question ID, question type, gold answer, evaluator function or grader labels to query(). The backend returns an ordered list of text and image items. The official harness truncates that list to a fixed memory-context token budget before building the reader messages.

Important ordering: post_query_hook() runs after query() but before truncation. It can record which evidence items were selected, but it cannot by itself prove which items reached the reader. The harness then truncates by an ordered prefix of complete context items, builds the final messages, and returns those messages plus the final memory_context and token counts. Therefore the adapter has two phases:

1. Pre-query selection receipt, emitted from query/post_query_hook, with ordered evidence item hashes and their use/lineage receipts.
2. Final reader-exposure receipt, emitted by an outer evaluator wrapper after the official prompt builder returns, using the exact post-truncation context prefix, exact reader request digest and reader response digest.

This design avoids modifying the official harness or changing its scoring. It also avoids falsely labelling the pre-truncation selected set as exposed.

## Receipt protocol

- Insert receipts preserve the official harness insertion order, tier and domain, plus trajectory, payload and source-snapshot commitments. The adapter rejects duplicate trajectories, order regressions and cross-domain mixing.
- Query input contains only run ID, opaque invocation ID, question-text digest and optional question-image digest.
- Pre-query selection records ordered text/image item digests, the MACE use-time and source-lineage receipt digests, and the canonical #13973 DecisionEvidenceUseLinkV1 digest associated with each item.
- Final exposure must equal an ordered prefix of the pre-query selection, because the official harness truncates by prefix. Only surviving items receive exposed-evidence links; the exact per-item #13973 link commitment is carried through to the final reader-exposure receipt.
- The final receipt binds the exact post-truncation reader request digest, reader response digest, fixed reader-model digest and reader-generation-configuration digest, along with original/final token counts, budget and memory/post-hook latency.
- Image item hashes must cover the image bytes used by the reader, not the filesystem path. The final reader-request hash should cover the fully serialized messages after image-to-data-URL conversion.

No question IDs, question types, gold answers, expected outcomes or evaluator labels are present in the memory-query contract. The evaluator may map the opaque invocation ID back to its private question record after the run.

## Benchmark execution plan

LongMemEval-V2 exposes two independent tiers (Small and Medium), and each operating point requires both web and enterprise domain runs. Preserve the official reader, official task/haystack assignment, official context-token cap and official aggregation. Produce a separate sidecar for each tier/domain/run; only the official leaderboard packaging utility combines web and enterprise metrics.

Recommended first experiment:

1. No-retrieval baseline.
2. Official RAG query-to-slice plus notes baseline.
3. One competitive external backend using the official Memory API.
4. MACE evidence-aware adapter with use-time revalidation and final-prefix exposure receipts.
5. The same MACE candidate with governance gates disabled only as a diagnostic ablation.

Keep multiple latency operating points for a single method/tier. Compare Small only with Small and Medium only with Medium. Use paired task IDs and conversation/trajectory-family clusters. Report the official LAFS score separately from governance sidecar counts, utility ablations and lifecycle cost.

## Important implementation constraint

The LongMemEval-V2 harness currently truncates after the backend post-query hook. Do not patch the upstream benchmark or infer actual exposure from the hook alone. The outer run wrapper can use the returned row's final context, token counts and exact messages to construct the second-phase receipt. The reader response digest is captured after the fixed reader call.

## Tests

Eighteen synthetic Rust tests cover strict insert order and preservation of repeated identical trajectory payloads, privacy-minimized query fields, opaque invocation IDs, pre-query binding, exact-prefix exposure, truncation and token-budget consistency, missing reader request/response, omitted tail items, cross-domain insertion, identical context bytes from distinct evidence sources, reader configuration binding, and receipt tampering.

## Validation and claim boundary

This contract has not yet run against a LongMemEval-V2 dataset or official reader. Synthetic Rust tests do not constitute a leaderboard result. Digests are integrity commitments, not signatures or independent evaluator attestations. Before submitting, freeze candidate, reader, grader, prompt, context cap, tier, dataset split and operating point; then retain raw run receipts and request/response artifacts under independent custody.

Official benchmark: https://github.com/xiaowu0162/LongMemEval-V2/
Official leaderboard: https://xiaowu0162.github.io/longmemeval-v2/