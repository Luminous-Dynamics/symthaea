# MACE Preference Exposure Inventory V0.1

**Status:** Research implementation on top of the #13973 decision-evidence exposure sidecar. It now distinguishes positive exposure from independently verified non-exposure. No production provider integration or benchmark result is claimed.

## Why this is necessary

A positive exposure link proves that a particular source-derived byte span appeared in one frozen model context. It cannot prove that a selected preference was absent.

For an Ignore decision, absence from retrieval results is insufficient: an item might enter through a summary, cache, profile preamble, tool result, prompt template, or another memory path. A missing source row is not evidence of non-exposure. The provider request must be tied to a complete inventory of all memory-origin spans in the exact context.

This module adds ContextExposureManifestV1, an independent selection verifier, a complete prompt-inventory verifier and a provider-request capture verifier. It reuses the #13973 ingress crate and does not create another evidence store or provenance ledger.

## Protocol

1. The existing #13153 action policy selects Ignore, Support or Dominate. The selection is bound to run, episode, subject, scope, purpose, audience, bitemporal validity, policy digest, authority receipt and the exact preference IDs.
2. The prompt assembler inventories every memory-origin span after all memory composition, summarization, profile injection and context formatting have completed. Every span maps to one or more source memory IDs, source receipts, lineage receipts, transformation-chain receipts and preference IDs.
3. The manifest binds the complete inventory to the exact model-context bytes, provider-request bytes, model invocation, prompt-assembler build, provider-adapter build and prompt/provider capture receipts.
4. The provider adapter captures the exact bytes submitted to the model. The verifier checks both byte digests and asks the independent provider-capture verifier to authenticate the capture and the transformation from context to request.
5. The verifier authenticates the #13153 selection receipt, the independent dominance authority if applicable, every source mapping, and the prompt inventory's completeness.
6. Only then does it return a private in-process VerifiedPreferenceExposureV1 capability.

The serialized selection and manifest are integrity-committed using SHA-256. These digests are not signatures. JSON deserialization never creates verifier custody; the host must rerun the injected verifier.

## Mode rules

- **Ignore:** none of the selected preference IDs may occur in any source mapping in the complete memory-origin inventory. If any selected preference appears—including inside a composed summary—the verifier rejects with ignored_preference_was_exposed.
- **Support:** every selected preference ID must be positively mapped to at least one exact memory-origin span.
- **Dominate:** same positive exposure rule as Support, plus a separately authenticated dominant-preference authority receipt. The selector cannot grant itself dominance.

Current user text is not classified as remembered memory merely because it contains a preference. It is included in the exact request bytes but does not count as a memory-origin span unless the prompt assembler derives it from stored memory. The trusted prompt-assembly verifier must preserve this distinction.

## Repetition and context accounting

A memory item can be repeated in different spans. This is not treated as a duplicate: VerifiedPreferenceExposureV1::preference_span_count reports the number of distinct spans containing each preference. That supports metrics for:

- repeated preference exposure per model invocation;
- source item fan-out through summaries/templates;
- irrelevant-memory token/byte share;
- selected-but-unexposed and unselected-but-exposed preferences;
- repeated detail exposure and prompt repetition.

Duplicate source mappings within the same span, duplicate span IDs, non-canonical span order and overlapping byte ranges are rejected.

## Hard resource bounds

- At most 256 preference IDs in a selection.
- At most 4,096 memory-origin spans per context.
- At most 256 source mappings per span.
- At most 256 preference labels per source mapping.
- Context and provider request each capped at 64 MiB.

The schema uses closed objects (additionalProperties: false) and carries the same collection limits.

## Trust boundary

The injected ContextExposureVerifierV1 must independently:
- authenticate that the selection is the exact #13153 action decision for the same run/episode/action sequence and the correct mode;
- authenticate the dominance receipt for Dominate;
- verify source, lineage and transformation receipts for every memory source;
- establish that the prompt assembler inventory is complete and includes all memory-derived spans after composition and truncation;
- authenticate the actual provider request capture and prove that it is the request submitted for the recorded model invocation.

A caller-provided memory_spans array, a self-reported prompt_assembly_receipt_sha256, a retrieval log, or an unsigned SHA-256 digest cannot establish completeness. Unknown injection paths must make verification fail closed.

This contract proves source-to-context inclusion/non-inclusion at the observed provider boundary. It does not prove model attention, comprehension, semantic adherence, causal influence, or the truth of a memory. Those remain separate measured outcomes.

## Composition

- #13154 / #13153 owns the cost-sensitive Ignore/Support/Dominate decision and hard authorization gates.
- #13973 owns source admissibility and the positive exact-byte exposure link.
- #14263 binds selected behavior to executor events and response commitment.
- #14264 owns cross-benchmark evaluation and rights-aware comparisons.
- #13388 owns the on-policy causal environment and user burden.
- #14280 owns the missing redb checkpoint/branch/read-only adapter capability.

An end-to-end episode should expose distinct receipts for retrieval, admissibility, influence selection, full context inventory, provider request, execution and user/task outcome. Do not collapse these into a single opaque memory-used boolean.

## Tests and validation

Fourteen deterministic tests are authored, covering:
- Ignore with complete inventory and unrelated memory only;
- Ignore rejecting a selected preference hidden inside a composed summary;
- incomplete inventory failing closed;
- Support requiring positive exposure of every selected preference;
- Dominate requiring separately verified authority and positive exposure;
- exact provider-request byte mismatch;
- source/transformation receipt rejection;
- canonical, non-overlapping and UTF-8-safe spans;
- repeated item exposure across separate spans and per-preference span counting;
- selection/manifest JSON round-trip with unknown-field rejection;
- negative bitemporal timestamps;
- manifest identity/snapshot rebinding;
- resource-limit rejection.

The dedicated Rust 1.96 workflow extends #13973's ingress suite and parses both exposure schemas. The workflow must run on the exact latest head before compilation, tests, Clippy or schema validation may be called passed. No MemPABench, OP-Bench, RPEval, or Agent Memory Leaderboard score is claimed.

## Research targets

On matched, frozen model/provider runs:
- **Ignore false-exposure:** zero selected-preference exposure in the complete inventory on deterministic controls.
- **Support/Dominate omission:** zero selected-preference omissions when the mode requires positive exposure.
- **Inventory completeness:** zero accepted manifests when any known prompt source/injection path is absent from the inventory.
- **Over-personalization:** OP-Bench at least the matched memory-free baseline, while MemPABench preference utility remains non-inferior.
- **Utility of influence:** positive paired decision utility for Support/Dominate over Ignore where preregistered applicability is positive; no utility claim where memory should be ignored.
- **Resource efficiency:** lower irrelevant-memory bytes and repeated-span count at non-inferior task quality; report tokens, tool calls, latency and cost separately.

These are prospective hard gates/objectives, not achieved results.
