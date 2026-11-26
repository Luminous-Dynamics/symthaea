# MACE context exposure receipt: Rust verifier v0.1

**Status:** Rust verifier implementation and deterministic unit tests authored on a public research branch. Compilation, rustfmt, tests and CI have not been executed in this environment. This is not a production provider interceptor, encrypted object store, trusted observer deployment or causal estimator.

## Implementation

Source: `research/broca-mace-transfer-bench-v0-1/aml-adapter/src/mace_context_exposure_receipt.rs`

Export: `aml_adapter::mace_context_exposure_receipt`

The module consumes the receipt contract proposed in [PR #13709](https://github.com/Luminous-Dynamics/symthaea/pull/13709) and the MON-005 adapter crate dependency set. It uses the existing serde, serde_json, serde_json_canonicalizer, sha2, base64 and ed25519-dalek dependencies.

## Independent request-body contract verification

The consumer now parses the exact immutable request artifact independently of the receipt producer. It verifies that:

- The request body's actual `model` selector equals the independently pinned model revision.
- The request has at least one message, and every message has a recognized role and a JSON-string content field.
- Every generic top-level provider parameter is a number or boolean. Free-form strings, nulls, arrays and nested objects fail closed; provider-specific dynamic fields require an adapter that source-maps their content.
- The generic numeric/boolean parameter map is reserialized deterministically and its SHA-256 profile is recomputed from the exact captured body. This recomputed digest must equal both the receipt and independent run-manifest pin.
- Every exposed span's `message_index` resolves to an actual message, its claimed role matches the parsed role, and its half-open byte interval is contained inside that message's exact serialized `content` string. The consumer uses the borrowed raw JSON string slice, not a search for matching text.

This is intentionally stricter than checking the request-body digest and trusting producer-supplied profile/span metadata. JSON Schema cannot prove these relationships.

## Provider parameter profile binding

The verifier binds `provider_parameter_profile_sha256` from the signed receipt to the independent run manifest. The value is the SHA-256 commitment to the canonical JSON serialization of the deterministic, lexicographically ordered provider parameter map. The capture adapter limits generic parameters to a fixed key/type allowlist: numeric `temperature`, `top_p`, `frequency_penalty`, `presence_penalty`, `max_tokens`, `max_completion_tokens`, `seed`, `n`, `top_logprobs`; boolean `stream`, `logprobs`, `parallel_tool_calls`. Unknown keys, wrong types, free-form strings, nulls, arrays and nested objects fail closed.

This closes a request-construction gap in which memory-derived instructions, metadata or tool definitions could be inserted through arbitrary JSON outside the message source map. Structured provider fields require a provider-specific typed adapter that source-maps every dynamic text/media leaf. The static profile must be frozen before the run and reused unchanged by all randomized arms, including true no-memory.

### Responsibilities implemented

- Strict typed parsing with unknown-field and duplicate-field rejection.
- Binding to independently supplied run, pair block, world cluster, query, arm, invocation, DecisionNeed, policy, source registry, lineage/lifecycle, renderer, transport, provider-parameter profile, model, clock and observer profile digests.
- Encrypted immutable request-body artifact profile binding, including storage, encryption, access, retention and custody digests.
- Request-body resolution through an explicit artifact resolver interface; the resolver owns ciphertext, custody and decryption verification.
- Exact plaintext byte-length and SHA-256 verification against both receipt and artifact commitments.
- Byte-range bounds, per-span SHA-256, duplicate exposure-ID, duplicate-parent and strict disjoint-span checks (v0.1 rejects all overlapping spans).
- Rejection when evidence appears in both the admitted-exposure and admission-denial sets.
- Delegated evidence identity, transitive lineage, source, authority, scope, estimand, rights and current-at-use checks through the existing canonical MACE owner interface.
- Ed25519 signature verification and a KMS/HSM-capable signer interface over `SHA-256("MACE-CONTEXT-EXPOSURE-V1\\0" || JCS(receipt_without_signature))`.
- Ed25519 key-ID derivation using the same SubjectPublicKeyInfo SHA-256 / first-16-byte convention used by MON-005.
- Signature verification before artifact retrieval or evidence-registry calls, avoiding expensive trust-boundary operations on unauthenticated receipt content.
- A strict no-memory-arm invariant: a `no_memory` invocation must have no memory exposures and no memory admission-denial records.

### Explicit owner boundaries

The module does not implement:
- encryption, object storage, retention enforcement or custody issuance;
- a trusted provider transport interceptor or complete-origin observer;
- a new observer key registry or trust root;
- a parallel canonical evidence registry, lineage verifier or lifecycle authority;
- provider-side hidden prompt inspection or server-side prompt transformation receipts;
- randomized intervention scheduling, evaluator custody or utility scoring.

These are deliberate interfaces to existing owners or separate deployment qualification work.

## Deterministic tests authored (not executed)

Twenty-two Rust unit tests cover:
1. valid signed receipt, body, span and evidence binding;
2. wrong run binding;
3. changed request-body rejection;
4. overlapping span rejection;
5. nonzero untracked-origin span rejection;
6. invalid Ed25519 signature rejection;
7. unknown observer key-ID rejection;
8. canonical evidence-admission rejection;
9. denied-and-exposed evidence collision;
10. request-artifact storage-profile substitution;
11. unavailable request-body artifact;
12. memory contamination of the no-memory arm;
13. duplicate and unknown JSON member rejection;
14. provider-parameter profile substitution rejection;
15. independent request-body parameter-profile recomputation;
16. free-form string and nested provider-parameter rejection;
17. model selector verification against the captured body;
18. exposure containment within the indexed message content;
19. message-role binding.

(The list is numbered by category; the exact source-level test count should be reconfirmed after any later test additions.)

These are authored tests only. They have not been compiled or executed. No passing test, Rust validation, CI, provider interception, security qualification or benchmark score is claimed.

## Remaining integration work

1. Implement the production artifact resolver against an independently pinned object-store custody profile; verify ciphertext and custody receipt before returning plaintext.
2. Implement the production observer-key resolver against a roster pinned outside candidate memory, logs and backup domains.
3. Bind the renderer's complete source map to exact UTF-8 request serialization; the boolean coverage declaration is an observer assertion until that implementation is independently qualified.
4. Implement the canonical admission verifier using #13264 and the existing source/authority/lifecycle owners.
5. Add a fake transport that serializes a provider request, captures bytes immediately before send, emits receipt plus encrypted artifact, and validates every memory-origin span.
6. Integrate one receipt per model invocation into #13141. A missing/invalid receipt makes a memory-consuming invocation non-qualifiable.
7. Keep randomized intention-to-treat as the primary memory-effect estimate. Receipt exposure is treatment-compliance/mechanism evidence, not proof of attention or causality.
8. Run `cargo fmt --check`, focused `cargo test`, JSON Schema validation and exact-head CI in a Rust-enabled environment before promoting this draft.

## Claim boundary

A verified receipt supports the statement: “under the pinned capture observer and custody profiles, these evidence-derived byte spans were present in the exact client request body associated with this model invocation.”

It does not support: “the model attended to, understood, relied on, or was causally changed by those spans.” Those require separate experiments and, where available, provider-side ingress attestations.
