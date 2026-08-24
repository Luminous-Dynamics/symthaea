# SCIP Strict LLM Adapter v1

Status: experimental Phase B bridge.

This document defines the first strict execution seam between a validated SCIP `CognitiveEnvelope` and Symthaea's existing `LLMBackend` interface.

## Purpose

The adapter exists because `LLMOrgan::query_async` has a useful legacy behavior that is unsafe for faithful SCIP translation: if its backend is absent or fails, the organ falls back to simulated generation. The synchronous `Translation` simulation path constructs a default `StructuredThought`, so successful return from that API does not prove that the supplied grounded SCIP meaning was translated.

The strict adapter therefore never calls `LLMOrgan::query`, `query_async`, `query_streaming_async`, or `translate_thought`.

Its execution path is:

```text
validated CognitiveEnvelope
        |
        +-- canonicalize semantically unordered collections
        |
        v
LlmTextFallback::compile
        |
        +-- resolve / verify exact GroundedConceptGraph
        +-- preserve semantic hash, confidence, provenance, evidence IDs
        +-- separate trusted system instruction from untrusted graph data
        |
        v
immutable ScipLlmRequest
        |
        +-- domain-separated request digest
        |
        v
LLMOrgan::get_backend
        |
        +-- no backend ----------> explicit MissingBackend error
        |
        v
LLMBackend::generate (one call)
        |
        +-- backend error -------> explicit redacted BackendFailure error
        +-- empty output --------> explicit EmptyOutput error
        +-- oversized output ----> explicit OutputTooLarge error
        |
        v
exact surface-text digest
        |
        v
ScipLlmOutput
```

There is no simulation fallback inside this path.

## Semantic authority

`GroundedConceptGraph` remains canonical. Model-produced text is a surface realization, not new grounded truth.

`ScipLlmOutput` carries:

- adapter profile (`symthaea.scip-llm-adapter/v1`)
- deterministic request digest
- deterministic exact-text surface digest
- SCIP message ID
- semantic hash
- source confidence
- evidence IDs
- provenance
- backend name
- fallback mode

alongside the generated text. This makes the execution auditable without claiming that the language model has independently verified its own translation.

A successful backend call is therefore evidence of **which exact request produced which exact UTF-8 surface bytes**, not evidence that the surface text is semantically faithful.

## Canonical request construction

SCIP semantic identity is insensitive to ordering that does not change meaning. The strict text adapter mirrors that rule before it creates model-facing bytes.

Before `LlmTextFallback::compile`, v1 canonicalizes:

- grounded graph node order;
- per-node grounding-reference order;
- edge order;
- per-edge evidence-reference order;
- envelope evidence-ID order;
- provenance feature-flag order;
- provenance transformation order.

This means two valid envelopes representing the same grounded state cannot acquire different strict request digests merely because vectors were populated in a different order.

`resolved_graph` is consumed only for payloads whose exact semantics are external to the envelope (`Hdc` and `Reference`). GroundedGraph and canonical StructuredJson are self-contained, so irrelevant caller-supplied resolution state is ignored rather than allowed to introduce a new failure path.

## Deterministic audit binding

The v1 request digest is domain-separated BLAKE3 over:

1. `symthaea-scip-llm-request-v1\0`
2. fallback-mode code (`0 = FaithfulTranslation`, `1 = GroundedReasoning`)
3. IEEE-754 `f32` temperature bits in little-endian order
4. `max_tokens` as little-endian `u64`
5. one byte binding `consciousness_context = None`
6. length-prefixed UTF-8 system prompt
7. length-prefixed UTF-8 grounded data content

Every variable-length field is prefixed by its byte length encoded as little-endian `u64`.

The accepted surface-text digest is domain-separated BLAKE3 over:

1. `symthaea-scip-llm-surface-v1\0`
2. little-endian `u64` UTF-8 byte length
3. exact returned UTF-8 bytes, without trimming or normalization

Changing request construction or privileged generation policy requires a new adapter-profile/digest version rather than silently reusing the v1 audit identity.

These digests are suitable inputs to later Xenia/transcript evidence binding. They do **not** authenticate the backend by themselves and do **not** provide confidentiality. Deterministic digests are equality/correlation identifiers; exposing them can reveal that two executions used the same request or output. Encryption, access control and disclosure policy remain transport/session concerns.

`LLMBackend::name()` is likewise only a human-readable runtime label. v1 does not cryptographically bind exact model weights, provider deployment, runtime configuration or inference implementation. A later protocol-agnostic backend-attestation seam can add that information without making `LLMOrgan` depend on SCIP.

## Instruction/data boundary

`LlmTextFallback` is the only compiler used by the strict adapter. All graph strings—including labels, relations, evidence identifiers, URLs, quoted text, code fragments, and provenance values—remain inside the untrusted data packet.

`ScipLlmRequest` keeps its `content` and `system_prompt` fields private. Callers can inspect them through read-only accessors, but cannot mutate a compiled request and then still claim it was produced by the strict adapter.

The adapter also fixes `GenerationParams` internally. v1 does not accept caller-provided system-prompt or consciousness-context overrides.

Current parameters:

| Mode | Temperature | Max tokens |
|---|---:|---:|
| FaithfulTranslation | 0.2 | 512 |
| GroundedReasoning | 0.3 | 768 |

These are adapter policy, not protocol semantics, and may be revisited independently of SCIP's canonical graph representation. A change must also advance the request-digest profile if it changes exact backend inputs.

## Diagnostic privacy

`ScipLlmRequest` and `ScipLlmOutput` implement custom redacted `Debug` views.

Ordinary structured/log debugging does **not** expose:

- grounded compatibility content;
- trusted system instructions;
- generated surface text;
- source message IDs;
- semantic hashes;
- request digests;
- surface digests;
- provenance strings.

The debug views expose only non-content operational metadata such as mode, byte counts, confidence, evidence/provenance entry counts, backend label and elapsed time. Callers that intentionally need sensitive values must request them through explicit fields/accessors and apply their own disclosure policy.

This does not make the objects confidential in memory; it prevents accidental disclosure through the normal `Debug` path.

## Backend execution

The adapter uses the backend explicitly configured on `LLMOrgan` and calls `LLMBackend::generate` exactly once.

It intentionally does **not** call `LLMBackend::is_available` first. A preflight availability probe would add another network operation and create a time-of-check/time-of-use race. The generation call itself is authoritative: success returns output, and failure remains an explicit failure.

An explicitly configured `SimulatedBackend` is still legal for tests. The safety invariant is not “simulation can never exist”; it is “simulation can never be selected silently because a supposedly faithful SCIP backend call failed.”

### Backend error privacy

Arbitrary provider/backend error strings do not cross the strict adapter error boundary. They may contain URLs, headers, request fragments, infrastructure details, or provider-specific data that should not automatically become part of a semantic-layer failure object.

`BackendFailure` therefore exposes the configured backend name but not `error.to_string()`. Operator-specific diagnostics remain the responsibility of the backend/runtime observability layer.

## Output resource limit

v1 rejects backend output larger than 1 MiB.

This is a **post-generation acceptance ceiling**, not an allocation guarantee: the current `LLMBackend::generate` trait returns a completed `String`, so an adapter cannot prevent a custom backend from allocating more internally before returning. The ceiling prevents oversized text from being accepted or propagated farther through this boundary. A future streaming/size-aware backend primitive can enforce an earlier hard cap.

## Accounting limitation

This first bridge depends only on public root APIs. Directly calling the backend therefore does not mutate `LLMOrgan`'s private statistics, embedding cache, conversation history, or distillation collector.

That is intentional for Phase B1. It proves the strict semantic and failure boundary before changing root internals.

A later root-side integration may add a generic strict execution primitive that preserves normal accounting while retaining all of these fail-closed properties. It must not reintroduce `query_async` simulation fallback and should remain protocol-agnostic so `LLMOrgan` does not depend on SCIP.

## Required validation

The adapter test suite must prove at least:

1. direct grounded envelopes preserve source identity through successful execution;
2. HDC/reference-derived requests require the exact matching grounded graph;
3. semantically equivalent graph/evidence/provenance ordering yields identical strict request bytes and digests;
4. self-contained payloads ignore irrelevant external graph resolution state;
5. instruction-like graph strings remain data, not system instructions;
6. missing backend is an error;
7. backend failure is redacted and never reaches organ simulation;
8. explicit simulated backends remain explicit test choices;
9. empty and oversized outputs fail closed;
10. strict execution does not mutate legacy `LLMOrgan` accounting through a hidden fallback path;
11. request digests are deterministic and change with semantic/policy changes;
12. accepted surface digests bind the exact returned UTF-8 bytes;
13. default `Debug` output redacts prompt/surface content and correlatable content-address identifiers.

## Non-claims

This adapter does not claim:

- that model output is semantically faithful merely because generation succeeded;
- that the LLM independently grounds or verifies graph contents;
- that generated text can replace the canonical graph;
- that request/surface digests authenticate the backend or transport;
- that deterministic digests provide confidentiality;
- that backend name proves exact model/runtime identity;
- that it preserves `LLMOrgan` conversation/statistics accounting yet;
- that the post-generation output ceiling constrains backend-internal allocation;
- that hosted LLMs natively consume HDC vectors.

The adapter is a compatibility execution boundary for today's text-oriented language models, not the final native cognitive interchange path.
