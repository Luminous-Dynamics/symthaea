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
        v
LLMOrgan::get_backend
        |
        +-- no backend ----------> explicit MissingBackend error
        |
        v
LLMBackend::generate (one call)
        |
        +-- backend error -------> explicit BackendFailure error
        +-- empty output --------> explicit EmptyOutput error
        +-- oversized output ----> explicit OutputTooLarge error
        |
        v
ScipLlmOutput
```

There is no simulation fallback inside this path.

## Semantic authority

`GroundedConceptGraph` remains canonical. Model-produced text is a surface realization, not new grounded truth.

`ScipLlmOutput` carries the source:

- SCIP message ID
- semantic hash
- source confidence
- evidence IDs
- provenance
- backend name
- fallback mode

alongside the generated text. This makes the surface output auditable without claiming that the language model has independently verified its own translation.

## Instruction/data boundary

`LlmTextFallback` is the only compiler used by the strict adapter. All graph strings—including labels, relations, evidence identifiers, URLs, quoted text, code fragments, and provenance values—remain inside the untrusted data packet.

`ScipLlmRequest` keeps its `content` and `system_prompt` fields private. Callers can inspect them through read-only accessors, but cannot mutate a compiled request and then still claim it was produced by the strict adapter.

The adapter also fixes `GenerationParams` internally. v1 does not accept caller-provided system-prompt or consciousness-context overrides.

Current parameters:

| Mode | Temperature | Max tokens |
|---|---:|---:|
| FaithfulTranslation | 0.2 | 512 |
| GroundedReasoning | 0.3 | 768 |

These are adapter policy, not protocol semantics, and may be revisited independently of SCIP's canonical graph representation.

## Backend execution

The adapter uses the backend explicitly configured on `LLMOrgan` and calls `LLMBackend::generate` exactly once.

It intentionally does **not** call `LLMBackend::is_available` first. A preflight availability probe would add another network operation and create a time-of-check/time-of-use race. The generation call itself is authoritative: success returns output, and failure remains an explicit failure.

An explicitly configured `SimulatedBackend` is still legal for tests. The safety invariant is not “simulation can never exist”; it is “simulation can never be selected silently because a supposedly faithful SCIP backend call failed.”

## Output resource limit

v1 rejects backend output larger than 1 MiB. Provider token limits are advisory and backend-specific; the byte ceiling is a local defensive boundary against a broken or hostile backend returning unbounded surface text.

## Accounting limitation

This first bridge depends only on public root APIs. Directly calling the backend therefore does not mutate `LLMOrgan`'s private statistics, embedding cache, conversation history, or distillation collector.

That is intentional for Phase B1. It proves the strict semantic and failure boundary before changing root internals.

A later root-side integration may add a private strict execution primitive that preserves normal accounting while retaining all of these fail-closed properties. It must not reintroduce `query_async` simulation fallback.

## Required validation

The adapter test suite must prove at least:

1. direct grounded envelopes preserve source identity through successful execution;
2. HDC/reference-derived requests require the exact matching grounded graph;
3. instruction-like graph strings remain data, not system instructions;
4. missing backend is an error;
5. backend failure is an error and never reaches organ simulation;
6. explicit simulated backends remain explicit test choices;
7. empty and oversized outputs fail closed;
8. strict execution does not mutate legacy `LLMOrgan` accounting through a hidden fallback path.

## Non-claims

This adapter does not claim:

- that model output is semantically faithful merely because generation succeeded;
- that the LLM independently grounds or verifies graph contents;
- that generated text can replace the canonical graph;
- that the adapter authenticates the backend or transport;
- that it preserves `LLMOrgan` conversation/statistics accounting yet;
- that hosted LLMs natively consume HDC vectors.

The adapter is a compatibility execution boundary for today's text-oriented language models, not the final native cognitive interchange path.
