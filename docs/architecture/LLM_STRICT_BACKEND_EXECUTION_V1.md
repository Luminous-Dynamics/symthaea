# LLM Strict Backend Execution v1

Status: experimental root-language primitive.

## Purpose

`LLMOrgan::query_async` intentionally preserves a legacy convenience policy: if a configured backend fails, the organ records the error and falls back to simulated generation. That policy is useful for interactive operation but is not suitable for callers that require a real backend result or an explicit failure.

`LLMOrgan::execute_backend_strict` adds a protocol-agnostic real-backend path with **no simulation fallback**.

It is deliberately defined in the root language organ rather than SCIP. SCIP, Broca, code-generation, evaluation, and future callers may use the same strict primitive without making `LLMOrgan` depend on any interchange protocol.

## Execution contract

```text
LLMQuery
   |
   v
configured LLMBackend?
   | no
   +--------> MissingBackend
   |
   yes
   v
GenerationParams from query/config
   |
   v
LLMBackend::generate exactly once
   |
   +-- error --> stats.errors += 1 --> Generation error
   |
   v
normal real-backend success accounting
   |
   +-- queries_processed
   +-- tokens_generated
   +-- average generation latency
   +-- output embedding / embedding cache
   +-- bounded conversation history when enabled
   |
   v
LLMGenerationResult
```

No branch in `execute_backend_strict` calls `query`, simulation helpers, or another fallback generator.

## Legacy compatibility

`query_async` keeps its existing behavior. Its backend-success block is factored through the strict primitive; if strict execution returns an error, `query_async` logs the operator-visible backend cause and then follows its existing simulation path.

Therefore the refactor gives strict callers a fail-closed API without silently changing existing interactive behavior.

## Accounting semantics

Successful strict backend execution intentionally matches the previous real-backend success path:

- increment `queries_processed` once;
- count generated whitespace tokens and add them to `tokens_generated`;
- update the running average generation time;
- create/cache the output embedding;
- when memory is enabled, append the user request and assistant response;
- trim conversation history to the existing 100-message bound;
- return confidence `0.9` and `FinishReason::EndOfSequence`.

A backend generation failure increments `errors` once and does not increment `queries_processed`, produce an embedding, or add conversation history.

A missing backend does not increment `errors`, matching the previous `query_async` no-backend behavior.

## Error privacy and operator diagnostics

`LLMBackendExecutionError` has two forms:

- `MissingBackend`
- `Generation { backend, source }`

Default `Display` and `Debug` expose the backend label but redact arbitrary provider error text. This prevents accidental propagation of URLs, request fragments, provider headers, or other infrastructure details through ordinary semantic/application errors.

The underlying `anyhow::Error` remains available through `std::error::Error::source()` for code that intentionally owns operator observability.

The backend label is not cryptographic model identity.

## Non-claims

This primitive does not:

- depend on or understand SCIP;
- guarantee that generated text is factually or semantically correct;
- authenticate the backend or exact model weights;
- impose an output-size ceiling;
- change streaming fallback behavior yet;
- change the native Broca/SSM direct-thought path;
- collect SCIP request/surface digests.

Those responsibilities remain in their existing layers.

## Required validation

The focused tests must prove:

1. successful strict execution preserves real-backend statistics and history accounting;
2. repeated output exercises the existing embedding cache;
3. backend failure returns an explicit error, increments `errors`, and never simulates;
4. default error Display/Debug redact provider detail while `Error::source()` retains it;
5. missing backend is explicit and does not mutate accounting;
6. `query_async` still follows its legacy backend-failure-to-simulation policy.

After this primitive is validated, the strict SCIP adapter can migrate from direct `get_backend().generate()` execution to `execute_backend_strict`, regaining normal organ accounting without coupling the organ to SCIP.
