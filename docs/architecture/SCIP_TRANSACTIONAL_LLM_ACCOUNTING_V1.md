# SCIP Transactional LLM Accounting v1

Status: experimental Phase B2.1 bridge.

This document defines a correctness-first composition between:

- Phase B1 `symthaea-scip-llm-adapter`, which proves strict SCIP realization and surface acceptance without mutating `LLMOrgan`; and
- Phase B2 `LLMOrgan::execute_backend_strict`, which performs one real backend call with normal organ accounting and no simulation fallback.

The problem is transactional ordering. B2 records a successful backend return immediately. B1 may then reject that returned surface because it is blank or larger than the SCIP 1 MiB acceptance ceiling. A naive B3 integration would therefore leave history, token counts, latency averages, or embedding-cache state for output that SCIP refused to accept.

## v1 staging algorithm

`execute_accounted_transactional` uses clone-and-commit staging:

```text
compiled ScipLlmRequest
        |
        v
clone current LLMOrgan
        |
        v
staged.execute_backend_strict(locked query)
        |
        +-- missing backend ------> MissingBackend; original unchanged
        |
        +-- backend failure ------> commit staged error counter only;
        |                           return redacted BackendFailure
        |
        v
backend surface on staged organ
        |
        +-- blank ----------------> EmptyOutput; discard staged organ
        |
        +-- > 1 MiB --------------> OutputTooLarge; discard staged organ
        |
        v
construct source-bound ScipLlmOutput
        |
        v
commit staged organ into caller
```

No path calls `LLMOrgan::query_async`, so backend failure can never trigger legacy simulation.

## Atomicity claim

For SCIP surface-acceptance failures, v1 is atomic with respect to `LLMOrgan` state visible through the cloned object:

- `queries_processed` does not advance;
- `tokens_generated` does not advance;
- generation latency average is unchanged;
- embedding-cache mutations from rejected text are discarded;
- conversation history receives no rejected user/assistant pair;
- `cache_hits` caused only by the staged rejected result are discarded.

Backend-internal state is not rolled back. The backend is held behind an `Arc` and may have its own interior state or remote side effects. The transactional guarantee is specifically about `LLMOrgan` accounting/state, not external model execution.

## Backend failures

Phase B2 increments `stats.errors` when the configured backend itself returns an error. That accounting is meaningful even though no surface was accepted.

On this path v1 commits the staged organ before mapping the root error to B1's redacted `ScipLlmError::BackendFailure`. The strict root executor performs no successful-generation accounting before returning that error, so the committed organ-level delta is the error counter only.

Missing backend follows existing root semantics and does not mutate counters.

## Locked request policy

The B1 request keeps its actual `GenerationParams` private by design. v1 must therefore reconstruct the same root `LLMQuery` policy:

| SCIP fallback mode | Root query type | Temperature | Max tokens |
|---|---|---:|---:|
| FaithfulTranslation | Translation | 0.2 | 512 |
| GroundedReasoning | Analysis | 0.3 | 768 |

The root strict executor currently uses query content, system prompt, temperature, max tokens, and `consciousness_context = None` as backend inputs. Query type is root-side metadata and is not passed into `LLMBackend::generate`.

Duplicating these values is intentionally guarded by an interoperability test: the same compiled `ScipLlmRequest` is executed through both B1 direct execution and v1 transactional execution against recording backends, and the observed backend prompt plus every relevant `GenerationParams` field must match for both modes. A B1 policy change therefore fails the v1 test instead of silently changing audit semantics.

## Source identity

After acceptance, v1 constructs the same `ScipLlmOutput` identity fields carried by B1:

- adapter profile;
- request digest;
- exact surface digest;
- source SCIP message ID;
- source semantic hash;
- confidence;
- canonical evidence IDs;
- provenance;
- backend name;
- fallback mode;
- generation latency.

The generated text remains unverified surface realization. The grounded concept graph remains canonical.

## Why clone-and-commit is not the final runtime design

`LLMOrgan` currently contains bounded conversation history and an embedding cache. Cloning that state per realization is acceptable as a correctness reference and integration staging mechanism, but it is unnecessary work for a production hot path.

The preferred future root primitive is protocol-agnostic pre-accounting acceptance, conceptually:

```text
backend.generate
      |
      v
caller-specified output acceptance policy
      |
      +-- reject --> explicit error; no success accounting
      |
      v
finish_backend_generation
```

Once such a root primitive is independently validated, this bridge can become a thin call-through or be retired without changing SCIP's externally tested transactional semantics.

## Required validation

v1 tests must prove:

1. B1 direct execution and transactional execution send identical backend content and generation policy in both fallback modes;
2. accepted output commits normal root accounting and preserves SCIP source identity;
3. blank output is rejected with no persistent successful-generation accounting/history/cache state;
4. oversized output is rejected with no persistent successful-generation accounting/history/cache state;
5. backend failure is called exactly once, never simulates, commits the root error counter, and exposes only the redacted SCIP error;
6. missing backend is state-preserving;
7. the bridge compiles against the exact B1+B2 stacked APIs.

## Non-claims

This bridge does not claim:

- zero-copy or production-optimal performance;
- rollback of backend-internal or remote provider state;
- semantic faithfulness of generated text;
- cryptographic model attestation;
- confidentiality from content hashes;
- native HDC consumption by text-oriented providers.

It is a transactional correctness bridge for the current Phase B integration stack.