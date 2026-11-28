# MACE AML Add/Search service runtime

Copyright (C) 2026 Luminous Dynamics  
SPDX-License-Identifier: AGPL-3.0-or-later

## Scope

The `mace_aml_server` binary exposes the participant-side AML API:

- `GET /health` — unauthenticated liveness response (204).
- `POST /add` — authenticated, synchronous Add.
- `POST /search` — authenticated Search returning memory evidence, never final answers.

It uses the durable Add/Search coordinator and the single-writer redb backend. The service is intentionally restricted to one in-flight request because the backend is a single-writer embedded database. Requests that arrive while busy receive 429 with `Retry-After: 1`.

This runtime is a research/development integration, not yet a validated AML submission or production-hardened service. It has not been compiled or tested on this exact commit in the current environment.

## Required environment

| Variable | Required | Meaning |
|---|---|---|
| `MACE_AML_DB_PATH` | yes | Persistent redb database file path. Protect the directory and backups. |
| `MACE_AML_TRACK` | yes | Exactly one of `coding` or `multimodal` for this binary. `textual` is deliberately refused until the official frozen Adapter word counter is integrated. |
| `MACE_AML_AUTH_MODE` | yes | Exactly one of `token`, `bearer`, `x-api-key`, or explicit `public-smoke`. |
| `MACE_AML_API_KEY` | for authenticated modes | Secret value; supplied through the environment/secret manager, never a URL. |
| `MACE_AML_BIND_ADDR` | no | Defaults to `127.0.0.1:3000`. |
| `MACE_AML_TLS_TERMINATED` | non-loopback only | Must be `1` when bound beyond loopback; asserts a trusted HTTPS reverse proxy terminates TLS. |
| `MACE_AML_ENABLE_PUBLIC_SMOKE` | public Smoke only | Must be `1` to permit `MACE_AML_AUTH_MODE=public-smoke`. Never set in a production deployment. |
| `MACE_AML_MAX_BODY_BYTES` | no | Defaults to 50,331,648 bytes (48 MiB); allowed range is 1 KiB through 67,108,864 bytes (64 MiB). Requests above the configured limit receive 413; values outside the allowed configuration range fail startup. No silent truncation. |

The process has no implicit no-auth default. If the authentication mode is omitted, startup fails. HTTP authentication scheme names are accepted case-insensitively; credential bytes remain exact and are compared using fixed-length secret digests. Reusing a request ID with a different payload is a permanent 422 contract error (no Retry-After); an in-flight or unknown commit outcome is a retryable 409 with bounded Retry-After.

## Build and run

From the repository root:

```bash
cargo run --manifest-path research/broca-mace-transfer-bench-v0-1/aml-adapter/Cargo.toml --bin mace_aml_server
```

Example local Coding-track configuration:

```bash
export MACE_AML_DB_PATH=/var/lib/mace-aml/coding.redb
export MACE_AML_TRACK=coding
export MACE_AML_AUTH_MODE=token
export MACE_AML_API_KEY='<load-from-secret-manager>'
export MACE_AML_BIND_ADDR=127.0.0.1:3000
cargo run --manifest-path research/broca-mace-transfer-bench-v0-1/aml-adapter/Cargo.toml --bin mace_aml_server
```

Do not put secrets into command history in an actual deployment. Inject them using a process supervisor or secret manager.

## HTTPS reverse proxy

Bind the Rust process to loopback and expose only the reverse proxy publicly. The proxy must:

- terminate TLS using a valid certificate;
- forward only the required Add, Search and health routes;
- preserve the configured authentication header;
- cap request body size at or below the Rust service limit;
- permit the official Add request time budget (up to 30 minutes);
- avoid logging authorization headers or request bodies;
- rate-limit connections without silently truncating accepted payloads;
- preserve 429 and Retry-After responses.

A non-loopback bind requires the operator to set `MACE_AML_TLS_TERMINATED=1`. This is an operator assertion, not cryptographic proof that TLS is configured correctly.

## API contract

The server uses the fixed track setting at deployment time; no request `track` field is accepted. Request parsing rejects unknown envelope, message and content-part fields.

Add preserves `request_id`, `user_id`, `session_id`, message order, Unix-millisecond `timestamp`, and ordered multimodal text/image parts. The existing storage coordinator must verify the durable receipt, read every visible ID back under the exact user scope, verify source request identity and payload digest, and finalize the idempotency row before returning HTTP 200.

Search uses the same `user_id` scope and returns ranked stored memories. It does not generate a final answer. The platform performs Answer/Score. Formal external Search evaluations use `top_k=100` across Textual, Coding and Multimodal tracks. The configured track determines content validation, not a higher result-count ceiling. The current redb backend uses deterministic BM25 lexical retrieval; it is an internal baseline, not a dense-embedding or SOTA result.

## Known restrictions and non-claims

- **Textual is disabled in the binary.** AML defines Textual chunking at 20 messages or 2,000 Adapter-counted words. The public guide describes a frozen Adapter counter but does not publish its implementation. No whitespace or guessed tokenizer is substituted. The library router accepts an injected `FrozenAdapterWordCounter`; the executable must remain gated until that exact counter is independently obtained and pinned.
- The redb receipt verifier validates local SHA-256 integrity hashes; it is not an external signer or independently witnessed storage attestation.
- redb is configured for one writer. This is not distributed multi-writer operation.
- No encryption-at-rest/key management, retention expiry, verified erasure across backups, external signed audit service or public infrastructure monitoring is included yet.
- The runtime does not itself prove AML Academic component eligibility or full evaluation performance. The official rules specify `text-embedding-v4` for embeddings and `gpt-4o-mini` for LLM-related components in Open-source Methods; rerankers are unrestricted. This BM25-only backend has no LLM Add component. Re-check the official rules before declaring eligibility.
- Evaluation memory and request contents must not be logged. Data-retention and erasure controls remain a separate release gate.

## Validation status

The route parser/authentication/serialization tests, redb tests and paired-runner tests are source-authored. They have not been executed here because no Rust toolchain (`cargo`, `rustc`, `rustfmt`) is available. GitHub workflows that report `skipped` are not passing checks. Do not call this runtime validated until exact-head `cargo fmt --check`, `cargo check`, `cargo test`, route-level negative controls, restart/crash tests and AML Smoke have actually passed.
