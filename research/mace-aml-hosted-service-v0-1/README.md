# MACE AML Hosted Service v0.1

**Status:** first Rust runtime slice; not production-qualified, not deployed, and not an AML submission.

This crate implements the missing HTTP/storage boundary behind the frozen Add/Search contract from [PR #13470](https://github.com/Luminous-Dynamics/symthaea/pull/13470) and [issue #13359](https://github.com/Luminous-Dynamics/symthaea/issues/13359). It is intentionally separate from the five-package research workspace so it can be built and deployed independently.

## Implemented in this tranche

- Axum `GET /health`, `POST /add`, and `POST /search` endpoints.
- Strict Serde request structs reject unknown fields.
- Required API-key authentication through `Authorization: Bearer`, `Authorization: Token`, or `X-Api-Key`; a route middleware checks the key before the JSON extractor consumes the request body, and the key is configured outside source and logs.
- One configured service principal and tenant per process; Search filters by principal, tenant, and the exact AML `user_id`; `/health` checks the database connection.
- SQLite WAL mode with `synchronous=FULL`, an immediate transaction, immutable request-ID/payload binding, and source rows plus searchable text projection committed together.
- Same request ID plus same normalized request payload returns the original Add response without another write; same request ID plus changed payload returns HTTP 422 and never overwrites the first write.
- Source event timestamps and service ingestion timestamps are stored separately; Search uses event time for its optional RFC 3339 created_at field and falls back to ingestion time when the source omitted it. Search reads the same committed rows and therefore provides read-after-write visibility for acknowledged Add requests.
- Deterministic lexical BM25 ranking with a stable memory-ID tie-break; Search returns source memory records only, not a generated answer.
- Original ordered multimodal content parts and their Base64 data URIs are stored unchanged. JPEG/PNG/WebP signatures and decoded per-image / aggregate payload limits are checked at ingress.
- A 48 MiB HTTP body bound, top-k bound of 100, and non-negative timestamp validation.
- Nine focused Rust unit tests authored for read-after-write, duplicate retry, changed-payload conflict, user isolation, top-k rejection, media-signature validation, file-backed reopen recovery, and concurrent duplicate retries.

## Run locally

Install the pinned project Rust toolchain used by the repository before building. The service refuses to start without all three identity/secret settings.

```bash
export MACE_API_KEY="$(openssl rand -hex 32)"
export MACE_PRINCIPAL_ID="mace-participant"
export MACE_TENANT_ID="mace-evaluation"
export MACE_TRACK="textual" # textual | coding | multimodal
export MACE_DB_PATH="./state/mace-aml.sqlite3"
export MACE_BIND="127.0.0.1:8080"

cargo run --manifest-path research/mace-aml-hosted-service-v0-1/Cargo.toml
```

The default bind is loopback. For any external deployment, terminate HTTPS at a hardened ingress, restrict network exposure, use a dedicated service identity, protect and back up the database, and inject secrets through a real secret manager rather than shell history.

## Wire examples

```bash
curl -fsS http://127.0.0.1:8080/health

curl -fsS -X POST http://127.0.0.1:8080/add \
  -H "Authorization: Bearer $MACE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id":"demo:session-1:chunk-0",
    "messages":[{"role":"user","timestamp":1795492800000,"content":"The blue notebook is in the desk drawer."}],
    "user_id":"demo-user",
    "session_id":"session-1"
  }'

curl -fsS -X POST http://127.0.0.1:8080/search \
  -H "Authorization: Bearer $MACE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"Where is the blue notebook?","user_id":"demo-user","top_k":10}'
```

A caller must reuse the exact logical `request_id` and same request payload for every retry. A caller receiving an ambiguous outcome must reconcile by retrying that same request, never by minting a new ID for a possibly committed write.

## Explicit limits — do not mistake this for qualification

1. **Rights and authority are not yet integrated.** The current local `active` lifecycle state means only that this service accepted and stored the authenticated Add. It does not prove source authority, consent, purpose compatibility, retention eligibility, correction status, retraction, erasure, or parent-lineage validity. No external evidence is exposed to consequential downstream actions by this prototype. The #13446 bitemporal rights projection and #13319 adversarial lifecycle/erasure witness must gate production Search before content is returned.
2. **Textual word-count conformance is not complete.** The service enforces the 20-message Textual chunk bound and HTTP byte limits, but does not yet embed or pin the organizer's exact frozen word-counter implementation. It must not be submitted until that implementation is version-bound and conformance-tested. It does not silently claim whitespace splitting is equivalent.
3. **Retrieval is a baseline.** BM25 is computed over the scoped in-process candidate set; this is adequate for a first deterministic correctness baseline, not a high-scale index. Image bytes are preserved but not decoded into visual features; image-only retrieval and multimodal semantic ranking are not implemented. Search's text projection uses text parts only.
4. **Lifecycle mutation is not exposed here.** There are no public correction, revocation, deletion, or backup-restore routes. These must be governed internal events, externally witnessed, and reconciled before restored data can be served.
5. **Operational hardening remains.** No rate limiter, distributed deployment, key rotation, database encryption, online backup protocol, migration protocol, observability/metrics endpoint, or independent tenant-key mapping is implemented. One process handles one principal/tenant. SQLite's single connection is serialized; capacity and latency must be measured.
6. **No executed validation is claimed in this environment.** Nine tests are source-authored, but `cargo`, `rustc`, and `rustfmt` are unavailable in the current execution environment. Compilation, formatting, tests, stress tests, API smoke, public deployment, and AML evaluation remain pending. A skipped workflow is not a pass.

## Required qualification ladder

1. Run `cargo fmt --check`, `cargo test`, and `cargo clippy -- -D warnings` on a pinned toolchain.
2. Add an organizer-compatible Textual word-counter adapter, pin the upstream protocol revision, and replay the official positive/negative API fixtures.
3. Add crash/restart tests around every transaction boundary, duplicate concurrent Add tests, process-kill tests, and persistent-disk read-after-write recovery tests.
4. Integrate #13446 rights and bitemporal projection before candidate content exposure; add adversarial cross-principal, cross-tenant, wrong-purpose, expired, corrected, superseded, retracted, erased, and restored-backup negative controls.
5. Add monotonic external erasure witness and mandatory restore reconciliation from #13319; demonstrate that old backups, compacted projections, caches, and index rebuilds cannot resurrect erased content.
6. Add a provider-neutral persistent ranking backend with measured BM25, dense, hybrid RRF, timeline, event graph, causal graph, static HDC, and HDC-LTC arms. Bind each run to data, source, model, index, prompt, tool, and evaluator digests.
7. Run exact official track-specific Add/Search Smoke and Full evaluations after the organizer publishes the next cycle contract. Do not retroactively claim Cycle 2.
8. Publish p50/p95/p99 latency, throughput, write amplification, storage bytes, retrieval quality, token/tool/compute/energy cost, failure recovery, and hard-gate outcomes.

## Why this is the right first slice

It makes the central contract executable: **an Add acknowledgement is not a queue receipt; it is a committed source record and a committed searchable projection**. The adapter is provider-neutral and the backend is replaceable. It establishes a reproducible lexical baseline before introducing embeddings, HDC, graph retrieval, or adaptive model-specific policies. This avoids attributing a storage/visibility bug to a retrieval model.

The strategic programme remains broader than AML ranking. MACE should aim to optimize decision value under rights and lifecycle constraints, not only retrieval scores. See [#13363 model-conditioned memory dose](https://github.com/Luminous-Dynamics/symthaea/issues/13363), [#13388 on-policy memory control](https://github.com/Luminous-Dynamics/symthaea/issues/13388), [#13319 adversarial memory lifecycle](https://github.com/Luminous-Dynamics/symthaea/issues/13319), and [#13261 matched benchmark baselines](https://github.com/Luminous-Dynamics/symthaea/issues/13261).
