# AML Text/Coding HTTP service deployment guide

This guide describes the optional participant-hosted HTTP service implemented by `memory_challenge_http.rs`. It is a deployment scaffold; it is **not a claim that a public endpoint is deployed or that organizer Smoke has passed**.

## API

- `GET /health`: unauthenticated JSON `{"status":"ok"}`; any 2xx health response is acceptable to the current organizer contract.
- `POST /add`: synchronous; returns HTTP 200 only after the SQLite transaction commits and the backend has confirmed the newly added records are searchable.
- `POST /search`: returns a top-level `{"data":[...]}` evidence array; the backend enforces the supplied `user_id` and rejects `top_k > 100`.

Add and Search accept the current typed AML Text/Coding JSON contract. Add preserves message order and echoes `request_id`, `user_id`, and `session_id`. Search does not accept `session_id`; `user_id` is its only isolation scope.

## Build and run

The server is feature-gated so normal Broca builds do not require Axum:

```sh
cargo run -p symthaea-broca --features aml-http-server --bin broca-aml-server
```

Required environment variable:
- `AML_DB_PATH`: local SQLite database path.
- `AML_API_KEY`: participant Memory System Key, at least 32 bytes of high-entropy secret material.

Optional environment variables:
- `AML_BIND`: default `0.0.0.0:8080`.
- `AML_PAYLOAD_ROOT`: default `/var/lib/symthaea-aml/payloads`.
- `AML_MAX_BODY_BYTES`: default 8 MiB; maximum 64 MiB.
- `AML_MAX_CONCURRENCY`: default 16; maximum 256.

The process opens the SQLite/WAL backend at startup. The SQLite database must be on local storage on the same host as the service; do not put a WAL database on NFS or another network filesystem.

## Public ingress requirements

The Axum listener is plain HTTP. **Do not expose it directly to the public Internet.** Put it behind a public HTTPS reverse proxy/load balancer, configure certificate renewal, close the listener from untrusted networks, and forward only the required paths. Configure the proxy with bounded request-body size, connection/request timeouts compatible with the organizer's 30-minute request ceiling, and suitable rate/concurrency limits. The service itself limits in-flight requests but is not a complete abuse/rate-limiting layer.

The health endpoint is deliberately unauthenticated. Add/Search accept `Authorization: Bearer`, `Authorization: Token`, or `X-Api-Key`. Never place the API key in the URL, command history, logs, screenshots, public CI output, or the public repository.

Do not log request bodies. The evaluation payload contains private test memories; retain only the minimum operational metadata needed to diagnose failures.

## Retention and deletion

The SQLite backend exposes explicit cutoff and full logical purge methods, but the HTTP server does **not** expose a public purge endpoint and does not yet run an automatic retention worker. Before a real AML run, an operator must schedule and verify `purge_stored_before` no later than 30 days after the evaluation finishes. Logical deletion is not forensic erasure: also handle SQLite WAL/SHM sidecars, backups, snapshots, and storage according to the organizer's deletion rules. Do not use this evaluation-only database for permanent user memory.

## Release gate before applying

1. Build the exact pinned commit with `--features aml-http-server`; run unit and HTTP router tests.
2. Deploy behind public HTTPS; verify `GET /health` from outside the hosting network.
3. Verify all accepted authentication schemes and reject missing/wrong keys.
4. Run official Add → Search Smoke with the same `user_id`; verify synchronous HTTP 200, exact echo fields, top-level `data`, top-K limit and no cross-user leakage.
5. Record capacity, concurrency, body limit, timeout, version/commit, endpoint URLs and authentication method in the organizer's application.
6. Pass the official Smoke and Full evaluation. A green local test or a merged PR is not a substitute.

## Official contract notes

The current Agent Memory Challenge guide requires participant-hosted Add/Search endpoints; repository-only and Docker-only submissions are not accepted as deployments. The public endpoint must satisfy sample isolation, return memory evidence rather than final answers, and preserve the fixed evaluation version. The Open-source Methods division requires `gpt-4o-mini` for LLM-related components and `text-embedding-v4` if an embedding component is used; lexical-only BM25 does not claim to use embeddings. Verify current organizer guidance again before final submission.
