# Broca AML Text/Coding service deployment scaffold

This directory provides a reproducible container build and a Caddy HTTPS ingress example for the optional \`aml-http-server\` service.

**This is a deployment scaffold, not a deployed endpoint.** Building the image, starting Compose, or passing the local health check does not qualify a competition submission. The participant must operate a publicly reachable endpoint and pass the organizer's official Smoke and Full process.

## Prerequisites

- Docker Engine with Compose v2 / BuildKit.
- A Linux host with persistent local storage.
- A public DNS \`A\`/\`AAAA\` record for the host.
- Inbound TCP ports 80 and 443 reachable from the public Internet.
- A unique API key with at least 32 bytes of high-entropy secret material.

SQLite WAL is local-host storage. Keep the database volume on a local filesystem; do not place it on NFS or another network filesystem. This one-container configuration is intentionally single-instance: do not horizontally scale replicas against the same SQLite file or assume process-local index coherence between replicas.

## Configure

From the repository root:

\`\`\`sh
cp deploy/aml/.env.example deploy/aml/.env
\`\`\`

Set the public DNS name in \`AML_DOMAIN\` and generate a unique \`AML_API_KEY\`, for example \`openssl rand -base64 48\`.

The \`.env\` file is excluded from the Docker build context and must never be committed. Keep the key in the host's protected secret store or deployment environment; avoid shell history and CI logs.

## Build and run

\`\`\`sh
docker compose -f deploy/aml/compose.yaml build
docker compose -f deploy/aml/compose.yaml up -d
docker compose -f deploy/aml/compose.yaml ps
\`\`\`

Caddy is pinned to the official 2.11.7-alpine image. It obtains and renews public TLS certificates automatically when the DNS name and inbound ports are correctly configured. The Broca listener is only exposed on the Compose network; port 8080 is not published directly to the host.

Verify health from outside the host:

\`\`\`sh
curl --fail https://YOUR_DOMAIN/health
\`\`\`

Expected body: \`{"status":"ok"}\`.

## Operational boundaries

- The listener is plain HTTP inside the private Compose network. Caddy provides public HTTPS ingress.
- The application health route is unauthenticated. Add and Search require the configured Memory System Key.
- The service applies request-body and concurrency limits, but this scaffold is not a complete DDoS protection or global rate-limiting system.
- The SQLite WAL database is persisted in the \`aml-data\` volume. Back up only when policy permits, encrypt backups, and apply explicit backup/snapshot expiration.
- The server's daily worker performs logical retention purges using a 29-day age cutoff. Logical deletion does not guarantee forensic erasure of WAL/SHM pages, backups, snapshots, host storage or provider snapshots.
- Do not reuse this evaluation database as permanent personal memory.
- Do not log raw Add/Search bodies or API keys.

## Release and competition checklist

1. Build the exact pinned commit using \`cargo build --locked --release --no-default-features -p symthaea-broca --features aml-http-server --bin broca-aml-server\`.
2. Run the repository's Rust unit and HTTP router tests on that same commit.
3. Test health, authentication failure, Add, Search, user isolation, top-k enforcement and read-after-write behavior from an external client.
4. Verify the organizer's current API guide, track-specific request format, model/embedding configuration, request timeout and retention obligations.
5. Submit fixed commit, configuration, public HTTPS endpoints, authentication mode, request/capacity limits and version details through the organizer's access/review process.
6. Obtain an Evaluation Key and pass official Smoke before requesting Full.
7. Keep the deployed version frozen during evaluation; do not report repository CI or local smoke as organizer acceptance.

No public deployment, challenge acceptance, benchmark score, capacity guarantee, availability SLA or SOTA result is claimed by this scaffold.

## External operator smoke check

After the public endpoint is reachable, run this from an external client with `curl`, `jq`, `openssl` and `sed` installed:

```sh
export AML_BASE_URL=https://YOUR_DOMAIN
export AML_API_KEY='YOUR_MEMORY_SYSTEM_KEY'
bash deploy/aml/external-smoke.sh
```

The script verifies public HTTPS health, that unauthenticated Add is rejected, authenticated Add response echoes, read-after-write Search visibility, and isolation from a second `user_id`. It creates one random, non-sensitive smoke message under a unique user ID. That record remains subject to the configured retention worker; the API does not provide a per-user immediate delete route.

This is a local operator check only. It is not the organizer's official Smoke/Full test, an access review, or evidence that the endpoint has been accepted for evaluation.
## Official Agent Memory Leaderboard Cycle 2 release gates

The official Cycle 2 materials deadline is **31 October 2026 at 23:59 UTC+8 (17:59 SAST)**. Evaluation closes **4 November 2026 at 23:59 UTC+8 (17:59 SAST)**. The deadline is close; this scaffold alone is not an eligible submission. The organizer requires stable, publicly reachable participant-hosted Add/Search endpoints, a fixed version and materials, organizer review, an issued Evaluation Key, a passed Smoke run, and then a Full run. Repository-only and Docker-only submissions are not deployed by the platform.

Before applying, verify the selected board and model rules against the current official guide:

- Open-source Methods (Academic) entries must use text-embedding-v4 for embeddings and gpt-4o-mini for LLM-related components. The official Full-evaluation checklist separately says open-source Methods are expected to use gpt-4o-mini during Add.
- This branch's current AML backend is lexical BM25 only: it does not call an embedding model or an LLM during Add/Search. Therefore it has no embedding model to declare; **do not self-certify the no-LLM Add path as compliant without confirming that interpretation with the organizer**. Any LLM-related Add component introduced for the Academic board must use gpt-4o-mini; any embedding component must use text-embedding-v4. The competition guide also describes the Industry board for self-trained/other open-weight model entries; confirm which board is correct before registering.
- The organizer's platform deterministically chunks ordinary Textual Add inputs at 20 messages or 2,000 adapter-counted words, whichever boundary is reached first. The participant server receives the chunks and must preserve their order and identifiers; do not re-chunk already formed API requests.
- Formal Search may request top_k=100; the service supports this limit. Add and Search must use the identical user_id. session_id organizes source provenance but is not a Search filter. Synchronous Add must return HTTP 200 only after durable persistence and immediate read-after-write visibility.
- The organizer accepts Token, Bearer, or X-Api-Key authentication. The unauthenticated Health GET is allowed; unauthenticated Add/Search are not allowed for a formal public endpoint.

Official references:
- Participation and eligibility: https://agentmemories.ai/evaluation
- Add/Search request and response contract: https://agentmemoryleaderboard.ai/api-guide
- Competition board and model requirements: https://agentmemories.ai/competition

## Current qualification status

**Not yet qualified.** This scaffold has not been built or deployed; there is no public endpoint, DNS/TLS check, organizer-issued Evaluation Key, successful Smoke, Full evaluation, or organizer compliance review. Do not submit it as a completed participant deployment. The external smoke script is an operator-side preflight only and does not substitute for the official Smoke.