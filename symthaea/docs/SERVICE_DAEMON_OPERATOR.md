# Symthaea Service Daemon Operator Guide

This guide covers the hardened control-plane contract for the Symthaea service daemon and the benchmark API.

## Service Daemon

### Security model

- The daemon accepts JSON-line requests over `--socket` or `--tcp`.
- Public TCP binds are refused unless you explicitly opt into one of:
  - `SYMTHAEA_SERVICE_BEARER_TOKEN`
  - `SYMTHAEA_SERVICE_INSECURE_ALLOW_UNAUTH=1`
- Authenticated requests use a top-level envelope field:

```json
{"authorization":"Bearer secret-token","type":"status"}
```

- `ping` and `protocol` are unauthenticated.
- Mutating commands are rejected over `execute_gated`.
- GUI bridge requests like `gui_widget_change` and `parse_nix_config` are recognized but intentionally return `not_implemented`.
- The `protocol` response includes `known_not_implemented_requests` so clients can detect these reserved surfaces programmatically.

### Request examples

Discover the contract:

```json
{"type":"protocol"}
```

Query daemon status:

```json
{"authorization":"Bearer secret-token","type":"status"}
```

Read recent audit events:

```json
{"authorization":"Bearer secret-token","type":"audit_events","limit":25}
```

Validate a command without executing it:

```json
{"authorization":"Bearer secret-token","type":"validate_command","command":"git status"}
```

Attempt read-only execution:

```json
{"authorization":"Bearer secret-token","type":"execute_gated","command":"git status"}
```

### Protocol references

- JSON Schema: [`api/service-protocol-v1.schema.json`](../api/service-protocol-v1.schema.json)
- Runtime discovery endpoint: `{"type":"protocol"}`

### Audit persistence

- In-memory retention keeps the most recent 256 events per process.
- To append events to disk as JSONL, set:

```bash
export SYMTHAEA_SERVICE_AUDIT_LOG_PATH=/var/lib/symthaea/service-audit.jsonl
```

- Without that variable, audit queries only return the in-memory ring buffer.
- Rotate the JSONL file with your normal log rotation tooling; the daemon appends one JSON object per line.
- The NixOS module defaults this to `${dataDir}/logs/service-audit.jsonl`.
- The NixOS module also installs weekly log rotation with compression and eight retained archives for that file.
- Keep the audit log readable only by trusted operators; it may contain request metadata, command validation details, and auth-failure context.
- If you centralize logs, preserve JSONL line boundaries and do not drop rejected or denied events.

## Benchmark API

### Security model

- The API is loopback-only by default.
- Public binds require either:
  - `SYMTHAEA_API_BEARER_TOKEN`
  - `SYMTHAEA_API_INSECURE_ALLOW_UNAUTH=1`
- Authenticated requests use the standard HTTP header:

```http
Authorization: Bearer secret-token
```

### Audit persistence

- The API also keeps the most recent 256 events in memory.
- To persist API audit events as JSONL, set:

```bash
export SYMTHAEA_API_AUDIT_LOG_PATH=/var/lib/symthaea/api-audit.jsonl
```

### Protocol references

- OpenAPI description: [`api/openapi.yaml`](../api/openapi.yaml)
- Audit endpoint: `GET /v1/audit/events?limit=25`

## Operational guidance

- Prefer Unix sockets for local automation.
- If you expose TCP, require bearer auth and keep the bind on loopback unless there is a strong reason not to.
- Treat the daemon and API audit logs as append-only operational records and rotate them outside the process.
- Decide retention and access policy before exposing audit queries to other tooling. The default ring buffer is for short-term inspection; the JSONL path is the durable record.
- Use the daemon `protocol` request and API OpenAPI spec as the source of truth for client generation.
- For deployment changes, keep both Nix checks green:
  - `eval-service-module`
  - `service-module-smoke`
