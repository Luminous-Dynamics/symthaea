# Hardening Note

This note records the current hardening boundary for Symthaea's service and API control planes.

## Stable expectations

- The service daemon uses a JSON-line protocol with top-level `authorization: "Bearer <token>"`.
- The benchmark API uses the standard HTTP `Authorization: Bearer <token>` header.
- Public TCP/API binds require explicit auth or an explicit insecure opt-in.
- Daemon `execute_gated` is read-only only.
- `gui_widget_change` and `parse_nix_config` are reserved daemon request types and intentionally return `not_implemented`.
- Audit events are queryable and can be persisted to JSONL.

## Public vs protected routes

- Public API routes:
  - `/health`
  - `/v1/health`
  - `/v1/leaderboard`
  - `/v1/datasets`
  - `/v1/results/:id` remains GET-accessible at the routing layer, with private-result checks enforced by the handler
- Protected API routes:
  - `/v1/submit`
  - `/v1/compare`
  - `/v1/dimensional-sweep`
  - `/v1/audit/events`

## Deployment defaults

- The NixOS module sets `SYMTHAEA_SERVICE_AUDIT_LOG_PATH=${dataDir}/logs/service-audit.jsonl`.
- The same module configures weekly compressed rotation with eight retained archives.
- The service module is expected to preserve core hardening flags like `NoNewPrivileges`, `PrivateTmp`, and `RestrictAddressFamilies=AF_UNIX`.
- The deployment test surface now includes both module evaluation and a VM-backed service smoke test.

## Required CI gate

- Changes touching the service daemon, API auth/privacy, shared control-plane code, or Nix deployment should keep these jobs green:
  - `Hardened Lib Regressions`
  - `Hardened Daemon Regressions`
  - `Hardened API Regressions`
  - `Hardened Nix Regressions`
- Treat those jobs as the merge gate for control-plane changes even if broader CI is still green.

## Change discipline

- Do not add new control-plane request types without updating:
  - runtime handling
  - protocol schema
  - protocol discovery response
  - focused regression tests
- Do not widen remote execution capability without explicit review.
- Do not add more “recognized but not implemented” request types unless they are intentionally reserved and documented as such.
- Prefer deployment smoke coverage or focused route/protocol tests over broad unstructured integration churn.
