# MON-045 — Evidence Dependency Graph and Scoped Invalidation v0.1

MON-045 models recovery evidence as a typed directed acyclic graph (DAG). A recovery claim remains qualified only while every required prerequisite remains valid at the exact digest and evaluation-time boundary it was built against.

## Paradigm

Treat a recovery claim like an incremental build artifact:
- evidence, source truth, capability, scope and policy are typed input nodes;
- claims are derived output nodes;
- required edges pin an exact prerequisite digest;
- optional context edges are visible but do not become implicit requirements;
- revocation, expiry, authentication loss or digest drift propagates only through required dependency edges;
- unrelated claims stay independently evaluable.

This makes invalidation precise rather than global.

## Rules

- Required edge: prerequisite must exist, match the expected SHA-256, be authenticated, be within its validity window, not be revoked, and have a qualified effective verdict.
- Missing required prerequisite: dependent becomes indeterminate.
- Required digest mismatch: dependent is blocked because its pinned dependency identity changed.
- Revoked prerequisite: dependent is blocked.
- Expired or unauthenticated prerequisite: dependent is indeterminate.
- Optional context: does not affect the dependent verdict; missing, stale or non-qualified context remains visible.
- Cycles, duplicate nodes/edges, invalid root kinds and malformed structure block the graph.
- The evaluator processes the DAG in deterministic topological order.
- Root claims, not orphan artifacts, determine the graph verdict.

## Output

The result includes:
- canonical graph digest;
- deterministic topological order;
- per-node own and effective verdicts;
- required and optional inbound edges;
- per-root claim dependency trace digest;
- invalidating prerequisite node IDs;
- impacted and unaffected root claims;
- all non-qualified nodes;
- campaign verdict and production-action boundary.

## Safety boundary

The input's authenticated bit and SHA-256 digests are caller-supplied claims. MON-045 does not bootstrap trust, verify signatures, retrieve artifacts, verify source truth, or enforce the production mutation boundary. campaign_verdict is always indeterminate and production_action_permitted is always false.

MON-041 custody must authenticate issuer identity, signatures, revocation state, nonce use and artifact bytes. Runtime adapters must populate the graph from independently retrieved source and configuration evidence.

## CLI

Run: go run ./cmd/qualify --input INPUT.json

Exit codes: 0 requested root claims internally qualified; 2 a requested root claim or graph is blocked; 3 the graph/claim is indeterminate; 64 usage; 65 malformed input; 66 input unavailable; 74 output failure.

Run bash CHECKS.sh for formatting, vet, unit, race, CLI, schema-if-available and benchmark checks.
