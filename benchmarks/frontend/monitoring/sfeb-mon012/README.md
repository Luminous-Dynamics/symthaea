# SFEB MON-012 — principal, operation and trust-boundary qualification

**Status:** research/semantic qualification increment; it does not modify or claim to secure the MON-009 PostgreSQL reference sink.
**Parent:** [MON-011 causal evidence join](../sfeb-mon011/README.md).

## Why this increment exists

A source read of the MON-009 reference schema identifies three qualification gaps:

1. The apply_action function accepts tenant and resource identifiers as arguments and records session_user, but does not map that authenticated database principal to a tenant/resource/action grant before mutation.
2. The observer_readback view selects all resource_state rows and contains no principal- or tenant-scoping predicate. Writer/observer role separation does not establish per-observer isolation.
3. operation_ledger is keyed by (tenant_id, operation_id) and its advisory lock includes the tenant. That is tenant-scoped idempotency, while MON-011 W1 currently calls for globally scoped operation identity.

These are structural gaps in a disposable reference profile, not proof of an exploit in a deployed product. The SQL explicitly excludes privileged database administrators from its trust boundary. MON-012 makes these assumptions explicit and adds adversarial semantic tests.

## Research contribution

MON-012 introduces three checks:

- **Principal-to-scope binding:** a verified session identity and separately verified grant must match the exact principal, tenant, resource, action, policy digest, authority digest and mutation time.
- **Destination-global operation identity:** (destination_id, operation_id) identifies one logical operation. Tenant, principal, resource, exact request digest, policy, authority revision, fence and expected prior revision are immutable bindings under that key. Identical retries return the original outcome; any changed binding conflicts.
- **Control-domain independence:** sink and observer independence is evaluated from evidenced identity administration, credential roots, key custody, cloud account, storage administration, operator administration, network administration and deployment pipeline. Different trust-domain labels are not evidence of independence. Missing control evidence yields independence_unverifiable.

The Go package is a standard-library semantic checker over normalized, adapter-verified evidence. It does not authenticate credentials, signatures, database sessions, grant issuers or control-domain declarations. Its Verified fields are trust-boundary inputs, not cryptographic checks.

## Local preflight

Environment: Go 1.23.2, linux/amd64.

- go test -race -count=5 ./... — pass.
- go vet ./... — pass.
- go test -count=1 -cover ./... — 96.6% statement coverage.
- Nine named test functions cover exact scope, identity/tenant/resource/action substitution, policy and authority digest mismatch, stale/expired/revoked grants, as-of revocation, cross-tenant and cross-principal operation-ID replay, changed request/fence/revision bindings, shared administrator roots and missing control evidence.

Run:

    cd principal-scope
    go test -race -count=5 ./...
    go vet ./...
    go test -count=1 -cover ./...

## Qualification boundary

No PostgreSQL server/client was available for this increment. No PostgreSQL isolation, role-escalation, concurrency, transaction, row-security or bypass test is claimed.

See TRUST-BOUNDARY-QUALIFICATION-v1.md for SQL hardening requirements and negative controls; VERSION-PINNED-INTEROP-v1.md for standards versions and adapter boundaries; MATCHED-BENCHMARK-EXTENSION-v1.md for benchmark additions.

## Next gates

1. Add an authenticated principal-to-tenant/resource grant table and a narrowly granted SECURITY DEFINER mutation wrapper. Do not trust caller-supplied tenant IDs or mutable session GUCs as identity.
2. Restrict readback by authenticated database principal and exact resource scope.
3. Enforce destination-global operation identity. A single-destination database can use a global unique constraint; a shared store uses (destination_id, operation_id).
4. Execute two-tenant concurrency, role-escalation, revocation-race, readback-disclosure and privileged-bypass tests on disposable PostgreSQL.
5. Run pinned OAR and SCITT/COSE conformance vectors through maintained verifiers.
6. Run matched W0-W5 product adapters. W6 remains a separate instrumented physical-effect qualification.

No CI pass, PostgreSQL integration, maintained standards-verifier conformance, external benchmark or production qualification is claimed.
