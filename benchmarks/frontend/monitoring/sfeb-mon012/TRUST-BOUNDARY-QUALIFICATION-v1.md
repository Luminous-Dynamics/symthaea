# MON-012 — trust-boundary qualification v1

**Status:** security design and semantic preflight. PostgreSQL implementation/integration is not yet run.
**Date:** 2026-11-26.

## 1. Problem

MON-011 checks whether normalized source, decision, approval, intent, authority, sink and readback records refer to the same causal operation. The next failure class is more basic: an attacker may be able to choose the principal, tenant, resource, operation namespace or observer scope that creates and reads those records.

A perfect causal join over records emitted under an over-broad authorization boundary is still an over-broad authorization boundary. Two separately named services may also share the administrator or key that controls both.

## 2. Findings in MON-009 reference SQL

| Surface | Current construction | Qualification gap |
|---|---|---|
| Mutation identity | apply_action accepts p_tenant_id, p_resource_id and p_operation_id; records session_user | No mapping binds session_user to an allowed tenant/resource/action before mutation |
| Readback | observer_readback selects resource_state through a security-barrier view | No principal or tenant predicate; observer role can read across tenants |
| Idempotency | Primary key (tenant_id, operation_id); advisory lock includes tenant | Tenant-scoped namespace conflicts with MON-011's globally scoped W1 target |
| Independence | Writer and observer roles are separate | Role names do not prove independent identity roots, key custody, operators, storage or cloud control |
| Capture | Function records calls that reach apply_action | It cannot establish all mutations entered the function; direct/privileged paths need independent observation |

These are structural gaps in a disposable test sink, not claims that PostgreSQL itself is insecure or proof of a production exploit. The DDL explicitly excludes privileged administrators.

## 3. Mutation authorization contract

Before mutation, the destination MUST bind authenticated session principal, tenant, resource, mutation path/action, destination, exact request digest and encoding profile, policy digest, authority-set digest/revision, grant issuer and trust domain, validity interval, revocation status/freshness, fencing token, expected prior revision and globally scoped operation identity.

The principal MUST NOT come from a caller-supplied tenant_id, principal_id, HTTP header, trace ID or mutable session setting. PostgreSQL session_user is useful where the deployment has distinct authenticated logins. A pooled shared login requires a separately authenticated, non-spoofable identity handoff; SET LOCAL app.principal_id alone is not such a handoff.

Missing or unverifiable authority is indeterminate at the evidence layer and fail-closed at the mutation boundary. It must never be rewritten as an empty grant or success.

A hardened wrapper should:
1. Keep the underlying mutation function inaccessible to ordinary writer callers.
2. Verify session-to-principal and principal-to-scope grants before calling the atomic mutation.
3. Bind authenticated principal into the immutable operation record.
4. Revoke direct table DML from writer and observer roles.
5. Protect grant tables from caller writes.
6. Pin a safe search_path for every SECURITY DEFINER function.
7. Treat owner/superuser bypass as an explicit, externally monitored trust-boundary exception.

## 4. Operation identity namespace

MON-011 W1 currently requires zero duplicate committed mutations per globally scoped operation ID. MON-009 instead keys by (tenant_id, operation_id).

MON-012 adopts:

    idempotency_key = (destination_id, operation_id)

Tenant, principal, resource, action, request digest, policy digest, authority revision/digest, fence and expected prior revision are immutable values under that key.

- Same key and identical immutable binding: return the original outcome; do not mutate again.
- Same key with any changed immutable binding: identity_conflict; do not mutate.
- Same operation ID in different destinations: distinct keys unless a broader namespace is configured.
- Missing identity fields or malformed digests: indeterminate, not a new operation.

A single-destination database can enforce this with UNIQUE(operation_id). A shared store can use UNIQUE(destination_id, operation_id). Lock acquisition and uniqueness must use the same namespace. If deployment semantics intentionally scope IDs to tenants, change W1 wording and benchmark vectors accordingly.

## 5. Observer authorization

A separate database role is not sufficient. Each readback must authorize the observer principal for the exact tenant/resource and observation purpose. It must bind the exact revision, state digest, read time and authorization evidence.

Mandatory controls:
- Tenant A observer cannot read tenant B state.
- Resource A observer cannot read resource B without a grant.
- Mutable GUC cannot impersonate another principal.
- A different revision cannot be relabelled as the committed revision.
- Writer credentials cannot be reused as an independent observer without downgrading the independence claim.

Use a principal-filtered view or narrowly scoped read function. Test the real database role and session_user, not only a Go predicate.

## 6. Independence as a control-dependency graph

For each evidence party, collect evidence-backed roots for identity administration, credential issuance/root, key custody/HSM administration, cloud account/organization, storage and backup administration, operator/incident-response control, network administration and build/deployment pipeline.

A policy selects which dimensions are mandatory. A shared mandatory root means parties are not independent under that policy. An unknown or unverifiable root makes independence uncheckable. A distinct service name, account alias, process, key ID or project label cannot grade the claim up.

Independence is claim-relative. Two services may be independent for key custody but dependent for network administration. Preserve the dependency evidence and policy with the verdict rather than collapsing it into a timeless boolean.

## 7. Bounded capture completeness

No finite log proves every action in the world was captured. Qualify a declared boundary:

    {source, ingress, destination, mutation paths, interval, high-water mark, excluded paths}

Reconcile independent evidence from ingress/dispatch counters, destination transaction or WAL/CDC high-water mark, immutable operation/attempt records, observer sequence, external transparency checkpoints/witnesses and declared bypass paths.

The high-water mark cannot be asserted solely by the same party whose completeness is being audited. Direct writes, trigger disablement, superuser actions, restore from an older snapshot, logical-slot truncation and unobserved API endpoints are negative controls.

Verdicts are complete_within_declared_boundary, incomplete or uncheckable. Never claim unqualified global completeness.

## 8. Mandatory PostgreSQL adversarial tests

After implementing the grants, wrapper and observer scoping, run on a fresh disposable PostgreSQL instance:

1. Tenant A writer targets tenant B: reject, no revision change.
2. Tenant A observer queries tenant B: zero rows disclosed.
3. Same destination operation ID concurrently submitted for two tenants: one identity wins; the other conflicts; no second resource changes.
4. Same operation ID retried with changed principal, resource, request, policy, authority revision, fence or expected revision: conflict, no mutation.
5. Exact retry after lost acknowledgement: original outcome, exactly one committed revision.
6. Spoofed session GUC/request principal: authorization still uses authenticated session identity.
7. Authority revokes immediately before commit: refuse or remain unresolved; never accept stale authority.
8. Different role names but shared admin/key root: W3 reports shared control.
9. Owner/superuser direct mutation outside function: independent capture detects bypass or completeness becomes uncheckable.
10. Rollback, restore, WAL/CDC lag, retention expiry and observer outage: no false-complete result.
11. Receipt for rejected transaction is not a committed effect.
12. Readback at a different revision is not relabelled as the committed revision.

Record database version, extensions, DDL/SQL hashes, isolation level, role graph, query plan, concurrency schedule and observed rows. These tests are not passed until actually executed against PostgreSQL.

## 9. What the Go preflight establishes

principal-scope checks normalized semantics for exact principal/scope binding, freshness/revocation, destination-global operation replay and control-root overlap. It does not prove PostgreSQL SECURITY DEFINER, row-level security, view-owner, role-membership or pooled-session behavior. It does not authenticate grants, revocation feeds, timestamps, signatures, control-root declarations or capture of privileged/bypass mutations. These remain separate integration and deployment qualifications.
