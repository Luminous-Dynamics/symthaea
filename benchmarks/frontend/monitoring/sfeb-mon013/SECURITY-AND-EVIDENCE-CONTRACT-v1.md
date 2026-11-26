# MON-013 PostgreSQL security and evidence contract v1

## Contract

The database reference profile keeps four states separate:

1. Dispatch/requested: no destination result has been established.
2. Commit pending: a function accepted the conditional mutation inside the caller's still-open transaction.
3. Observed committed: a separate observer sees the immutable operation record and exact resulting state-history revision after commit.
4. Qualified committed: the effect is observed and authorization/policy validity at the actual database commit time is also established.

commit_pending MUST NOT be mapped to OAR committed. observed_committed establishes the database effect; qualification can still be not_qualified or indeterminate. Physical effect remains a separate W6 question.

## Database identity

The only supported database principal is authenticated session_user from a distinct login. The implementation MUST NOT use caller-supplied principal IDs, tenant IDs, headers, trace IDs or mutable session settings as authentication.

A pooled shared login is an explicit unsupported mode in this prototype. An eventual pooled mode requires a separate authenticated identity exchange whose assertion signature, audience, expiry, replay nonce and tenant/resource/action scope are verified by a maintained adapter before reaching the mutation function.

## Atomicity and retries

The destination key is (destination_id, operation_id). All immutable bindings are compared: database principal, logical principal, tenant, resource, action, exact request digest, policy digest, authority digest/revision, fence and expected prior revision.

An exact retry returns the original operation outcome. A different binding conflicts. The global advisory lock serializes lookup/mutation for the key; the primary key remains the durable uniqueness authority. A transaction rollback leaves no committed operation row; any client result received before COMMIT remains provisional.

## Authority and commit time

The function locks the matched grant row FOR SHARE, causing an explicit revocation update to serialize against the transaction. A deferred constraint trigger rechecks the grant/policy by transaction end, but clients able to force constraints immediate are not prevented by this reference profile.

The separate readback obtains transaction commit time from pg_xact_commit_timestamp(operation_ledger.xmin). A grant is valid at commit only when the timestamp falls within its not-before/expiry/freshness window and any revocation occurred after that commit; the accepted policy/authority snapshot must also still be valid at that time.

PostgreSQL commit timestamps require track_commit_timestamp=on, apply only to transactions committed after it was enabled, and are routinely removed during vacuum. A missing timestamp yields authority_at_commit=uncheckable, never valid. For durable portable audit, export and seal the readback evidence promptly with an external witness.

## RLS and privileged identities

State and state history use FORCE ROW LEVEL SECURITY. The object owner is a dedicated NOLOGIN, NOSUPERUSER, NOBYPASSRLS role. Ordinary writer and observer roles receive no direct table mutation rights. Observer SELECT is RLS-filtered.

Superusers and BYPASSRLS roles always bypass RLS. Database owners, privileged host operators, snapshot/restore administrators, replication and WAL access are separate trust roots. They cannot be made independent by database role naming. Control-domain evidence is still required for W3.

## Cryptographic/profile boundaries

The database accepts a grant row from a trusted provisioning path; it does not validate the grant's cryptographic issuer proof. It hashes exact request bytes but does not implement JCS. State digests use the versioned local postgres-jsonb-text-v1 profile. Receipt payloads are unsigned. COSE/SCITT registration and OAR mapping remain external maintained-library adapters.

## Failure outcomes

- Missing or malformed request binding: invalid/indeterminate; no mutation.
- Missing, stale, expired, revoked or mismatched grant: unauthorized; no mutation.
- Operation key with changed immutable binding: identity conflict; no mutation.
- Expected revision/fence/policy conflict: refused; no target-state revision change.
- Function return before caller COMMIT: commit pending, not committed.
- No post-commit observer record: not observed / unresolved.
- Exact observer digest/revision mismatch: contradicted.
- Effect observed but commit-time grant proof missing: observed committed, authority uncheckable, qualification indeterminate.
- Effect observed with invalid authority at commit: observed committed, authority invalid, qualification not qualified.
- Effect observed with exact state and valid commit-time grant: observed committed, qualified digital database effect; no physical effect claim.
