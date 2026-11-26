# MON-013 multi-session concurrency and privileged-bypass protocol v1

**Status:** test protocol; none of these multi-session tests have yet run.

## Required independent sessions

Use at least four authenticated database sessions:

- writer A: tenant A / resource A / mutate grant A
- writer B: tenant B / resource B / mutate grant B
- observer A: tenant A / resource A / readback grant A
- grant administrator: revocation authority

A separate DBA session configures roles, test grants, server settings and audit collection. Record the DBA identity and include it in the control-dependency graph.

## Test schedule A — global operation identity

1. Synchronize writer A and writer B on a barrier.
2. Both submit the same operation UUID to the same destination, but with different tenant/resource/request bytes.
3. Both calls race through the same destination-global advisory-lock key.
4. Commit both caller transactions.
5. Read from the operation ledger and both resources using observer/DBA access.

Expected: exactly one immutable operation-ledger row; at most one resource revision advances; the losing caller receives identity_conflict or a transaction-level uniqueness conflict; no second mutation commits.

Repeat with exactly identical request bytes and immutable bindings. One transaction creates the operation; the other receives the original outcome. No second revision.

## Test schedule B — acknowledgement loss

1. Writer A invokes the function.
2. Database transaction commits, but the client connection is terminated before it receives the commit acknowledgement.
3. New writer A connection retries the same operation key and exact request.
4. Observer A reads the exact operation and resulting revision.

Expected: retry returns the immutable original operation result; observer reads one resulting revision; no duplicate state transition. If the client cannot establish whether COMMIT succeeded, it must read back or retry with the identical key rather than create a new operation ID.

## Test schedule C — revocation serialization

Run both orderings:

- Mutation obtains grant FOR SHARE first; grant administrator attempts revoke and blocks until mutation commit. Readback commit timestamp must be earlier than revoked_at.
- Revocation update commits first; mutation then attempts the grant and is refused.

Repeat with the grant expiring between function return and outer transaction commit. Test both normal deferred constraint mode and a caller that executes SET CONSTRAINTS ALL IMMEDIATE before the mutation and then delays COMMIT. The latter must not be marked qualified if actual commit time is after expiry.

## Test schedule D — commit pending and rollback

1. Writer A begins a transaction.
2. Calls apply_action_authorized and receives commit_pending.
3. Observer A on a separate connection queries the exact operation before writer A commits.
4. Writer A rolls back.
5. Observer A queries again.

Expected: the pre-commit query is not_observed; after rollback it remains not_observed; no target revision or operation row is persisted. A function return alone never creates a committed outcome.

Repeat with COMMIT instead of ROLLBACK. Only after commit may observer return observed_committed.

## Test schedule E — RLS and connection identity

- Tenant A writer directly attempts INSERT/UPDATE/DELETE/TRUNCATE on target and ledger tables.
- Tenant A observer directly queries tenant B state and state history.
- Writer A sets arbitrary application GUCs purporting to be tenant B or principal B.
- Writer A attempts SET ROLE to any owner, grant-admin, superuser or BYPASSRLS role it does not hold.
- A shared connection-pool login is used while two application principals are passed in request payloads.

Expected: direct mutation is denied; cross-tenant reads are filtered/denied; GUC/request identity never changes session_user; role escalation fails; the pooled shared-login configuration is unqualified unless a distinct authenticated token-exchange profile is added.

## Test schedule F — privileged paths and capture completeness

- Run direct target DML as the object owner, database owner, superuser and BYPASSRLS role.
- Disable a test trigger, truncate/restore a table, rewind a backup, interrupt WAL/CDC and cause observer lag.
- Compare transaction ledger and observer state with independently collected WAL/CDC or external audit evidence.

Expected: every privileged bypass is detected by the separately declared monitor, or completeness becomes uncheckable/incomplete. Do not claim that RLS contains superusers or host/database administrators.

## Evidence to retain

For each run record:
- PostgreSQL version, image/package digest, extensions and relevant GUCs;
- schema/function/policy/trigger hashes and role membership graph;
- session identities and control-root owners;
- exact SQL request bytes/digests and operation IDs;
- synchronization/barrier schedule, transaction IDs, commit timestamps and lock waits;
- before/after resource revisions and readback digests;
- expected result authored independently from the implementation;
- raw transcripts, exit codes, and artifact hashes.

No run is qualified by a diagram or a sequential SQL smoke test alone. Publish failed schedules and transaction traces, not only aggregate pass counts.
