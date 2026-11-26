# SFEB MON-013 — PostgreSQL principal and effect enforcement

**Status:** reference schema and executable integration suite authored; PostgreSQL execution is not yet qualified.
**Parent:** [MON-012 trust-boundary qualification](../sfeb-mon012/README.md).

## What changed

MON-012 identified structural gaps in the MON-009 disposable sink: its operation identity was tenant-scoped, authenticated database sessions were not mapped to resource grants, and observer reads were not tenant/resource scoped.

MON-013 supplies a fresh-database PostgreSQL reference profile with:

- Authenticated database session_user mapped to explicit principal/tenant/resource/action grants. Caller-controlled GUCs and request principal fields are not identity.
- Grant validity, freshness and revocation checks with a FOR SHARE row lock that serializes explicit revocation against the mutation transaction.
- A destination-global (destination_id, operation_id) key, global advisory-lock namespace and immutable request/principal/policy/authority/fence/revision binding.
- FORCE ROW LEVEL SECURITY on mutable state and immutable state history, with sfeb_owner explicitly NOBYPASSRLS.
- No direct writer mutation privileges; observer SELECT is scoped by RLS and an exact-revision readback function.
- Append-only operation, attempt and state-history records.
- An explicit two-phase outcome: mutation returns commit_pending; a separate observer can return observed_committed only after the operation ledger and exact state-history revision are visible.
- A deferred transaction-end authority recheck plus post-commit evaluation against PostgreSQL's recorded commit timestamp when configured and retained.

## Run

Use a NEW disposable PostgreSQL database. Do not point this at a production database.

1. Configure track_commit_timestamp = on in PostgreSQL's server configuration and restart the server.
2. Set MON013_TEST_DSN to the disposable database connection string. The script runs DDL, creates test roles, and seeds test grants/data.
3. Run:

    sh run-integration.sh

The suite expects a superuser/DBA DSN because it creates non-login group roles, test login roles, schema objects and test grants, and uses SET SESSION AUTHORIZATION to simulate authenticated sessions.

## Local tool availability

The development environment used for this draft has Go 1.23.2 but no psql, PostgreSQL server, Docker or Podman. The PostgreSQL suite is provided but has not been executed here. Do not interpret SQL presence or static inspection as database integration success.

## Critical semantics

The mutator deliberately returns commit_pending, not committed: a function may return before the caller's transaction commits or can still roll back. A separate authorized observer checks the exact destination operation, request digest, resulting revision and state digest after commit.

observed_committed establishes a digital database state observation, not physical-world effect. The qualification field independently reports whether the action's grant/policy validity at the actual database commit time could be established.

track_commit_timestamp must be enabled before the transaction. PostgreSQL documentation warns that commit timestamp information is routinely removed during vacuum. If it is unavailable or aged out, the effect may still be observed, but authority-at-commit remains uncheckable and qualification is indeterminate. Long-term offline audit therefore requires a durable external witness/seal of the readback record; PostgreSQL's local timestamp is not itself an external trust anchor.

## Trust boundary and limitations

- The scope-grant table is a trusted local authorization cache. Its issuer evidence fields are metadata; this SQL does not verify remote grant signatures, OAuth tokens, COSE, OAR or SCITT.
- session_user is suitable only when each security principal has a distinct authenticated PostgreSQL login. Shared connection-pool logins collapse users into one database principal; do not replace this with a spoofable set_config GUC. A separate authenticated token-exchange adapter is future work.
- PostgreSQL superusers, BYPASSRLS roles, host/cloud administrators, physical backup/restore control and database-owner paths remain privileged control roots. FORCE ROW LEVEL SECURITY does not constrain superusers or BYPASSRLS.
- A separate observer login in the same PostgreSQL deployment is principal separation, not independent administration. W3 independence remains unqualified until external control-root evidence is assessed.
- The deferred authority trigger is a defense-in-depth check. A caller able to force the constraint immediate can run it early; post-commit timestamp comparison is the authoritative check in this profile. Missing commit timestamps never grade up.
- postgres-jsonb-text-v1 is a local state-digest profile, not a universal canonical JSON encoding. Request bytes are hashed exactly; a version-pinned caller adapter must validate the selected canonicalization profile (e.g. JCS) before calling the database.
- The receipt JSON is unsigned. A maintained COSE/SCITT/OAR adapter must sign/register the exact profile bytes after readback qualification.
- The integration suite is sequential. Concurrent same-key delivery, revocation races across two sessions, owner/restore bypasses and WAL/CDC completeness need separate multi-session fault tests.

## Next gates

1. Execute the disposable PostgreSQL suite on supported PostgreSQL versions and record database/version/config/DDL digests.
2. Add two-session concurrency tests for same-key cross-tenant contenders, lost acknowledgements and revoke-vs-commit races.
3. Add direct-writer, BYPASSRLS, owner, superuser, restore and connection-pool negative controls.
4. Add external commit-readback sealing so audit remains reproducible after VACUUM removes commit timestamp metadata.
5. Feed exact committed/readback bytes through the MON-011 joiner; then add maintained OAR / Agent Action Capsule / SCITT verifiers.
