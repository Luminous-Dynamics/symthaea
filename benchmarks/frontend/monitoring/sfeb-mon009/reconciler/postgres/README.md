# MON-009 PostgreSQL test sink

This directory contains an executable PostgreSQL reference sink and an opt-in integration harness. It is designed for an **isolated, disposable test database** only. `schema.sql` drops and recreates the `sfeb` schema and creates two `NOLOGIN` roles. Run it only after explicitly setting `SFEB_ALLOW_DESTRUCTIVE_RESET=yes`.

## What it tests

- One immutable logical outcome per `(tenant_id, operation_id)`.
- A transaction-scoped advisory lock serializes concurrent attempts that reuse one operation ID, including conflicting attempts targeting another resource.
- Same-ID/same-binding delivery returns the original receipt payload as `replayed`, without a second state mutation.
- Same-ID/changed request/resource/authority/fence/revision returns `identity_conflict`.
- A first accepted or rejected operation advances one per-resource event sequence in the same transaction as the ledger entry.
- Stale authority revision/freshness, stale fence and expected-revision conflicts are recorded as rejected operation outcomes.
- A separate append-only attempt ledger records committed, rejected, replayed, conflicting and invalid invocations that reach the procedure. This is distinct from the one-effect-per-operation ledger.
- The writer role can execute the security-definer mutation procedure but has no direct table write privilege. The observer role can read a restricted view but cannot mutate.

## Important boundaries

- The destination receipt is **unsigned JSON evidence derived from the committed ledger**, not a signature, COSE Receipt or SCITT registration. A separately governed signer/adapter must authenticate it after commit.
- `request_canonical` is hashed exactly as supplied. The adapter must validate RFC 8785/JCS canonical bytes; PostgreSQL checks semantic JSON equivalence but is not a JCS canonicalizer.
- The result digest profile is PostgreSQL `jsonb::text` serialization (`postgres-jsonb-text-v1`), not a cross-language universal JSON digest. A production profile should pin an interoperable state serialization and verifier.
- Separate PostgreSQL roles do not establish independent administration, storage, host, network or operator domains. The first sink demonstrates privilege separation and exact readback, not organizational independence.
- PostgreSQL superusers, schema owners, host/storage administrators, backup/restore operators and cloud control-plane administrators remain privileged bypasses outside this test boundary.
- The attempt ledger is transactional. A transaction deliberately rolled back does not leave a durable attempt record. External dispatch/receipt/watermark systems are needed for a stronger completeness claim.
- This is a reversible digital-state sink. It does not prove that the resulting state is physically true, safe or wise.

## Run against PostgreSQL

Use a disposable database and a connection role with privileges to create an extension and roles, and to `SET ROLE` to `sfeb_writer` and `sfeb_observer`. Then run:

```sh
SFEB_PGURI='postgresql://...' \
SFEB_ALLOW_DESTRUCTIVE_RESET=yes \
  ./run-integration.sh
```

The integration script runs idempotency, changed-binding conflict, stale-fence, stale-authority, observer privilege and concurrent same-ID delivery checks. It requires `psql`. The harness has **not been run** in the current environment because no PostgreSQL server/client, Docker or Podman is available.
