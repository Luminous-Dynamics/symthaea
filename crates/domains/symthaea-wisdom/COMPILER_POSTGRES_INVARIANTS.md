# Compiler and PostgreSQL Readiness Invariants

Series XVI deliberately narrows its scope to defects that can be justified
without pretending this sandbox compiled the crate.

## Toolchain identity

- `Cargo.toml` declares Rust 1.94 as the minimum supported Rust version.
- `rust-toolchain.toml` pins the exact 1.94.1 toolchain with the minimal profile,
  rustfmt, and Clippy.
- The crate has no workspace-inherited dependency and no unused Serde dependency.
- Default features expose hardened daemon startup. The legacy direct-startup
  escape hatch remains subordinate when both features are enabled.

`scripts/validate-build-contract.py` checks these facts without Cargo. It does
not replace compilation.

## PostgreSQL materialization bounds

The concrete synchronous driver must reject startup material before returning
it to the generic production backend when either condition holds:

1. more than `MAX_PRODUCTION_ROTATION_BUNDLES` rows are returned; or
2. the ledger frame, startup identity, and rotation payloads cumulatively exceed
   `MAX_PRODUCTION_SNAPSHOT_BYTES`.

The fixed rotation query requests one row beyond the accepted count. Receiving
that sentinel row is an explicit overflow, not silent truncation.

These checks bound material retained by the Rust adapter after PostgreSQL has
returned a row. They do **not** impose a PostgreSQL protocol frame limit and do
not prevent the driver from allocating one oversized `bytea` value while
receiving it. Production PostgreSQL roles still need statement, memory, and
transport limits appropriate to the deployment.

## Server identity continuity

Every transaction re-reads:

- `current_database()`;
- `current_user`; and
- `server_version_num`.

The observed identity must match the identity captured when the executor was
constructed. A role change, database substitution, or version-changing
failover is rejected before application statements execute.

The operator-provided connection identity is a bounded non-secret label. It
accepts only ASCII letters, digits, dot, dash, and underscore so a DSN or
credential-bearing URI cannot be retained accidentally.

## Commit-stage uncertainty

Statement/start failures and commit failures are separate error variants.
Serialization failures and deadlocks are known aborts and may be retried within
the bounded retry policy. A different error returned by `COMMIT` is treated
conservatively as outcome-ambiguous: callers must reconcile durable state before
retrying the operation.

The adapter does not claim distributed exactly-once execution. It gives callers
the information needed to avoid treating a lost commit acknowledgement as a
known rollback.

## Mandatory merge gates

Run from the real Symthaea workspace:

```bash
scripts/check-series-xvi.sh
```

That command executes the static contracts first, then the complete Cargo
feature matrix, tests, rustfmt, and Clippy. The PostgreSQL live campaign from
Series XV remains mandatory against a disposable database and again against the
intended deployment topology.
