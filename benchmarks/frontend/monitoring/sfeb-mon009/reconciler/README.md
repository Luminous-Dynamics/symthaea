# MON-009 normalized evidence coverage reconciler

This standalone Go package reconciles three normalized, pre-authenticated evidence streams: caller intents, destination sink events and separate-principal observations. It checks operation binding, tenant/resource/path scope, monotonic sequence coverage under an externally anchored watermark, exact state-revision/digest readback, declared observer and watermark failure-domain separation, freshness, and continuity from an independently established initial state revision/digest.

## Verify

From this directory:

```sh
./reproduce.sh
```

The v0.2 reference profile accepts only canonical lowercase 64-character SHA-256 hex digests. It validates committed state-transition digests, binds the first committed event to a manifest-pinned initial revision/digest, checks subsequent revision/digest continuity, and rejects fencing-token regression. The suite contains 29 named test functions. Expected local checks: five race-enabled test runs, `go vet`, and statement coverage.

## PostgreSQL test sink

See `postgres/README.md`, `postgres/schema.sql` and `postgres/integration.sql`. The reference sink enforces conditional state changes, stable operation IDs, authority revision/freshness, fencing and prior revision in a transaction. It records one immutable logical operation outcome and a separate attempt row for replay, conflict, rejection and invalid input. The destination emits an unsigned receipt payload; the signer/SCITT adapter remains external.

Run `postgres/run-integration.sh` only against a disposable database with `SFEB_ALLOW_DESTRUCTIVE_RESET=yes`. It drops and recreates the `sfeb` schema, creates roles, and requires `psql` plus privileges to create pgcrypto and roles.

## Trust boundary

This Go package is **not** a cryptographic verifier, COSE/SCITT implementation, PostgreSQL driver, or proof that evidence sources are honest. It accepts normalized facts already validated by signature/receipt/source adapters. In particular, `Watermark.Authenticated` is an adapter result; setting it to `true` does not authenticate anything by itself.

The package cannot prove universal completeness. It checks only the sequence range and mutation paths declared in a boundary manifest. Privileged PostgreSQL superusers, host/storage/cloud administrators, rolled-back transactions and unmediated external paths remain excluded unless independent controls cover them. The watermark's initial state revision/digest must itself be established by a separately trusted read/source; this package only checks continuity from those supplied anchors.

`CanonicalManifestDigest` is a domain-separated, length-prefixed digest for this narrow v1 manifest profile, not a general-purpose canonical JSON or CBOR scheme. The caller must authenticate the manifest bytes/profile with an appropriate external trust mechanism.

## Database test status

The PostgreSQL integration harness is **not yet executed** in the current environment. There is no `psql`, PostgreSQL server, Docker or Podman available. Local Go tests and shell syntax checks do not substitute for database integration, GitHub CI, independent review or production qualification.
