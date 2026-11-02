# VIS-005 isolated contract test workspace

This nested Cargo workspace compiles the exact production module at
`../src/action_conditioned.rs` using only its direct serialization and SHA-256
dependencies. The source is not copied or forked.

Run from this directory:

```sh
cargo test --all-targets
```

The monorepo root workspace currently has wildcard members whose directory
contents do not all contain Cargo manifests. The isolated workspace allows the
contract itself to be compiled and tested without hiding or weakening the
root workspace-integrity checks. The root workspace issue remains independent.

The CI job retains the resolved Cargo.lock as an artifact so it can be reviewed
and pinned in a follow-up commit if the resolver selects the expected versions.
