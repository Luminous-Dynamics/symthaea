# Applying Symthaea Music Theory Patch Series 20

## Expected base

Apply this mail series after Patch Series 19. The expected Git tree before
application is:

```text
3136970a475d4e70adb6f0eaf292c1eb7e103910
```

Confirm the base with:

```text
git rev-parse HEAD^{tree}
```

## Apply the patches

Extract the patch archive, enter its directory, and run:

```text
git am --3way patches/*.patch
```

The patches target the `symthaea-music-theory` crate root used to produce the
archive. When applying inside the larger Symthaea workspace, run the command
from the matching crate directory or use the workspace path expected by your
normal patch workflow.

## Canonical verification

Run in the canonical Nix or Rust development shell:

```text
cargo fmt --all -- --check
cargo test -p symthaea-music-theory
cargo clippy -p symthaea-music-theory --all-targets -- -D warnings
cargo check -p symthaea-music-theory --examples
```

Focused checks:

```text
cargo test -p symthaea-music-theory incident
cargo test -p symthaea-music-theory quarantine
cargo test -p symthaea-music-theory recovery
cargo test -p symthaea-music-theory incident_response
cargo test -p symthaea-music-theory schema
```

## Operator sequence

1. Verify a conflict-bearing Series-19 continuity bundle.
2. Build and verify a publication incident report.
3. Create and externally authenticate necessary quarantine decisions.
4. Confirm that the ordinary outgoing-plus-incoming rotation path is unsafe or
   unavailable before invoking exceptional recovery.
5. Build an explicit catalog lineage rooted in the incident history.
6. Create the recovery plan and export its exact canonical bytes.
7. Collect distinct recovery-authority and incoming-witness signatures.
8. Build and verify the recovery bundle.
9. Build and audit the recovered witness-policy anchor.
10. Build and verify the portable incident-response package.
11. Obtain fresh checkpoint witnesses under the recovered policy.

## Important trust limits

- Incident evidence does not establish intent or guilt.
- Observers reporting a fork are not automatically responsible for it.
- Quarantine is containment, not adjudication.
- Recovery selects one authorized branch; it does not prove universal
  canonicality or absence of hidden forks.
- Logical epochs are not wall-clock timestamps.
- External verifiers define signature acceptance.
- The crate does not manage keys, establish signer independence, or implement
  distributed consensus.

See `EVIDENCE_INCIDENT_RECOVERY_RELEASE.md` for the complete contract.
