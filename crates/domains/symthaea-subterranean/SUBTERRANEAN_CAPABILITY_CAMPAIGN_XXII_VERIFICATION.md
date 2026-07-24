# Campaign XXII Verification Record

**Campaign:** Byzantine Team Containment and Trusted-Quorum Continuity
**Incremental patch range:** 351–374
**Baseline Git tree:** `93f8306f7de6114cf5a04a8f303de06de6bb92be`
**Environment limitation:** no Cargo, rustc, Rustfmt, Clippy, or Nix executable was available.

## Claims supported by this record

Campaign XXII was verified through source parsing, static consistency checks, Git patch replay, exact tree comparison, and artifact hashing.

This record does **not** claim that the new code compiled or that its Rust tests executed. The complete Rust 1.94 workspace remains the authoritative build and test gate.

## Static source checks

- 172 Rust source files were parsed with `tree-sitter-rust`.
- Parse failures: 0.
- Rust source lines: 57,757.
- Test or ignored-test annotations: 452.
- `git diff --check`: passed before packaging.
- Campaign-changed Rust files: 19.
- Production forbidden-marker findings in changed files: 0.

The production scan checked code before each file's `#[cfg(test)]` module for:

- `unsafe {`
- `panic!(`
- `todo!(`
- `unimplemented!(`
- `.unwrap()`
- `.expect(`

## Requirement and traceability checks

- Canonical requirement codes: 77.
- Unique canonical requirement codes: 77.
- Campaign XXII requirements: 6.
- Campaign XXII traceability links: 6.
- Public Campaign XXII modules exported by `lib.rs`: 8.

Campaign XXII requirement codes:

- `SUB-BYZ-001`
- `SUB-BYZ-002`
- `SUB-BYZ-003`
- `SUB-BYZ-004`
- `SUB-BYZ-005`
- `SUB-BYZ-006`

Exported modules checked:

- `peer_trust`
- `claim_quorum`
- `rescue_claim_consistency`
- `team_leadership`
- `trusted_quorum`
- `byzantine_containment`
- `byzantine_validation`
- `byzantine_bundle`

## Cross-layer hardening checks

The final campaign includes explicit corrections for:

1. partition logic erasing restrictive team directives;
2. a stricter Byzantine hold being mislabeled as weaker coordination;
3. legacy evidence deserialization using misleading false or zero defaults;
4. split-brain containment surviving the distributed-recovery checkpoint path;
5. simultaneous resource offers exceeding a lender's finite reserve;
6. accepted-but-untransferred offers being counted as locally available.

## Deterministic contract inventory

The source defines eight deterministic contracts:

1. unauthenticated peers have no team authority;
2. contradictory claims require reconciliation;
3. split brain removes motion;
4. lender overcommitment is rejected;
5. contradictory rescue requests are rejected;
6. persistent trusted-quorum loss selects return;
7. recovery actuators survive containment;
8. checkpoint restoration preserves quarantine authority.

These contracts were structurally inspected and linked into the top-level certification validator. They were not executed in this environment.

## Git reproducibility procedure

The packaging phase performs two independent replays:

### Incremental path

1. Start from the canonical Campaign XXI repository at cumulative patch 350.
2. Apply patches 351–374 using `git am`.
3. Compare the resulting Git tree with the prepared Campaign XXII source tree.

### Complete-history path

1. Extract the original uploaded `symthaea-subterranean.tar.gz` snapshot.
2. Create the canonical initial import commit.
3. Confirm that the import tree matches the root tree of the campaign history.
4. Apply patches 0001–0374 using `git am`.
5. Compare tracked paths, modes, and blob identities with the prepared Campaign XXII tree.

The final package-level verification record contains the resulting final tree identity and archive hashes.

## Unverified production gates

The following remain mandatory:

```text
cargo fmt --check -p symthaea-subterranean
cargo clippy -p symthaea-subterranean --all-targets -- -D warnings
cargo test -p symthaea-subterranean
```

Additional qualification remains necessary for:

- external cryptographic peer authentication and revocation;
- Sybil-resistant enrollment;
- protected monotonic counters and secure time;
- HIL replay, duplication, reordering, corruption, and split-brain campaigns;
- physical resource-transfer metering and interlocks;
- prolonged network-partition and denial-of-service tests;
- independent distributed-systems and safety review.
