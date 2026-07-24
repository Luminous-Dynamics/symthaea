# Campaign XXIII Verification Record

**Campaign:** Human Rescue Ethics Under Uncertain Team State
**Incremental patch range:** 375–401
**Baseline Git tree:** `851732b459ad215d4f7abd5292cf9f5017ff5fd8`
**Environment limitation:** no Cargo, rustc, Rustfmt, Clippy, or Nix executable was available.

## Claims supported by this record

Campaign XXIII was checked through Rust syntax parsing, static consistency analysis, requirement and traceability inspection, production-marker scanning, Git patch replay, exact tree comparison, and artifact hashing.

This record does **not** claim compilation or Rust test execution. The complete Rust 1.94 workspace remains the authoritative build and test gate.

## Static source checks

- Rust source files: 179.
- Rust source lines: 60,392.
- Test or ignored-test annotations: 470.
- Campaign-changed Rust files: 18.
- `tree-sitter-rust` parse failures: 0.
- `git diff --check`: passed before packaging.
- Production forbidden-marker findings in changed source: 0.

The production scan checked code before each file's `#[cfg(test)]` module for:

- `unsafe {`
- `panic!(`
- `todo!(`
- `unimplemented!(`
- `.unwrap()`
- `.expect(`

## Requirement and traceability checks

- Canonical requirement identifiers: 82.
- Unique canonical requirement identifiers: 82.
- Canonical requirement codes: 82.
- Unique canonical requirement codes: 82.
- Campaign XXIII requirements: 5.
- Campaign XXIII traceability links: 5.
- Public Campaign XXIII modules: 7.

Campaign XXIII requirement codes:

- `SUB-HRE-001`
- `SUB-HRE-002`
- `SUB-HRE-003`
- `SUB-HRE-004`
- `SUB-HRE-005`

Exported modules checked:

- `rescue_consent`
- `rescue_emergency_authority`
- `rescue_subject_claim`
- `rescue_triage`
- `rescue_ethics`
- `rescue_ethics_validation`
- `rescue_ethics_bundle`

## Deterministic contract inventory

The source defines eight Campaign XXIII release contracts:

1. replayed consent is rejected;
2. withdrawal stops active rescue motion;
3. emergency intervention requires independent roles;
4. conflicting identity claims require reconciliation;
5. refusal dominates urgency;
6. triage excludes protected attributes structurally;
7. physical recovery actuators survive an ethics hold;
8. checkpoint restoration preserves consent authority.

These contracts are linked into the top-level certification validator. They were structurally inspected but not executed in this environment.

## Cross-layer hardening checks

The final campaign explicitly corrects or prevents:

1. unbounded emergency-approval vectors;
2. consent records without an explicit external-authentication assertion;
3. expired subject claims entering through an old trust timestamp;
4. inconsistent triage counts or ineligible selected candidates;
5. invalid rescue-only authority without a selected authorized subject;
6. mixed mutable and immutable field borrowing in triage construction;
7. withdrawal being lost across distributed-recovery checkpoint restoration;
8. rescue ethics recreating authority removed by Byzantine containment or physical safety.

## Git reproducibility procedure

### Incremental path

1. Reconstruct the canonical Campaign XXII repository through cumulative patch 374.
2. Confirm baseline tree `851732b459ad215d4f7abd5292cf9f5017ff5fd8`.
3. Apply patches 375–401 using `git am`.
4. Compare the resulting tracked tree with the prepared Campaign XXIII source.

### Complete-history path

1. Extract the original uploaded `symthaea-subterranean.tar.gz` snapshot.
2. Create the canonical initial import commit.
3. Apply patches 0001–0401 using `git am`.
4. Compare tracked paths, modes, and blob identities with the incremental reconstruction.

The package-level `VERIFICATION.md` records the final tree identity and archive hashes after both replays complete.

## Unverified production gates

The following remain mandatory:

```text
cargo fmt --check -p symthaea-subterranean
cargo clippy -p symthaea-subterranean --all-targets -- -D warnings
cargo test -p symthaea-subterranean
```

Additional qualification remains necessary for:

- cryptographic consent and reviewer provenance;
- accessible consent, refusal, and withdrawal interfaces;
- protected clocks and monotonic counters;
- medical, legal, disability-access, and human-factors review;
- HIL replay, expiry, contradictory-claim, and withdrawal-during-motion campaigns;
- physical extraction and transport qualification;
- independent safety review and jurisdiction-specific authorization.
