# Checkpoint-durability integration status (2026-07-30)

Follow-up to `TRACK_B_RECOVERY_PLAN_2026-07-30.md`'s Series 20 implementation. Records exactly
which of the `checkpoint_*.rs` files are wired into `lib.rs` and compile/test clean under
the `checkpoint-durability` feature, and why the rest aren't yet. Correction: earlier docs in
this arc (Track B recovery plan, checkpoint recovery scope, verification ledger) said "13
`checkpoint_*.rs` files" — the real count is **15** (`ls src/checkpoint_*.rs | wc -l`); the
13 figure appears to have undercounted from the start and was never re-verified. Not correcting
those historical docs retroactively, but the wired/unwired fractions below use the real total.

## Update (same day): power-loss cluster CLOSED, 8/15 wired

The two gaps flagged below under "found to need MORE than originally scoped" are now resolved.
Per explicit instruction, semantics were frozen in writing FIRST
(`POWER_LOSS_CLUSTER_SEMANTICS_FREEZE_2026-07-30.md`, including an honest in-place correction
once 4 more missing fields turned up beyond the 2 first spotted), adversarial contract tests
were written second, and the wiring/build-verification came last.

- `CheckpointPowerLossCampaignPlan` gained all 6 real missing fields (not just the 2 originally
  spotted — `storage_profile_authority_key_id`, `power_loss_evidence_authority_key_id`,
  `test_harness_digest`, `power_controller_binding`, `power_controller_calibration_digest`,
  `operator_protocol_digest`), all validated non-zero.
- `validate_partial_against` implemented as `validate_against` minus the one trial-completeness
  check, exactly as this doc's "Next steps" #2 predicted.
- `journal_concurrency_tests` resolved as a minimal empty-by-construction scaffold
  (`CheckpointPowerLossJournalConcurrencyTest`) — only 1 real usage site, not enough evidence to
  invent real semantics for; documented as an explicit scope decision, not silently stubbed.
- 8 new adversarial tests in `checkpoint_power_loss_operations.rs` (partial-accepted /
  complete-rejected, every OTHER malformation still rejected by the lenient method, multi-lab
  merge success + rejects empty/inconsistent/overlapping input) + 1 new end-to-end integration
  test in `checkpoint_power_loss_federation.rs` proving `validate_partial_against` is actually
  exercised by the federation layer, with a negative control.
- Found+fixed 3 pre-existing test bugs (`issue_lease` windows exceeding
  `maximum_lease_seconds`) that had never been caught because the file was never wired into
  `lib.rs` before this pass.

Now **8/15 wired**, 154/154 lib tests pass with `--features checkpoint-durability`, clippy and
rustfmt clean, default features unaffected (106/106). See commit "wire power-loss
operations/federation cluster (Track B)".

## Wired in, compiling, tested (8/15)

`checkpoint_audit_archive.rs`, `checkpoint_platform.rs`, `checkpoint_power_loss_federation.rs`,
`checkpoint_power_loss_operations.rs`, `checkpoint_replay.rs`,
`checkpoint_series20_public_verifiability.rs`, `checkpoint_storage_evidence.rs`,
`checkpoint_trusted_time.rs` — 154/154 lib tests pass with `--features checkpoint-durability`,
clippy clean under `-D warnings`, default features unaffected (106/106).

Real bugs found and fixed while getting these files to compile:

- **`checkpoint_platform.rs` (new)**: `effective_uid()`/`lock_exclusive()`, the two missing
  utility functions Track B's plan already flagged. This crate's one deliberate
  `#![allow(unsafe_code)]` exception (raw `flock(2)`/`geteuid(2)`, no safe wrapper exists in
  this crate's dependency set) — every `unsafe` block has a `// SAFETY:` comment, and there's a
  real round-trip test proving the lock guard actually releases on drop (not just that
  acquiring a lock once works).
- **`checkpoint_audit_archive.rs`**: wrote `CheckpointAuditError`/`CheckpointAuditExportDurability`/
  `CheckpointKeyAuditExportReceipt` locally (confirmed via `git log --all -S` these never
  existed anywhere in history — Group C per the earlier fork's classification, i.e. NOT a
  Series 20 dependency, just a local gap in this one file). Fully derived from real call-site
  usage (exact variant names, exact struct fields), not guessed. Also fixed a real
  `From<std::io::Error>` gap (`?` couldn't convert) and 3 pre-existing clippy lints (unneeded
  `mut`, an unused-on-Linux parameter, a redundant `as u64` cast).
- **`checkpoint_power_loss_operations.rs`**: found `checkpoint_power_loss_federation.rs`
  already assumed `CheckpointPowerLossOperationsEvidence` carries a
  `sealed_result_evidence_digest: [u8; 32]` field (used at 3 call sites) that the struct didn't
  actually have — added it, plus a cross-check that every proof's own receipt digest matches.
  Also wrote `merge_checkpoint_power_loss_operations_evidence` (the Group C free function
  `checkpoint_power_loss_federation.rs` needs to combine several labs' partial evidence into
  one complete evidence object) — deliberately does NOT call the existing `validate_against` on
  each individual (partial) input, since that method requires full trial coverage and a
  single lab's evidence only covers its own allocation; instead it checks the inputs agree on
  shared metadata, concatenates proofs, and validates the FINAL merged result once, reusing
  the existing completeness/correctness/dedup logic rather than reimplementing it.
  **This file is NOT wired into `lib.rs` yet** — see below, more real gaps were found past this.
- **`checkpoint_storage_evidence.rs`**: found `CheckpointPowerLossTrialResult` was missing a
  `power_event_evidence_digest: [u8; 32]` field that its OWN `#[cfg(test)]` module already
  expected (a genuine pre-existing gap — the test file was apparently written slightly ahead of
  a struct edit that never landed, not something introduced by this pass). Added the field plus
  a non-zero validation check, matching `recovered_state_digest`'s existing pattern. This is
  what let `checkpoint_storage_evidence.rs`'s own tests compile at all under `cargo test`
  (they'd never actually been run before — `cargo check` alone doesn't compile `#[cfg(test)]`
  modules, so this gap was invisible until `cargo test` was actually tried).

## Deliberately NOT wired in (7/15)

**Blocked on a separate, larger Merkle transparency-log primitive** (not part of Series 20,
not built this pass — a real RFC-6962-style log with consistency proofs, more design work than
a quick addition): `checkpoint_gossip_archive.rs`, `checkpoint_gossip_transport.rs`,
`checkpoint_transparency_gossip.rs`.

**Blocked on the `fips204` post-quantum ML-DSA crate** — a new dependency, not added without
checking in first (same caution class as the ed25519-dalek decision, but for a
less-established post-quantum library): `checkpoint_hardware_signing.rs`,
`checkpoint_hybrid_public_verifiability.rs`.

**Blocked on BOTH of the above** (references types from files in both blocked groups):
`checkpoint_series21_public_verifiability.rs`, `checkpoint_series22_public_verifiability.rs`.

Per explicit instruction, this pass did not add `fips204`, design the transparency log, or
touch the gesture layer. All 7 remaining unwired files are blocked on one of those two
prerequisites; see `SYMTHAEA_VOCAL_TRACT_CHECKPOINT_ARCHITECTURE_RECOMMENDATION_2026-07-30.md`
for whether pursuing them is worthwhile at all.

## Next steps

The power-loss cluster (this doc's prior blocker) is closed — see "Update" section above.
Remaining unwired files all need either the transparency-log primitive or `fips204`; per
explicit instruction, Track B stops here. See the architectural recommendation doc for whether
either is worth pursuing, or whether this whole subsystem should be extracted/retired instead.
