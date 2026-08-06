# Power-loss cluster: frozen semantics (2026-07-30)

Written BEFORE any implementation, per explicit instruction. Covers the 3 remaining gaps
blocking `checkpoint_power_loss_operations.rs`/`checkpoint_power_loss_federation.rs` from
compiling, identified in `CHECKPOINT_DURABILITY_INTEGRATION_STATUS_2026-07-30.md`. Each gap
below is resolved with either (a) a fully-determined answer derived from real, existing call
sites (no design freedom), or (b) an explicitly-flagged, deliberately minimal resolution where
the evidence is too thin to justify inventing real business logic.

## 1. `CheckpointPowerLossCampaignPlan`'s two missing fields — fully determined

`checkpoint_power_loss_operations.rs:199-200` (inside `CheckpointPowerLossOperationsPlan::
validate_against`) already reads:

```rust
if lab.test_harness_binding != campaign.test_harness_digest
    || lab.power_controller_binding != campaign.power_controller_binding
{
    return Err(CheckpointPowerLossOperationsError::CampaignBindingMismatch);
}
```

`lab` here is a `CheckpointPowerLossLabManifest`, whose real, already-defined fields include
`test_harness_binding: [u8; 32]` and `power_controller_binding: [u8; 32]` (both required
non-zero by `CheckpointPowerLossLabManifest::validate`) — these are opaque commitment digests
scoping a lab to a specific test harness / power controller, the same pattern as that struct's
sibling `organization_binding`/`operator_group_binding`/`facility_binding` fields.

**Resolution (no ambiguity)**: add to `CheckpointPowerLossCampaignPlan`
(`checkpoint_storage_evidence.rs`):
- `pub test_harness_digest: [u8; 32]`
- `pub power_controller_binding: [u8; 32]`

Both validated non-zero in `CheckpointPowerLossCampaignPlan::validate`, matching the existing
`storage_profiles`/`campaign_id` non-zero-style checks in that method. Field NAMES are fixed by
the existing call site (note the asymmetric `_digest` vs. `_binding` suffix on the campaign
side even though both are `_binding` on the lab side — this is already how the real code
refers to them, not a naming choice made here).

## 2. `validate_partial_against` — fully determined by refactoring `validate_against`

The one real call site (`checkpoint_power_loss_federation.rs:740-742`,
`CheckpointPowerLossFederatedLabEvidence::validate_against`):

```rust
self.operations_evidence
    .validate_partial_against(campaign, operations, result_evidence)
    .map_err(CheckpointPowerLossFederationError::Operations)?;
```

Same three-argument shape as the existing `CheckpointPowerLossOperationsEvidence::
validate_against`, propagated through the same `Operations(CheckpointPowerLossOperationsError)`
error variant `checkpoint_power_loss_federation.rs` already uses elsewhere. This call happens
on a SINGLE lab's evidence — which by federation design only covers that lab's own trial
allocation, not the full campaign — so it structurally cannot pass the existing
`validate_against`'s completeness requirement.

Reading the existing `validate_against` closely, exactly ONE of its checks is a completeness
requirement (every trial in `result_evidence.results` must have exactly one proof):

```rust
if proof_ids.len() != results.len()
    || !results.keys().all(|trial_id| proof_ids.contains(trial_id))
{
    return Err(CheckpointPowerLossOperationsError::InvalidReceipt);
}
```

Every other check in `validate_against` — schema, campaign_digest, operations_plan_digest,
operations_authority_key_id, `sealed_result_evidence_digest` non-zero, the proof-count upper
bound (`self.proofs.len() > campaign.trials.len()`, itself NOT a completeness check — it's just
a sanity cap, already correct for partial evidence too), per-proof digest-consistency,
per-proof `receipt.validate_against(...)`, and duplicate-trial-or-attempt rejection — is
equally valid and necessary for PARTIAL evidence. A lab submitting incomplete-but-otherwise-
malformed evidence should be rejected exactly as readily as a lab submitting complete-but-
malformed evidence.

**Resolution**: refactor so `validate_partial_against` contains everything `validate_against`
currently does EXCEPT the completeness check, and `validate_against` becomes
`self.validate_partial_against(...)` followed by that one completeness check. This is not a new
design — it is extracting an already-existing method into two, with the dividing line drawn
exactly at the one check that structurally cannot apply to partial evidence. No new invariants
are invented; nothing existing is weakened for the full-evidence path.

This is directly consistent with (and validates in hindsight) the
`merge_checkpoint_power_loss_operations_evidence` function written in the previous pass, which
calls `.validate_against(...)` (the completeness-checked variant) exactly once, on the final
MERGED evidence — the merge function was already correctly designed around this split, before
this split was explicit.

## 3. `journal_concurrency_tests` — explicitly NOT invented, minimal inert scaffold only

`checkpoint_power_loss_federation.rs:806-811` is the ONLY reference to this concept anywhere in
the crate (confirmed via `grep -rn "journal_concurrency\|JournalConcurrency\|concurrency_test"
*.rs` — zero other hits, no doc comment, no related type, no test):

```rust
if self
    .operations_evidence
    .journal_concurrency_tests
    .iter()
    .any(|test| test.lab_id != self.lab_id)
{
    return Err(CheckpointPowerLossFederationError::InvalidLabEvidence);
}
```

This tells us only: `CheckpointPowerLossOperationsEvidence.journal_concurrency_tests` is some
`Vec<T>` where `T` has a `lab_id: CheckpointPowerLossLabId` field, and every entry's `lab_id`
must match the federated evidence's own `lab_id`.

**This is not enough evidence to justify inventing real business logic for what a "journal
concurrency test" actually records or verifies** (unlike items 1-2 above, where the intended
shape is fully pinned down by real usage). Per the standing instruction to freeze semantics
honestly rather than invent unsubstantiated design, the resolution here is deliberately
minimal:

- Add `pub struct CheckpointPowerLossJournalConcurrencyTest { pub lab_id:
  CheckpointPowerLossLabId }` — the smallest type that satisfies the one real usage site.
- Add `pub journal_concurrency_tests: Vec<CheckpointPowerLossJournalConcurrencyTest>` to
  `CheckpointPowerLossOperationsEvidence`, defaulting to `Vec::new()` at every real construction
  site (including inside `merge_checkpoint_power_loss_operations_evidence`, where merged
  evidence concatenates each input's list — trivially correct for the empty case, and correct
  in general since concatenation is the natural merge for a list keyed by lab).
- This makes the one real check (`.any(|test| test.lab_id != self.lab_id)`) vacuously pass for
  every currently-real code path (empty vec), while making the type system honest about the
  field existing. No claim is made that this represents real "concurrency testing" capability —
  it is a compile-unblocking scaffold, not a feature. Flagged explicitly as a candidate for
  proper design later, separate from this pass.

## Correction (same day, found while implementing item 1): 4 more fields, not 2

Item 1 above was INCOMPLETE. While adding `test_harness_digest`/`power_controller_binding` and
updating construction sites, both `checkpoint_power_loss_operations.rs`'s and
`checkpoint_power_loss_federation.rs`'s own test fixtures (`campaign()`/`fixture()`) turned out
to already construct `CheckpointPowerLossCampaignPlan` with FOUR more fields neither compile
error list had surfaced yet (rustc had not gotten far enough to report them):

- `storage_profile_authority_key_id: CheckpointStorageProfileAttestationKeyId` (type already
  exists in `checkpoint_storage_evidence.rs`)
- `power_loss_evidence_authority_key_id: CheckpointPowerLossEvidenceKeyId` (type does NOT exist
  anywhere -- referenced only, same as everything else in this cluster; needs to be written,
  matching `CheckpointStorageProfileAttestationKeyId`'s exact template: `pub struct
  CheckpointPowerLossEvidenceKeyId(pub [u8; 16])` + `::new()` rejecting all-zero)
- `power_controller_calibration_digest: [u8; 32]`
- `operator_protocol_digest: [u8; 32]`

Both test fixtures (operations.rs's and federation.rs's, written independently in different
files) agree on the exact same 6-field set in the same order/shape -- strong evidence this was
the real intended design, not two divergent drafts.

**Unlike `test_harness_digest`/`power_controller_binding`, none of these 4 fields are ever
READ anywhere** (confirmed via `grep -n "\.storage_profile_authority_key_id\|\.power_loss_
evidence_authority_key_id\|\.power_controller_calibration_digest\|\.operator_protocol_digest"
*.rs` across all 13 checkpoint files -- zero hits). They're constructor-only, same
under-evidenced shape as `journal_concurrency_tests` -- except unlike that field, a type and
exact field name/order IS pinned down by two independent, consistent test fixtures, so there's
no ambiguity about the field's existence or shape, only about what (if anything) should ever
read it.

**Resolution**: add all 4 fields for real (needed for the fixtures to compile, and consistent
across two independently-written call sites is real signal). Apply this crate's own
established convention -- every `_binding`/`_digest`-suffixed commitment field on this exact
struct is already non-zero-validated -- uniformly to these 4 too, in `CheckpointPowerLossCampaignPlan::
validate`. This is applying an existing, already-established pattern consistently, not
inventing new business logic. No cross-check against any other struct is added for these 4
(unlike `test_harness_digest`/`power_controller_binding`'s real lab-manifest cross-check) since
no evidence anywhere calls for one.

## What this freeze authorizes

Implement exactly the three items above, in this form, with adversarial contract tests
covering: the CampaignPlan/LabManifest cross-check (both fields, both directions of mismatch);
`validate_partial_against` accepting genuinely-partial evidence that `validate_against` would
reject for incompleteness, while still rejecting every other form of malformed evidence
identically to the full method; and the `merge_...` + `validate_against` interaction still
producing a complete, correctly-validated result when partial evidence from multiple labs is
combined. Then wire both files into `lib.rs` and verify the full `checkpoint-durability`
feature build (8/13 files) compiles and tests pass.

Does NOT authorize: `fips204`, the transparency-log primitive, or any Track A / gesture-layer
work — explicitly out of scope per instruction.
