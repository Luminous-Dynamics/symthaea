# EKM-073 — Restored Revision Authority Contract

## Purpose

EKM-066 restores complete typed revision-audit semantics, while EKM-064 can historically re-execute the decisions that actually authorized persisted mutations. The remaining gap is what authority historical revision receipts should have after a restart.

EKM-073 defines the conservative answer:

**historical receipts are archival only; every new post-restart mutation requires a freshly evaluated receipt bound to the new restart epoch.**

## Why this is necessary

`BeliefRevisionGate` can depend on more than the proposal basis. In particular:

- unresolved-contradiction blocking scans all claim evidence;
- current-uncertainty freshness scans all claim evidence to find the latest observation cycle.

EKM-028 retained the complete mutation-time claim/evidence census for decisions that actually became mutations. It did not retain an equivalent full census for every rejected or eligible-but-unapplied revision receipt.

Therefore reconstructing every historical receipt as fresh mutation authority would overclaim what the persisted evidence proves.

## Historical receipt classes

V1 treats all historical classes as `ArchivalOnly`:

- applied receipts — their effects are already represented and replay-verified in support state;
- rejected receipts — they never had mutation authority;
- eligible-but-unapplied receipts — they may be stale after restart and cannot be replayed as new authorization.

## Required invariants

The canonical contract requires:

- preserve historical receipt identity;
- preserve next-receipt-ID continuity;
- preserve historical applied-mutation linkage;
- forbid replay of historical applied authorization;
- forbid replay of historical eligible-but-unapplied authorization;
- require fresh post-restart evaluation for every new mutation;
- bind fresh receipts to a restart epoch.

## Current blocker

The current `BeliefRevisionReceipt` schema does not yet carry a restart-epoch identity. EKM-073 therefore records:

- `restart_epoch_binding_required = true`
- `current_receipt_schema_has_restart_epoch_binding = false`
- `operational_history_restore_implemented = false`

A later schema tranche should add an explicit authority epoch without retroactively turning historical receipts into live capability tokens.

## Authority boundary

The contract records:

- `historical_receipts_mutation_authority = false`
- `applied_authorization_replay_allowed = false`
- `eligible_unapplied_authorization_replay_allowed = false`
- `mutation_authority_exported = false`
- `activation_authorized = false`

No receipt constructor, arbitrary persisted-ID insertion path, mutation authorization, operational history restore, or activation path is added here.

## Relationship to EKM-072

EKM-072 requires an operational revision-history component in the eventual atomic activation bundle. EKM-073 clarifies what that component must mean:

- historical lineage is preserved for audit and ID continuity;
- historical receipts do not regain mutation authority;
- the post-restart operational continuation begins in a new authority epoch.

This avoids the unsafe shortcut of making persisted historical eligibility equivalent to fresh post-restart authorization.

## Qualification status

This tranche is stacked on EKM-072. GitHub Actions remains the executable authority. Static/API review and authored tests are not rustfmt, compile, Clippy, unit-test, integration-test, runtime, or deployment evidence.
