# EKM-067 — Sealed Split-State Hydration Sandbox

## Purpose

EKM-064 proves that persisted support mutations can be reproduced through the real EKM-026 belief-mutation firewall using protected mutation-time evidence censuses. EKM-066 separately restores the complete typed EKM-025 revision audit surface without creating an operational `BeliefRevisionHistory`.

EKM-067 combines those results into one sealed restart sandbox while preserving that separation.

The central invariant is:

**writable support state may exist inside the sandbox only after an independent second firewall replay, while complete revision history remains a non-authoritative immutable audit overlay.**

## Independent second replay

EKM-067 first re-verifies the exact EKM-064 report and EKM-065/EKM-066 source chain.

It then independently rebuilds the support state instead of importing EKM-064's temporary store.

For every persisted mutation it:

1. rebuilds the historical ledger membership from the protected EKM-063 census;
2. rebuilds the source revision decision through the typed EKM-040 policy schema;
3. reconstructs the persisted authorization identity/cycle;
4. executes the real EKM-026 `BeliefMutationFirewall::apply`;
5. compares the resulting mutation receipt field-for-field with persistence;
6. retains only the resulting `EpistemicSupportStore`.

The temporary full ledger, historical ledgers, `BeliefRevisionHistory`, firewall, authorizations and mutation outcomes are discarded before the sandbox is returned.

This intentionally gives the retained store a second implementation path relative to EKM-064. EKM-064 is still required to pass first; EKM-067 then independently reaches the same persisted support state.

## Historical membership

As in EKM-063/064, historical evidence membership comes from the protected EKM-028 census, not `observed_at_cycle`.

The reconstructed historical ledger uses the contiguous global evidence-ID prefix through the largest protected evidence ID and expands claim/provenance prefixes sufficiently to satisfy all records in that evidence prefix. It then requires the target claim's evidence census to equal the protected projection exactly.

Backdated evidence that was appended later remains excluded when it is absent from the protected census.

## Split-state design

`SealedSplitStateHydrationSandboxV1` contains:

- a private real `EpistemicSupportStore`;
- the exact source EKM-034 base snapshot needed for self-verification;
- the immutable EKM-066 revision-audit restoration;
- source/replay/eligibility/audit digests;
- no operational revision history.

Public inspection is limited to:

- support-state count;
- mutation count;
- revision-audit record count;
- immutable per-claim support summaries;
- immutable EKM-066 audit records;
- sandbox digest;
- `verify()`.

There is no public support-store reference, mutable handle, firewall, authorization, operational `BeliefRevisionHistory`, ledger, or decoded full restart snapshot.

## Self-verification

`verify()` recaptures the private support store through `BeliefMutationPersistenceCapsuleV1::capture` and requires exact equivalence with the source EKM-030-shaped support/mutation persistence:

- baseline and current support;
- state revision;
- initialization/update cycles;
- last mutation ID;
- mutation IDs;
- source revision IDs;
- deltas;
- before/after support;
- before/after state revisions;
- authorization IDs/labels/cycles;
- applied cycles;
- consumed authorization count.

The immutable audit overlay is also compared against the source EKM-034 base receipts and EKM-040 typed policy/decision schemas.

## Typed revision comparison

The independent replay does not use the legacy V1 `policy_debug` / `decision_debug` strings as its semantic authority.

It rebuilds policy from the canonical typed policy schema and compares the re-evaluated decision to the typed decision snapshot. Calibration and uncertainty are compared structurally, including exact floating-point bit patterns.

## Authority boundary

The sandbox reports:

- `support_store_constructed = true`
- `support_store_independently_replayed = true`
- `operational_revision_history_constructed = false`
- `immutable_revision_audit_attached = true`
- `writable_state_export_authorized = false`
- `activation_authorized = false`

Trusted state is not mutated.

EKM-067 does **not**:

- return `&mut EpistemicSupportStore`;
- return any support-store reference;
- construct an operational restored `BeliefRevisionHistory`;
- create new revision/mutation authority;
- swap the sandbox into `KnowledgeManager`;
- update restart-anchor/verifier/checkpoint state;
- touch legacy `TemporalFact::confidence`;
- touch causal/world-model/action authority;
- perform filesystem or network I/O.

## Resource bounds

V1 independently replays at most 4096 persisted belief mutations. Existing upstream wire/record bounds remain in force.

## Qualification status

EKM-067 is stacked on EKM-066 / PR #3952. GitHub Actions remains the executable authority. Parent EKM-066 exact-head CI #7377 was still queued when this tranche was authored.

Static/API review is not rustfmt, compilation, Clippy, unit-test, integration-test, runtime or deployment evidence.

## Next boundary

Do not add activation directly on top of this sandbox.

A safer next tranche after executable qualification is an **activation-preflight receipt** that proves:

1. the sandbox still self-verifies;
2. protected currentness evidence has not expired;
3. the currently live epistemic epoch has not advanced since sandbox construction;
4. the candidate's trust context remains the expected successor;
5. activation would not commit trusted checkpoints before the live-state swap succeeds.

The actual swap and trusted-checkpoint commit should remain separate transactions.