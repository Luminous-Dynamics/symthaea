# Checkpoint Federation and Revocation Campaign

**Program:** Symthaea vocal-tract checkpoint trust

**Campaign schema:** `symthaea.checkpoint-power-loss-federation-plan.v1`

**Operational evidence schema:** `symthaea.checkpoint-operational-evidence.v8`

## Purpose

Series 17 proves that a preregistered laboratory followed an authenticated,
resumable execution history. It does not prove that the laboratories are
independently administered, that every trial was assigned exactly once across
all sites, or that a compromised laboratory or evidence key was checked against
a current revocation decision before promotion.

Series 18 adds a federation authority above the existing storage, result, and
operations authorities. The federation authority freezes the campaign roster
and epoch, assigns every trial a globally ordered allocation, publishes an
explicit signed revocation list, and requires each laboratory to authenticate
its own partial operations evidence with a distinct key.

This campaign does **not** replace the physical sudden-power-loss campaign. A
federation pass is meaningful only when the underlying Series 16 result evidence
and Series 17 execution evidence are themselves valid.

## Independent trust roles

The campaign uses four distinct authority classes:

1. **Storage-profile authority** — authenticates the tested storage topology.
2. **Result-evidence authority** — authenticates recovered trial outcomes.
3. **Operations authority** — issues bounded leases and authenticates journals.
4. **Federation authority** — freezes the member roster, allocation order, epoch,
   and revocation state.

Each member laboratory additionally owns a unique **lab-evidence key**. No two
members may share a lab-evidence key or administration binding. The federation
key must not be reused as a lab-evidence key.

## Preregistered roster

Before any trial begins, publish one sealed federation plan containing:

- nonzero federation identifier;
- campaign and operations-plan digests;
- nonzero federation epoch;
- federation authority key identifier;
- at least two federation members;
- one member for every lab in the operations plan;
- unique lab identifiers, lab-evidence key identifiers, and administration
  bindings;
- bounded validity interval;
- maximum accepted clock offset;
- maximum accepted clock uncertainty; and
- the minimum number of independently administered labs required for promotion.

Changing any roster field, validity interval, epoch, or clock policy creates a
new federation plan and invalidates allocations issued under the old digest.

## Global allocation lane

The federation authority issues exactly one sealed allocation for every
preregistered campaign trial. Allocations must:

- be numbered contiguously from sequence 1;
- contain unique allocation and trial identifiers;
- bind the exact campaign, federation plan, epoch, storage-profile digest, lab,
  lab-evidence key, and attempt number;
- define issued, not-before, and expiry times within the member and federation
  validity windows; and
- contain an attempt number permitted by the operations plan.

Wall-clock timestamps are not used to order trials across laboratories. The
federation allocation sequence is the authoritative global order.

## Revocation lane

The federation authority publishes a signed revocation-list artifact even when
the list is empty. This prevents “no revocation file was supplied” from being
interpreted as “no members were revoked.”

Each entry binds a lab, its exact lab-evidence key, an effective timestamp, a
reason digest, and one of two scopes:

- `FutureAssignments` — allocations issued at or after the effective time are
  rejected; evidence from assignments that began before the effective time may
  remain valid.
- `AllEvidence` — all evidence under the member/key pair is rejected, including
  evidence created before the effective time.

The verifier must authenticate and apply the revocation list before merging any
lab evidence.

## Per-lab evidence lane

Each laboratory signs one partial evidence bundle with its own lab-evidence key.
The bundle contains:

- only execution proofs and concurrency tests belonging to that laboratory;
- the exact signed allocations corresponding to those proofs;
- the digest of the signed revocation list applied by the lab;
- the exact sealed result-evidence artifact digest; and
- one clock attestation covering the allocation-sequence interval in the bundle.

A lab bundle is rejected if a proof lacks an allocation, if the attempt or
storage profile differs, if the lease falls outside the allocation window, if a
revocation applies, or if evidence from another lab is included.

## Clock-reconciliation lane

Every member bundle includes a clock attestation binding:

- the lab and federation identities;
- the lab-evidence key;
- a nonzero sample identifier;
- the first and last allocation sequences covered;
- lab and federation timestamps;
- a bounded uncertainty; and
- the observation time.

Promotion requires all lab clock offsets and uncertainties to remain within the
frozen federation policy. Clock evidence is diagnostic and bounds timestamp
interpretation; it does not replace global allocation sequencing.

## Negative controls

The campaign must demonstrate rejection of at least the following:

1. duplicate lab identifier;
2. duplicate lab-evidence key;
3. duplicate administration binding;
4. missing operations-plan lab in the federation roster;
5. wrong federation authority key;
6. altered federation plan body;
7. missing allocation sequence;
8. duplicate allocation sequence;
9. duplicate allocation or trial identifier;
10. allocation assigned to the wrong lab or key;
11. lease outside the allocation validity window;
12. signed `AllEvidence` revocation;
13. post-effective `FutureAssignments` allocation;
14. lab bundle signed by the wrong lab key;
15. lab bundle containing another lab’s proof;
16. missing allocation for a completed proof;
17. result-evidence artifact digest mismatch;
18. excessive clock offset;
19. excessive clock uncertainty;
20. duplicate lab bundle; and
21. fewer independently administered labs than the preregistered minimum.

## Artifact layout

The strict evaluator accepts:

- canonical campaign artifact;
- canonical operations-plan artifact;
- sealed result-evidence artifact and result key;
- sealed federation plan and federation key;
- sealed revocation-list artifact;
- directory of sealed allocation artifacts;
- directory of sealed lab-evidence artifacts;
- private lab-key directory, with each filename equal to the lowercase
  hexadecimal lab-evidence key identifier plus `.key`; and
- an explicit verification timestamp.

Private key files are exactly 48 bytes (`key-id || key`) and must be regular,
owned by the effective user, private, and opened without following symlinks.

## Promotion gates

Series 18 passes only when all of the following are true:

- federation plan authenticated;
- one valid allocation exists for every completed trial;
- allocations form one contiguous global sequence;
- the required number of independent lab signatures verify;
- every lab bundle is bound to the exact result-evidence artifact;
- the signed revocation list was authenticated and applied;
- every allocation is represented exactly once across the lab bundles;
- all per-lab clock attestations verify;
- maximum clock offset is at or below the frozen requirement;
- maximum clock uncertainty is at or below the frozen requirement; and
- the existing Series 17 operations merger accepts the combined proofs.

Missing federation artifacts remain `not_exercised`; Series 17 evidence alone
cannot satisfy Series 18 promotion gates.

## Non-claims and remaining boundaries

This campaign does not prove legal or corporate independence merely from a
32-byte administration binding. External governance evidence must establish
who controls each binding and key. Symmetric authentication also means the
verifier possesses key material capable of producing signatures; production
federation deployments should migrate these artifact contracts to asymmetric
signatures or hardware-backed verification keys before public third-party
verification is claimed.


## Series 19 rollover boundary

This Series 18 campaign intentionally evaluates one federation epoch. Authority
rotation, member succession, epoch-ledger continuity, and prior-evidence
disposition are preregistered separately in
`CHECKPOINT_FEDERATION_LIFECYCLE_CAMPAIGN.md`. Results from this document alone
cannot satisfy operational evidence schema V9.
