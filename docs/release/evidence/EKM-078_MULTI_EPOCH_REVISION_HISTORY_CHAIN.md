# EKM-078 — Multi-Epoch Revision History Chain

## Purpose

EKM-066 restores the complete immutable legacy V1 revision audit. EKM-077 defines one passive V2 receipt segment per restart authority epoch. EKM-078 binds those representations into one globally contiguous audit lineage without constructing an authority-bearing `BeliefRevisionHistory`.

The central invariant is:

**legacy audit identity remains intact, receipt IDs remain globally contiguous across the V1→V2 boundary and every later epoch segment, while epoch issuance authority remains a separate proof obligation.**

## Chain inputs

`MultiEpochRevisionHistoryChainV2::capture` accepts:

- an existing complete EKM-066 immutable revision audit restoration
- zero or more ordered EKM-077 epoch segments
- the chain capture cycle

The legacy audit is accepted only when it remains non-operational and non-authorizing.

## Legacy prefix

The chain binds:

- EKM-066 restoration digest
- legacy receipt count
- derived legacy next global receipt ID

Legacy receipt IDs are independently checked as contiguous from 1 before the V2 handoff cursor is accepted.

No V1 receipt is converted to V2.

## V2 segment continuity

Each segment must:

- independently pass EKM-077 verification
- begin at the exact previous global next receipt ID
- use the next contiguous authority-epoch sequence
- not be captured after the chain capture cycle
- have a capture cycle that does not regress relative to the previous segment

For a complete chain, the first V2 authority epoch is sequence 1. Empty epoch segments remain meaningful and preserve epoch chronology even when no receipt was evaluated in that epoch.

## What EKM-078 proves

A successful chain records:

- `receipt_cursor_continuity_proven = true`
- `epoch_sequence_continuity_proven = true`

This proves structural audit continuity.

## What EKM-078 does not prove

EKM-077 segments carry an epoch digest and sequence but do not yet carry the successful activation-commit receipt that issued each epoch or the previous-epoch digest from that issuance event.

Therefore EKM-078 deliberately records:

- `epoch_issuance_chain_verified = false`
- `operational_history_constructed = false`
- `mutation_authority = false`
- `activation_authorized = false`

Contiguous sequence numbers and hashes are not substitutes for proving actual epoch issuance.

## Canonical chain identity

The chain digest is domain-separated with:

`symthaea-ekm-multi-epoch-revision-history-chain-v2`

It binds:

- chain capture cycle
- EKM-066 legacy restoration digest
- legacy receipt count and next cursor
- ordered EKM-077 segment digests
- final global next receipt ID
- all structural/authority claim flags

`verify_against(...)` reconstructs the chain from the exact legacy audit plus the retained segments and requires exact structural/digest equality.

## Authority boundary

EKM-078 does not:

- issue an authority epoch
- prove activation-commit lineage for epochs
- convert legacy receipts into V2
- construct `BeliefRevisionHistory`
- construct `PreparedBeliefMutation`
- authorize mutation
- expose writable history
- activate restart state

## Next boundary

A later passive tranche should bind each EKM-077 segment to an explicit successful activation-commit/epoch-issuance receipt and prove exact predecessor epoch linkage. Only after that lineage is executable-qualified should operational history continuation be considered.

## Qualification status

This tranche is stacked on EKM-077 / PR #4096. EKM-077 exact-head CI #7550 remained queued when this evidence contract was prepared.

GitHub Actions remains the executable qualification authority. Static/API review and authored checks are not rustfmt, compilation, Clippy, unit-test, integration-test, runtime, or deployment evidence.
