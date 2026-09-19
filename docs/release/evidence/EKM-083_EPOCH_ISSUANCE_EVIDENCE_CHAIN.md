# EKM-083 — Durable Epoch-Issuance Evidence Chain

## Purpose

EKM-078 proves global receipt-cursor continuity and contiguous epoch-segment numbering, but deliberately leaves `epoch_issuance_chain_verified = false` because sequence numbers and segment hashes are not proof that an epoch was actually issued.

EKM-079 through EKM-082 add progressively stronger activation evidence:

- EKM-079: passive activation-commit / authority-epoch record
- EKM-080: all nine EKM-072 activation phases accepted by one stable verifier profile
- EKM-081: local verifier policy plus checkpoint-relative trust continuity
- EKM-082: independent current-head attestation for the exact verifier trust snapshot

EKM-083 binds those layers to the exact EKM-077 segment and then chains the resulting durable bindings across EKM-078.

The central invariant is:

**activation evidence, verifier policy, verifier currentness, epoch/segment identity, predecessor continuity, CAS-generation continuity, and full verifier correctness remain separate claims.**

## Durable per-epoch binding

`bind_epoch_issuance_evidence(...)` requires one exact:

- EKM-079 issuance record
- EKM-077 receipt segment
- EKM-080 activation-evidence receipt
- EKM-081 verifier trust-review receipt
- EKM-082 verified currentness receipt

The binding is created only while EKM-082 currentness is still fresh.

It verifies that:

- the EKM-080 receipt names the exact EKM-079 record, activation-commit digest, and authority-epoch digest
- all activation phases were provider-accepted under one stable profile
- EKM-081 names that exact EKM-080 receipt and EKM-079 record
- EKM-081 local policy, checkpoint continuity, and profile freshness are all proven
- EKM-082 internally verifies and names the exact EKM-081 review, EKM-080 receipt, EKM-079 record, canonical verifier profile, trust policy, and caller checkpoint
- EKM-082 current-head proof is still unexpired when the durable binding is captured
- the EKM-077 segment sequence/digest equals the EKM-079 epoch sequence/digest
- the segment was captured no earlier than activation
- every V2 receipt in the segment was evaluated at or after the epoch activation cycle

The durable binding stores the exact identities and cycles under its own domain-separated BLAKE3 digest.

## Why the binding is durable

EKM-082 currentness is intentionally time-bounded. Requiring historical audit validation to re-use an old current-head assertion forever would make old epochs unverifiable after that assertion expires.

EKM-083 therefore consumes the fresh currentness result once, while valid, and produces an immutable evidence-binding receipt.

This does **not** make currentness permanent. It records that currentness was valid at the binding cycle.

## Multi-epoch evidence chain

`validate_epoch_issuance_evidence_chain(...)` binds the durable receipts to the exact EKM-078 history chain and proves:

- one binding per EKM-077 segment
- receipt-cursor continuity inherited from EKM-078
- contiguous epoch sequence beginning at 1
- exact epoch digest equality between segment and issuance evidence
- genesis has no predecessor
- every later epoch names the exact previous authority-epoch digest
- each epoch's expected live generation equals the previous epoch's committed generation
- every activation commit advances generation by exactly one
- activation cycles strictly increase
- no binding postdates the EKM-083 chain capture

The chain also records the final receipt cursor and latest bound epoch identity.

## Empty-chain semantics

An EKM-078 chain with zero V2 epoch segments is valid as an archival-only legacy history.

For that case EKM-083 does **not** claim that verifier current-head evidence was bound. The chain may still have structural continuity, but:

`verifier_current_head_evidence_bound = false`

until at least one epoch binding exists.

## Claim boundary

A successful non-empty EKM-083 chain may state:

- receipt cursor continuity proven = true
- epoch sequence continuity proven = true
- epoch predecessor continuity proven = true
- live generation continuity proven = true
- activation-cycle monotonicity proven = true
- issuance evidence chain consistent = true
- verifier current-head evidence bound = true

It deliberately continues to state:

- verifier correctness independently proven = false
- epoch issuance chain verified = false
- active epoch registry constructed = false
- operational history constructed = false
- mutation authority = false
- activation authorized = false

## Why full epoch issuance remains false

EKM-080 proves provider acceptance of phase evidence.

EKM-081 proves local policy and checkpoint-relative trust continuity.

EKM-082 proves that the verifier trust snapshot was current.

None of those independently proves that the verifier implementation and external trust root are correct, uncompromised, and sufficient to elevate the activation claim into authority.

EKM-083 therefore proves a complete **evidence-consistency chain**, not yet a complete **authority-issuance chain**.

That prevents `current verifier` from silently becoming `correct verifier`, and prevents `consistent activation evidence` from silently becoming `active mutation authority`.

## Deterministic repair discovered during this tranche

The unexecuted EKM-081/EKM-082 stack contained a real Rust module-path defect: it referenced a knowledge-root `epistemic_restart_authority_epoch_issuance_record` module that is not declared there.

The actual module lives under:

`epistemic_restart_continuity::authority_epoch_contract::issuance_record`

EKM-081 was corrected to use its actual ancestor path, EKM-082's bridge was corrected to use the nested path, and those exact fixed sources were propagated into EKM-083 before this tranche was frozen.

This was a deterministic static defect; it is not being mislabeled as a CI failure because no CI job had executed it.

## CI status

The previous EKM-082 CI run #7618 eventually completed as `cancelled`, but all 30 jobs were cancelled without step data. That is not executable evidence for rustfmt, compilation, Clippy, tests, or runtime behavior.

New exact-head runs created after the path repairs remain the qualification authority.

## Authority boundary

EKM-083 does **not**:

- establish verifier correctness
- establish an external root of trust
- set EKM-078's `epoch_issuance_chain_verified` flag
- install an active authority epoch
- construct operational `BeliefRevisionHistory`
- produce fresh V2 revision receipts
- integrate with `BeliefMutationAuthority`
- export mutation authority
- authorize activation

## Next boundary

The next meaningful trust boundary is not another receipt hash. It is an independently grounded verifier-correctness / trust-root attestation that can justify elevating a consistent EKM-083 evidence chain into a verified epoch-issuance chain.

Only after that should an active-epoch registry be considered, and active-epoch installation must remain separate from mutation-facade enforcement.
