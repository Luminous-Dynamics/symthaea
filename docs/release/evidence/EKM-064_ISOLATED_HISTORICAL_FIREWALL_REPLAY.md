# EKM-064 — Isolated Historical Firewall Replay Verification

## Purpose

EKM-055 showed that replaying persisted belief mutations against the final ledger can fail legitimately when later evidence has been appended to a claim. EKM-056 through EKM-063 establish a protected, current and read-only reconstruction of the exact mutation-time evidence census.

EKM-064 is the first tranche that uses that historical census to execute the existing EKM-026 belief-mutation firewall again.

The central invariant is:

**a persisted belief mutation is historically reproducible only when the real EKM-026 firewall, driven by the protected mutation-time claim/evidence census, reproduces the exact persisted mutation receipt and final support state inside disposable private state.**

## Source chain

EKM-064 requires the complete read-only chain:

1. exact restart-v2 snapshot
2. EKM-057 mutation-seal sidecar
3. exact EKM-044 restart validation receipt
4. EKM-059 protected mutation-seal checkpoint
5. EKM-060 protected-source admission receipt
6. EKM-061 current-head evidence
7. EKM-062 historical replay eligibility receipt
8. EKM-063 historical evidence projection

EKM-063 is re-derived from those exact inputs before any replay occurs.

## Historical ledger reconstruction

For one persisted mutation, EKM-064 constructs a private minimal append-only ledger prefix sufficient to evaluate the source revision and EKM-026 firewall:

- provenance IDs are replayed as a contiguous prefix through the largest provenance ID referenced by the required evidence prefix
- claim IDs are replayed as a contiguous prefix through the largest claim ID referenced by that evidence prefix or the target claim
- evidence IDs are replayed as a contiguous global prefix through the largest evidence ID in the protected mutation-time census

The resulting target-claim evidence IDs must equal the protected EKM-063 census exactly.

This is stronger than filtering the final ledger by observation timestamp. A final record absent from the protected census remains excluded even if `observed_at_cycle <= sealed_at_cycle`.

## Revision receipt identity

Persisted mutation receipts reference EKM-025 revision receipt IDs. Those IDs include rejected and unapplied decisions.

EKM-064 deliberately does **not** add a private arbitrary-ID constructor to `BeliefRevisionHistory`.

Instead it recreates the receipt ID space in order:

- revision receipts that source persisted mutations are semantically re-evaluated against their protected historical ledger projection and compared exactly with the persisted wire/schema receipt
- all other revisions are represented only by private zero-delta ID-cursor placeholders

Those placeholders exist solely to preserve receipt numbering. Their semantics are not treated as historical evidence and are never passed to the mutation firewall.

The report therefore exposes `full_nonmutation_revision_history_replayed = false` whenever placeholder history exists.

## Actual firewall replay

EKM-064 then:

1. creates a private `EpistemicSupportStore`
2. registers persisted support baselines through the ordinary public store API
3. preserves one private `BeliefMutationFirewall` across the entire mutation sequence
4. for each persisted mutation, reconstructs the exact source revision receipt and historical ledger
5. recreates the persisted authorization identity/label/cycle through `BeliefMutationAuthorization::new`
6. calls the real `BeliefMutationFirewall::apply`
7. requires a newly applied mutation, not idempotent replay
8. compares every mutation receipt field exactly with persistence
9. verifies the final support states
10. verifies the consumed authorization count

No direct support-store field injection is used.

## Resource boundary

V1 refuses histories containing more than 4096 persisted mutations. This is a conservative verification bound because a distinct historical ledger prefix may be rebuilt for each mutation.

This bound is not an epistemic limitation and can later be raised or replaced by an optimized persistent-prefix representation after qualification.

## Report semantics

For each mutation the read-only report binds:

- mutation ID
- source revision receipt ID
- claim ID
- EKM-063 projection-record digest
- protected historical evidence count
- final-ledger-only evidence count
- final-only evidence carrying an observation timestamp not after the historical seal
- support-before/support-after bit patterns
- support revisions
- exact persisted mutation receipt reproduction

The aggregate report binds:

- restart capture cycle
- EKM-063 projection digest
- EKM-062 eligibility digest
- mutation and firewall-invocation counts
- non-mutation revision placeholder count
- every per-mutation replay digest
- final support-state equivalence
- consumed-authorization equivalence

## Authority boundary

EKM-064 returns only an immutable verification report.

The private historical ledgers, revision history, support store, authorizations, firewall and mutation outcomes are discarded before return.

A successful report still has:

- `writable_state_export_authorized = false`
- `writable_hydration_authorized = false`
- `activation_authorized = false`

It does not:

- expose an `EpistemicLedger`
- expose `EpistemicSupportStore`
- expose a mutation authorization
- hydrate the live restart quarantine
- advance any trusted checkpoint
- mutate legacy `TemporalFact::confidence`
- mutate the causal DAG, world model or action system
- perform file/network I/O or key custody

## Qualification status

This tranche is stacked on EKM-063. GitHub Actions remains the executable authority. Parent EKM-063 exact-head CI #7310 was still queued when this evidence note was prepared.

Static/API review is not rustfmt, compilation, Clippy, unit-test, integration-test, runtime or deployment evidence.

## Next boundary

If EKM-064 receives executable qualification, the next safe step is **not activation**.

A later tranche should make EKM-055 consume a qualified EKM-064 replay report and the protected historical projection to build the sealed writable hydration sandbox, then prove that the hydrated support/mutation state is exactly the state already reproduced by EKM-064. The writable objects should still remain non-exportable and non-activatable.
