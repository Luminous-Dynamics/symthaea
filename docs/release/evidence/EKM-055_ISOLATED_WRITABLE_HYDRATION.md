# EKM-055 — Isolated Writable Hydration Sandbox

## Purpose

EKM-053 admits externally supplied restart state only into a sealed read-only quarantine. EKM-054 can externally protect the joint trust-context digest, but neither layer reconstructs the writable belief-revision state used by EKM-024 through EKM-029.

EKM-055 introduces the first isolated hydration of the real `BeliefRevisionHistory` and `EpistemicSupportStore` types while keeping both objects sealed inside a non-activating sandbox.

The central invariant is:

**externally supplied restart state may become writable EKM state only inside an isolated, non-exporting sandbox after the exact read-only quarantine and exact protected trust-context checkpoint are re-verified, and the hydrated state must re-export to the source persistence semantics exactly.**

## Construction path

The sandbox constructor consumes the EKM-053 read-only quarantine handle and requires:

- the exact EKM-042/V2 source snapshot;
- a valid EKM-053 quarantine whose checksum, V2 digest and capture cycle match that snapshot;
- an EKM-054 verified protected checkpoint whose context digest exactly matches the quarantine trust-context digest;
- a hydration cycle that does not predate checkpoint verification and occurs before checkpoint expiry;
- a fresh EKM-043 semantic validation pass over the exact source snapshot.

The consumed quarantine cannot continue to be used independently by the caller after successful hydration.

## Revision-history hydration

EKM-055 does not deserialize private receipt fields directly.

For every persisted revision it:

1. rebuilds the canonical typed policy from the EKM-040 schema;
2. reconstructs the proposal, calibration and uncertainty inputs;
3. independently re-runs `BeliefRevisionGate`;
4. requires the resulting typed decision to equal the persisted EKM-036 decision snapshot;
5. records the decision through the ordinary `BeliefRevisionHistory::evaluate_and_record` path;
6. requires the regenerated receipt ID, claim, delta, rationale, evidence snapshots, duplicate-evidence diagnostic, eligibility, provenance-root count and V1 compatibility policy/decision representations to match the source wire receipt.

Rejected decisions remain part of the hydrated append-only history.

## Support-store hydration

Each persisted support state is initialized through the ordinary `EpistemicSupportStore::register_claim` API at its persisted baseline.

Each applied mutation is then replayed through the existing `BeliefMutationFirewall` using:

- the regenerated source revision receipt;
- the persisted authorization ID and authority label;
- the persisted authorization cycle;
- the current replayed support state;
- the persisted mutation cycle.

The newly generated mutation receipt must exactly match the source mutation record, including stable mutation ID, source receipt, support transition, revisions, authorization identity and cycles.

After replay, EKM-055 re-captures both:

- `BeliefMutationPersistenceCapsuleV1`; and
- `BeliefRevisionHistoryCapsuleV1`.

Those re-captured objects must exactly reproduce the source support/mutation/history semantics.

## Historical replay limitation discovered

The current restart payload stores `EvidenceRecord.observed_at_cycle`, but it does **not** store a distinct ledger insertion/availability cycle for each evidence record.

The existing mutation firewall intentionally rejects a revision if evidence currently linked to the claim postdates the frozen revision decision. Consequently, a final restart image containing evidence added after an already-applied historical mutation does not always contain enough chronology to prove the exact historical firewall input state.

EKM-055 therefore fails closed with `HistoricalLedgerStateInsufficient` whenever final retained evidence prevents exact firewall replay.

It does **not** bypass the firewall and does **not** write private support-store fields directly merely to make restoration succeed.

A future restart schema should record evidence-ledger availability chronology or an equivalent exact historical evidence census for every applied revision before general hydration is claimed.

## Public capability boundary

`IsolatedEpistemicRestartHydrationV1` is non-Clone and keeps these real writable objects private:

- `EpistemicSupportStore`;
- `BeliefRevisionHistory`;
- reconstructed `EpistemicLedger`;
- consumed EKM-053 quarantine;
- source V2 snapshot;
- EKM-054 protected checkpoint receipt.

Public inspection is limited to counts, immutable scalar support summaries and stable digests.

No `&mut EpistemicSupportStore`, `&mut BeliefRevisionHistory`, mutable ledger, raw quarantine or activation handle is exposed.

The sandbox explicitly reports:

- `isolated_hydration_performed = true`;
- `writable_state_export_authorized = false`;
- `activation_authorized = false`;
- trusted-state mutation remains false.

## Non-authorities

EKM-055 does not:

- replace live Symthaea state;
- expose writable hydrated objects;
- advance restart-anchor state;
- advance verifier-trust state;
- persist or advance the EKM-054 protected checkpoint;
- authorize evidence or belief mutation outside replay reconstruction;
- mutate the legacy `EnhancedKnowledgeGraph` or `TemporalFact::confidence`;
- alter the causal DAG, world model or action policy;
- perform file/network I/O;
- perform signing or key custody.

## Qualification status

This tranche is stacked on EKM-054. GitHub Actions remains the executable authority. At creation time the EKM-054 exact-head CI and the explicitly re-run historical Format job remain queued; no format, compile, Clippy, test or runtime PASS is inferred from queue state or static review.
