# Mycelix as a Receipt-Checkpoint Witness: Pinned Source Audit (v0.1)

**Tracking:** EVAL-ORACLE-004 / Symthaea #9642  
**Cross-repository source finding:** Mycelix #5630  
**Mycelix source subject:** `Luminous-Dynamics/mycelix@512ba1e9a9ec7371739835410559305f7c7497cb`  
**Status:** source inspection complete for the RB-BFT prototype and workspace architecture; no Holochain conductor experiment, multi-agent adversarial test, deployment audit or snapshot-restore campaign executed

## 1. Decision

Do **not** qualify the current standalone Mycelix RB-BFT zome as a rollback-resistant checkpoint witness service.

Mycelix remains a plausible **audit publication, identity, provenance and peer-validation layer**. The actual security decision accepting a Symthaea evaluator checkpoint must remain anchored by a separately qualified monotonic mechanism (independent remote witnesses and/or TPM 2.0 NV monotonic state) until a successor implementation demonstrates the required quorum and storage properties.

This is a source-level qualification decision for one inspected code path, not a claim that all Mycelix applications or Holochain itself are insecure.

## 2. Pinned architecture evidence

At the pinned Mycelix commit:

- The repository README describes the overall application as a pre-alpha research project. It explicitly distinguishes unit-test coverage from real multi-agent Tryorama/Sweettest coverage, and notes that its own maturity matrix was not independently rerun.
- The architecture notes describe Holochain as agent-centric: each agent has its own signed source chain, while public entries propagate through DHT validation/peer storage. The notes explicitly distinguish this from global consensus.
- The architecture document describes a separate RB-BFT-style consensus overlay for particular system-critical roles. That overlay must be assessed on its own source and threat model; it cannot inherit consensus guarantees merely from DHT replication.

Relevant Mycelix paths:
- `README.md`
- `mycelix-workspace/mycelix-core/docs/10-research/Mycelix Prototypes and Architecture.md`
- `mycelix-workspace/mycelix-core/docs/architecture/Mycelix Protocol Architecture_ A Framework for Scalable, Interoperable, and Agent-Centric Decentralized Systems.md`

The current architecture document describes sequencers and aggregators as an additional overlay. It also labels local finality and global finality separately. This reinforces the design requirement to state which consistency/finality guarantee a specific application path actually provides.

## 3. RB-BFT prototype: source-visible gaps

Inspected code:
- `mycelix-workspace/happs/consensus/zomes/rbbft/coordinator/src/lib.rs`
- `mycelix-workspace/happs/consensus/zomes/rbbft/integrity/src/lib.rs`

### 3.1 Vote admission and attribution

The coordinator's `vote_on_proposal` constructs a Vote using the caller's agent key and creates a new entry/link. There is no visible per-voter/per-proposal/per-round/per-phase uniqueness guard in that function.

The integrity callback routes Vote creation into `validate_create_entry`, but its action argument is unused for the entry. The Vote branch checks only a reputation range and minimum. It does not bind `Vote.voter` to the Holochain action author, nor validate the proposal/round/phase relationship against an authoritative record.

The coordinator path setting the voter to the caller is useful but not sufficient: the integrity layer is the network validation boundary and must independently enforce claimed-author semantics.

### 3.2 Tally is over linked records, not a fixed validator snapshot

`get_proposal_votes` fetches every vote linked under a proposal. `calculate_consensus` sums all linked vote weights and accepts when weighted-yes / weighted-total reaches `QUORUM_THRESHOLD`.

The observed tally does not deduplicate by validator identity and does not filter by VoteType or round. Thus repeated entries from one voter can contribute multiple times, and PreVote / PreCommit records are combined in the same calculation path.

The threshold is a ratio over returned records, not a required quorum of a pinned eligible validator set. A positive one-record all-yes set yields a 1.0 observed-record ratio. That is not a validator-set quorum-intersection proof.

### 3.3 CommittedBlock is a scalar summary, not a quorum certificate

The committed entry stores proposal hash, round, height, weighted yes total, weighted total and timestamp. Integrity checks positive total and the ratio. It does not reconstruct the tally from unique, valid, phase-correct votes or carry the exact validator membership digest and signer set.

The observed callback also does not establish proposal/height/parent continuity or bind a finalizing author to a committed certificate.

### 3.4 The 45% fault-tolerance parameter is not proved by these checks

The integrity code declares a 45% Byzantine threshold constant and 2/3 quorum constant and uses reputation-squared weights. Those constants and arithmetic tests do not establish a 45% Byzantine fault bound.

Such a claim requires an explicit adversary and identity/weight model, a fixed membership/weight snapshot, a quorum-intersection argument under the precise weighted voting rule, a safety/liveness protocol including view changes and phase separation, unique signer and equivocation handling, and model/property-based plus multi-agent attack testing.

The current ratio over submitted records does not supply those missing premises.

## 4. Federated-learning path is not a drop-in fix

The federated-learning workspace has a separate commit-reveal aggregation coordinator. Its source has additional controls, including a check against duplicate validator registration and a comment explaining that self-reported trust is ignored in favor of on-chain reputation.

That is useful progress, but it is a different application path with different entries and APIs. The source inspected here does not establish a durable independent witness anti-equivocation database, a monotonic counter outside the conductor's rollback domain, or checkpoint quorum certificates suitable for EVAL-ORACLE-003.

Do not transfer FL claims or tests to the RB-BFT zome without exact path and version evidence.

## 5. Required successor certificate contract

A qualified checkpoint witness protocol must bind DNA/network ID and protocol version; evaluator ID and evaluator key ID; checkpoint epoch and sequence/counter high-water mark; previous checkpoint digest and receipt-chain head; consensus instance, round, view and phase; immutable validator membership/weight snapshot digest; unique validator identities and exact weights at that snapshot; proposal/checkpoint digest and canonical signed payload; explicit quorum rule and exact signer set; challenge/request ID and freshness; and key-registry revision and membership transition certificate.

Use integer/rational weights with checked arithmetic. Avoid floating-point tally thresholds.

Integrity validation must verify that each vote is authored by its claimed validator, comes from an eligible member of the pinned snapshot, names the exact evaluator/checkpoint/epoch/round/view/phase, has a valid signature over canonical bytes, is not a duplicate or conflicting second vote for that validator/instance/phase, and is within the permitted protocol phase.

The final certificate must let an independent verifier reconstruct exactly why the quorum is valid without trusting coordinator-supplied aggregate scalar totals.

## 6. Witness failure and storage model

The Holochain DNA can publish signed votes/certificates and distribute validation evidence. The application must separately prove per-witness atomic anti-equivocation under concurrent calls; same-epoch conflicting-head rejection; partition behavior and fixed-threshold availability; cloned-agent-key and source-chain-fork treatment; conductor restart and full data-root snapshot-restore behavior; key rotation/revocation and membership transition; durable state outside the evaluator rollback domain.

If a witness's complete durable state can be rolled back with its conductor data root, the DHT may preserve an earlier signed history but it does not prevent the witness from signing a new conflicting history after restore. The protocol needs external monotonic state, independent failure domains, or both.

A local source chain is valuable for tamper-evident author history. DHT peer validation is valuable for rule-based publication and replication. Neither should be described as an irreversible global monotonic counter without separate evidence.

## 7. Required comparison experiments

Compare three anchors under the same checkpoint interface and failure schedule:

1. **Simple remote notary:** fixed allow-list, durable append-only/compare-and-set epoch store, challenge-bound signatures, fsync/transaction semantics, independent backup.
2. **TPM 2.0 NV monotonic anchor:** policy-bound NV counter/index, authorization and reset/clear semantics, backup/re-enrollment process.
3. **Mycelix witness overlay:** fixed membership, signed checkpoint proposals/votes, atomic anti-equivocation, BFT certificate and DHT audit publication.

Inject simultaneous same-epoch conflicting proposals; duplicate requests and retries; one Byzantine signer; partition before/after quorum; delayed/reordered attestations; replay under new challenges; cloned keys; conductor crash at every write boundary; snapshot restore of one/all witnesses; old/new membership overlap and rotation; witness loss and disaster recovery.

Measure accepted conflicting checkpoint count (target 0), stale/rollback acceptance count (target 0), quorum availability by fault/partition pattern, commit and verification p50/p95/p99, durable-write latency and energy, recovery time and operator intervention, signer diversity and independent storage-domain proof.

Do not collapse availability and safety into one score. During partition, below-quorum behavior must be a fail-closed Indeterminate/Unavailable result, not silent threshold reduction.

## 8. Current evidence / limits

This audit used repository source at the pinned commit listed above and focused on the RB-BFT coordinator and integrity modules. It did not execute a live exploit, Holochain conductor, Sweettest campaign, cloned-key test, partition test, snapshot restore or TPM adapter. The repository's own README labels the overall project pre-alpha and says multi-agent coverage varies substantially by cluster.

The result is therefore a source-bound implementation review and a precise qualification blocker, not a claim about a production incident or every Mycelix cluster.

## 9. Tracking

- Symthaea: EVAL-ORACLE-004 / #9642
- Mycelix security issue: https://github.com/Luminous-Dynamics/mycelix/issues/5630
- Receipt verifier: Symthaea PR #9640
- OS-level evaluator isolation: Symthaea PR #9630

No Mycelix code path is authorized to act as the sole rollback-resistant checkpoint authority until a successor implementation and the comparison campaign satisfy the contract above.