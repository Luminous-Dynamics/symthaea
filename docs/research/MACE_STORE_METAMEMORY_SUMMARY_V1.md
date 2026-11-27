# MACE Store Metamemory Summary V1

**Snapshot:** 2026-11-27  
**Status:** Value-free store summary, integrity verifier, JSON Schema and deterministic tests authored. Rust compilation, test execution and official benchmark execution are not claimed.

## 1. Research premise

A memory agent should not have to infer its backend's lifecycle state from retrieved content. Before retrieval or answer synthesis, the store should expose what it can reliably say about its own state: schema, available attributes, lifecycle counts, source/index sequence and integrity commitments.

This is the core idea behind *Knowing the Store: What a Memory Backend Must Write Down Before an Agent Can Read It*. The public artifact is now available at https://github.com/ansh348/knowing-the-store. Its README describes 148 questions, each applied to three versions of one store that differ in one attribute (444 store-version cases). The supplementary archive is published with an offline reproduction path; the accompanying ICLR 2027 submission is marked under review, not accepted.

The paper reports a particularly important limitation: for the short attribute lists present in tested stores, several model readers did not reliably beat a cosine-similarity lookup over attribute names. The public abstract says the tested backends omit or misstate metadata needed to identify lifecycle state. Better language modelling cannot compensate for a state the store never wrote down.

## 2. Implemented value-free snapshot

Module:

`research/broca-mace-transfer-bench-v0-1/aml-adapter/src/temporal_projection.rs`

Schema:

`data/schemas/mace_store_metamemory_summary_v1.schema.json`

Public API:

`DerivedProjectionIndexV1::metamemory_summary(&StreamingSessionLedgerV1)`

The summary contains:
- schema version;
- a hash commitment to the user/session scope (raw identifiers are not emitted);
- source ledger sequence and projection-index sequence;
- count of currently active source events;
- total projection count;
- lifecycle-state counts for Current, Stale, Quarantined, Revoked, ErasureRequested, Rebuilt and Erased;
- projection-kind counts for Summary, Embedding, PreferenceModel, Plan, FactIndex, AnswerCache and Other;
- a fixed, ordered inventory of eight metadata attributes;
- ledger snapshot, projection-index state and summary SHA-256 commitments.

The eight declared metadata attributes are:
1. `lifecycle_status`
2. `projection_kind`
3. `built_at_sequence`
4. `source_dependency_count`
5. `parent_dependency_count`
6. `invalidation_receipt_present`
7. `replacement_link_present`
8. `erasure_receipt_present`

The output is value-free: it does not include source text, projection content hashes, record IDs, user identifiers or individual dependency IDs. The raw store remains available only behind its ordinary authorization boundary.

## 3. Fail-closed semantics

The store refuses to generate the summary when:
- the ledger and projection index have different user/session scope;
- the projection index has not consumed the current ledger sequence.

The summary verifier checks:
- schema version and exact attribute inventory/order;
- exact lifecycle and projection-kind inventory/order;
- checked arithmetic over counts (overflow is rejected);
- count totals equal the declared projection total;
- ledger and projection-index sequence equality;
- valid digest shape and the final summary commitment.

**Important:** the summary's self-integrity check proves only that its fields agree with their commitment. It does not prove that the data source is authentic, that the snapshot is still the newest head, or that the caller is authorized to see even aggregate metadata. A consumer must obtain it from a trusted store boundary, compare its sequence/ledger commitment to the known current head and enforce caller scope before disclosure.

## 4. Knowing the Store comparison design

The upstream artifact uses 148 unique questions, each asked against three store versions with a one-attribute difference. The resulting counterfactual set contains 444 store-version items. It studies supersession, retraction and expiry as separate lifecycle obligations. All stores are synthetic. The supplementary repository publishes a 36 MB archive with 1,438 files and an offline command that checks its packaged reproduction; the README says the number gate reports 232 checks with zero mismatches. Its supplementary README is careful that some values are recomputed, some regenerated, some independently re-derived and some merely accounted against a ledger. MACE should preserve those distinctions rather than claim a full rebuild when the artifact says otherwise.

Run the official evaluator unchanged first. Then add a MACE-specific metadata fault-injection extension. Compare:
- the official store-summary attribute list and lifecycle counts;
- a lexical/name-list lookup baseline;
- a count-only baseline;
- a status-and-attribute metadata baseline;
- a language-model metamemory reader using the same summary;
- an oracle metadata reader, reported separately.

Do not treat a larger attribute list as automatically better. The paper shows that extra names can help or distract depending on naming, length and lifecycle-state counts. Test controlled omissions, reordering, misleading extra fields, wrong counts, stale sequence numbers and missing lifecycle dimensions.

## 5. Adversarial fault-injection matrix

| Fault | Required response |
|---|---|
| One required metadata attribute omitted | Reject the summary or mark the store view incomplete; never infer the attribute exists |
| Attribute list reordered or duplicated | Reject exact schema conformance |
| One lifecycle count changed but the digest is stale | Integrity verification fails |
| Count arithmetic overflows | Integrity verification fails without panic |
| Projection index is behind the source ledger | No summary is emitted |
| Summary scope differs from caller scope | Authorization boundary rejects disclosure |
| Revocation/correction applied but index not caught up | Sequence mismatch blocks summary |
| Metadata schema version is unknown | Consumer abstains from lifecycle-dependent claims |
| Summary is internally valid but older than current store head | Freshness check rejects it |
| Aggregate status says Current but authoritative record is revoked | Independent consistency check fails; never trust aggregate alone for record-level use |

## 6. Qualification targets

These are proposed MACE targets, not existing results:
- 100% exact schema conformance on frozen valid summaries.
- 100% rejection of malformed, reordered, omitted and duplicate attribute inventories.
- Zero summaries emitted while the projection index is behind the authoritative ledger.
- Zero arithmetic panics or wraparound on adversarial count values.
- Zero stale projection acceptance after a ledger transition has been committed.
- 100% agreement between summary counts and full index enumeration on deterministic fixtures.
- 100% freshness/sequence detection in controlled stale-snapshot tests.
- Explicit caller authorization before any store metadata is disclosed.

Report these separately from model question accuracy. A correct-looking answer from an incomplete store summary is not evidence that the system knew the store state.

## 7. Relationship to the other MACE work

The summary becomes the metamemory input for:
- the streaming event ledger (source sequence and active evidence);
- dependency-aware projections (projection count, type and lifecycle);
- claim-relative revalidation (preserve/rebuild versus invalidate);
- non-regressive learning (a model must not learn from stale or unauthorized memory state);
- adaptive policy acquisition (the agent can query the backend for missing metadata instead of guessing).

The metamemory summary does not authorize record use, decide claim truth, or replace direct per-record checks. It is a compact map of what the store says it knows about its own structure.

## 8. Current implementation boundary

The summary is an in-memory reference view. Production still requires:
1. signed/attested summary receipts from the durable store;
2. tenant- and role-aware authorization over metadata disclosure;
3. distributed consistency across vector, lexical, relational, cache and replica backends;
4. freshness tokens or a linearizable head comparison;
5. lifecycle reconciliation and unknown-outcome recovery;
6. record-level proofs linking aggregate counts to enumerated authorized records;
7. deletion verification for replicas and backups;
8. fault-injection tests across crash, partition, delayed revocation and concurrent update scenarios.

No deployment or physical-erasure claim is made.


## 9. Freshness verification against the current store

An unkeyed self-digest is only an internal consistency check: anyone able to alter a summary can recalculate it. The projection index now also exposes `verify_metamemory_summary(ledger, summary)`. It first checks the summary's internal shape/commitment, then recomputes the expected value from the exact supplied ledger and projection-index instance and requires exact structure equality. This rejects a summary that is well-formed and internally self-consistent but no longer reflects the current ledger/index state.

This still depends on the caller obtaining the authoritative current ledger and enforcing scope authorization. It is not a signature or proof against a malicious store implementation. Three additional tests check exact current acceptance, a valid-but-stale summary after a new event, and a self-resealed count forgery. Eight summary tests are now authored in total; the full projection module has twenty tests, not executed.
