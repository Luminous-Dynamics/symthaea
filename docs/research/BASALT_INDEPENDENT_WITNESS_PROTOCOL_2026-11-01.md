# BASALT Independent Witness Protocol v1 — Design Specification

**Status:** research specification and schemas. No witness service, signing key, signature verifier, deployment, or anti-rollback qualification is implemented by this increment.

## Purpose and claim boundary

A local database can be internally consistent and still be stale: an attacker or operator can restore an older complete snapshot, including the local revision counter and an unkeyed state digest. The restored database cannot prove its own freshness.

BASALT therefore separates four observations:

1. **Observed store revision:** what the database currently reports, including a restored stale value.
2. **Witness-anchored authority revision:** the revision carried by the latest checkpoint established by an independent witness quorum.
3. **Receipt acceptance revision:** the historical revision at which one immutable transition was first accepted.
4. **Current lookup revision:** the current authoritative head returned by a lookup/retry.

They must never be substituted for one another.

This document composes established mechanisms rather than introducing a new signature format:
- [DSSE](https://github.com/secure-systems-lab/dsse) provides a payload-type-bound pre-authentication encoding and allows multiple signatures over the same payload.
- [RFC 8785 — JSON Canonicalization Scheme](https://www.rfc-editor.org/rfc/rfc8785.html) defines repeatable canonical JSON bytes for hashing/signing. It is an Informational RFC, not an Internet Standards Track RFC.
- [TUF v1.0.36](https://theupdateframework.github.io/specification/v1.0.36/) supplies a mature pattern for pinned trust roots, threshold signatures, monotonically versioned root metadata and dual-threshold root rotation.
- [RFC 9052 — COSE](https://www.rfc-editor.org/rfc/rfc9052.html) is a standards-track signed-CBOR option worth benchmarking for constrained devices. This v1 draft chooses DSSE+JCS to simplify multi-witness evidence and readable inspection; it does not prohibit a later COSE profile.

These references are not evidence that BASALT currently provides the guarantees described here. They are the foundations proposed for an independently reviewed implementation.

## Artifacts added

- `benchmarks/basalt/schema/witness-dsse-envelope.v1.schema.json`
- `benchmarks/basalt/schema/witness-checkpoint-payload.v1.schema.json`
- `benchmarks/basalt/schema/witness-head-response-payload.v1.schema.json`
- `benchmarks/basalt/schema/witness-set-metadata-payload.v1.schema.json`
- `benchmarks/basalt/schema/witness-trust-root-payload.v1.schema.json`
- `benchmarks/basalt/schema/authority-state-manifest.v1.schema.json`

The JSON Schemas describe structure only. A verifier must additionally enforce uniqueness, sorting, quorum arithmetic, signatures, hash calculations, sequence continuity, trust-root transitions, and cross-record relationships. JSON Schema success is not a signature or authority verdict.

## Cryptographic envelope

### DSSE payload types

The envelope schema accepts four explicit payload types:

- `application/vnd.luminous.basalt.witness-checkpoint.v1+json`
- `application/vnd.luminous.basalt.witness-head-response.v1+json`
- `application/vnd.luminous.basalt.witness-set-metadata.v1+json`
- `application/vnd.luminous.basalt.witness-trust-root.v1+json`

The envelope follows the DSSE structure: `payloadType`, base64 `payload`, and a list of `{ keyid, sig }` signatures. Every signature in one envelope authenticates the same payload bytes and payload type.

The signature input is DSSE pre-authentication encoding:

```text
PAE(type, payload) =
  "DSSEv1 " || LEN(type) || " " || type || " " ||
  LEN(payload) || " " || payload
```

`LEN` is the ASCII decimal byte length, without leading zeros. Lengths are byte lengths, not Unicode character counts. The signature is computed over the exact PAE bytes. Do not sign pretty-printed JSON or a re-serialized object.

### Canonical payload bytes

Before enveloping a payload:

1. Parse JSON with duplicate object keys rejected.
2. Validate it against the payload schema and semantic rules.
3. Restrict all identifiers to the declared interoperable character set; do not normalize Unicode after parsing.
4. Represent all 64-bit counters, epochs, revisions and sequence values as canonical decimal strings (no leading zeroes except the value `"0"`). This avoids JSON number precision ambiguity.
5. Canonicalize the validated JSON using RFC 8785 JCS.
6. Put those exact canonical bytes in the DSSE `payload` field using standard Base64.

A verifier must decode the payload, reject duplicate keys and out-of-profile values, re-canonicalize the parsed JSON, and require byte-for-byte equality with the decoded payload bytes before checking DSSE signatures.

### Key identity and algorithm selection

- `keyid` is an untrusted lookup hint, not proof of identity or authorization.
- Resolve every `keyid` only through the separately trusted root / witness-set metadata.
- V1 permits Ed25519 only. The algorithm is selected by trusted key metadata; an envelope cannot select or override it.
- Public-key identity is `SHA-256(JCS({ "keytype": "ed25519", "public": BASE64_PUBLIC_KEY }))`, rendered as 64 lowercase hexadecimal characters. A verifier recomputes it and rejects mismatches or duplicate key IDs.
- A key ID contributes at most one valid signature to a threshold, even if duplicated in the envelope. Unknown, revoked, malformed or duplicate key IDs never contribute.

The v1 schema uses standard Base64 for DSSE payloads, signatures and Ed25519 public keys. The challenge nonce uses unpadded Base64URL because it is generated independently by the verifier.

## Canonical authority state manifest

The witness must bind the complete accepted state, not only a local integer revision or the latest cursor. The canonical input is a `basalt.authority-state.v1` manifest as described by `authority-state-manifest.v1.schema.json`.

The manifest includes:

- authority ID, incarnation ID and epoch;
- authority revision and active policy revision, signer/witness policy versions, policy bundle digest and its policy digest binding;
- all source cursors, keyed by the full stable tuple (provisioned host ID, source ID, partition ID);
- all retained transparency-log cursors;
- the complete accepted-receipt history, including immutable acceptance revision, predecessor, receipt/seal digests, profile/boot identities, all profile/boot/log-origin/policy transition-evidence references, checkpoint identity/evidence and original signer/witness policy versions and bundle digest.

Do not hash only the latest receipt or cursor. This would fail to bind historical acceptance state and permit deletion or substitution of earlier history to go unnoticed.

Canonical array ordering is normative:
- source cursors sorted lexicographically by the UTF-8 byte sequence of `(provisioned_host_id, source_id, partition_id)`;
- log cursors sorted by UTF-8 bytes of `log_origin_id`;
- accepted receipts sorted by their numeric `receipt_acceptance_revision` (parsed from canonical decimal strings), then by receipt digest as a defensive tie-break. Equal acceptance revisions are invalid and never legitimized by the tie-break.

Every counter is encoded as a canonical decimal string. No floating point values are allowed in the state manifest. The top-level `policy_digest` MUST equal the active `policy_bundle_digest`; the signer/witness version tuple is retained separately so that the digest does not erase the policy revision context. The state digest is:

```text
state_digest =
  "sha256:" || hex_lower(
    SHA-256(
      ASCII("BASALT-AUTHORITY-STATE-V1") || 0x00 ||
      JCS(authority_state_manifest)
    )
  )
```

The complete manifest is reproducible across SQLite, PostgreSQL, etcd-backed state and other adapters. Backend page files, query order, internal row IDs and database-specific serialization are not part of the digest.

## Witness checkpoint record

The checkpoint payload schema is `witness-checkpoint-payload.v1.schema.json`. Its fields bind:

- stable authority ID;
- authority incarnation ID and monotonic authority epoch;
- monotonic witness sequence;
- current authority revision;
- state schema ID and canonical state digest;
- policy revision and policy digest;
- predecessor checkpoint digest (null only for a separately authorized genesis record);
- witness-set ID/version and trust-root version.

A checkpoint may also carry optional transition-availability references in a later compatible extension: immutable accepted-receipt digest and content-addressed receipt/history bundle digest. These references are necessary if BASALT chooses to rebuild a restored database by replaying witness-accepted transitions. The witness signs the digest; the full data remains retrievable from independently replicated immutable storage.

The checkpoint payload also carries `transition_bundle_digest`: a digest of the canonical ordered set of accepted transition receipts since the predecessor checkpoint. It is null only for an authorized genesis record. This permits a recovery service to fetch transition content by digest and rebuild a projection; the content itself must live in separately replicated immutable storage. A missing bundle makes rebuilding incomplete even though rollback detection can still succeed.

The checkpoint digest is independent of the signature set so it remains stable when the same payload is re-signed by an overlapping key set:

```text
checkpoint_digest =
  "sha256:" || hex_lower(
    SHA-256(
      ASCII("BASALT-WITNESS-CHECKPOINT-V1") || 0x00 ||
      JCS(checkpoint_payload)
    )
  )
```

The DSSE signatures authenticate the canonical payload bytes through PAE. The checkpoint digest identifies the payload in the predecessor chain.

### Append rule

For a witness member to sign a new checkpoint it must:

1. verify the trusted root and current witness-set metadata, including the initial/rotated witness-set digest chain;
2. verify the proposed authority ID/incarnation/epoch is allowed;
3. require `sequence = current_head.sequence + 1`;
4. require `previous_checkpoint_digest = current_head.checkpoint_digest`;
5. reject same-sequence/different-digest equivocation;
6. validate revision and policy monotonicity (or an explicit, separately authorized new-incarnation transition);
7. durably persist the new head before returning its signature.

A witness member must never sign two different checkpoint digests for the same authority, incarnation and sequence. A signer that detects a fork must stop signing and emit an auditable equivocation alert.

## Independent witness quorum

V1 proposes the classical Byzantine-fault-tolerant sizing relation:

```text
n = 3f + 1
q = 2f + 1
read_quorum = write_quorum = q
```

Here `f` is the maximum number of Byzantine witness members assumed by the threat model. For example, four members with quorum three tolerate one Byzantine member under the protocol assumptions. The schema carries `member_count`, `fault_tolerance`, `write_quorum` and `read_quorum`; the verifier MUST check `n = 3f+1` and both quorums equal `2f+1`. The threshold is not a magic constant and must be justified against the actual deployment threat model.

Each member must have:
- a unique member ID;
- a unique key ID;
- a unique administrative / operational failure domain;
- an independently controlled signing key;
- durable, monotonic witness-head storage;
- a separately monitored backup/restore process.

Four VMs in one cloud account, one administrator group, one KMS tenancy or one backup domain do not constitute four independent witnesses. Administrative diversity is part of the quorum claim, not deployment decoration.

### Write quorum

A candidate checkpoint becomes witness-committed only after a write quorum of distinct active members has durably stored and signed exactly the same canonical checkpoint payload. A client that receives fewer than the required signatures has an indeterminate/uncommitted witness result and must resolve by querying the quorum; it must not report a witness commit.

### Fresh head read (anti-freeze and replay)

A hash chain alone does not prove a response is the latest head. An attacker could replay an old, correctly signed chain head or suppress new records. V1 therefore uses a client-generated challenge:

1. The verifier creates a cryptographically random 32-byte nonce.
2. It sends the nonce and authority identity to the configured independent witness set.
3. Each member reads its latest durably committed checkpoint and returns a DSSE-signed `witness-head-response` containing the exact nonce, authority/incarnation/epoch, witness-set and root versions, latest sequence, checkpoint digest, authority revision, state digest and policy digest.
4. The verifier requires `q = 2f+1` valid nonce-bound responses from distinct active members. Responses must agree on authority ID/incarnation/epoch and the pinned witness-set/trust-root versions, but **they are not required to report identical heads**: honest members can temporarily lag a just-committed write.
5. For every response, the verifier fetches and verifies the referenced checkpoint envelope and its write-quorum certificate, then verifies the predecessor chain to the pinned genesis/known checkpoint.
6. It rejects two different certified checkpoint digests at the same authority/incarnation/sequence. Otherwise it selects the highest certified sequence among the valid responses and compares that checkpoint's canonical state/policy digests to the restored store's observed manifest.

The `challenge_nonce` field carries the exact fresh nonce. It prevents replay of an earlier response to a different challenge. The read protocol is a quorum read of a monotonic register, not an all-replicas-equal barrier. With `n=3f+1`, `q=2f+1`, and at most `f` Byzantine members, every read quorum intersects a previously committed write quorum in at least `f+1` members, including at least one honest member. An honest member that durably stored the committed head before signing cannot return a lower head to a later challenge. Therefore selecting the highest certified head in the fresh read quorum cannot miss a write that completed before the read began. A concurrent write may linearize before or after the read; admission still needs the scoped authority CAS and checkpoint predecessor to close the race.

This argument depends on honest members persisting a checkpoint before signing, never signing conflicting digests for the same sequence, and returning their durable local head for each fresh challenge. If the verifier cannot obtain a valid response quorum, cannot verify the referenced checkpoint certificates, or observes an unresolved authority-incarnation transition, it reports `anti_rollback_unknown`. Conflicting same-sequence certified heads are `rejected_witness_equivocation`. The verifier never falls back to the local DB revision, an old cached witness pointer, or a best-effort single-node response.

The nonce is not a replacement for the witness quorum; it prevents replay while quorum intersection prevents a stale quorum from hiding a previously committed write under the stated fault model. Requiring all returned replicas to report the same head would unnecessarily reject a valid read merely because some honest replicas are behind; the maximum-certified-head rule preserves the quorum-intersection freshness argument without assuming instantaneous replication.

## Trust roots, key rotation and revocation

Trust bootstrap must happen outside the database being qualified:
- the verifier has a pinned trust-root digest/version delivered through a separate provisioning/release channel; the root's `initial_witness_set_digest` pins the initial witness-set metadata digest;
- the root private keys are offline or in a separately controlled signing ceremony;
- witness members do not control the root keys that authorize their own replacement;
- the DB administrator and its backup operator cannot modify root metadata or witness membership.

The root payload is described by `witness-trust-root-payload.v1.schema.json`. Its signatures are verified against the previously pinned root metadata. Every successor root must set `previous_trust_root_digest` to the digest of the currently trusted root and increment `trust_root_version` by exactly one.

### Root rotation

Follow TUF's double-threshold rotation pattern:
- root version advances exactly by one;
- the new root metadata is signed by a threshold of keys from the previously trusted root;
- the same new metadata is also signed by a threshold of keys declared by the new root;
- count each unique key ID at most once toward each threshold;
- verify the predecessor root digest and reject skipped versions, rollback or an untrusted keyset.

If the prior root threshold is compromised, a normal online rotation is no longer trustworthy. Emergency recovery requires an independently authenticated, out-of-band ceremony and a new pinned trust-root digest. This is an explicit governance event, not a database restore option.

### Witness-set rotation

Witness membership metadata is signed by the trusted root threshold and has a monotonic version and predecessor digest. The new set must satisfy the quorum arithmetic and independence requirements before it is used. A removed/revoked member cannot contribute signatures after the new set is active.

Do not count the same key ID twice. Do not infer key validity from an envelope. V1 fixes the allowed algorithm to Ed25519; any future algorithm migration requires a versioned trust-root change and test vectors.

## Restore/recovery state machine

Compare the store's observed values against the latest challenge-bound quorum response and its checkpoint chain.

| Condition | Required result |
|---|---|
| Store incarnation, revision, state digest and policy digest equal the witnessed head | Freshness can be reported only after full signature, chain and trust-root verification |
| Store revision or digest is behind the witnessed head | `rejected_anti_rollback`; optionally rebuild by replaying independently retained transition receipts |
| Store revision appears higher but has no matching witness chain | Reject as unanchored; never trust the larger local integer |
| Authority incarnation/epoch is older or differs unexpectedly | Reject; require explicit incarnation migration evidence |
| Witness set disagrees on sequence or digest | `rejected_witness_equivocation`; stop writes and investigate |
| Witness quorum unavailable, stale or cannot answer the fresh nonce | `anti_rollback_unknown`; fail closed |
| Trust root unknown, rolled back, expired under an enforced clock policy, or invalidly rotated | `rejected_trust_root`; no freshness verdict |
| Signature invalid, signer revoked, duplicate key ID, threshold not met | `rejected_signature`; no freshness verdict |
| Witness head is newer than the DB but immutable transition content is unavailable | Freshness failure is still detected; recovery is `recovery_incomplete`, not success |

The observed store revision and witness-anchored authority revision remain distinct fields in every result.

### Rebuilding a restored store

A possible architecture is to treat the witness-certified sequence as the authoritative commit log and databases as rebuildable projections:
- every checkpoint references the immutable receipt/content bundle required to reconstruct that state;
- a database is rebuilt by replaying the witnessed receipt sequence through the reference transition model;
- the rebuilt projection is compared against the latest canonical state digest;
- only after the quorum confirms the current head can the projection become available for authoritative reads.

This is a paradigm shift worth benchmarking, but it changes the commit latency and failure modes. The alternative—database-first commit followed by witness attestation—can leave an un-witnessed database advance after witness outage. Those two commit protocols must be modelled and compared explicitly before selecting one. Never acknowledge an authoritative transition merely because the local DB transaction committed when the witness claim is still indeterminate.

## Threat model and explicit limits

V1 aims to tolerate up to `f` Byzantine witness members under the `3f+1 / 2f+1` quorum assumptions, provided:
- no more than `f` members are compromised;
- failure domains are genuinely independent;
- honest witness state is durable and monotonic;
- cryptographic keys are not extracted from honest members;
- trust-root bootstrap and root rotation remain secure;
- the verifier's challenge nonce is unpredictable.

It does not protect against compromise of more than `f` members, compromise of the pinned trust-root threshold, a malicious verifier, or coordinated control of the required independent domains. These are explicit residual risks. Key rotation, backup/restore of witness members themselves, operator changes and emergency re-provisioning need separate audited ceremonies.

## Required negative controls

- restore an old complete DB snapshot and its old local witness pointer;
- replay an old DSSE head response against a new nonce;
- provide a stale but valid checkpoint chain to only one member;
- create two different checkpoints at the same sequence;
- break predecessor digest linkage;
- duplicate one key ID to attempt to satisfy a quorum twice;
- supply a revoked key or a key not in the pinned root;
- use an algorithm different from the trust metadata;
- roll back witness-set or trust-root metadata version;
- rotate root without both old-root and new-root thresholds;
- return a DB revision higher than the witnessed head with no chain proof;
- take the witness quorum offline;
- restore a witness member from an older local disk image;
- provide a quorum whose members share one administrative domain;
- omit immutable transition content for a witnessed head and attempt to claim recovery success.

Every negative control must be rejected or classified unknown/incomplete, never pass.

## Benchmark programme

Measure safety before efficiency:

- 0 false-freshness acceptances across all rollback/replay/fork/identity/policy/key mutants;
- 100% rejection of invalid signatures, duplicated signer identities and below-threshold certificates;
- 100% detection of conflicting signed heads under the declared threat model;
- exact append retries are idempotent; same sequence with a different payload digest is rejected;
- 100% agreement with a simple exhaustive reference model for small witness-set schedules;
- fresh challenge read results replay deterministically from nonce, member responses, set metadata and verifier build.

Then report p50/p95/p99 of checkpoint append, quorum read and challenge verification; bytes signed/transmitted; CPU; member outages; quorum-loss time; key-rotation duration; recovery and projection-rebuild time. Compare 4-member/3-of-4, 7-member/5-of-7 and geographically distributed failure domains. Do not compare a single local file write against a quorum round trip as equivalent durability.

## Implementation plan

1. Validate these schemas with an independent Draft 2020-12 implementation and semantic constraints.
2. Add canonical JCS test vectors and DSSE PAE test vectors.
3. Add a pure Rust verifier for canonical bytes, DSSE PAE, Ed25519 signatures, key IDs and threshold counting.
4. Add a deterministic witness-member state machine with crash/restart/fork/partition simulation.
5. Test read/write quorum intersection, replay prevention and key rotation against exhaustive small-state schedules.
6. Integrate with SQLite, PostgreSQL and etcd restore arms; later add Turso.
7. Only then deploy physically independent witness members and run recovery/failure campaigns.

No phase is complete because a schema parses. Cryptographic and physical claims require executed signatures, independently controlled keys, quorum records, attack tests and reproduced artifacts.
