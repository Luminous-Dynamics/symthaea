# Replicator Safety Kernel — Verified Evidence Test Plan v0.1

Status: **normative verification plan; authored design evidence only**

This plan defines the minimum deterministic, black-box, cryptographic, trust-lifecycle, failure-domain, replay, and property evidence required for the RSK verified-positive-evidence boundary.

It contains no physical replication mechanism.

---

## 1. Test-family goals

The verified-evidence implementation is acceptable only if it proves two independent statements:

1. **invalid/untrusted positive claims cannot become capability-bearing verified values**;
2. **even valid verified positive capabilities cannot override current hard negative facts or inherited constitutional constraints**.

A valid signature alone is never sufficient evidence of either statement.

---

## 2. Canonical encoding vectors — VE-CAN

Required vectors:

- minimal valid grant payload;
- maximal bounded valid grant payload;
- minimal/maximal approval payload;
- monitor observation payload;
- risk policy snapshot;
- trust snapshot with one and many signer records;
- stable deterministic ordering of signer/failure-domain records;
- every authority-relevant field flipped one at a time changes digest;
- unknown critical tag rejects;
- duplicate canonical field/record rejects where forbidden;
- alternate/noncanonical integer/length/order encoding rejects;
- over-limit message/signature/key ID/approval/domain count rejects before large allocation.

Golden vectors must include exact canonical bytes and digest outputs tied to schema version.

---

## 3. Signature/profile vectors — VE-SIG

For each admitted signature profile:

- valid signature accepted;
- one-bit message mutation rejects;
- one-bit signature mutation rejects;
- wrong key rejects;
- wrong profile rejects;
- unknown/disallowed profile rejects;
- empty signature rejects;
- oversized signature rejects;
- profile/key encoding mismatch rejects;
- signature over another evidence-family domain-separation prefix rejects.

Algorithm-specific tests supplement but do not replace common envelope tests.

---

## 4. Trust snapshot/lifecycle vectors — VE-TRUST

Required cases:

- fresh valid snapshot accepted;
- sequence zero/invalid epoch rejected;
- issued-at >= expiry rejected;
- snapshot not-yet-valid/expired rejected;
- sequence rollback rejected;
- same sequence/different digest collision rejected;
- issued-at regression rejected when policy forbids it;
- predecessor mismatch rejected where chaining is enabled;
- unknown key rejected;
- key not-yet-valid rejected;
- key expired rejected;
- key retired rejected for new positive authority;
- key revoked rejected;
- wrong role/usage rejected;
- duplicate canonical key rejected;
- legitimate key rotation succeeds;
- revoked key cannot be revived by replaying older snapshot;
- restart cannot forget latest accepted snapshot without anti-rollback failure/freeze.

---

## 5. Grant verification vectors — VE-GRANT

Required cases:

- valid grant produces opaque `VerifiedReplicationGrant`;
- raw `ReplicationGrant`/signed payload cannot call production evaluator directly;
- wrong subject rejects;
- wrong lineage rejects;
- wrong generation rejects;
- invalid/not-yet-valid/expired window rejects;
- wrong safety-case digest rejects;
- wrong containment digest rejects;
- wrong policy/trust epoch rejects;
- unknown capability schema rejects;
- unknown resource-accounting schema rejects;
- invalid/overflowing ceilings reject;
- signer lacks grant-issuer role rejects;
- signer is forbidden subject-controlled domain rejects when external independence required;
- signer later revoked makes grant unusable according to current revocation policy;
- verified grant cannot be deserialized from raw bytes without re-verification.

Threat coverage: T01, T05, T11, T17.

---

## 6. Quorum/failure-domain vectors — VE-QUORUM

Required cases:

- valid independent quorum produces opaque `VerifiedIndependentQuorum`;
- same approval duplicated rejects/de-counts;
- same key duplicated rejects;
- two keys mapped to one signer identity cannot count as two identities;
- two identities in same required admin domain fail a distinct-admin requirement;
- two identities in same required key-root domain fail a distinct-root requirement;
- missing required failure-domain metadata fails closed;
- signer in requesting subject's forbidden control domain cannot satisfy external-independent requirement;
- revoked/retired/wrong-role signer cannot count;
- approval payload mismatch cannot combine;
- approval expiry intersection enforced;
- stronger verified policy raises threshold;
- requester self-restriction can raise but never lower threshold;
- R4/R5 constitutional floors cannot be lowered by policy input;
- raw numeric approval count cannot enter production evaluator as verified quorum.

Threat coverage: T02, T03, T04, T11, T17.

---

## 7. Runtime monitor vectors — VE-MON

Required cases:

- valid monitor observation produces opaque verified witness;
- raw booleans/digests cannot enter production evaluator as trusted witness;
- wrong monitor identity/role rejects;
- signature/attestation failure rejects;
- stale observation rejects;
- observation sequence rollback rejects;
- wrong subject/scope rejects;
- wrong containment digest rejects;
- wrong safety-case/policy digest rejects;
- monitor revoked/retired rejects;
- missing required failure-domain metadata fails closed;
- two common-domain monitors cannot satisfy required independent-monitor policy;
- verified negative monitor evidence vetoes otherwise valid positive evidence;
- monitor unavailability freezes authority rather than widening it.

Threat coverage: T14, T15, T16, T29.

---

## 8. Verified policy vectors — VE-POLICY

Required cases:

- valid policy snapshot produces opaque `VerifiedRiskPolicySnapshot`;
- requester cannot construct a verified policy;
- invalid policy authority signature rejects;
- stale/expired policy rejects;
- policy sequence/version rollback rejects;
- same version/different digest collision rejects;
- superseded policy cannot regain authority through replay;
- policy below constitutional floor is clamped/rejected according to frozen semantics;
- stronger policy dominates requester values;
- unknown capability/resource/signature profile registry rejects;
- policy authority wrong role/lifecycle rejects;
- policy digest retained in downstream verified capabilities.

Threat coverage: T17, T18, T27.

---

## 9. Negative-state composition vectors — VE-NEG

For each fully valid verified-positive bundle, independently inject:

- subject quarantine;
- subject revocation;
- ancestor quarantine/revocation;
- lineage quarantine/revocation;
- grant revocation after verification;
- stale cursor/head;
- fork/non-operational state;
- expired trusted time;
- containment drift/current negative monitor evidence;
- unadmitted build/runtime identity.

Expected result in every case: **no new positive replication authority**.

This is the key composition theorem preventing verified positive evidence from becoming an override channel.

---

## 10. Durable replay vectors — VE-REPLAY

Once canonical durable evidence exists:

- persist raw signed grant/quorum/monitor/policy/trust evidence;
- restart;
- prove trusted capability is reconstructed only by verification;
- attempt to persist/deserialize a fake `Verified*` representation -> rejected/nonexistent API;
- replay historical grant whose signer is now revoked -> no current positive authority;
- replay historical snapshot older than latest accepted -> rollback/freeze;
- replay valid historical audit evidence remains auditable even when no longer authorizing;
- checkpoint replay and full replay produce equivalent accepted trust/authority state.

Threat coverage: T10, T11, T18, T19, T27.

---

## 11. API-surface compile tests — VE-API

Use compile-fail/API-review tests where practical to establish:

- verified type fields are private;
- no public `new_unchecked` or equivalent exists in production feature set;
- no `Deserialize` directly into verified capability;
- raw signed evidence cannot satisfy production evaluator trait/type requirements;
- verification reports cannot be converted into verified capability without verifier success;
- failure-domain facts cannot be set by an approval object and automatically trusted;
- reference-only constructors are unavailable from production adapter feature/profile.

A security boundary that relies only on caller discipline is insufficient.

---

## 12. Property/state-machine tests — VE-PROP

Generate bounded histories over:

- trust snapshots/rotation/revocation;
- signer identities and key rotations;
- policy versions;
- grants/generations;
- approvals and failure-domain mappings;
- monitor observations/sequences;
- negative facts;
- time/freshness;
- ledger cursor changes.

Global properties:

1. no raw value becomes trusted without verifier success;
2. verified capability provenance always names an accepted trust/policy snapshot;
3. retired/revoked/wrong-role signer never supports new positive authority;
4. duplicate identity/failure-domain collapse never increases policy-defined independent count;
5. trust/policy rollback never restores authority;
6. verified positive bundle never overrides current hard negative state;
7. verified capability validity never extends beyond the earliest required evidence/policy/trust expiry;
8. replay after restart never trusts serialized capability state directly;
9. requester-controlled inputs can only narrow, never widen, trusted policy requirements.

---

## 13. Fault/availability tests — VE-FAULT

Inject:

- signature verification provider unavailable;
- trust snapshot source unavailable;
- revocation source stale/unavailable;
- policy source unavailable;
- one quorum signer unavailable;
- one required failure-domain metadata source unavailable;
- monitor unavailable;
- time/continuity source unavailable;
- storage restart during trust rotation.

Expected posture:

- no implicit trust relaxation;
- existing verified evidence usable only within pre-existing validity/policy;
- missing required freshness/independence freezes new authority;
- unavailability does not authorize destructive action by itself.

---

## 14. Fuzzing/bounds — VE-FUZZ

Target all bounded decoders/verifiers with:

- arbitrary bytes;
- truncation at every byte offset;
- length-prefix extremes;
- unknown discriminants;
- duplicate/sortedness violations;
- deeply repeated records within and beyond bounds;
- large signature/key/domain fields;
- invalid UTF-8 if textual identifiers exist;
- semantically invalid but structurally canonical messages.

Required properties:

- no panic/UB;
- bounded memory/CPU according to declared limits;
- malformed input cannot produce verified capability;
- unknown critical schema/profile/role/domain kind fails closed.

---

## 15. Exact evidence subject

Every executed verification run intended as production evidence must bind:

- exact Git commit/tree;
- Cargo.lock;
- Rust/Nix/toolchain identity;
- features/target/config;
- crypto provider/profile versions;
- schema/registry versions;
- trust/policy test fixtures and digests;
- test seeds/model bounds;
- artifact/runtime digest where applicable.

This prevents T27 from separating passing evidence from the deployed verifier.

---

## 16. Implementation promotion gate

Do not promote a verifier implementation merely because the happy path works.

Minimum promotion requires green executed evidence for:

- VE-CAN;
- VE-SIG;
- VE-TRUST;
- VE-GRANT;
- VE-QUORUM;
- VE-MON;
- VE-POLICY;
- VE-NEG;
- VE-API;
- applicable VE-PROP/VE-FAULT/VE-FUZZ;
- VE-REPLAY once durable persistence exists.

**Production admission remains DENIED / NOT YET ELIGIBLE.**
