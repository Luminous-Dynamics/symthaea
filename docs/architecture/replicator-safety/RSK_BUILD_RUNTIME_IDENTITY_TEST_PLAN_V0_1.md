# Replicator Safety Kernel — Build/Runtime Identity Test Plan v0.1

Status: **normative verification plan; authored evidence only until executed**

This document defines adversarial verification families for the exact build/artifact/runtime identity boundary.

It contains no physical replication mechanism.

---

## 1. Goal

Prove that positive RSK authority is available only when the running enforcement process matches the exact admitted source/build/artifact/policy/schema profile.

The tests are organized by stable family identifiers so code, property tests, formal models, incident reports, and admission records can cite the same requirement.

---

## 2. BI-SOURCE — source candidate identity

Required cases:

- BI-SOURCE-001: exact Git commit/tree accepted when all other evidence matches.
- BI-SOURCE-002: different commit with same package version rejected.
- BI-SOURCE-003: same commit claim with different tree rejected.
- BI-SOURCE-004: `Cargo.lock` substitution rejected.
- BI-SOURCE-005: Rust toolchain manifest substitution rejected.
- BI-SOURCE-006: `flake.lock` substitution rejected when Nix participates.
- BI-SOURCE-007: qualification receipt for another source subject rejected.
- BI-SOURCE-008: qualification receipt with valid digest but wrong candidate lineage rejected.

Invariant:

```text
source-label equality != source-content equality
```

---

## 3. BI-BUILD — build-definition identity

Required cases:

- target triple substitution;
- Cargo feature enable/disable substitution;
- `RUSTFLAGS` substitution;
- linker/profile/codegen substitution;
- build recipe/build-type substitution;
- resolved dependency substitution;
- builder identity substitution;
- untrusted external parameter outside admitted policy;
- missing mandatory provenance field;
- unsupported provenance schema;
- provenance statement for correct source but different output.

Property:

```text
same source + different authority-relevant build definition != same admitted candidate
```

---

## 4. BI-NIX — Nix identity semantics

Required cases:

- BI-NIX-001: store path + expected NAR hash accepted under matching profile.
- BI-NIX-002: correct store path claim + wrong NAR hash rejected.
- BI-NIX-003: same content claim + wrong derivation/build identity rejected when derivation identity is required.
- BI-NIX-004: input-addressed store path is never treated as content hash by type/API confusion.
- BI-NIX-005: profile requiring content-addressed output rejects ordinary input-addressed evidence.
- BI-NIX-006: unsupported/unpinned `path-info` JSON format rejected.
- BI-NIX-007: parser does not silently accept unknown identity-critical fields/shape changes.
- BI-NIX-008: dependency closure mismatch rejected when closure binding is required.
- BI-NIX-009: binary-cache signature alone cannot replace artifact/content identity.
- BI-NIX-010: Nix version/profile mismatch starts a new evidence lineage.

---

## 5. BI-ART — artifact identity

Required cases:

- exact executable digest accepted under admitted release;
- one-byte artifact mutation rejected;
- valid signature over unadmitted artifact rejected;
- admitted artifact with missing SBOM/provenance required by profile rejected;
- SBOM digest substitution rejected;
- provenance subject digest different from measured artifact rejected;
- executable matches but required runtime closure differs -> rejected;
- executable and closure match but architecture/profile differs -> rejected.

Where multiple digest algorithms exist, algorithm identifiers are part of the typed identity; digest bytes are never interpreted without their algorithm/profile.

---

## 6. BI-REPRO — independent rebuild evidence

Required cases:

- two genuinely independent builders produce identical required content identity -> reproducibility predicate may pass;
- two executions on one builder do not satisfy a two-domain requirement;
- two builders under one administrative/control plane do not automatically satisfy independence;
- build outputs disagree -> admission blocked;
- three builders with 2-vs-1 output split -> admission remains blocked unless policy explicitly defines a separately justified dispute process;
- builder provenance missing failure-domain metadata -> independence requirement fails closed;
- reproducibility evidence for another source/build definition rejected;
- normalization profile drift rejected.

The test suite MUST ensure there is no default majority-vote admission rule.

---

## 7. BI-ADMIT — admitted release capsule

Required cases:

- exact capsule canonical digest verifies;
- any field mutation invalidates canonical digest/signature;
- unknown schema version rejected;
- missing required identity root rejected;
- stale admission rejected;
- revoked admission rejected;
- superseded admission rejected where profile forbids rollback;
- release capsule for another ledger/recovery epoch rejected;
- old qualification evidence cannot be mixed with post-drift build evidence unless the new profile explicitly re-qualifies it;
- release signer role cannot be inferred from generic signing capability.

---

## 8. BI-RUNTIME — running instance identity

Required cases:

- measured executable + policy + schemas match admitted release -> runtime identity predicate may pass;
- executable mismatch -> positive authority denied;
- policy mismatch -> denied;
- capability schema mismatch -> denied;
- resource accounting schema mismatch -> denied;
- runtime profile/feature mismatch -> denied;
- runtime attestation stale -> denied;
- runtime attestation unavailable where required -> denied;
- process filename/path matches but measured bytes differ -> denied;
- image/container tag matches but digest differs -> denied;
- current runtime is admitted but local quarantine exists -> quarantine still dominates;
- valid runtime attestation cannot clear revocation, stale cursor, fork state, or expiry.

---

## 9. BI-ROLLBACK — rollback/supersession

Required histories:

```text
admit A -> run A -> admit B superseding A -> attempt run A
```

Expected: A cannot regain positive authority when policy makes B's supersession monotonic.

Other cases:

- rollback to historically valid signing key;
- rollback to historically valid policy;
- rollback to old trust snapshot;
- rollback to old capability/resource schema;
- rollback to old ledger epoch/head;
- rollback after recovery into a fresh epoch;
- clock rollback combined with old release replay.

The result MUST be deny/freeze, not destructive action.

---

## 10. BI-DRIFT — evidence-lineage drift

Generate candidate histories where exactly one authority-relevant field changes after evidence begins.

Fields include:

- source tree;
- dependency lock;
- Rust toolchain;
- Nix/flake lock;
- Cargo feature set;
- compiler/link flags;
- target triple;
- build type;
- policy digest;
- capability schema;
- resource schema;
- runtime-attestation profile.

Property:

```text
change -> new candidate lineage
```

The verifier MUST reject a synthetic capsule that combines evidence from both sides as if it described one candidate.

---

## 11. BI-PROV — provenance authenticity and scope

Required cases:

- valid signed provenance from allowed builder identity;
- valid signature from wrong/untrusted builder;
- revoked builder key;
- stale trust snapshot;
- provenance signed after builder role expiry;
- statement subject does not match artifact;
- source material does not match admitted source;
- external parameter outside policy;
- duplicate/conflicting provenance statements;
- provenance schema parse ambiguity;
- unknown critical algorithm/profile.

The suite MUST prove:

```text
valid signature != admitted build
```

---

## 12. BI-TRANS — transparency non-authority

Required cases:

- valid transparency receipt for unadmitted artifact -> denied;
- valid transparency receipt for revoked release -> denied;
- transparency log equivocation/fork -> admission incident/freeze where policy requires;
- missing transparency receipt where optional -> does not create positive evidence;
- missing required transparency receipt -> deny admission;
- receipt cannot replace provenance signature/trust verification.

---

## 13. BI-CONFIG — configuration identity

Authority-relevant configuration mutation tests must include:

- lower quorum floor;
- different trust root;
- different monitor policy;
- different time policy;
- different containment profile;
- different recovery root;
- altered capability interpretation;
- altered resource accounting interpretation.

A runtime with correct executable but wrong authority configuration MUST be denied.

---

## 14. BI-FAULT — fault/crash tests

Once durable admission storage exists:

- crash during capsule persistence;
- crash between artifact promotion and capsule commit;
- crash between capsule commit and runtime activation;
- torn write of release state;
- stale cache serving superseded capsule;
- registry/network partition during runtime revalidation;
- corrupted local artifact cache;
- disk rollback/snapshot restore.

Recovery MUST never create a state in which an artifact is treated as admitted without a complete committed admission record.

---

## 15. BI-PROP — generated state-machine histories

Generate arbitrary sequences over:

```text
source_change
qualify
build
attest
rebuild
admit
supersede
revoke
deploy
restart
config_change
policy_change
schema_change
rollback
fork
recover
```

Properties include:

- no runtime positive authority without a current matching admitted release;
- no admission from mismatched artifact/build/source identities;
- supersession/rollback constraints preserved;
- drift starts new evidence lineage;
- negative facts dominate matching build/runtime evidence;
- recovery epoch does not inherit old admission implicitly.

---

## 16. BI-FORMAL — formal refinement targets

The RSK v0.2+ formal model should include abstract variables for:

- source candidate;
- build definition;
- artifact identity;
- admission release ID;
- runtime release ID;
- supersession/revocation state;
- current epoch;
- environment-drift generation.

Target invariants:

```text
RuntimeAuthorityImpliesCurrentAdmission
RuntimeIdentityMatchesAdmission
DriftInvalidatesPriorEvidenceLineage
SupersededReleaseCannotAuthorize
ArtifactMismatchCannotAuthorize
PolicyMismatchCannotAuthorize
RecoveryCarriesNoAdmissionImplicitly
```

---

## 17. BI-API — type/API misuse tests

Production APIs should make category confusion difficult.

Compile-fail or equivalent tests SHOULD prove that code cannot freely substitute:

- `GitCommit` for `ArtifactDigest`;
- `NixStorePath` for `NarHash`;
- raw provenance for `VerifiedBuildProvenance`;
- raw admission record for `VerifiedAdmittedRelease`;
- file path for `VerifiedRuntimeIdentity`;
- historical admission for current runtime capability.

Opaque verified types should have private construction and no trusted deserialization shortcut.

---

## 18. BI-FUZZ — parser/verifier fuzzing

Fuzz targets should include:

- admitted release capsule parser;
- SLSA/in-toto adapter;
- Nix path-info adapter;
- SBOM adapter;
- runtime attestation envelope;
- canonical digest encoder;
- artifact manifest/closure parser.

Required properties:

- bounded allocations;
- no path traversal;
- no duplicate-key ambiguity;
- no numeric overflow;
- no unknown-schema permissive admission;
- malformed evidence never becomes positive authority.

---

## 19. Evidence hierarchy

Passing these tests does not itself establish production safety.

Executed evidence should eventually include:

1. unit/integration results;
2. generated/property histories;
3. fuzz evidence;
4. formal bounded-model results;
5. independent reproducibility evidence;
6. signed provenance and release evidence;
7. deployment/runtime-attestation evidence;
8. branch/release governance enforcement evidence.

Each item binds to the exact candidate lineage.

---

## 20. Current status

These are target tests for the production-admission implementation.

No build/runtime identity implementation is introduced by this document.

Production admission remains **DENIED / NOT YET ELIGIBLE**.