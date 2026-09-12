# Replicator Safety Kernel — Exact Build and Runtime Identity v0.1

Status: **normative production-admission contract; not an admission record**

This document defines how the Replicator Safety Kernel (RSK) binds reviewed source and qualification evidence to the exact build artifact and runtime configuration that may serve positive replication authority.

It contains no physical replication mechanism, fabrication recipe, molecular design, biological implementation, or autonomous manufacturing path.

---

## 1. Constitutional purpose

RSK evidence is meaningful only if the process enforcing authority is demonstrably the process that the evidence describes.

The governing rule is:

```text
reviewed source
    != qualified source automatically
    != built artifact automatically
    != admitted artifact automatically
    != running authority process automatically
```

Positive authority requires an unbroken, verified chain across those boundaries.

A green source-level test or formal result MUST NOT authorize a different:

- commit or source tree;
- dependency lock;
- compiler/toolchain;
- feature set;
- build recipe;
- target architecture;
- linked dependency closure;
- capability/resource schema;
- risk policy;
- runtime configuration;
- executable artifact;
- deployment instance.

A mismatch freezes **new positive replication authority**. It is not, by itself, permission for destructive action.

---

## 2. Identity layers

RSK distinguishes at least five identities.

### 2.1 Source candidate identity

The exact reviewed source subject.

At minimum it binds:

- repository identity;
- Git commit;
- Git tree;
- `Cargo.lock` digest;
- `rust-toolchain.toml` digest;
- `flake.lock` digest when Nix participates in the build;
- RSK crate versions;
- selected workspace/build manifests;
- qualification profile/schema versions.

### 2.2 Build definition identity

The exact instructions and environment inputs used to transform source into an artifact.

At minimum it binds:

- build system and build-type/schema version;
- builder identity;
- target triple / architecture;
- feature set;
- relevant `RUSTFLAGS`, linker flags, profile and codegen settings;
- Nix derivation identity when used;
- resolved build dependencies/materials;
- external build parameters;
- sandbox/hermeticity claims with evidence status;
- build provenance statement digest.

### 2.3 Artifact identity

The exact output bytes / filesystem object to be admitted.

At minimum it binds:

- primary executable cryptographic digest;
- complete runtime closure identity where policy requires it;
- Nix output store path when applicable;
- NAR/content digest when applicable;
- SBOM digest;
- dependency/provenance statement digests;
- reproducibility evidence root;
- artifact signatures/attestations and trust snapshot.

### 2.4 Admitted release identity

A governed decision over one exact source/build/artifact/policy tuple.

It binds:

- all identities above;
- capability-schema digest/version;
- resource-accounting-schema digest/version;
- verified risk-policy digest/version;
- canonical RSK evidence schema versions;
- qualification receipt roots;
- formal/property/fuzz evidence roots required by the admission profile;
- release/admission decision identifier;
- validity/revocation/supersession state;
- required runtime-attestation profile.

### 2.5 Runtime instance identity

Evidence about the process that is actually serving authority now.

It binds, as applicable:

- admitted release identifier;
- executable/artifact digest;
- loaded policy/config/schema digests;
- runtime feature/configuration profile;
- host/process/container/VM identity according to the deployment profile;
- boot/runtime measurement evidence where required;
- runtime attestation freshness;
- current epoch / ledger head binding;
- runtime verifier policy version.

A valid artifact on disk is not sufficient if the running process or loaded configuration cannot be bound to it.

---

## 3. No single identifier substitutes for the chain

RSK MUST NOT treat any one of the following as complete production identity by itself:

- Git commit;
- Git tree;
- Cargo package version;
- Nix store path;
- container tag;
- image name;
- SBOM;
- CI run number;
- SLSA level string;
- provenance statement presence;
- code-signing certificate;
- TPM/TEE quote;
- Kubernetes deployment name;
- process path;
- hash of only the top-level executable.

Each can be useful evidence. None independently proves the full admitted release and runtime relationship.

---

## 4. Nix evidence semantics

Nix is a strong fit for RSK build identity, but its evidence must be interpreted precisely.

### 4.1 Store path is not always output-content identity

Ordinary input-addressed Nix outputs identify the derivation/input construction path. They do not, by themselves, substitute for an independent cryptographic digest of the resulting store object.

Therefore an admitted Nix build SHOULD bind both:

```text
how it was produced  -> derivation/build identity
what was produced    -> NAR/content/artifact digest
```

when the deployment profile can provide both.

### 4.2 Content-addressed derivations are not assumed

Floating content-addressed derivations remain an opt-in/experimental Nix feature in current Nix documentation. RSK MUST NOT silently assume that an arbitrary store path is content-addressed.

If a profile requires content-addressed Nix outputs, that requirement MUST be explicit and verified.

### 4.3 Nix path-info schema must be pinned

Nix's JSON output format evolves. Evidence generation MUST pin the parser/output schema version supported by the admitted toolchain rather than relying on an unversioned default.

Where used, the capsule SHOULD record at least:

- Nix version;
- requested JSON format version;
- store path;
- `narHash`;
- `narSize` where useful;
- signatures/content-address metadata where applicable;
- references / closure root;
- derivation or build recipe identity.

### 4.4 Closure identity

If dynamic/runtime dependencies can affect authority behavior, policy MUST define whether admission binds:

- only the executable;
- executable + selected dependencies;
- the complete Nix runtime closure;
- the entire VM/container/system image.

This is a deployment-profile decision and cannot be left implicit.

---

## 5. Build provenance

RSK SHOULD represent build provenance using an interoperable attestation model such as in-toto/SLSA where practical.

Current SLSA 1.2 provenance separates:

- output subject;
- build definition;
- external parameters;
- resolved dependencies;
- builder/run details.

RSK uses that structure as **supply-chain evidence**, not as replication authority.

### 5.1 Builder identity means trust boundary

A `builder.id` or equivalent identifier MUST correspond to a defined security boundary, not merely a human-readable CI service name.

The admission policy SHOULD know:

- who controls the builder;
- what credentials/control plane can affect it;
- isolation properties;
- which inputs are external/untrusted;
- provenance signing identity;
- builder lifecycle/revocation state;
- relevant failure domain.

### 5.2 External parameters are not implicitly trusted

A provenance statement faithfully recording attacker-controlled build parameters is still useful provenance, but it does not make those parameters acceptable.

Downstream admission MUST verify external parameters against the release profile.

### 5.3 Provenance is evidence, not permission

Even valid signed provenance cannot override:

- quarantine;
- revocation;
- stale/forked ledger state;
- unverified policy;
- expired admission;
- containment failure;
- runtime identity mismatch;
- trusted-time failure;
- missing required reproducibility evidence.

---

## 6. Independent reproducibility

For high-consequence production profiles, one successful build is insufficient evidence against builder compromise or nondeterministic output substitution.

A reproducibility record SHOULD distinguish:

```text
number of build executions
number of distinct builder identities
number of distinct failure domains
```

Two builds on the same compromised control plane are not automatically independent.

### 6.1 Strong comparison target

Preferred comparison is the strongest stable content identity the build profile supports, for example:

- executable SHA-256;
- NAR hash;
- canonical image digest;
- closure manifest digest.

If bit-for-bit reproduction is impossible for a justified reason, the profile MUST explicitly define the normalization/comparison rule and its residual risk. “Semantically equivalent” is not an acceptable undefined fallback.

### 6.2 Disagreement

Rebuild disagreement MUST produce a blocked/non-operational admission result until investigated.

The system MUST NOT:

- majority-vote arbitrary binaries into production;
- choose the newest artifact;
- choose the artifact from the preferred builder;
- ignore unexplained nondeterminism;
- accept the first signed output.

---

## 7. Admission capsule

An admitted release is represented conceptually as:

```text
AdmittedReleaseCapsule {
    schema_version,
    source_identity,
    qualification_evidence_root,
    build_definition_identity,
    provenance_root,
    artifact_identity,
    sbom_root,
    reproducibility_evidence,
    capability_schema_digest,
    resource_schema_digest,
    risk_policy_digest,
    evidence_schema_set,
    runtime_attestation_profile,
    admission_policy_digest,
    admission_decision_id,
    issued_at,
    expires_at,
    supersedes,
}
```

The canonical capsule digest becomes the release identity consumed by runtime admission.

The capsule MUST be verified/authenticated under the release/admission trust policy before it can become positive authority evidence.

---

## 8. Runtime admission

A production RSK process MUST begin in a state equivalent to:

```text
PositiveAuthorityEnabled = false
```

It may enable positive authority only after the runtime-admission verifier proves that the running instance matches one current admitted release capsule and all other constitutional prerequisites are satisfied.

### 8.1 Required equality

Conceptually:

```text
runtime.executable_digest == admitted.artifact.executable_digest
runtime.policy_digest     == admitted.risk_policy_digest
runtime.capability_schema == admitted.capability_schema_digest
runtime.resource_schema   == admitted.resource_schema_digest
runtime.release_id        == admitted.release_id
```

plus every additional field required by the deployment profile.

### 8.2 Runtime configuration is part of identity

Security-sensitive configuration MUST NOT be treated as an unrelated operational detail.

Examples include:

- quorum floors;
- accepted trust roots;
- monitor source policy;
- time-source policy;
- containment profile;
- capability schema registry;
- resource accounting schema;
- recovery-policy roots;
- feature flags that change authority semantics.

Changing such configuration either selects another already-admitted profile or creates a new admission lineage.

### 8.3 Loaded bytes, not filename claims

Where feasible, runtime evidence SHOULD identify the bytes actually mapped/executed, not merely the path from which the process claims to have started.

Platform mechanisms may include, depending on deployment profile:

- measured boot;
- TPM-backed measurements;
- fs-verity/IMA;
- signed container/VM image digests;
- launcher-supervisor measurements;
- independent local attestation agents.

No one mechanism is universally required by this v0.1 contract. The profile must state what it trusts and why.

---

## 9. Rollback protection

An older artifact can be perfectly signed and byte-identical to a historically admitted release while being invalid **now**.

Runtime admission MUST therefore bind freshness/supersession state, not only artifact signature validity.

Rollback defenses include:

- release sequence / epoch;
- current admission-policy state;
- revocation/supersession records;
- trusted-time/freshness evidence;
- ledger epoch/head binding;
- minimum admitted release version where policy requires it.

A rollback MUST NOT restore:

- expired grants;
- revoked policy;
- old recovery authority;
- superseded trust roots;
- old ledger cursors;
- previously cleared incident state.

---

## 10. Evidence-lineage reset on drift

Once qualification/admission evidence starts for candidate `C`, any authority-relevant environment drift creates candidate `C'`.

Examples:

- source/tree change;
- lockfile change;
- Rust/Nix/toolchain change;
- feature/profile change;
- link flag change;
- dependency closure change;
- policy/schema change;
- build-type/provenance schema change;
- runtime-attestation profile change.

Evidence collected for `C` MUST NOT be silently mixed with evidence for `C'`.

A new lineage may reference old evidence as historical context, but promotion requires the current candidate's required gates to be satisfied explicitly.

---

## 11. Build and release trust separation

The following authorities SHOULD be separable where risk warrants it:

- source/change approval;
- qualification executor;
- build executor;
- provenance signer;
- reproducibility witness;
- release/admission authority;
- deployment authority;
- runtime attestation verifier;
- recovery authority.

A production profile MUST document any collapsed roles and the resulting common-mode risk.

No builder, CI runner, artifact registry, Nix cache, transparency service, or runtime attester may unilaterally mint replication authority merely because it controls one stage.

---

## 12. Transparency and external receipts

Admission/provenance artifacts MAY be included in transparency systems.

Transparency is valuable for:

- equivocation detection;
- auditability;
- incident reconstruction;
- externally witnessed checkpoints.

But inclusion or receipt presence is evidence that a statement was recorded, not proof that the statement was safe or authorized.

A transparency receipt MUST NOT override local negative facts or missing admission predicates.

---

## 13. Failure semantics

The following conditions deny new positive replication authority:

- no admitted release capsule;
- capsule verification failure;
- artifact digest mismatch;
- runtime executable mismatch;
- policy/config/schema mismatch;
- stale or superseded admission;
- unverified provenance when required;
- unsigned/untrusted build provenance when required;
- missing mandatory build input;
- reproducibility disagreement;
- unadmitted target architecture/profile;
- runtime attestation unavailable/stale when required;
- rollback evidence;
- unknown evidence schema/profile;
- evidence lineage mixing after drift.

Denial/freeze is the default consequence. Destructive response remains a separate governance decision.

---

## 14. Relationship to RSK qualification receipts

`RSK_QUALIFICATION_CAPSULE_V0_1` and its independent verifier establish evidence that declared Rust gates ran for one exact source subject/toolchain/input state.

This contract consumes those receipts as one input to release admission.

The transition is:

```text
verified qualification receipt
    -> one admission evidence input
    != admitted release by itself
```

The admitted release additionally requires build, artifact, policy/schema, reproducibility, governance, and runtime-identity evidence according to policy.

---

## 15. Relationship to existing Symthaea patterns

The repository already uses an operational-resilience pattern that distinguishes:

- canonical build recipes/materials/SBOMs;
- independent build replicas;
- build output agreement;
- runtime observations;
- authenticated release evidence;
- fail-closed release assessment.

RSK SHOULD reuse that architectural shape where practical instead of inventing incompatible supply-chain semantics.

RSK nevertheless maintains its own constitutional rule:

```text
supply-chain evidence != positive replication authority
```

---

## 16. External standards alignment

This contract is designed to compose with, without depending constitutionally on:

- SLSA v1.2 provenance;
- in-toto attestation envelopes/statements;
- Nix derivation/store/NAR identity;
- SBOM formats such as SPDX or CycloneDX;
- Sigstore-style signing/transparency systems;
- TPM/TEE or OS measurement systems.

Exact technologies remain deployment-policy choices.

The constitutional semantics depend only on verified identities and their relationships, not on one vendor or cryptographic ecosystem.

---

## 17. Production-admission invariants

The production implementation and formal/test suites SHALL target at least:

1. `QualifiedSourceDoesNotImplyAdmittedArtifact`
2. `ArtifactDigestMustMatchAdmittedRelease`
3. `RuntimeMustMatchAdmittedArtifact`
4. `RuntimePolicyMustMatchAdmittedPolicy`
5. `RuntimeSchemasMustMatchAdmittedSchemas`
6. `UnverifiedProvenanceCannotAdmit`
7. `RebuildDisagreementBlocksAdmission`
8. `StaleAdmissionCannotAuthorize`
9. `SupersededReleaseCannotRegainAuthorityByRollback`
10. `EnvironmentDriftStartsNewEvidenceLineage`
11. `NixStorePathAloneIsInsufficientWhenContentDigestRequired`
12. `TransparencyReceiptCannotMintAuthority`
13. `RuntimeIdentityFailureCannotClearNegativeFacts`
14. `BuildSignerCannotSelfPromoteToRecoveryAuthority`
15. `AdmissionEvidenceCannotCrossEpochImplicitly`

---

## 18. Production admission gate

This document does not admit any artifact.

RSK remains:

```text
Production admission status: DENIED / NOT YET ELIGIBLE.
```

until the separately governed admission process records executed evidence for the exact candidate artifact/runtime profile and every required gate.