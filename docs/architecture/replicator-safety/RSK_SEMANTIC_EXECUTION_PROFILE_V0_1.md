# Replicator Safety Kernel — Semantic Execution Profile v0.1

Status: **normative semantic/runtime-identity contract; not a production admission record**

This contract defines the compact identity that binds RSK semantic schemas to the exact validator interpretation and representation/arithmetic profiles used by an admitted runtime.

It contains no physical replication mechanism, fabrication recipe, biological or molecular design, physical resource model, or autonomous manufacturing path.

---

## 1. Problem

A source schema digest is necessary but insufficient runtime identity.

The same semantic schema can be interpreted by different runtime representations or arithmetic engines. A runtime can also carry the correct schema digest while using a weakened or caller-constructed validation table.

The governing distinction is:

```text
valid schema identity
    != deterministic validator interpretation
    != admitted runtime representation/arithmetic semantics
```

For positive authority, all three must agree.

---

## 2. Semantic execution profile

The v0.1 profile is conceptually:

```text
SemanticExecutionProfile {
    schema,
    capability: {
        schema_id,
        representation_profile,
        validator_rule_table_sha256,
    },
    resource: {
        schema_id,
        representation_profile,
        validator_rule_table_sha256,
    },
}
```

The canonical identity is:

```text
SemanticExecutionProfileId = SHA256(canonical(SemanticExecutionProfile))
```

The reference schema tag is:

```text
symthaea.rsk.semantic-execution-profile.v1
```

The profile is deliberately compact. It commits the semantic source identity, the runtime interpretation profile, and the deterministic validator image without embedding the full schemas or rule tables into every admission record.

---

## 3. Current Rust constituent profiles

The current reference target composes:

```text
capability representation:
  symthaea.rsk.capability-representation.rust-u64.v1

resource representation/arithmetic:
  symthaea.rsk.resource-representation.rust-u64-sum-exact.v1
```

These profile identifiers are part of semantic execution identity.

A future wider capability vector, wider resource quantity, different aggregation engine, conversion-aware fixed-point implementation, or different rounding semantics is a distinct execution profile even if some source schemas remain unchanged.

---

## 4. Derivation, not declaration

A runtime or caller MUST NOT mint positive semantic evidence by supplying a profile identifier or constituent hashes directly.

The production derivation path is conceptually:

```text
VerifiedSchemaRegistrySnapshot
    -> exact canonical capability schema bytes
    -> exact canonical resource schema bytes
    -> recompute schema IDs
    -> enforce admitted runtime representation profiles
    -> deterministically derive validator rule tables
    -> hash validator rule tables canonically
    -> construct SemanticExecutionProfile
    -> hash profile canonically
    -> compare with admitted release
```

There is no production shortcut:

```text
caller_claimed_profile_id -> trusted semantic execution identity
```

or:

```text
correct schema digest + arbitrary local validator rules -> trusted validator
```

---

## 5. Constituent validator digests

The capability and resource `validator_rule_table_sha256` fields bind the exact deterministic structural-validator configuration derived from canonical schema bytes.

For the current reference adapters, these tables include the admitted runtime representation/arithmetic profile identifiers themselves.

This provides two useful checks:

1. the source schema has the expected meaning;
2. the runtime derives the same executable validator interpretation from that meaning.

A source schema digest match with a validator-table digest mismatch is a semantic execution failure.

---

## 6. Registry provenance remains separate

`SemanticExecutionProfileId` is not a registry signature, trust snapshot, governance approval, or freshness proof.

The semantic execution profile answers:

```text
What exact schema-derived semantics does this runtime execute?
```

Verified registry evidence answers:

```text
Under what trusted provenance/policy context were these exact schemas accepted?
```

Production admission requires both.

The capability and resource schemas used to derive one production profile MUST be resolved under one coherent verified registry snapshot/policy context. A caller MUST NOT opportunistically mix independently accepted schemas from incompatible registry epochs or policy states.

The admitted release SHOULD therefore bind both:

- `semantic_execution_profile_id`;
- the verified registry snapshot/trust evidence required by the release policy.

A registry key rotation or fresh signature over unchanged schema bytes does not, by itself, change semantic meaning. Conversely, unchanged trust provenance does not excuse a semantic-profile mismatch.

---

## 7. Relationship to exact build/runtime identity

`RSK_BUILD_AND_RUNTIME_IDENTITY_V0_1` defines the broader source -> build -> artifact -> admitted release -> runtime identity chain.

The semantic execution profile is one authority-relevant constituent of that chain.

An admitted release capsule should bind, at minimum:

```text
semantic_execution_profile_id
```

in addition to artifact, source, toolchain, policy, provenance, evidence, and runtime-attestation identities.

The running process must independently derive or be independently measured against the same semantic execution profile before new positive authority can be enabled.

A binary digest match is insufficient if authority-relevant schemas or their execution profiles differ from the admitted release.

A semantic profile match is insufficient if the binary/artifact/runtime identity does not match the admitted release.

---

## 8. Runtime admission comparison

Conceptually:

```text
runtime_derived_semantic_execution_profile_id
    == admitted.semantic_execution_profile_id
```

is a mandatory equality for positive authority.

The runtime-derived side must come from the exact loaded/verified semantic inputs and qualified runtime interpretation, not from a configuration string that merely repeats the admitted ID.

Failure to derive the profile is denial.

Failure to verify its schema provenance is denial.

Profile mismatch is denial.

Unknown profile schema/version is denial.

These failures freeze new positive authority; they do not independently authorize destructive action.

---

## 9. Restart and cache semantics

A serialized statement such as:

```text
verified = true
semantic_execution_profile_id = X
```

MUST NOT recreate authority after restart by itself.

After restart, the production path must re-establish the evidence required by policy, including:

- verified registry/trust state;
- exact canonical schema bytes;
- deterministic validator derivation;
- current runtime profile eligibility;
- admitted release binding;
- freshness/revocation/supersession state.

Caches may accelerate reconstruction, but cached results cannot substitute for required verification.

---

## 10. Upgrade and migration semantics

A change to any constituent below creates a different semantic execution profile identity:

- capability schema ID;
- capability representation profile;
- capability validator-table digest;
- resource schema ID;
- resource representation/arithmetic profile;
- resource validator-table digest.

A different `SemanticExecutionProfileId` requires explicit admission under a release/profile policy.

Existing grants, budgets, checkpoints, or durable evidence MUST NOT silently acquire authority under the new profile merely because identifiers or low-level numeric values appear compatible.

Cross-schema or cross-profile authority carryover requires the separately verified conservative transition machinery defined by the relevant RSK semantic contracts.

---

## 11. Build lineage and implementation changes

Two builds may derive the same semantic execution profile while having different source or artifact identities.

That does not make them interchangeable production binaries.

Semantic execution identity captures semantic interpretation; exact build/runtime identity separately captures the implementation and artifact enforcing it.

Therefore:

```text
same SemanticExecutionProfileId
    != same admitted artifact
```

and:

```text
same admitted artifact digest
    != semantic profile automatically valid
```

Both relationships must be verified by the admission process.

---

## 12. Canonical golden vector

The reference cross-language corpus is:

```text
docs/architecture/replicator-safety/golden/
  RSK_SEMANTIC_EXECUTION_PROFILE_GOLDEN_V0_1.json
```

For the abstract test-only capability/resource schemas currently used by the semantic corpus, the expected profile identity is:

```text
fc53377e5dad0dc7b29be6dc9bb2d050911bd04467ca4769c29a42dd4400c23a
```

This is test evidence only. It is not a production schema approval, physical resource definition, or admission decision.

Independent language implementations should derive the profile from canonical schema bytes and compare their output to the golden corpus. They MUST NOT obtain parity by simply deserializing a precomputed trusted validator table.

---

## 13. Required invariants

The implementation/test/formal program SHALL target at least:

1. `SemanticExecutionProfileIsDerivedNotDeclared`
2. `SchemaChangeChangesSemanticExecutionIdentity`
3. `ValidatorInterpretationChangeChangesSemanticExecutionIdentity`
4. `CapabilityRepresentationChangeChangesSemanticExecutionIdentity`
5. `ResourceArithmeticProfileChangeChangesSemanticExecutionIdentity`
6. `UnsupportedCurrentRuntimeSchemaCannotProduceProfile`
7. `CallerSuppliedValidatorRulesCannotProduceTrustedProfile`
8. `ProfileMismatchDeniesPositiveAuthority`
9. `UnknownProfileVersionDeniesPositiveAuthority`
10. `ProfileMatchCannotSubstituteForRegistryProvenance`
11. `ProfileMatchCannotSubstituteForArtifactIdentity`
12. `RestartCannotRestoreVerifiedProfileByBooleanCache`
13. `CrossProfileAuthorityDoesNotCarryImplicitly`
14. `OneProductionProfileUsesOneCoherentVerifiedRegistryContext`
15. `SemanticProfileFailureCannotClearNegativeFacts`

---

## 14. Adversarial targets

Tests should include at least:

- same schema ID claim with altered validator rule table;
- correct profile ID string with different loaded schema bytes;
- capability semantic-description change;
- capability width 64 vs 65 under the current Rust profile;
- resource numeric-ID remap;
- resource `u64::MAX` vs `u64::MAX + 1`;
- resource `sum` vs unsupported `max` aggregation;
- `exact` vs unsupported rounding;
- tampered validator-table digest;
- tampered representation-profile identifier;
- capability/resource schemas sourced from incompatible verified registry snapshots;
- stale/superseded admitted semantic profile;
- runtime profile cache replay after restart;
- admitted artifact with wrong loaded semantic profile;
- correct semantic profile under wrong executable artifact.

---

## 15. Non-claims

This contract does not:

- verify registry signatures or trust roots;
- approve a production capability/resource vocabulary;
- define physical resource measurements;
- implement production runtime attestation;
- prove a binary is the admitted artifact;
- implement cross-schema translation authority;
- grant replication authority;
- close build/runtime identity, registry provenance, or production admission blockers.

It defines one exact semantic identity that those systems must bind.

---

## 16. Production admission status

This document and its reference tooling do not admit any artifact or runtime.

```text
Production admission status: DENIED / NOT YET ELIGIBLE.
```
