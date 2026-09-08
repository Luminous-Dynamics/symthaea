# Canonical Scientific Artifact Identity v1

**Status:** architecture contract only; non-authorizing; non-qualifying.

**Series:** SCI-002, stacked on SCI-001.

**Parent:** `architecture/scientific-method-kernel-audit-v1@ea0f794142802486549f0b33eb5aeb1495e747a7`

## 1. Purpose

SCI-001 identified content-addressed scientific artifact identity as the first implementation prerequisite for the Scientific Method Kernel.

Symthaea already contains several strong local identity patterns:

- RCA derives serializer-independent BLAKE3 identities from validated graph semantics;
- NeuroBridge distinguishes raw observation roots, normalized closure roots, source/input/execution/transform/scientific roots;
- Matter uses strict SHA-256-shaped external evidence references while keeping them `ReferenceOnly` until external bytes are independently verified;
- Earth-observation work binds exact content digests, typed interpretation metadata, transformation lineage, and canonical floating-point conventions;
- generic observation admission accepts only canonical `sha256:<64 lowercase hex>` / `blake3:<64 lowercase hex>` textual forms;
- scientific proposition work separately treats proposition semantic identity as a domain-owned semantic target rather than a raw byte digest.

The repository therefore does not need another ad-hoc hash helper. It needs a small common contract that says **what an artifact identity means, how it is constructed, what verification is required, and what authority it explicitly does not carry**.

This document freezes those semantics before any shared Rust implementation exists.

---

## 2. Core theorem

The v1 identity model preserves the following non-equivalences:

```text
human-readable artifact label
    != canonical artifact identity

well-formed digest string
    != bytes-to-digest verification

verified bytes
    != correct interpretation of those bytes

content identity
    != semantic identity

semantic identity
    != scientific truth

same content identity
    != independent observation

different content identity
    != independent evidence

canonical artifact identity
    != execution evidence
    != measurement validity
    != scientific qualification
    != replication
    != action authority
```

The most important design rule is:

> A cryptographic digest identifies bytes or a deliberately canonical semantic encoding. It does not, by itself, establish what those bytes mean or whether they are scientifically valid.

---

## 3. Identity layers

SCI-002 defines four distinct layers.

### 3.1 Locator / human label

Examples:

```text
"harmonic_oscillator.discovery-bound.smt2"
"AME2020"
"Workbench output"
"experiment-17"
https://example.org/data.csv
```

These are navigation/provenance metadata only.

They may help retrieve an artifact but do not define canonical scientific identity.

### 3.2 Raw byte content identity

This identifies one exact byte sequence.

Conceptually:

```text
RawArtifactContentIdentityV1 {
    identity_profile
    digest_algorithm
    digest
    byte_len
}
```

The identity is about the bytes only.

If one bit changes, the raw byte content identity changes.

### 3.3 Canonical semantic artifact identity

Some scientific objects have multiple byte encodings that are intentionally semantically equivalent. A domain may therefore define a versioned canonical semantic encoding.

Conceptually:

```text
CanonicalSemanticArtifactIdentityV1 {
    identity_profile
    domain_namespace
    semantic_schema_profile
    canonicalization_profile
    canonical_payload_digest
    canonical_payload_len
}
```

This identity is valid only with respect to the exact registered semantic schema and canonicalization profile.

The shared kernel does **not** define a universal canonical representation for arbitrary scientific objects.

### 3.4 Composite / manifest identity

Many scientific artifacts are collections or graphs rather than one payload.

Examples:

- an execution receipt plus stdout/stderr sidecars;
- a scientific input snapshot;
- an evidence-lineage graph;
- a model plus exact training receipt;
- a proof target plus witness plus solver result;
- a theory/disposition generation.

A composite identity must bind its exact child identities and relation semantics under a versioned profile.

Conceptually:

```text
CompositeScientificArtifactIdentityV1 {
    identity_profile
    composite_kind
    semantic_profile
    ordered_or_canonicalized_edges
    composite_digest
}
```

A manifest is not allowed to inherit authority from its children merely by listing them.

---

## 4. Digest algorithms and canonical textual form

### 4.1 Stable algorithm identity

V1 should support at least stable identifiers for:

```text
SHA-256
BLAKE3-256
```

The exact implementation names are not authority-bearing. The algorithm identifier must denote one exact digest algorithm and output size.

The common representation should preferably store the digest algorithm as a typed enum/profile identifier and the digest as fixed-size bytes rather than rely internally on free-form strings.

### 4.2 Textual boundary

Where a textual form is required, v1 should use one canonical spelling per algorithm, consistent with the stricter identity work already present in shared observation types:

```text
sha256:<64 lowercase hex>
blake3:<64 lowercase hex>
```

The parser must reject:

- uppercase-hex aliases;
- leading/trailing whitespace;
- wrong digest length;
- non-hex characters;
- unknown algorithms;
- ambiguous separators;
- silent normalization of malformed input.

A successfully parsed digest remains only a **declared digest identity** until exact bytes are checked.

### 4.3 Algorithm agility without identity rewriting

If the repository later adopts a different digest algorithm, historical identities must not be silently rewritten.

A new algorithm creates a new content identity representation. Cross-algorithm correspondence requires an explicit multi-digest or equivalence artifact produced from the same verified bytes.

```text
old digest
    != automatically upgraded digest
```

Historical evidence addresses remain immutable.

---

## 5. Raw-byte verification boundary

The common kernel must distinguish a declared reference from bytes that have actually been checked.

Suggested states:

```text
DeclaredArtifactReferenceV1
ObservedArtifactBytesV1
VerifiedArtifactContentV1
```

The naming is illustrative.

### 5.1 Declared reference

May contain:

```text
locator
claimed digest
claimed byte length
artifact role
subject label
issuer label
claimed time
```

Maximum local interpretation:

```text
ReferenceOnly
```

This is compatible with the current Matter external-evidence boundary.

### 5.2 Observed bytes

An observer/loader has acquired a concrete byte sequence.

The observation should retain enough execution/provenance information to identify what was read, but simply possessing bytes does not prove they match a declared reference.

### 5.3 Verified content

A verifier recomputes the exact digest over the exact observed bytes and requires:

```text
recomputed_digest == declared_digest
```

and, when declared:

```text
observed_byte_len == declared_byte_len
```

Only then may it issue a positive `VerifiedArtifactContentV1`-style wrapper.

The positive wrapper should be private-fielded and should not be directly deserializable into authority.

Archived serialization is audit material. Reconstructing current trusted state requires revalidation or a separately qualified persisted-receipt boundary.

---

## 6. Canonical semantic encoding boundary

### 6.1 Domain-owned semantics

The Scientific Method Kernel may standardize the envelope, but the scientific domain owns the semantic payload.

Examples:

```text
nuclear mass-table row semantics
causal estimand semantics
ALife perturbation schedule semantics
Lean theorem target semantics
scientific proposition semantics
raster payload interpretation
```

These must not be forced into one universal JSON structure.

### 6.2 Canonicalization profile

A semantic artifact identity is meaningful only if an exact profile defines how the semantic object becomes canonical bytes.

The profile must specify every representation choice that can affect identity, including where relevant:

- field order;
- collection ordering;
- duplicate handling;
- string encoding and normalization rules;
- integer encoding;
- enum/tag encoding;
- optional-field encoding;
- map key ordering;
- floating-point encoding;
- signed-zero handling;
- NaN handling;
- infinity policy;
- timestamp/time-zone representation;
- unit representation;
- schema version;
- domain-separation prefix;
- child identity encoding;
- length-prefix / framing rules.

The profile itself must have immutable identity.

### 6.3 No implicit `serde_json` authority

Ordinary serializer output must not accidentally define scientific identity.

In particular:

```text
serde serialization today
    != stable canonical scientific encoding forever
```

A domain may deliberately choose canonical JSON, CBOR, protobuf-like bytes, an explicit manual encoding, or another representation, but the exact canonicalization rules must be part of the identity profile.

Changing serializer/library behavior must not silently mutate scientific identity.

### 6.4 Floating-point values

There is no safe universal float canonicalization rule for all science.

Different domains may legitimately require different semantics:

- preserve exact IEEE-754 bit patterns;
- canonicalize `-0.0` to `+0.0`;
- prohibit NaN;
- preserve a specific NaN payload;
- encode rational/decimal values exactly instead of binary floats.

Therefore v1 requires the **domain canonicalization profile** to own float semantics.

The shared kernel must not globally normalize floats.

---

## 7. Semantic identity is not raw content identity

Two important cases must remain representable.

### 7.1 Different bytes, same declared semantics

Examples:

- equivalent JSON formatting;
- different archive/container layout around the same normalized scientific payload;
- equivalent theorem pretty-printing;
- a normalized closure identity derived from richer raw Nix metadata.

This may yield:

```text
RawContentIdentity A != RawContentIdentity B
CanonicalSemanticIdentity A == CanonicalSemanticIdentity B
```

but only because an exact canonicalization profile establishes that projection.

### 7.2 Same bytes, different interpretation

The same byte sequence may be interpreted differently under different schemas, units, coordinate frames, endian assumptions, calibration rules, or theorem languages.

Therefore:

```text
same raw bytes
+ different semantic profile
    -> different semantic scientific identity
```

This is why a semantic identity must bind the exact schema/canonicalization profile rather than only a digest.

---

## 8. Artifact kind and domain separation

Every semantic or composite identity must be domain separated.

A 32-byte digest for one object must not be reusable as a different object merely because both fields have the same width.

At minimum, identity computation should bind stable versioned tags for:

```text
identity profile
artifact kind
domain namespace
semantic schema/profile
```

Illustrative non-equivalences:

```text
scientific input snapshot
    != verification target
    != solver witness
    != execution receipt
    != measurement record
    != evidence contribution
    != proposition
    != disposition generation
```

Even when their payload digests happen to match.

---

## 9. Composite identity rules

### 9.1 Child identity vs child bytes

A composite may bind verified child content identities rather than re-embed every child byte sequence, provided the composite profile makes that transitive dependency explicit.

### 9.2 Ordered vs unordered collections

Order must never be chosen implicitly.

For every repeated field the profile must say whether order is:

```text
identity-significant
```

or:

```text
canonically order-insensitive
```

If order is semantically irrelevant, entries must be canonicalized by a specified stable key before hashing.

### 9.3 Graph identity

Graphs require explicit semantics for:

- node identity;
- edge identity/type;
- edge direction;
- duplicate edges;
- node/edge ordering;
- closed-world vs partial graph assumptions;
- whether unrelated node additions change generation identity.

SCI-002 should reuse the design lesson from RCA: complete validated graph semantics can define a generation identity while legacy producer labels remain ordinary metadata.

### 9.4 Closure identity

A closure/manifest identity should distinguish:

```text
raw observation root
    != normalized semantic closure root
```

where normalization intentionally discards non-semantic metadata.

The NeuroBridge pattern is the primary example.

---

## 10. Translation and equivalence receipts

The kernel must not silently treat transformations as identity preservation.

Examples:

```text
Expr -> SymExpr
SMT problem -> Lean theorem
raw raster -> calibrated raster
CSV -> normalized table
source graph -> canonical semantic graph
specification -> extracted implementation semantics
```

For a scientifically meaningful transformation, retain:

```text
source artifact identity
target artifact identity
transformation implementation identity
configuration/profile identity
execution identity or receipt
claimed relation
verification status
```

Suggested relation classes may include:

```text
BytePreserving
LosslessCanonicalization
SemanticsPreservingWithinDeclaredProfile
InformationReducingProjection
ModelDerivedTransformation
EquivalenceNotEstablished
```

No generic transformation should be allowed to self-declare `SemanticsPreserving` without a qualified domain/verifier boundary.

This requirement directly supports #738 and #786: proof-target translation ancestry must eventually be artifact-bound rather than inferred from in-process control flow.

---

## 11. Identity and provenance

Identity answers:

> Which exact artifact or semantic object is this?

Provenance answers:

> Where did it come from, what produced it, and what does it depend on?

They are complementary but not interchangeable.

```text
same identity
    != same acquisition path

different provenance
    != different content

same content
    != independent provenance
```

An evidence-lineage graph may contain multiple observations of identical bytes. Whether those count as independent evidence is a separate lineage/replication theorem.

---

## 12. Identity and time/currentness

Content identity is immutable with respect to time.

Currentness is not.

Examples:

- an artifact can remain byte-identical after its signing key is revoked;
- a reference can remain the same while its source is retracted;
- a policy artifact can retain content identity after expiration;
- a dataset can remain immutable while newer versions supersede it.

Therefore:

```text
artifact identity
    != freshness
    != trust currentness
    != lifecycle eligibility
```

Temporal/lifecycle layers should refer to immutable artifact identities rather than mutating identity records in place.

---

## 13. Identity and authority

No identity type should expose convenience transitions such as:

```text
verified()
trusted()
scientifically_valid()
independent()
causal()
novel()
authorized()
```

solely because content identity exists.

A verified artifact-content wrapper proves only the declared bytes-to-digest relation under the exact digest algorithm/profile.

It does not prove:

- source authenticity;
- scientific validity;
- measurement correctness;
- semantic truth;
- causal identification;
- independent replication;
- novelty;
- safety;
- action authority.

---

## 14. Proposed v1 shared surface

Names are illustrative. Semantics are normative.

```text
DigestAlgorithmV1
ContentDigestV1
RawArtifactContentIdentityV1
ArtifactDomainV1
ArtifactKindV1
CanonicalizationProfileIdentityV1
SemanticSchemaProfileIdentityV1
CanonicalSemanticArtifactIdentityV1
CompositeScientificArtifactIdentityV1
DeclaredArtifactReferenceV1
VerifiedArtifactContentV1
ArtifactTransformationReceiptV1
```

The implementation should be split so domains can consume only what they need.

A first Rust tranche should be much smaller than this full vocabulary.

---

## 15. First implementation tranche

After architecture review, the smallest useful implementation is:

```text
ContentDigestV1
RawArtifactContentIdentityV1
VerifiedArtifactContentV1
```

with strict digest parsing and exact bytes-to-digest verification.

Required properties:

1. stable algorithm IDs;
2. exact digest length;
3. canonical textual form;
4. byte length bound into identity;
5. digest recomputation from supplied bytes;
6. private-field positive verified wrapper;
7. no direct positive deserialization;
8. domain-separation/version profile;
9. no `DefaultHasher`, Rust `Hash`, Debug string, or incidental serializer defining identity;
10. tests proving that malformed/aliased representations fail closed.

This tranche should not yet implement arbitrary semantic canonicalization.

---

## 16. Second implementation tranche

Add a registered semantic canonicalization profile interface and one narrow pilot.

Preferred pilot candidates are artifacts with already-clear semantics, for example:

- the Ramanujan discovery-bound proof target/witness lineage;
- one NeuroBridge normalized closure identity;
- one RCA canonical graph generation;
- one deterministic scientific input snapshot.

The pilot must prove that the shared abstraction does not weaken the domain-specific identity theorem.

Do not migrate multiple domains in the same first semantic-identity PR.

---

## 17. Migration policy for existing identity schemes

Existing identities remain valid historical/domain-native identities.

The shared SCI-002 contract does **not** retroactively reinterpret them.

Migration requires an explicit adapter or correspondence receipt:

```text
legacy/domain identity
    + exact artifact / semantic object
    + SCI-002 profile
    -> SCI-002 identity
    + correspondence evidence
```

A new shared identity must not silently replace a historical evidence address.

This is especially important for:

- RCA graph/witness IDs;
- NeuroBridge execution/closure roots;
- Matter external evidence references;
- Earth-observation artifact manifests;
- existing proposition/evidence IDs.

---

## 18. Adversarial requirements

Any future implementation should include at least these regressions.

### Digest syntax and bytes

- uppercase alias rejected;
- whitespace alias rejected;
- wrong digest length rejected;
- unknown algorithm rejected;
- one-byte mutation changes content identity;
- wrong declared byte length rejected;
- digest mismatch cannot issue a verified wrapper.

### Type/domain separation

- same payload under different artifact kind produces different semantic/composite identity;
- same payload under different schema profile produces different semantic identity;
- same payload under different canonicalization profile produces different semantic identity;
- copying a digest from one domain into another does not recreate the same typed identity.

### Canonicalization

- order-insensitive collections are stable under permutation only when the profile declares that behavior;
- order-sensitive collections change identity under permutation;
- duplicate handling is deterministic/fail-closed;
- float edge cases obey the exact domain profile;
- serializer formatting changes do not change identity when serializer bytes are not normative.

### Persistence / authority

- serialized `VerifiedArtifactContentV1` cannot deserialize directly into positive authority;
- caller-supplied digest metadata cannot mint a verified wrapper;
- a valid content identity cannot become `trusted`, `experimental`, `independent`, `causal`, `novel`, or action-authorized without a separate layer.

### Transformation

- source/target substitution changes transformation receipt identity;
- transformation implementation/config substitution changes receipt identity;
- `SemanticsPreserving` cannot be self-asserted by an unqualified caller;
- same output bytes from different transform ancestry remain distinguishable in provenance even when content identity is equal.

---

## 19. Relationship to other SCI tranches

### SCI-003 execution capsule

Execution capsules should consume SCI-002 artifact identities for:

```text
source
runtime/dependency closure
inputs
configuration
external tools
```

but capsule identity remains distinct from execution occurrence.

### SCI-004 experiment contract

Experiment contracts should bind exact scientific proposition, input/data, measurement, analysis, decision-rule, and preregistration artifact identities.

### SCI-006 evidence dependency graph

Dependency graphs should use immutable artifact identities as nodes/references while preserving domain-specific dependency semantics.

### SCI-011 learned grammar provenance

Learned grammar/macros must carry exact originating artifact/model/dataset/verification identities so downstream rediscovery cannot hide prior knowledge.

### SCI-014 Theory Atlas

Theory Atlas proposition/evidence/disposition generations should be content-addressed without confusing identity with truth or current disposition.

### #786 verification authority receipts

`VerificationReceiptV1` should bind SCI-002 identities for:

```text
verification target
semantic target
proof/witness artifact
verifier implementation
configuration
execution capsule
output/result artifacts
```

A qualified verification receipt must not rely on `config_hash()` or any non-cryptographic diagnostic hash as scientific identity.

---

## 20. Required implementation dependency order

The safe order is:

```text
SCI-001 common/non-common audit
    -> SCI-002a strict digest/content identity
    -> SCI-002b bytes-to-digest verified-content wrapper
    -> SCI-002c one domain semantic-canonicalization pilot
    -> SCI-003 execution capsule
    -> #786 qualified verification receipts
```

where #786's positive authority layer additionally remains gated by its verifier-soundness prerequisites.

There is no need to wait for every later SCI tranche before implementing the basic raw-byte identity boundary.

There **is** a need to avoid using the current evidence-plane `DefaultHasher`/Debug diagnostic configuration hash as the canonical identity root for any of these artifacts.

---

## 21. Review boundary

Review SCI-002 on this question:

> Does this contract define a sufficiently precise, reusable identity substrate for exact scientific artifacts without confusing bytes, semantics, provenance, verification, scientific authority, or action authority—and without weakening stronger domain-native identity schemes already present in Symthaea?

A positive architecture review does not qualify any Rust implementation and does not transfer qualification from any referenced domain PR.
