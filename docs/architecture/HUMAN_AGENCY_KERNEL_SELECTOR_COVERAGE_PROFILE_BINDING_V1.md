# HAK-016 — Selector Coverage Canonical Profile Binding v1

Status: candidate audit/evidence tooling. No runtime authority changes.

## Purpose

The core HAK-016 selector coverage record names:

```text
canonicalization_profile = hak.canonical-json.v1
digest_domain = hak.selector-coverage.v1
```

HAK-015 established that a symbolic profile name is not sufficient identity for the semantics behind a digest:

```text
ProfileName
!=
ProfileContent
```

HAK-016 therefore adds an **adjacent profile-binding artifact** rather than changing the historical meaning of the selector coverage record itself.

```text
SelectorCoverageRecord
        +
SelectorCoverageProfileBinding
        ↓
Coverage digest with exact canonical-profile provenance
```

## Exact HAK-015 profile identity

The v1 binding identifies the inherited HAK-015 profile through both:

```text
exact Git artifact ref
+
raw profile-file SHA-256
```

For this stacked tranche:

```text
artifact_ref =
git:Luminous-Dynamics/symthaea@cf440ce4f813bb30a6b1948e5caac96feda10610:docs/architecture/hak/canonical-json-v1.profile.json

raw_sha256 =
sha256:f76152db3cce567bb4d24ca46b5ed94d95db698daa8882a6fcdb38828d40e830
```

The raw byte digest is intentionally not computed using the canonicalization profile it identifies. This avoids making profile identity depend recursively on the profile's own semantics.

```text
RawProfileIdentity
!=
ProfileCanonicalizationResult
```

The profile's semantic object is still validated by HAK-015; the raw SHA-256 independently identifies the exact inherited file bytes.

## Binding artifact

`hak.selector-coverage-profile-binding.v1` binds:

- the HAK-016 coverage schema;
- the exact `coverage_digest`;
- `hak.canonical-json.v1` profile id;
- the exact inherited HAK-015 Git profile artifact;
- the exact raw SHA-256 of that profile file;
- explicit profile-identity semantics;
- a profile-bound HAK canonical binding digest.

The implementation is:

```text
scripts/hak_selector_coverage_profile_binding.py
```

and its strict schema is:

```text
docs/architecture/hak/selector-coverage-profile-binding-v1.schema.json
```

## Deterministic validation

The validator requires:

```text
CoverageRecord.coverage_digest
==
recomputed HAK-016 coverage digest
```

and:

```text
current inherited profile raw SHA-256
==
bound expected profile raw SHA-256
```

before deriving the binding.

A submitted binding is accepted only if it is exactly the deterministic binding derived from the supplied coverage record and inherited profile artifact.

Therefore coherent local redigesting does not authorize substitution of:

- the profile id;
- profile artifact ref;
- raw profile-file digest;
- coverage digest;
- binding semantics.

## Why both Git ref and raw digest?

They answer different questions.

```text
GitArtifactRef
```

provides repository/path/commit provenance.

```text
RawSHA256
```

identifies the exact file bytes independently of Git object naming or HAK canonicalization semantics.

Thus:

```text
ArtifactLocationIdentity
!=
ArtifactContentIdentity
```

Strong interpretation benefits from both.

## Composition boundary

The profile binding does **not** make an invalid coverage record valid. HAK-016 semantic acceptance still requires the coverage record to be deterministically derived from a validated HAK-014 receipt, and strong HAK-016 acceptance additionally requires HAK-014 replay against the exact normalization inputs.

```text
ProfileBindingValid
!=
CoverageSemanticsValid
```

Likewise:

```text
CoverageSemanticsValid
!=
UpstreamHAK014Qualified
!=
UpstreamHAK015Qualified
```

Each exact-head evidence subject retains its own qualification state.

## Migration

A future HAK canonicalization profile must use an explicit new profile artifact/version or an explicit migration relation. HAK-016 must not silently reinterpret an old coverage digest under new profile bytes that happen to carry the same human-readable name.

```text
NewProfileBytes
!=
OldDigestSemantics
```

## Non-claims

The profile binding does not:

- prove HAK-015 globally correct;
- prove portability beyond HAK-015's qualified implementations/corpus;
- authenticate provider observations;
- establish semantic truth;
- establish scientific validity;
- grant runtime authority.
