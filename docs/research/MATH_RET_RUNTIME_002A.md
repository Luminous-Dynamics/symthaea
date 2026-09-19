# MATH-RET-RUNTIME-002A — Candidate Artifact Loader

Status: draft production-hardening implementation

Authority: `MeasurementOnly`

This standalone crate removes the caller-trust seam between the auditable
candidate-set artifact from MATH-RET-CANDIDATE-001A and the pre-materialization
`MembershipGuardBackend` from MATH-RET-RUNTIME-001C.

## Trust boundary

Before 002A, production code could conceptually provide these independently:

```text
claimed candidate_set_sha256
claimed candidate_count
Vec<SourceObjectDigest>
```

and then construct a `FrozenCandidateUniverse`.

002A instead requires:

```text
exact candidate artifact bytes/path
          +
qualified expected binding
          ↓
CandidateArtifactLoader
          ↓
SHA-256(actual bytes)
strict candidate-set-v1 parse/validation
qualified metadata comparison
          ↓
LoadedCandidateArtifact
          ↓
FrozenCandidateUniverse
```

There is no loader API that accepts a caller-supplied candidate vector.

## Qualified expected binding

The caller may provide only the values that must be checked against the
artifact and are expected to come from the already-qualified graph/index
package:

- `candidate_set_sha256`;
- `candidate_count`;
- `corpus_snapshot_sha256`;
- `knowledge_boundary_sha256`;
- `candidate_eligibility_policy_sha256`.

Candidate identities are parsed exclusively from the artifact.

## Artifact semantics

The Rust implementation mirrors the existing
`validate-math-retrieval-candidate-set.py` contract:

- exact root field set;
- `version = math-retrieval-candidate-set-v1`;
- `authority = MeasurementOnly`;
- non-empty candidate-set ID;
- `SourceObjectDigest` source identity;
- `SourceObjectDigestAscending` canonical ordering;
- positive count exactly equal to array length;
- strict lowercase `sha256:<64 hex>` identities;
- no duplicates;
- strict ascending source order.

The SHA-256 commitment is over the **exact input bytes**, not a reserialized
JSON object.

## TOCTOU rule

`load_file` performs exactly one file read and immediately delegates to the
byte loader. `LoadedCandidateArtifact` retains those exact bytes plus the
immutable parsed universe.

The loader must never implement:

```text
hash path
... time passes ...
reopen same path for candidates
```

A test loads artifact A, replaces the file with artifact B, and verifies the
already-loaded universe and retained bytes remain A.

## Query exclusion

The frozen candidate artifact is the static eligible universe. It is allowed
to contain the source object that happens to be the current query. Per-query
self-exclusion remains a later `RetrievalExecutor` invariant and is not folded
into artifact loading.

## Dependency/reproducibility boundary

This crate remains standalone like the earlier runtime seam/guard. Registry
versions are exact-pinned and the standalone `Cargo.lock` is checked with
`--locked`.

The lock resolution is deliberately small:

- `serde_json = 1.0.151`;
- `sha2 = 0.10.9`;
- their minimal transitive dependencies;
- path dependencies on the frozen runtime seam and membership guard.

## Required canaries

The Rust tests require:

1. SHA-256 matches the NIST `abc` vector;
2. exact bytes reconstruct the expected universe;
3. one-byte mutation fails against the frozen artifact digest;
4. unsorted candidates fail even if the artifact hash is rebound;
5. duplicate candidates fail;
6. count mismatch fails;
7. unexpected root fields fail;
8. authority escalation fails;
9. non-lowercase digest spelling fails;
10. corpus snapshot mismatch fails;
11. knowledge boundary mismatch fails;
12. eligibility policy mismatch fails;
13. qualified count mismatch fails;
14. query-source membership remains separate from query exclusion;
15. post-load path substitution cannot mutate the loaded universe;
16. malformed JSON / empty candidate artifacts fail closed.

The dedicated workflow also reruns the frozen Python candidate-set self-test so
Rust and Python semantics remain aligned.

## Next production tranche

MATH-RET-RUNTIME-002B should bind the `CanonicalSourceMaterializer` itself to
an exact implementation/policy identity and reject mismatches **before the
first source fetch**.

002C can then compose:

```text
CandidateArtifactLoader
      ↓
FrozenCandidateUniverse
      ↓
MembershipGuardBackend
      ↓
identified CanonicalSourceMaterializer
      ↓
RetrievalExecutor
      ↓
payload audit
```

## Nonclaims

002A proves only candidate-artifact provenance and immutable universe loading.
It does not establish retrieval relevance, HDC advantage, mathematical
equivalence, proof success, theorem truth, evidence score, formal authority,
or production readiness of the full retrieval system.
