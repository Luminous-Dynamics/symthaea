# MATH-RET-RUNTIME-002C — Production Retrieval Composition

Status: draft production-hardening implementation

Authority: `MeasurementOnly`

002C composes the previously separate production gates around the already-frozen
retrieval executor. It adds no representation-specific scoring logic and does
not alter the research qualification subjects.

## Single production entry point

`ProductionRetrievalRuntime<B, M>` owns:

- the exact loaded candidate artifact bytes;
- a `MembershipGuardBackend<B>` constructed from that artifact's universe;
- a `QualifiedMaterializer<M>`;
- the frozen `MaterializerBinding`.

Construction is allowed only through the 002A candidate loader:

```text
candidate artifact bytes/path
        ↓
CandidateArtifactLoader
        ↓
LoadedCandidateArtifact
        ↓
FrozenCandidateUniverse
        ↓
MembershipGuardBackend<B>
```

Per-request execution then runs:

```text
request
  ↓
preflight request ↔ loaded candidate universe
  ↓
materializer identity bind
  ↓
only then backend retrieval may execute
  ↓
raw backend IDs checked by MembershipGuardBackend
  ↓
frozen RetrievalExecutor
  ↓
canonical payload materialization
  ↓
packing + trace + actual payload audit
  ↓
atomic evidence commit
```

The membership guard repeats the candidate/request binding when the backend is
actually called. The earlier preflight is deliberate defense in depth and also
prevents needless materializer binding when the request names the wrong
candidate universe.

## Failure ordering

The composition tests require:

### Candidate artifact mutation

A one-byte mutation of the candidate artifact fails during runtime construction.
The backend and materializer do not yet exist inside an admitted production
runtime.

### Wrong request candidate binding

The request is rejected before:

- backend execution;
- materializer fetch;
- staged or committed evidence.

### Wrong materializer implementation identity

The 002B bind fails before:

- backend execution;
- materializer fetch;
- staged or committed evidence.

### Illegal backend source

The backend runs once, but the membership guard rejects its source identity
before:

- source materialization;
- trace/payload staging;
- evidence commit.

### Legal execution

A legal single-index execution must:

- execute the backend once;
- materialize the selected source through the bound canonical materializer;
- commit exactly one evidence bundle;
- emit no residual staged state;
- retain the candidate artifact digest in the trace;
- retain the actual delivered canonical payload bytes in the payload audit.

## Exact-byte fixture

The fixture candidate artifact is a real repository file, not a Rust string
constructed at runtime. Its exact raw-byte SHA-256 is frozen as:

`sha256:e92003a3eeb556515927dae44e2c95847fa64a65d9ac7c43535f1f410576b77a`

The qualification workflow independently checks that digest with `sha256sum`
before Rust tests run.

## What 002C deliberately reuses

002C does not reimplement:

- candidate-set parsing;
- candidate membership;
- source materializer identity checks;
- retrieval scoring/fusion;
- canonical payload packing;
- trace construction;
- payload auditing;
- transactional evidence commit.

It composes the frozen components from 002A, 001C, 002B, and the original runtime
seam.

## Remaining boundary after 002C

The production shell is now ready for a real backend only after the relevant
representation/index lineage itself qualifies.

For structural retrieval the safe order remains:

```text
qualified/extracted canonical sparse representation
      ↓
real S index/backend
      ↓
ProductionRetrievalRuntime
      ↓
end-to-end evidence qualification
      ↓
only then real structural HDC H as a one-variable successor
```

No real S/H backend is added here.

## Nonclaims

002C establishes only mechanical composition when its exact workflow passes. It
does not establish that 002A/002B or the research runtime are already qualified,
does not establish retrieval relevance, HDC advantage, equivalence, proof
success, theorem truth, evidence score, formal authority, production deployment
readiness, or novelty.
