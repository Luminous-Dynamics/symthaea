# MATH-RET-RUNTIME-002B — Materializer Identity Binding

Status: draft production-hardening implementation

Authority: `MeasurementOnly`

002B closes the second production trust assumption recorded in #4360: the
frozen retrieval request already names the source-object contract, source-fetch
policy, and payload serialization, but the predecessor
`CanonicalSourceMaterializer` trait itself exposes only:

```text
materialize(SourceObjectDigest) -> canonical payload bytes
```

It therefore cannot independently prove that the concrete implementation
realizes the identities carried by the request.

## Two-phase type boundary

002B adds:

```text
MaterializerIdentity {
    source_object_contract_sha256,
    source_fetch_policy_sha256,
    payload_serialization_sha256,
    implementation_sha256,
}

MaterializerBinding {
    identity,
}

MaterializerIdentityProvider

QualifiedMaterializer<M>
        ↓ bind(request, qualified binding)
BoundMaterializer<'_, M>
```

`QualifiedMaterializer<M>` intentionally does **not** implement
`CanonicalSourceMaterializer`.

Only `BoundMaterializer<'_, M>` implements the predecessor trait, and that type
has no public constructor. It can be obtained only after a successful identity
check.

This turns "check before fetch" from documentation into a type-level API
property.

## Identity comparison

Before returning a bound materializer, 002B requires exact equality between the
concrete implementation's self-reported identity and the qualified
`MaterializerBinding` for all four fields.

It then independently requires the retrieval request graph to equal the same
binding for:

- `source_object_contract_sha256`;
- `source_fetch_policy_sha256`;
- `payload_serialization_sha256`.

The existing graph contract does not contain an implementation identity, so
`implementation_sha256` remains an explicit production qualification binding
rather than being falsely attributed to the predecessor graph.

## Why identity comes from the implementation

`QualifiedMaterializer::new` accepts only the inner materializer. It does not
accept a second caller-provided `MaterializerIdentity` value.

The concrete implementation must implement:

```text
MaterializerIdentityProvider::materializer_identity()
```

This prevents a wrapper caller from trivially pairing implementation A with an
unrelated identity B at construction time.

A later production implementation/build/config qualification must define how
its `implementation_sha256` is derived. 002B freezes the runtime enforcement
boundary; it does not invent an implementation hash from a Rust type name.

## Required canaries

The Rust tests require all of these to reject before the first inner
`materialize` call:

1. wrong request source-object contract;
2. wrong request source-fetch policy;
3. wrong request payload serialization;
4. wrong implementation identity;
5. concrete materializer source-contract substitution;
6. concrete materializer fetch-policy substitution;
7. concrete materializer serialization substitution;
8. a binding attempting to relabel the actual implementation.

The positive canaries require:

- successful bind exposes the exact frozen identity;
- payload bytes pass through unchanged;
- the same source under one deterministic identified materializer yields the
  same canonical bytes.

The final payload audit remains responsible for independently hashing the bytes
that actually reached downstream cognition.

## Separation from backend representation

002B has no retrieval-backend, HDC, canonical-AST, normal-form, memory, Phi, or
proof dependency. Representation-specific backends still return source IDs
only. They cannot supply payload bytes or materializer identity.

## Production composition

002C should compose the now-separated gates in this order:

```text
candidate artifact bytes
      ↓
002A CandidateArtifactLoader
      ↓
FrozenCandidateUniverse
      ↓
MembershipGuardBackend
      ↓
backend ranking
      ↓
002B QualifiedMaterializer.bind(request, binding)
      ↓
BoundMaterializer
      ↓
RetrievalExecutor
      ↓
actual payload-byte audit
```

002C should provide one entry point that performs both the universe/request
binding and materializer binding before downstream materialization, so a
production caller cannot accidentally omit one of the gates.

## Nonclaims

002B proves only that one materializer instance matched the frozen mechanical
identity supplied for one qualified request before source fetching was allowed.
It does not establish mathematical relevance, equivalence, proof success,
theorem truth, evidence score, formal authority, HDC advantage, or full
production readiness.
