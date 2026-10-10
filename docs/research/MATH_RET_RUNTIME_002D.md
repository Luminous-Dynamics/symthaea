# MATH-RET-RUNTIME-002D — Materializer Implementation Receipt

Status: draft production-hardening implementation

Authority: `MeasurementOnly`

002D closes the next trust seam exposed after MATH-RET-RUNTIME-002B/002C.

002B correctly requires an `implementation_sha256` before the first source
fetch, but the expected implementation identity is still supplied by the
production caller. 002D changes the source of that identity:

```text
qualified receipt SHA-256
        +
exact receipt bytes
        +
exact implementation artifact bytes
        ↓
MATH-RET-RUNTIME-002D loader
        ↓
validated build/policy provenance
        ↓
MaterializerBinding derived from admitted bytes
        ↓
002B pre-fetch identity enforcement
```

The caller no longer supplies an implementation digest independently of the
qualified receipt.

## Receipt contract

Version:

`math-retrieval-materializer-implementation-v1`

Authority:

`MeasurementOnly`

The receipt freezes:

- receipt identity;
- implementation artifact kind;
- SHA-256 of the exact implementation artifact bytes;
- exact artifact byte count;
- source-object contract identity;
- source-fetch policy identity;
- payload-serialization identity;
- exact source-bundle SHA-256;
- exact `Cargo.lock` SHA-256;
- Rust toolchain identity;
- target triple;
- build profile;
- lexicographically sorted unique feature set;
- rustflags SHA-256;
- build-recipe SHA-256;
- build-environment SHA-256.

Supported implementation kinds in v1 are:

```text
ExecutableArtifact
WasiComponent
SharedLibrary
StaticBuildArtifact
```

These names describe the frozen artifact form. They do not themselves prove
that an arbitrary in-process Rust value was instantiated from those bytes.

## Qualified input

The loader receives:

```text
MaterializerImplementationReceiptBinding {
    receipt_sha256,
    source_object_contract_sha256,
    source_fetch_policy_sha256,
    payload_serialization_sha256,
}
```

`implementation_sha256` is intentionally absent.

The exact qualified receipt digest commits the implementation digest and build
metadata. The loader then independently computes SHA-256 over the exact
implementation artifact bytes and requires equality with the digest inside the
receipt.

## Derived 002B binding

Only after both byte sequences validate does 002D construct:

```text
MaterializerBinding {
    MaterializerIdentity {
        source_object_contract_sha256,
        source_fetch_policy_sha256,
        payload_serialization_sha256,
        implementation_sha256,
    }
}
```

This is the binding consumed by MATH-RET-RUNTIME-002B.

The implementation digest therefore has one mechanical provenance chain:

```text
qualified receipt digest
    commits
receipt implementation_sha256
    equals
SHA256(actual supplied implementation artifact bytes)
```

## Exact-byte / TOCTOU rule

`load_files` reads the receipt and implementation artifact once, then delegates
to the byte loader. The admitted object retains both exact byte sequences.

Changing either path after successful admission cannot alter:

- the retained receipt;
- the retained implementation artifact;
- the derived implementation digest;
- the derived materializer binding.

This is intentionally analogous to the candidate-artifact rule in 002A.

## Build environment

002D is build-system-neutral. `build_environment_sha256` must identify the
canonical build-environment receipt used by the production qualification.

For a Nix-qualified materializer, that receipt should eventually commit the
relevant flake lock / derivation / store or NAR identity rather than treating a
human-readable Nix path as sufficient provenance. Nix-specific receipt
semantics should be frozen in a separate contract rather than hidden inside
this generic loader.

## Mechanical canaries

The Rust tests require:

1. NIST SHA-256 `abc` vector;
2. exact receipt + artifact derive the expected 002B binding;
3. receipt-byte substitution fails against the qualified receipt digest;
4. one-byte implementation artifact mutation fails against the frozen receipt;
5. unexpected receipt fields fail;
6. authority escalation fails;
7. unknown implementation kinds fail;
8. artifact-size mismatch fails;
9. unsorted feature sets fail;
10. duplicate features fail;
11. noncanonical/uppercase digests fail;
12. empty required build metadata fails;
13. each source/fetch/serialization policy mismatch fails independently;
14. implementation digest is derived from admitted bytes rather than caller input;
15. path substitution after loading cannot mutate the admitted receipt/artifact;
16. malformed/empty inputs fail closed.

## Relationship to 002C

002D does not mutate the frozen 002C production-composition subject.

A later composition successor should replace caller construction of
`MaterializerBinding` with:

```text
qualified implementation receipt
      ↓
002D
      ↓
derived MaterializerBinding
      ↓
002C/002B production path
```

If the production materializer is later moved behind a separately instantiated
WASI component, executable helper, or other content-addressed module, the
retained artifact bytes should become the actual instantiation source. That
would close the remaining distinction between "qualified artifact provenance"
and "this in-process implementation was actually created from those bytes."

## Evidence-session hygiene remains separate

002D does not address the evidence-session concern recorded on #4360. A later
runtime tranche should establish transaction-scoped sink ownership or an
explicit clean-session invariant so early preflight failure cannot leave stale
staged evidence under caller control.

## Nonclaims

002D proves mechanical provenance relationships only:

```text
these exact receipt bytes were qualified
these exact artifact bytes hash to the implementation identity in that receipt
this 002B materializer binding was derived from those admitted bytes
```

It does **not** prove that arbitrary in-process Rust code was loaded from those
bytes, mathematical relevance, theorem truth, proof success, evidence score,
formal authority, HDC advantage, or full production readiness.
