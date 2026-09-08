# Workbench Nix Closure Identity v1

Status: **candidate pure identity compiler only; no Nix closure has been realized, captured, or qualified by this specification**

Schema: `symthaea-workbench-nix-closure-identity-v1`

## Purpose

This specification defines the deterministic interpretation layer between a future raw Nix capture and the Workbench execution capsule defined by `WORKBENCH_EXECUTION_CAPSULE_PROFILE_V1.md`.

It intentionally does **not** execute Nix, query the store, build Workbench, execute `wb_command`, access the network, inspect scientific inputs, or claim scientific authority.

The theorem in scope is only:

```text
already-observed store records
    { store path, NAR SHA-256, references[] }
        -> closed-world validation
        -> canonical graph normalization
        -> root-reachability/minimality check
        -> canonical closure digest
```

The theorem is not:

```text
selection -> realization
realization -> correct execution
execution -> valid transform
transform -> atlas correctness
atlas correctness -> FMQ-010
FMQ-010 -> neural alignment
neural alignment -> consciousness evidence
```

## Input contract

The compiler accepts a declared closure root and a non-empty list of observed store-object records. Every record has exactly:

```json
{
  "path": "/nix/store/<32-nix32>-<name>",
  "nar_sha256": "sha256:<64-lowercase-hex>",
  "references": []
}
```

Unknown fields are rejected. Duplicate JSON object keys are rejected before semantic interpretation.

### Store paths

v1 is deliberately bound to `/nix/store`. The 32-character digest uses the Nix32 alphabet. Embedded `/`, carriage-return, and newline characters are rejected from the store-object name component.

This compiler treats store paths as observed identifiers. It does not attempt to recalculate Nix store-path fingerprints.

### NAR hash representation

v1 accepts exactly one internal representation:

```text
sha256:<64 lowercase hexadecimal digits>
```

This is **not** an assertion that every Nix command emits that representation. A future capture layer must retain the raw observed hash representation and deterministically translate it to this canonical form. That translation is outside this compiler and must be separately qualified.

## Closure graph semantics

For canonical entries `E` and declared root `R`, v1 requires all of the following:

1. `R` is an entry in `E`.
2. Every reference in every entry is itself an entry in `E`.
3. No store path occurs twice.
4. No reference occurs twice within one entry.
5. Every supplied entry is reachable from `R` by repeatedly following references.

The fifth invariant is important. Merely proving that all references are present would permit arbitrary unrelated store objects to be appended to a claimed closure. v1 instead requires:

```text
SuppliedEntries = Reachable(R)
```

Self-references and reference cycles are allowed; reachability is cycle-safe.

## Canonicalization

The compiler canonicalizes only representation, never semantics:

- references are sorted lexicographically;
- entries are sorted lexicographically by store path;
- object schemas are closed-world;
- stored identities must already use the canonical order;
- `entry_count` must be a positive JSON integer and must equal the actual entry count.

A stored identity that is semantically non-canonical remains invalid even if an attacker recomputes its SHA-256.

## Closure digest

The digest payload is:

```text
{
  "schema": "symthaea-workbench-nix-closure-identity-v1",
  "root": R,
  "entries": canonical_entries
}
```

serialized as UTF-8 JSON with sorted keys and compact separators, then hashed with SHA-256:

```text
ClosureDigest = SHA256(CanonicalJSON(schema, root, entries))
```

Including the schema string provides version/domain separation. `entry_count` is intentionally redundant and is validated independently from the entries.

## Observation versus interpretation

This layer accepts already-observed records. The future capture layer must not discard its raw evidence after normalization.

A qualified capture receipt should retain, at minimum:

- exact capture implementation identity;
- exact command/argument contract;
- exact Nix version;
- root store path as observed;
- raw stdout/stderr digests for each capture surface;
- process exit status;
- raw NAR-hash representation;
- normalized NAR-hash representation;
- normalized closure identity produced by this compiler.

Therefore:

```text
RawObservationRoot != NormalizedClosureRoot
```

The normalized root is an interpretation of the raw observation, not a replacement for it.

## Authority boundary

Successful compilation or validation establishes only:

```text
canonical_closure_identity_compiled = true
```

It does not establish any of the following:

```text
closure_observed = false
closure_realized = false
closure_capture_qualified = false
closure_qualified = false
execution_environment_qualified = false
workbench_executed = false
transform_executed = false
scientific_execution_qualified = false
atlas_correctness_established = false
fmq010_established = false
neural_alignment_established = false
consciousness_evidence = false
```

The CLI therefore reports `validated-identity-only`, not `qualified`.

## Scientific identity versus qualification identity

The closure compiler is evidence machinery, not a scientific generator merely because it authenticates the runtime environment.

Future Lineage-B provenance should preserve at least these separate roots:

```text
GeneratorImplementationRoot
    code that can alter scientific transform bytes

InputSnapshotRoot
    immutable scientific input custody

ExecutionCapsuleRoot
    realized Workbench closure + fixed execution environment + platform

TransformOutputRoot
    exact transform outputs

ScientificCommitmentRoot
    method + atlas namespace + inputs + generator + execution + outputs

QualificationEnvelopeRoot
    capture/compiler/verifier implementations + verification receipts
```

A verifier or capture-tool upgrade should normally create a new `QualificationEnvelopeRoot`, not silently redefine the underlying scientific result when the scientific inputs, generator, execution capsule, and outputs are unchanged.

## Qualification invariants

- **WNCI-001 Closed-world entries:** every entry has exactly `path`, `nar_sha256`, and `references`.
- **WNCI-002 Canonical store path:** every path is a canonical `/nix/store/<32-nix32>-<name>` representation accepted by v1.
- **WNCI-003 Canonical NAR SHA-256:** every NAR identity is `sha256:<64 lowercase hex>`.
- **WNCI-004 Unique objects:** duplicate store paths are rejected.
- **WNCI-005 Unique references:** duplicate references are rejected.
- **WNCI-006 Closed references:** every reference names a supplied closure entry.
- **WNCI-007 Exact root reachability:** every supplied entry is reachable from the declared root.
- **WNCI-008 Deterministic ordering:** input and reference order do not affect the compiled identity.
- **WNCI-009 Canonical stored representation:** a persisted identity must already be normalized.
- **WNCI-010 Count integrity:** `entry_count` is a positive integer equal to the canonical entry count.
- **WNCI-011 Digest integrity:** NAR content, reference topology, root, or canonical entry changes alter or invalidate the closure identity.
- **WNCI-012 Rehash resistance:** recomputing a digest over semantically non-canonical data does not restore validity.
- **WNCI-013 Duplicate-key rejection:** ambiguous JSON objects are rejected before interpretation.
- **WNCI-014 Encoding-boundary humility:** translation from Nix-emitted hash encodings is outside this compiler and requires separate capture qualification.
- **WNCI-015 No observation laundering:** successful compilation does not prove the records were observed from a Nix store.
- **WNCI-016 No scientific laundering:** closure identity does not authorize execution, atlas correctness, FMQ-010, neural alignment, or consciousness claims.

## Promotion sequence

```text
#624 selection/execution-capsule profile
        ->
this pure closure-identity normalizer
        ->
raw Nix closure capture receipt
        ->
independent raw-receipt verification
        ->
execution capsule qualification
        ->
#576 snapshot consumed by derive()
        ->
new Lineage-B scientific commitment
        ->
#525 archival reconstruction
        ->
real A <-> B FMQ-010 comparison
```

Only the second step is in scope here.
