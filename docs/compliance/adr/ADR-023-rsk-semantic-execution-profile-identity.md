# ADR-023: RSK semantic execution profile identity

**Status**: Proposed

**Change Class**: A

**Date**: 2026-09-13

## Context

The Replicator Safety Kernel already distinguishes semantic schema identity from structural validation and, in later hardening tranches, from the runtime representation/arithmetic profiles that interpret those schemas.

A remaining admission gap was that `RSK_BUILD_AND_RUNTIME_IDENTITY_V0_1` conceptually bound capability/resource schema digests directly. That is not sufficient to prove how an admitted runtime interprets those schema bytes.

The same source schema may be:

- executed by different representation widths;
- interpreted by different resource arithmetic engines;
- paired with a caller-constructed or weakened validator table;
- correctly identified semantically while being unsupported by the current runtime representation.

Therefore source semantic identity and runtime semantic execution identity must be distinct evidence dimensions.

This ADR contains no physical replication mechanism, fabrication recipe, biological/molecular design, physical resource definition, or autonomous manufacturing path.

## Decision

Introduce the normative v0.1 semantic execution profile:

```text
symthaea.rsk.semantic-execution-profile.v1
```

The profile commits, for both capability and resource semantics:

- exact source schema digest;
- exact admitted runtime representation/arithmetic profile identifier;
- canonical SHA-256 of the deterministically derived structural-validator rule table.

The canonical profile digest is the `SemanticExecutionProfileId`.

The profile is **derived, not declared**. Production code must reconstruct it from exact verified schema bytes through the admitted deterministic adapter/runtime-profile logic. A caller-supplied profile ID, representation-profile string, or validator-table hash is not trusted evidence by itself.

The admitted release/runtime identity model must eventually bind `SemanticExecutionProfileId` in addition to artifact/build/configuration and verified schema-registry provenance.

## Current reference composition

The current Rust-target profile composes:

```text
capability:
  symthaea.rsk.capability-representation.rust-u64.v1

resource:
  symthaea.rsk.resource-representation.rust-u64-sum-exact.v1
```

The profile therefore fails closed if the canonical schemas are not executable under those current target profiles.

## Registry provenance

Semantic execution identity does not absorb registry trust/provenance.

Production profile derivation must begin from one coherent verified schema-registry snapshot/policy context, but registry signatures, trust roots, freshness, signer lifecycle, and governance remain separate evidence.

This separation prevents trust-key rotation from silently becoming a semantic change while still preventing semantic-profile equality from bypassing provenance verification.

## Runtime/build relationship

`SemanticExecutionProfileId` is one constituent of exact build/runtime admission, not a replacement for it.

The following implications are intentionally false:

```text
same semantic execution profile -> same admitted artifact
same artifact digest -> same semantic execution profile
```

Both must be verified by the release/runtime-admission process.

A profile mismatch, inability to derive the profile, unknown profile version, or unsupported constituent runtime profile denies new positive authority.

## Restart behavior

Cached `verified=true` state or a serialized profile ID cannot restore positive authority after restart by itself.

Required registry/trust evidence, exact schemas, deterministic derivation, current-profile eligibility, release binding, and freshness/revocation state must be re-established according to admission policy.

## Golden evidence

Add:

```text
docs/architecture/replicator-safety/golden/
  RSK_SEMANTIC_EXECUTION_PROFILE_GOLDEN_V0_1.json
```

The current abstract test vector derives:

```text
SemanticExecutionProfileId =
fc53377e5dad0dc7b29be6dc9bb2d050911bd04467ca4769c29a42dd4400c23a
```

This is test-only semantic evidence, not a production admission identity.

## Reference implementation

Extend `scripts/rsk_schema_adapter.py` with deterministic profile construction, profile-ID derivation, and exact verification against independently reconstructed schema-derived semantics.

Extend `scripts/test_rsk_schema_adapter.py` to cover:

- committed golden reconstruction;
- semantic schema changes;
- resource runtime-ID remapping;
- representation-profile tampering;
- validator-digest tampering;
- unsupported capability width;
- unsupported resource width/aggregation/rounding;
- historical resource schema rejection.

The existing RSK governance workflow already treats the adapter and its self-test as Class A and runs that self-test in focused governance CI.

## Alternatives considered

### Bind only source schema digests

Rejected. It fails to commit runtime interpretation and deterministic validator semantics.

### Put full schemas and validator tables directly in every admitted release record

Not required for the compact identity layer. Canonical full evidence should remain recoverable/verifiable, but a digest-bound profile avoids unnecessary duplication while preserving exact identity.

### Include registry signature/trust snapshot directly in `SemanticExecutionProfileId`

Rejected for v0.1. Provenance and semantic execution are related but distinct lifecycles. Production admission must bind both rather than conflating them.

### Trust a runtime-reported profile identifier

Rejected. This would turn an evidence result into a self-asserted configuration claim.

## Consequences

Positive consequences:

- source semantic identity and runtime interpretation are explicitly separated;
- build/runtime admission gains one compact equality target for semantic execution;
- validator-rule substitution becomes detectable;
- future wider/different runtime semantics require a new explicit execution identity;
- cross-language Rust/Python parity has a single derived profile target;
- profile changes naturally start a new release/admission lineage.

Costs:

- production registry resolution must preserve enough exact schema evidence to reconstruct the profile;
- Rust must independently implement and qualify the same derivation before production use;
- admission capsules and runtime attestation profiles eventually need one more bound identity;
- future profile evolution requires explicit compatibility/migration review.

## Evidence status

At authoring time:

- the profile schema, reference derivation, golden vector, and adversarial tests are authored;
- the canonical golden profile digest was independently recomputed during authoring;
- exact-head GitHub workflow execution is not yet claimed because the repository RSK/ordinary CI queue remains backlogged;
- Rust/Cargo production qualification remains blocked by the controlled workspace-lockfile work tracked in #1926;
- verified registry provenance remains tracked separately under #1969.

No production gate is checked off by this ADR.

## Related work

- #1682 — exact build/runtime identity
- #1668 — verified positive authority evidence
- #1678 — capability semantic integrity
- #1679 — resource semantic integrity
- #1926 — controlled Cargo.lock qualification blocker
- #1969 — verified schema-registry provenance
- #2038 / #2057 — deterministic schema-derived validator type state
- #2064 — shared schema-adapter golden corpus
- #2071 / #2076 — current Rust capability representation profile
- #2078 / #2080 — current Rust resource representation/arithmetic profile

## Production admission

Production admission remains:

```text
DENIED / NOT YET ELIGIBLE
```
