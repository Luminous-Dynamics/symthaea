# ADR-012: RSK Exact Build, Artifact, and Runtime Identity

**Date**: 2026-09-12  
**Status**: Proposed  
**Change Class**: A (Safety-Critical)

## Context

The Replicator Safety Kernel now has increasingly strong source-level qualification, evidence verification, trust-boundary, governance, and temporal contracts.

Those guarantees remain insufficient for production if the process enforcing authority cannot be shown to be the exact artifact/configuration that the evidence describes.

A green source commit, signed provenance statement, Nix store path, SBOM, container tag, or code-signing certificate is individually useful evidence but does not by itself establish exact production runtime identity.

This ADR contains no physical replication mechanism.

## Decision

RSK production admission will use an explicit identity chain:

```text
SourceCandidate
    -> BuildDefinition
    -> ArtifactIdentity
    -> AdmittedRelease
    -> VerifiedRuntimeIdentity
```

No transition is implicit.

The runtime authority path may consume a verified admitted-release/runtime-identity capability only after the production verifier establishes exact equality for all policy-required identity dimensions.

## Source candidate

The candidate binds at least:

- repository;
- Git commit and tree;
- `Cargo.lock`;
- Rust toolchain;
- Nix `flake.lock` when applicable;
- RSK package/build manifests;
- qualification profile/schema.

Environment drift after evidence begins starts a new evidence lineage.

## Build definition

The build definition binds at least:

- build-type/schema;
- builder identity and trust boundary;
- resolved dependencies/materials;
- external parameters;
- target triple;
- feature set;
- relevant compiler/link/profile flags;
- Nix derivation/build identity where applicable;
- build provenance statement.

The design is compatible with SLSA v1.2/in-toto provenance but does not depend constitutionally on one provenance vendor or service.

## Nix decision

A Nix store path is retained as useful derivation/store identity, but RSK does not assume that an ordinary store path is an output-content hash.

Where Nix is used, production evidence should bind both the derivation/build identity and an output-content identity such as a NAR/artifact digest according to the release profile.

Content-addressed derivations are not assumed by default because they remain opt-in/experimental in current Nix documentation.

Nix machine-readable output formats used as evidence must be explicitly version-pinned; RSK must not rely on an evolving unversioned JSON default.

## Artifact identity

The admitted artifact binds the exact executable/output content plus every additional runtime dependency/closure dimension required by deployment policy.

The profile decides whether that means:

- executable only;
- executable plus selected libraries;
- complete Nix closure;
- complete container image;
- complete VM/system image.

The choice must be explicit and evidence-backed.

## Independent reproducibility

High-consequence release policy should require independent reproduction according to a declared failure-domain requirement.

The system distinguishes:

```text
build execution count
builder identity count
builder failure-domain count
```

Output disagreement blocks admission by default. There is no implicit majority-vote release rule.

## Admitted release capsule

A canonical admitted-release capsule binds:

- source identity;
- verified qualification evidence roots;
- build definition and provenance;
- artifact identity;
- SBOM/dependency roots;
- reproducibility evidence;
- capability/resource schema identities;
- risk policy;
- evidence schema set;
- runtime attestation profile;
- admission policy/decision;
- validity, supersession, and revocation state.

The canonical capsule digest is the release identifier used by runtime admission.

## Runtime identity

Production starts with positive authority disabled.

Before enabling new positive authority, the runtime verifier must prove that the running process and authority-relevant configuration match a current admitted release.

At minimum, deployment policy should bind the running executable/artifact, policy, capability schema, resource schema, release ID, and any required host/process/image measurements.

Runtime path names, tags, or self-reported versions are not trusted substitutes for measured identity.

## Rollback and supersession

Historical artifact validity is not current authority.

A previously admitted artifact may become ineligible because of:

- supersession;
- revocation;
- policy/trust changes;
- recovery epoch changes;
- time/freshness constraints;
- ledger state.

Rollback must not restore expired grants, revoked policy, old recovery authority, or old ledger state.

## Supply-chain tools remain non-authoritative

Nix, SLSA, in-toto, Sigstore, SBOM systems, transparency logs, TPM/TEE mechanisms, container registries, and CI services may provide evidence.

None of them independently mints positive replication authority.

Current negative facts and constitutional predicates remain dominant.

## Governance hardening

Normative files under:

```text
docs/architecture/replicator-safety/
```

are themselves part of the safety boundary. This tranche therefore classifies the normative RSK architecture directory as Class A in the generic detector and focused workflow.

A future weakening of a constitutional/admission contract must require a Class A ADR just like a change to the Rust authority kernel.

## Verification plan

`RSK_BUILD_RUNTIME_IDENTITY_TEST_PLAN_V0_1.md` defines BI-* families covering:

- source substitution;
- build-definition substitution;
- Nix store/NAR identity confusion;
- artifact mutation;
- independent reproducibility;
- release capsule integrity;
- runtime mismatch;
- rollback/supersession;
- evidence-lineage drift;
- provenance authenticity/scope;
- transparency non-authority;
- configuration identity;
- crash/fault histories;
- property/model/fuzz/API misuse tests.

## External design references

This decision is informed by:

- SLSA v1.2 provenance and build-platform trust-boundary concepts;
- in-toto attestation models;
- current Nix derivation/store/NAR identity semantics;
- the repository's existing operational-resilience build/provenance/reproducibility architecture.

These are design references, not certification claims.

## Alternatives considered

### Treat Git commit as release identity

Rejected. Build environment, feature/profile, dependency, artifact, configuration, and runtime substitution remain possible.

### Treat Nix store path as complete artifact identity

Rejected. Input-addressed store paths identify construction inputs/derivation context rather than serving universally as exact output-content digests.

### Treat signed provenance as admission

Rejected. Provenance can faithfully describe a build that violates release policy. Verification and policy evaluation remain separate.

### Trust a container/image tag

Rejected. Mutable/human-readable tags are not exact artifact identity.

### Admit once and trust rollback forever

Rejected. Revocation, supersession, policy evolution, recovery epochs, and freshness make authority time/state dependent.

### Allow majority output among reproducibility builders

Rejected as a default. Unexplained output disagreement is a safety incident/admission blocker, not a voting problem.

## Consequences

### Positive

- source evidence is bound to the executable actually enforcing authority;
- supply-chain substitutions become explicit typed mismatches;
- Nix is used precisely rather than overclaimed;
- reproducibility becomes failure-domain aware;
- runtime policy/schema drift cannot silently preserve admission;
- rollback/supersession semantics become explicit;
- normative safety documents receive Class A governance protection.

### Costs

- production admission requires more metadata and verification;
- deployment profiles must define what runtime closure/host identity is trusted;
- release processes need reproducibility/provenance/SBOM evidence;
- runtime attestation mechanisms are platform-specific;
- independently reproduced builds may increase build cost.

### Residual risk

- a sufficiently compromised admitted builder/release/runtime trust root may still forge evidence within its trusted boundary;
- physical containment remains outside software build identity;
- hardware/firmware attestation may itself have implementation flaws;
- reproducibility does not prove semantic correctness;
- exact identity does not replace the other RSK constitutional gates.

## Production status

No artifact is admitted by this ADR.

```text
Production admission status: DENIED / NOT YET ELIGIBLE.
```

Admission remains blocked until #1682 and the other production gates have executed evidence on the exact candidate lineage.