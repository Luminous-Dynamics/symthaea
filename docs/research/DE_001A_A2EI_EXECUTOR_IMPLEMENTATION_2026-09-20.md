# DE-001A2EI — Exact Raw Executor Implementation Contract

Status: `preregistered-contract-only`

Scientific claim: `NONE`

Authority: `raw-optimizer-executor-implementation-qualification-contract-only`

## Purpose

DE-001A2EI freezes the implementation-qualification boundary that must exist between the already-preregistered DE-001A2E raw-execution contract and any future real optimizer invocation.

This tranche does **not** implement an optimizer executor and does not execute Cobaya, DESI likelihoods, samplers, minimizers, or cosmological calculations. Its only positive result is contract consistency.

The key theorem is:

```text
qualified execution contract
!= qualified executor implementation
!= optimizer authorization
!= optimizer execution
!= optimizer reproduction
!= cosmological evidence
```

A2E deliberately keeps `executor_implementation_authorized=false`; therefore a future raw result cannot become authoritative merely by matching the expected JSON shape. The exact executable that produces it must first be qualified against A2EI.

## Required upstream evidence

A future A2EI implementation qualifier must consume one explicit successful `DE-001A2E-CONTRACT-QUALIFICATION-CAPSULE-v1` root and preserve its transitive A2 authorization, scientific subject, effective sampler, sampled-coordinate order, and one-invocation budget.

No latest-run discovery, backend substitution, option rewriting, normalization/migration, network access, or runtime installation is permitted.

## Exact executor identity

Qualification must bind an exact implementation identity, including source HEAD/TREE, a SHA-256 content root, release executable SHA-256 and size, Cargo.lock, Rust toolchain, runtime closure, execution-contract identity, raw-result protocol identity, command-construction identity, and postflight implementation identity.

Git object IDs are provenance identifiers. SHA-256 remains the content/evidence identity.

## Fixture-only qualification

Before any real optimizer process can be launched, the production executor implementation must be exercised with inert deterministic fixture executables.

The frozen fixture matrix covers:

- successful single invocation;
- nonzero exit with no automatic retry;
- invalid local evidence;
- attempted second invocation;
- command/argv digest divergence;
- stdout/stderr digest divergence;
- executable/source postflight mutation;
- duplicate manifest roles/paths;
- final-file, intermediate-directory, and result-root symlink attacks;
- current-directory and parent-path aliases;
- forbidden reproduction classification;
- contract-hash mismatch;
- authorization/sampler/coordinate lineage mismatch.

A2EI qualification must prove these behaviors using the same production execution path that a later real invocation would use. A toy implementation that bypasses production command construction, capture, manifesting, or postflight logic is insufficient.

## Result-tree confinement

Manifest paths are identities, not convenience strings. The frozen rule is therefore stricter than ordinary path normalization:

```text
result root is a real directory
+
all manifest path components are Component::Normal
+
no intermediate symlinks
+
terminal entry is a regular non-symlink file
-> eligible for content verification
```

`./foo`, parent traversal, absolute/prefix components, result-root symlinks, intermediate symlink traversal, and duplicate role/path aliases are invalid evidence.

The same confinement helper should ultimately be shared by the raw executor and A2Q so generation and verification cannot diverge semantically.

## Runtime isolation

`network_allowed=false` is not satisfied merely because the implementation does not intentionally make a network call.

A real-execution-qualified implementation must demonstrate enforceable network isolation and runtime-package-installation prohibition through a receiptable mechanism such as a Nix sandbox or another fail-closed equivalent.

If isolation cannot be mechanically established, `real_optimizer_execution_authorized` remains false.

## Raw-result binding

A future `DE-001A2E-RAW-OPTIMIZER-EXECUTION-v1` result must additionally bind the exact A2EI qualification receipt, executor source HEAD/TREE, executable SHA-256, A2E contract receipt, A2 authorization receipt, scientific subject, effective sampler, and sampled-coordinate order.

The raw executor must always emit:

```text
reproduction_verdict=UNASSESSED
scientific_claim=NONE
```

It may not contain the historical reproduction tolerances or choose scientific success metrics. Those belong exclusively to preregistered A2Q qualification.

## Promotion boundary

This contract-consistency tranche always keeps:

```text
executor_implementation_qualified=false
optimizer_execution_authorized=false
a2_execution_authorized=false
real_optimizer_execution_authorized=false
a2q_execution_authorized=false
scientific_claim=NONE
```

The future order is:

```text
qualified A1X AGREE
+ qualified A2R/A2G/A2M provenance
+ qualified exact A2 authorization
+ qualified A2E execution contract
+ qualified A2EI exact executor implementation
-> exactly one raw A2 optimizer invocation
-> preregistered A2Q qualification
```

No engineering gate in that chain establishes LambdaCDM validity, dynamic dark energy, an observational anomaly, phenomenological dark-energy dynamics, or a physical mechanism.
