# PHI-SEM-001A — Phi / integration authority-flow inventory v1

Parent: #5401  
Implementation contract: #5402

## Status

This tranche is **measurement-only**.

It introduces a deterministic lexical census of tracked Phi/integration surfaces. It does
not rename public APIs, change thresholds, alter estimator math, change control behavior,
or grant/revoke any authority.

The product is the audit mechanism itself. A separate exact-subject qualification run
must execute it and preserve the generated JSON before `PASS_INVENTORY` can be claimed.

## Core theorem

```text
lexical witness exists
!= runtime reachability

path appears in census
!= estimator correct

two values are both called Phi
!= same estimator
!= same units
!= same proposition

PASS_INVENTORY
!= IIT validated
!= consciousness demonstrated
!= authority repaired
```

## Why this tranche exists

Symthaea currently uses the name `Phi`/`phi` across several non-equivalent surfaces:

- HDC partition-loss estimation;
- entropy-from-Hamming-similarity estimation;
- spectral and hierarchical MIP proxies;
- structural micro/meso/macro integration;
- algebraic connectivity used as an optimization objective;
- composite `ConsciousnessState::phi` values;
- feedback, attention, confidence, action-gating, telemetry, and documentation surfaces.

Later migrations cannot be safely scoped from a hand-written list. The first requirement
is therefore a reproducible census bound to an exact source tree.

## Audit profile

The executable profile is:

```text
phi-sem-001a-lexical-v1
```

Run it from a clean tracked worktree:

```bash
python3 scripts/audit_phi_authority_flow.py \
  --pretty \
  --output docs/research/evidence/phi_sem_001a_inventory_v1.json
```

By default the tool refuses tracked worktree modifications.

`--allow-dirty` exists only for exploratory local use and is not acceptable for
qualification evidence.

## Determinism

The report intentionally contains no wall-clock timestamp.

It binds:

- `git rev-parse HEAD`;
- `git rev-parse HEAD^{tree}`;
- tracked path set from `git ls-files -s -z`;
- the audit script Git blob, SHA-256, and byte length;
- Git blob identity for each matched path;
- SHA-256 and byte length for each matched path;
- exact lexical match line numbers and bounded line text;
- the audit profile SHA-256;
- every mandatory witness selector;
- the two Phi Oracle subtree fingerprints;
- lexical Cargo dependency edges naming `symthaea-phi-oracle`.

Entries, categories, paths, selectors, and witness groups are sorted before serialization.

Pretty and compact serialization have the same semantic content; qualification should
bind the exact emitted evidence bytes it preserves.

## Self-reference exclusion

The census deliberately excludes its own measurement artifacts:

```text
scripts/audit_phi_authority_flow.py
docs/research/PHI_SEM_001A_INVENTORY_V1.md
docs/research/evidence/phi_sem_001a_*
```

This keeps the product from inflating its own Phi/authority witness count merely by
explaining the audit. The exclusions are themselves profile-hashed.

## Classification vocabulary

The v1 census uses the following non-exclusive categories:

```text
Estimator
TransformOrComposite
ScientificLabel
SyntheticValidation
OptimizationObjective
FeedbackController
MetacognitiveSignal
ConfidenceOrStatusProducer
AttentionOrWorkspaceControl
ExpressionControl
ActionOrReadinessGate
GovernanceBridge
ExternalExport
SerializationOrCheckpoint
CompatibilityAlias
DocumentationClaim
UnclassifiedPhiLexeme
```

These are lexical classifications, not proofs about runtime semantics.

A single line or path may belong to multiple categories.

## Mandatory witness selectors

The profile fails closed if **any configured selector** is absent.

An exact-file selector requires that exact tracked path. A directory selector ending in
`/` requires at least one tracked path under that prefix.

The initial witness set covers:

- core HDC integrated-information code;
- the consciousness pipeline;
- legacy core Phi-guided search;
- the workspace `symthaea-phi-search` crate;
- Phi feedback;
- Phi optimization;
- metacognitive Phi use;
- Phi-aware attention/action thresholds;
- both synthetic Phi validation and synthetic-state paths;
- cognitive-loop consciousness engine;
- late consciousness integration;
- NixOS action gates;
- both named Nixward Phi surfaces;
- workspace-active Phi Oracle;
- the root-level duplicate Phi Oracle;
- telemetry sink and telemetry gRPC surfaces.

Mandatory witnesses are a floor, not an allowlist. The lexical search runs across all
tracked candidate text files.

## Phi Oracle duplicate witness

The audit fingerprints both:

```text
crates/domains/symthaea-phi-oracle/
crates/symthaea-phi-oracle/
```

For each subtree it records the ordered relative path set, Git blob identities, SHA-256
identities, and an aggregate subtree fingerprint.

It also reports whether the relative file sets and bytes are equal.

In addition, every tracked `Cargo.toml` line naming `symthaea-phi-oracle` is recorded with
manifest identity and line number. This provides a bounded source witness for which copy
current dependency edges name.

This tranche **does not delete or choose between the copies**. Dependency ownership and
cleanup remain later product changes.

## Candidate text scope

The v1 profile scans tracked UTF-8 source/research/configuration text with extensions
such as:

```text
.rs .toml .md .py .json .yaml .yml .nix .sh .txt .csv .ron .proto
```

plus common extensionless build files.

Binary or non-UTF-8 files are ignored.

This scope is intentionally explicit and profile-hashed. Expanding it later creates a
new audit profile rather than silently changing the meaning of old evidence.

## Exact source versus generated evidence

The audit reads the checked-out tracked bytes, while the clean-worktree requirement
binds those bytes to the index/commit subject.

Qualification should additionally record:

```text
product commit
product tree
audit script Git blob / SHA-256
generated report SHA-256
generated report byte length
toolchain identity
exit status
```

A future never-merge qualifier may wrap those checks, but the product PR must not claim
a qualification PASS merely because the script exists.

## Interpretation discipline

A hit such as:

```text
required_phi
```

supports the bounded statement:

```text
this exact source line contains a lexical Phi-related action-gating witness
```

It does not establish:

```text
this path executes in production
this condition is security-critical
the gate can be reached
the gate is sufficient for authorization
```

Those are later source-authoritative/runtime questions.

Likewise, a documentation hit connecting Phi to consciousness establishes only that the
claim language exists in the frozen source.

A Cargo dependency-edge witness establishes only that a frozen manifest line names a
path. It does not by itself prove the package is built in every workspace/profile.

## Intended next tranche

After a qualified inventory, #5403 (`PHI-SEM-001B`) can introduce typed estimator
identity and observation semantics in the dependency-light cognitive telemetry layer.

That later work should be driven by **inventory-demonstrated estimator families only**.

The inventory itself must not pre-commit to a universal replacement scalar.

## Qualification sketch

A separate qualifier should:

1. bind an exact product commit/tree;
2. verify the audit script is the expected blob;
3. run the audit from a clean checkout;
4. require exit status 0;
5. require zero missing mandatory witness selectors;
6. preserve canonical JSON output;
7. bind the output bytes and toolchain identity;
8. report only `PASS_INVENTORY` on success.

The qualifier should not reinterpret a successful inventory as scientific validation.

## Nonclaims

This tranche does not establish:

- that Symthaea is conscious or unconscious;
- that IIT is true or false;
- that any current Phi estimator is valid or invalid;
- that all runtime Phi flows have been proven reachable;
- that all dynamic/reflection/generated-code paths are captured;
- that high or low integration is desirable;
- epistemic confidence from integration;
- execution authority from integration;
- governance authority from integration;
- clinical interpretation of any HDC value.

Its purpose is narrower and foundational:

> freeze what the source actually says and where it says it, before changing meaning.
