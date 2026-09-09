# Qualification Execution Identity Recurrence Review v1

Date: 2026-09-09
Status: architecture evidence review / non-authorizing
Series: SCI-Q4 recurrence gate

## Purpose

This review asks whether the shared execution-identity vocabulary proposed by SCI-Q4 has actually earned implementation under Symthaea's rule-of-three.

The answer must distinguish:

```text
design recurrence
    != landed production recurrence
    != qualified shared abstraction
```

Open source candidates are useful architectural evidence, but they are not allowed to count as though they were already merged, qualified, or operational.

## Stable-main recurrence

### 1. Muse reproducibility / execution evidence

Landed Muse code records execution-relevant identity across study, analysis, rendering, replication, and release surfaces.

Representative `ReproductionEnvironment` coordinates include:

- operating system;
- architecture;
- Nix version;
- flake-lock SHA-256;
- toolchain-evidence SHA-256;
- execution-environment SHA-256;
- commands SHA-256.

Muse additionally binds source/input/output commitments at multiple evidence boundaries.

Status for recurrence review: **landed independent domain**.

### 2. Fabrication deterministic replay

Landed Fabrication code records `ReplayEnvironment` coordinates including:

- kernel version;
- source revision;
- target triple;
- rustc version;
- Cargo.lock digest;
- canonical feature flags;
- deterministic seed;
- algorithm/version inventory.

It separately binds fabrication manifest/trust/audit semantics.

Status for recurrence review: **landed independent domain**.

## Open-candidate recurrence

### 3. Neuro / Connectome Workbench execution capsule — PR #624

PR #624 independently identifies the same execution-boundary problem for a dynamically linked scientific tool.

Its proposed stronger future identity includes:

- exact root flake selection;
- pinned nixpkgs revision and NAR root;
- Workbench package/source metadata;
- realized recursive Nix runtime closure;
- per-path NAR hashes and references;
- actual `wb_command` SHA-256;
- exact version-output SHA-256;
- Nix version;
- platform identity;
- controlled locale/timezone/OpenMP/HOME/XDG/TMP environment.

Its own authority boundary explicitly says the current profile does **not** establish realized closure or scientific execution qualification.

Current repository status on 2026-09-09:

- open;
- draft;
- mergeable;
- not merged.

Status for recurrence review: **independent design evidence, not landed production recurrence**.

### 4. Genesis / ALife Earth-forced execution capsule — PR #1052

PR #1052 independently composes a profile-specific execution capsule from:

- validated population snapshot;
- encounter scheduler snapshot;
- Earth forcing snapshot;
- Genesis behavior/lifecycle checkpoint;
- environment/behavior tick consistency;
- fixed-partner identity/lifecycle relationships;
- persisted runner/scheduler/environment state;
- inheritance/mutation RNG and construction seeds;
- identity allocation and evolutionary lifecycle state.

Its split-run regression aims to prove serialize/revalidate/restore/continue equivalence across nontrivial stochastic/evolutionary execution.

Its own boundary says the work remains a source candidate until exact CI executes and excludes cryptographic provenance and arbitrary environment/closure serialization.

Current repository status on 2026-09-09:

- open;
- draft;
- mergeable;
- not merged.

Status for recurrence review: **independent design evidence, not landed production recurrence**.

## Merged-history check

A repository search for merged pull requests carrying the literal `execution capsule` lineage currently returns zero results.

This does not mean the concepts are absent from main; Muse and Fabrication clearly contain related replay/reproduction mechanics under different names.

It does mean that the newer explicit capsule architecture has not yet crossed the merge boundary.

## Recurrence matrix

| Mechanic | Muse landed | Fabrication landed | Neuro #624 candidate | Genesis #1052 candidate |
|---|---:|---:|---:|---:|
| source/artifact identity | yes | yes | yes | yes |
| declared execution profile | partial/yes | yes | yes | yes |
| realized environment identity | yes | partial/yes | planned explicitly | profile-specific causal state |
| dependency/runtime closure | partial | partial | strong planned Nix closure | not generic |
| command/invocation identity | yes | replay contract | yes | runner semantics |
| input snapshot identity | yes | manifest | yes | yes |
| stochasticity/seed state | study-dependent | yes | mostly deterministic tool profile | yes, extensive |
| platform identity | yes | yes | yes | profile-specific |
| replay/restore semantics | replication evidence | yes | future execution | yes |
| output/evidence identity | yes | audit outputs | future receipt | persisted capsule/tests |
| authority separated from identity | yes | yes | yes | yes |

This matrix supports the SCI-Q4 **shape**, but only the first two columns are currently landed independent domains.

## Rule-of-three interpretation

Symthaea's rule-of-three should be applied at two levels.

### Design abstraction threshold

There are now at least four independently motivated domain lines supporting the broad distinction:

```text
artifact/source
+ execution profile
+ realized runtime/state
+ inputs/stochasticity
+ occurrence/output
```

Therefore the Q4 architecture census is justified.

### Production shared-runtime threshold

Only two independent landed domains currently implement sufficiently strong execution/replay identity mechanics on stable main.

Therefore a generic production runtime type in `symthaea-evidence-plane` is **not yet earned solely by landed recurrence**.

Open candidates do not count as merged evidence.

## Consequence for SCI-Q4

Keep SCI-Q4 non-authorizing and docs-first for now.

Do not yet introduce a universal:

- `ExecutionEnvironmentHash`;
- `ExecutionCapsule`;
- dependency-closure schema;
- Nix-only execution model;
- generic remote-attestation type.

The next implementation should occur only after one of these gates is met:

1. a third independent execution-identity implementation lands and its semantics survive review; or
2. a deliberately minimal shared vocabulary is independently qualified against at least three concrete domain fixtures without forcing domain-specific fields into the common layer.

Gate 2 still requires explicit justification for why source candidates are acceptable as fixtures while remaining non-authoritative.

## Common fields that appear safest to abstract later

The recurrence review suggests a small future common envelope may include references rather than owning domain internals:

```text
ExecutionSubjectRef
DeclaredExecutionProfileRef
RealizedExecutionEnvironmentRef
InputSetRef
StochasticityDisposition
ExecutionOccurrenceDisposition
OutputSetRef
```

These should be identity-bearing references to qualified domain/profile objects, not one enormous generic environment struct.

This is deliberately narrower than copying all fields from Muse, Fabrication, Neuro, or Genesis.

## Fields that should remain profile/domain-specific

Examples:

- Workbench recursive Nix closure structure;
- Muse study release and participant evidence;
- Fabrication trust/audit manifests;
- Genesis population/lifecycle/fixed-partner invariants;
- accelerator/device details when scientifically irrelevant to other profiles;
- domain-specific causal state.

Common infrastructure should bind these objects by identity rather than reinterpret their semantics.

## Strong recurring theorem

Across all four lines, the most stable shared rule is not a particular field list. It is:

```text
what was requested
    != what was realized
    != what actually executed
    != what output was produced
    != what scientific claim the output supports
```

That non-equivalence is ready for shared architecture even while concrete execution schemas remain local.

## Relationship to SCI-Q5

SCI-Q5 anti-replay binding should therefore depend on a **verified execution occurrence identity**, not on a prematurely universal execution-environment record.

A domain/profile-specific execution verifier may eventually produce the shared opaque witness.

This allows Neuro, Genesis, Muse, and Fabrication to retain different execution details while still satisfying one anti-replay theorem:

```text
receipt observations belong to this exact verified occurrence
and
this occurrence executed this exact artifact subject
```

## Exit gate for production Q4 implementation

Before introducing shared runtime code, record all of the following:

- at least three concrete execution-identity fixtures;
- status of each fixture: landed, qualified candidate, or architecture-only;
- exact fields common to all selected fixtures;
- exact fields intentionally left local;
- adversarial mismatch cases;
- proof that the shared type does not mint scientific/materialization authority;
- migration impact on existing domain types.

If one fixture is unmerged, the resulting shared implementation remains experimental until that dependency is resolved or replaced with landed evidence.

## Governing principle

> Architectural recurrence may justify a shared question before production recurrence justifies a shared answer. Symthaea should abstract the invariant only as far as the evidence has actually converged.
