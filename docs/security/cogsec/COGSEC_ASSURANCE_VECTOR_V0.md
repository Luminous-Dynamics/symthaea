# CogSec Assurance Vector v0

Status: **experimental qualification contract**

Applies to: frozen K0 candidate PR #143 and stacked shadow-evidence PR #195

This document prevents a common security-assurance failure: collapsing several independent questions into one green/red status.

A CogSec result is represented as an **assurance vector**, not a scalar score.

## 1. Why a vector

The following claims are different:

1. the reference-monitor kernel compiles and its invariants pass;
2. every scoped runtime transition was observed;
3. each observation was attributed to the correct monitor evaluation and protected resource;
4. enabling observation did not change legacy cognition;
5. exported evidence is complete and internally consistent;
6. durable evidence cannot be silently rewritten;
7. enforcement actually blocks unauthorized state transitions;
8. recovery/replay preserves the same security semantics.

Passing one does not imply another.

For example:

- a perfectly tested kernel can have zero live coverage;
- complete event counts can still be misattributed to the wrong resource;
- correctly attributed shadow evidence can still perturb cognition;
- a behaviorally non-interfering shadow run can still lose evidence;
- complete local evidence can still be forgeable after export;
- a shadow monitor can correctly say `Deny` while legacy code still performs the mutation by design.

Therefore CogSec MUST NOT use a single `validated`, `secure`, `full_coverage`, or maturity boolean as a substitute for the individual claims below.

## 2. Assurance vector

A qualification record SHOULD report at least:

```text
CogSecAssuranceVector {
    kernel,
    observation_coverage,
    attribution,
    non_interference,
    evidence_completeness,
    evidence_integrity,
    enforcement,
    recovery,
}
```

Each dimension is independently evidenced.

Recommended state vocabulary:

```text
NotClaimed
Designed
MechanismTested
Qualified
IndependentlyReviewed
```

`Qualified` means the claim-specific exit gate below passed against a recorded immutable revision and qualification environment. It does not promote any other dimension.

## 3. K — Kernel correctness

Question:

> Does the small deterministic reference-monitor package implement its stated local transition algebra and authority boundary?

Evidence includes:

- exact package compilation under the pinned toolchain;
- package unit/property/compile-fail tests;
- package documentation tests;
- `clippy --all-targets -- -D warnings`;
- policy/permit/domain-seal invariants;
- later Kani/TLA+/formal obligations where appropriate.

Repository-wide CI is contextual evidence, not a substitute for package-level K evidence.

### K0 exit gate

The exact frozen kernel source must have recorded package-level compile/test/doc/clippy evidence. A red unrelated workspace job may block repository merge policy, but it must not be misreported as a kernel semantic failure. Conversely, a green root-package workflow that never compiles the package must not be misreported as kernel qualification.

## 4. O — Observation coverage

Question:

> Did shadow instrumentation observe every transition in the independently declared scope?

Evidence includes:

- scenario-owned exact/relational event-count contracts;
- an independently fixed `QualificationManifest`;
- event-sequence completeness;
- required-event loss accounting;
- event-derived P0 denominator;
- generic evidence-plane counter reconciliation.

### O0 exit gate

For the declared scenario scope:

```text
scenario expected counts
    == typed retained event counts
    == mechanism counters
```

and required evidence loss is zero.

A runtime-produced snapshot may not define or shrink its own denominator.

## 5. A — Attribution integrity

Question:

> Does each shadow evaluation actually describe the transition that the outer event claims, and does each paired legacy mutation concern that same protected resource/lineage?

This is the #199 gate.

Evidence includes:

- receipt stage matches the event stage;
- unambiguous outer event kind ↔ kernel `MutationKind` mappings agree;
- outer protected resource equals receipt resource;
- policy root/epoch agree;
- authorization and revocation epochs agree;
- state-root/freshness facts agree when present;
- the paired observed mutation targets the same protected resource;
- future transaction identity matches across prepare/evaluate/observe lineage.

### A0 exit gate

No scoped event can qualify by relabelling a valid receipt for a different transition or by pairing an evaluation of resource A with a mutation of resource B.

Unresolved mappings such as `WorkingStateInfluenceEvaluated` and `DreamMergeEvaluated` MUST be reported as taxonomy limitations rather than coerced into unrelated mutation classes.

## 6. N — Behavioral non-interference

Question:

> Does turning shadow observation on leave legacy cognition unchanged?

Evidence is the deterministic `LegacyBehaviorProjection` comparison defined in `COGSEC_SHADOW_QUALIFICATION_V0.md`.

### N0 exit gate

For identical initial state, deterministic seeds, inputs, and normalized external time:

```text
LegacyBehaviorProjection(audit_off, tick_n)
    == LegacyBehaviorProjection(audit_on, tick_n)
```

for every qualified tick and scenario.

No field is excluded merely because it differs. The exclusion/normalization list is closed-by-review.

## 7. C — Evidence completeness

Question:

> Is the local evidence set structurally complete enough to support the claimed observation/attribution result?

Evidence includes:

- no required buffer loss;
- no duplicate event identities;
- no missing assigned sequence;
- no missing causal parent;
- no invalid causal ordering;
- no event/counter mismatch;
- explicit `Complete` / `DegradedEvidence` / `InvalidQualification` state.

### C0 exit gate

Any missing required evidence invalidates the affected qualification claim rather than being silently ignored or synchronously blocking legacy cognition.

## 8. I — Evidence integrity / authenticity

Question:

> Can exported or durable evidence be trusted as having the claimed producer, ordering, and immutable history?

Current #195 portable snapshots and `MutationReceiptRecord`s are ordinary data. Therefore I is intentionally **NotClaimed** in the current tranche.

Future evidence includes:

- authenticated producer envelope;
- tamper-evident checkpoint/hash chain;
- exact schema/policy/environment roots;
- restart/epoch continuity evidence;
- Xenia-backed signature/identity verification where appropriate;
- selective-disclosure/privacy-preserving export where needed.

### I0 exit gate

A verifier can detect removal, insertion, reordering, substitution, or cross-run splicing at the assurance level being claimed.

## 9. E — Enforcement closure

Question:

> Can an unauthorized protected mutation still occur through any live write path?

Shadow mode deliberately cannot satisfy this claim.

Evidence eventually includes:

- protected state owned behind permit-consuming APIs;
- no raw mutator handles outside the protection boundary;
- complete mutation census;
- authorization/precommit/commit under one serialized owner boundary;
- stale state/policy/revocation rejection;
- single-use permit replay rejection;
- bypass tests demonstrating zero unmediated protected writes.

### E0 exit gate

For the declared protected-resource scope:

```text
privileged_mutations_without_valid_commit_permit == 0
```

and bypass count is zero.

## 10. R — Recovery semantics

Question:

> After restart, revocation, rollback, or selective recomputation, does the security history remain truthful and do stale authorities remain invalid?

Evidence eventually includes:

- explicit owner/ledger epochs;
- monotonic revocation epochs;
- immutable status-change events;
- checkpoint parent/result roots;
- deterministic replay/selective recomputation;
- tests showing revoked/stale permits cannot revive after restart.

## 11. Claim composition

High-level statements are conjunctions over explicit dimensions.

Examples:

### “Kernel qualified”

Requires K only.

### “Shadow instrumentation coverage qualified”

Requires at least:

```text
K + O + C
```

for the declared scope.

### “Shadow evidence attribution qualified”

Requires:

```text
K + O + A + C
```

### “Behaviorally safe shadow qualification”

Requires:

```text
K + O + A + N + C
```

### “Authenticated shadow evidence”

Requires:

```text
K + O + A + N + C + I
```

### “Enforcement-qualified”

Requires at least:

```text
K + O + A + N + C + I + E
```

for the declared protected-resource scope, plus the relevant recovery claim when restart/replay is part of the threat model.

No missing dimension may be replaced by a weighted average or confidence score.

## 12. Current truthful state

At the time this contract was introduced:

- **K:** not yet qualified; frozen #143 has successful Showroom evidence but its repository-wide CI is red on unrelated workspace gates, and focused package-level CogSec evidence is still required;
- **O:** designed / mechanism-test substrate exists in stacked #195, but no `ContinuousMind` runtime hooks have executed;
- **A:** designed; #199 defines the next binding gate and is not yet implemented;
- **N:** designed; `LegacyBehaviorProjection` is specified but not yet executed;
- **C:** local typed ledger/reconciliation substrate exists, but package CI/runtime evidence is pending;
- **I:** not claimed; portable records are not authenticated/tamper-evident yet;
- **E:** not claimed; shadow mode intentionally does not block legacy mutations;
- **R:** designed at architecture level, not qualified in this tranche.

This section is a snapshot, not a permanent registry. Durable qualification records should bind exact commits, toolchains, policies, manifests, scenarios, and evidence roots.

## 13. Constitutional rule

> CogSec assurance is monotonic only through evidence, never through naming.

Adding a crate, event, policy, counter, proof-shaped object, or CI badge does not increase an assurance dimension unless the claim-specific qualification gate actually executes and records evidence.
