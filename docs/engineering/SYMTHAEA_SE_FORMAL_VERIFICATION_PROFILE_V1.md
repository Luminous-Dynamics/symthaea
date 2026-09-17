# Symthaea Systems Engineering Formal Verification Profile v1

Status: architecture profile only. No formal result, solver run, proof artifact, engineering evidence, qualification, certification, or physical authority is established by this document.

Tracking: #3677, #3681, #3685

## 1. Purpose

Define a conservative formal-verification layer for Symthaea systems engineering that composes existing Lean, Z3, proof-audit, and future temporal-model-checking capabilities without allowing solver availability, parser success, fallback selection, bounded exploration, or proof-tool output to become engineering authority by itself.

Canonical laws:

```text
formalization accepted by a tool
!= faithful engineering formalization

solver executed
!= property established

property established in a formal model
!= physical validation
!= requirement satisfaction
!= safety-case closure
!= design qualification
```

ETK remains the authority boundary for evidence admission, currentness, and assurance consequences.

## 2. Existing substrate to reuse

Symthaea already contains:

- a Z3 bridge in the runtime/formal surface;
- a root/runtime Z3-facing compatibility surface;
- `symthaea-lean-bridge`;
- proof-audit infrastructure;
- HDC logic/proposition machinery;
- existing formal-safety and ETK work.

Do not create a second proof-assistant framework when the existing Lean/Z3 surfaces can be hardened and projected through one typed formal-artifact contract.

## 3. Immediate semantic correction

The existing Z3 result model includes a fallback-selection state. A backend-selection event must never itself be interpreted as a positive logical verdict.

Freeze:

```text
preferred solver unavailable
!= SAT
!= UNSAT
!= VALID
!= INVALID
```

Likewise:

```text
fallback selected
!= fallback proved anything
```

A fallback backend may produce its own typed verdict, but backend identity and logical verdict are orthogonal fields.

## 4. Core types

Future implementation should distinguish at least:

```text
FormalPropertyId
FormalModelId
FormalModelRevision
FormalToolIdentity
FormalQuery
FormalRunId
FormalVerdict
Counterexample
ProofArtifact
FormalAssumptionRef
FormalBound
```

### 4.1 FormalProperty

A `FormalProperty` should bind:

- stable property identity;
- human-readable intent;
- exact formal encoding identity;
- source requirement/hazard/interface/mode references when present;
- assumptions;
- units/types where applicable;
- model/configuration snapshot;
- property kind;
- declared scope and limits.

Property kinds may include:

- invariant;
- satisfiability/consistency condition;
- implication/refinement condition;
- deadlock freedom;
- reachability/unreachability;
- liveness/fairness;
- state-transition legality;
- bounded numeric property;
- theorem/proof obligation.

## 5. Verdict semantics

A common normalized verdict should not overload execution-path metadata.

Recommended logical outcomes:

```text
Sat
Unsat
Valid
Invalid
Unknown
```

Recommended execution outcomes/metadata:

```text
Completed
Timeout
Unavailable
ResourceLimit
MalformedOutput
Unsupported
ParserFailure
BackendFailure
```

Backend selection should be explicit:

```text
Backend::Z3
Backend::Lean
Backend::InternalDpll
Backend::Tlc
Backend::Apalache
Backend::Other(...)
```

A positive helper such as `is_sat()` may return true only for an actual `Sat` logical verdict. It must not return true for `Unavailable`, `Unknown`, `FallbackUsed`, or parser/execution metadata.

## 6. Formal execution provenance

Every formal run should eventually bind:

- exact tool/package/executable identity;
- exact tool version;
- adapter version;
- exact query/model/configuration identity;
- property identity;
- solver options;
- random seed where relevant;
- timeout/resource policy;
- working directory/environment identity;
- stdout/stderr/raw artifact digests;
- parser/normalizer version;
- normalized verdict;
- exact SE graph/configuration snapshot;
- boundedness/exploration limits where relevant.

Formal-result provenance answers which exact computation produced a verdict. It does not establish the engineering validity of the model or property.

## 7. Tool roles

### 7.1 Lean

Reuse the existing Lean bridge for theorem/proof-oriented work.

A Lean proof can establish a theorem about the encoded definitions and assumptions. It cannot establish that those definitions faithfully represent a physical system without separate validation.

### 7.2 Z3 / SMT-LIB

Use the existing Z3 surface for SMT properties such as consistency, constraint satisfaction, implications, bounded arithmetic relationships, and selected proof obligations.

SMT-LIB input and output should be retained/digested as first-class artifacts.

### 7.3 TLA+ / TLC

Add a read-only, bounded temporal/state-machine verification path for:

- protocol behavior;
- safety interlocks;
- mode transitions;
- failover logic;
- configuration/currentness state machines;
- distributed approval;
- authority/token lifecycle;
- degraded-mode behavior;
- coordination/concurrency.

Canonical law:

```text
no TLC counterexample in explored state space
!= universal proof
!= physical safety validation
```

The explored state-space configuration, symmetry/reduction settings, bounds, fairness assumptions, and TLC version must be retained.

### 7.4 Apalache

Optional later symbolic TLA+ checker. Do not add it until the common TLA property/model artifact contract qualifies.

## 8. SE graph projection

The formal layer should consume exact SE semantics, not parse arbitrary narrative into authoritative properties.

Possible projections:

```text
Requirement -> candidate FormalProperty
Hazard -> candidate safety invariant
Interface -> compatibility/contract property
Component mode -> transition-system property
Configuration -> formal model context
Change -> property applicability review
```

Projection is proposal-level unless accepted through the appropriate engineering boundary.

Freeze:

```text
automatically generated formal property
!= accepted statement of stakeholder intent
```

Unsupported semantics must be machine-visible rather than silently omitted.

## 9. Counterexamples and failed proofs

Counterexamples are first-class engineering artifacts.

A counterexample should preserve:

- tool/model/property identity;
- state/action trace when available;
- variable valuations;
- initial-state assumptions;
- bounds/configuration;
- raw tool output identity;
- normalized trace/parser identity.

Counterexamples may trigger candidate changes, hazard review, requirement clarification, or additional analysis. They do not directly authorize design changes.

Unknown, timeout, unsupported, and failed-proof outcomes must also be retained. They must not be filtered out of engineering memory.

## 10. Candidate evidence boundary

A formal result may become a `CandidateEvidence` artifact only when bound to:

```text
exact property
+ exact formal model
+ exact SE model/configuration snapshot
+ exact tool/run identity
+ assumptions
+ bounds/validity envelope
```

ETK then decides whether the artifact is admissible/current/applicable for a particular assurance claim.

Canonical law:

```text
formal result
-> candidate evidence
-> ETK admission/currentness
-> possible support for a claim
```

Never:

```text
formal result
-> requirement satisfied
```

## 11. Proposed PR sequence

```text
SE-FORMAL-000  formal verification profile (this document)
SE-FORMAL-001  harden Z3 verdict/backend semantics and canonicalize duplicate surfaces
SE-FORMAL-002  formal execution provenance envelope
SE-FORMAL-003  SE graph -> candidate formal-property projection
SE-FORMAL-004  TLA+ / TLC read-only adapter
SE-FORMAL-005  temporal-transition engineering corpus
SE-FORMAL-006  optional Apalache adapter
SE-FORMAL-007  ETK candidate-formal-evidence bridge
```

Runtime work that depends on SE graph identities must remain blocked until the required lower SE contracts have executable qualification.

## 12. SE-FORMAL-001 minimum regression corpus

At minimum prove:

- unavailable Z3 does not produce SAT merely because fallback is selected;
- fallback backend identity is separate from verdict;
- malformed output fails closed;
- unknown is not truthy;
- timeout is not truthy;
- partial/truncated output is not accepted;
- contradictory status tokens fail closed;
- one canonical implementation/re-export path is established for duplicate Z3 surfaces;
- changed query/property identity cannot reuse an old result as current.

## 13. Temporal verification fixtures

Initial deterministic fixtures should include:

1. redundant-controller failover;
2. two-channel safety interlock;
3. engineering requirement lifecycle transitions;
4. configuration supersession/currentness protocol;
5. distributed/two-person approval workflow;
6. telemetry loss -> degraded mode -> recovery;
7. stale evidence after model revision;
8. deliberately broken liveness case with preserved counterexample.

These fixtures are benchmark/verification artifacts, not claims about real deployed systems.

## 14. Integration with cognition

HDC, Broca, causal reasoning, memory, LTC/CfC, or other cognitive subsystems may:

- propose candidate formal properties;
- propose abstractions;
- suggest invariants;
- prioritize failed properties;
- cluster counterexamples;
- recall analogous proof/model-checking episodes.

They may not:

- rewrite a failed verdict as success;
- suppress a counterexample;
- silently weaken bounds/assumptions;
- silently replace an accepted formal property;
- promote a proof artifact to engineering authority.

## 15. Qualification discipline

Every formal adapter should demonstrate:

```text
exact tool identity
+ exact property identity
+ exact model identity
+ exact options/bounds
+ bounded execution
+ raw artifact preservation
+ strict parser behavior
+ normalized verdict reproducibility
+ adversarial negative corpus
+ immutable postflight
```

For nondeterministic or heuristic formal tools, record the source of nondeterminism and seed/options instead of asserting determinism.

## 16. Non-goals

- no formal-tool output -> deployment authority shortcut;
- no SAT/UNSAT -> safety certification shortcut;
- no proof -> physical validation shortcut;
- no generated formalization silently treated as faithful requirements;
- no second ETK;
- no arbitrary command execution facility;
- no replacement for simulation, test, physical observation, or expert review where those are required.

## 17. Closure criterion

The formal program succeeds when Symthaea can take one exact accepted engineering subject, derive or ingest an explicitly scoped formal property, bind it to an exact model/configuration snapshot, execute an identified formal tool, preserve verdicts/counterexamples/bounds/provenance, and hand a candidate formal-evidence artifact to ETK without any ambiguity between:

```text
what was intended
what was formalized
what was checked
what the checker concluded
what assumptions/bounds applied
what evidence may support
what authority exists
```
