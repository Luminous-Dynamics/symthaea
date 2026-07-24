# Subterranean Capability Campaign XV

## Human–Machine Operational Accountability

**Date:** 2026-07-20  
**Baseline:** hardened v14 / patch 186  
**Campaign range:** patches 187–205

## Executive summary

Campaign XV closes the gap between “the machine acted safely” and “a human reviewer can reconstruct, question, correct, and audit why it acted that way.”

The campaign adds a bounded trace of every command-authority transformation, observational counterfactual explanations, replay-resistant challenges, retained near-miss review, safety-monotonic appeals, response deadlines, append-only evidence amendments, checkpoint continuity, release-blocking requirements, and a self-consistent independently reviewed accountability bundle.

The central constraint is unchanged:

> Accountability can expose and reassess decisions, but it cannot create actuator authority or weaken physical safety.

## Scope

### Decision reconstruction

Every deployed frame now records:

- requested and effective mission;
- selected controller policy;
- each command-authority stage;
- each stage’s input and output command;
- modifying effect and maximum delta;
- structured reasons;
- final hazard, safety tier, fallback, and actual command.

Trace completeness is mechanically checked against the final command.

### Explainability

A bounded counterfactual explainer answers why forward movement, productive work, return, or candidate-policy control was not selected. It reports only blockers found in retained evidence and cannot alter the control path.

### Challenges and response accountability

Operator challenges are:

- externally authenticated and hardware-backed;
- role-scoped;
- epoch/sequence replay-resistant;
- bounded in size and capacity;
- tied to one retained decision;
- subject to an explicit response deadline.

Unanswered challenges become overdue evidence after the deadline rather than disappearing or remaining silently open forever.

### Append-only evidence correction

Evidence corrections require an accepted correction challenge and an independently verified response. The original evidence remains immutable. A correction is appended with stable provenance and may supersede only one matching prior amendment without branching history.

### Near misses and appeals

Near-miss precursors create retained review obligations. Corrective near misses cannot close without an attributable corrective-action reference.

Appeals can request reassessment of bounded administrative decisions, but physical hazards, final-command invariants, actuator isolation, return reserve, retirement, and decommissioning remain non-appealable.

### Persistence and evidence

Operational checkpoint schema v8 preserves all accountability state. Runtime evidence now records trace completeness, unresolved reviews, overdue challenges, near misses, and evidence amendments.

### Release assurance

Seven canonical release-blocking requirements and eight deterministic accountability contracts are integrated into the crate’s existing requirement registry, traceability matrix, and certification validator.

A release accountability bundle requires distinct authenticated:

- Operator Representative;
- Safety Reviewer;
- Independent Auditor.

The bundle recomputes its snapshot and rejects unresolved review state or incomplete evidence.

## Ordered patch sets

### 187 — Bounded command-decision traces

Adds structured per-stage traces over the actual deployed authority chain.

### 188 — Counterfactual “why not” explanations

Adds observational blocker and prerequisite explanations for retained decisions.

### 189 — Replay-resistant operator challenges

Adds bounded authenticated challenge envelopes, independent responses, and replay rejection.

### 190 — Near-miss review ledger

Detects and retains bounded near-miss obligations with corrective-action closure rules.

### 191 — Safety-monotonic appeals

Allows administrative reassessment without making physical safety appealable.

### 192 — Composite accountability supervisor

Composes traces, explanations, challenges, appeals, and near misses without actuator authority.

### 193 — Runtime decision-trace integration

Records every command transformation in the live deployment path.

### 194 — Checkpoint persistence

Advances operational checkpoint schema to v8 and validates review state before activation.

### 195 — Accountability evidence

Adds bounded frame-level and aggregate evidence for decision and review truth.

### 196 — Canonical accountability requirements

Registers `SUB-ACC-001` through `SUB-ACC-005` and traceability links.

### 197 — Deterministic release contracts

Adds cross-system acceptance tests and makes accountability release-blocking.

### 198 — Overdue challenge escalation

Adds response deadlines and persistent overdue state.

### 199 — Challenge deadline requirement

Registers `SUB-ACC-006` and its deterministic release contract.

### 200 — Self-consistent accountability bundle

Binds supervisor state, reviewers, requirements, and validation into one deployment-bound bundle.

### 201 — Append-only evidence amendment ledger

Adds independent verifier provenance, replay resistance, and linear supersession.

### 202 — Challenge-bound correction requirement

Registers `SUB-ACC-007` and verifies an end-to-end correction workflow.

### 203 — Accountability protocol

Documents authority ordering, trust boundaries, workflows, non-claims, and residual risks.

### 204 — Campaign publication

Publishes this campaign scope and ordered change record.

### 205 — Verification and packaging

Publishes exact-tree reconstruction evidence, final validation results, and artifact identities.

## Release gates

The campaign is accepted only if:

- warning-denied offline type checking passes;
- the full deterministic unit-test suite passes;
- the controlled-hardware runtime benchmark remains explicitly ignored outside its qualified environment;
- rustfmt and `git diff --check` pass;
- production panic-marker audit remains clean before test modules;
- patches 187–205 apply cleanly over canonical patch 186;
- all 205 patches reconstruct from the original uploaded snapshot;
- both reconstruction paths produce the exact same Git tree;
- packaged artifact checksums verify.

## Explicit non-claims

Campaign XV does not claim:

- cryptographically authenticated reviewer identities;
- legal sufficiency of an appeal process;
- psychological completeness or fairness of explanations;
- truth of externally referenced correction evidence;
- secure immutable storage;
- human-factors certification;
- production qualification in the omitted Rust 1.94 workspace.

Those remain external integration and qualification responsibilities.
