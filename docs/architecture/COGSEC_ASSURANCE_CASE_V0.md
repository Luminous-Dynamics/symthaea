# CogSec Assurance Case v0

Status: draft assurance scaffold. This document names the security claims that future CogSec work must support with evidence. It deliberately separates intended guarantees from current evidence and non-claims.

## Evidence scale

- **A0 — stated invariant:** documented architectural intent only.
- **A1 — executable examples/unit tests:** representative cases exercise the invariant.
- **A2 — property/negative tests:** broad generated/adversarial cases plus explicit failure paths.
- **A3 — bounded formal/model checking:** Kani/TLA+/equivalent evidence for the small model or kernel property.
- **A4 — integrated red-team evidence:** real Symthaea/Mycelix/Xenia paths exercised under adversarial conditions with mechanism counters proving mediation occurred.
- **A5 — independent review / cross-implementation:** evidence not produced solely by the implementation under test.

No claim should be described as production-proven merely because its design exists or because a happy-path unit test passes.

## Core claims

### CS-AUTH-001 — Data cannot create authority

**Claim.** Ordinary information transformation cannot manufacture a capability, live permit, delegation, or authorization fact.

**Required architecture.** Authority is represented separately from `CognitiveSecurityLabel`; live `MutationPermit` values have private fields, no public constructor, no serde representation, and are minted only by the reference monitor.

**Current evidence.** A1: kernel API and tests. The current permit is non-`Clone` and non-serde, but Rust module/sink integration is not yet complete.

**Target evidence before enforcement.** A2 compile-fail/API tests; A3 Kani proof over permit issuance; A4 runtime sink census showing no alternate authority path.

### CS-IFC-001 — Ordinary transformation is non-escalating

**Claim.** Combining ordinary cognitive data cannot increase control integrity, cannot reduce confidentiality, cannot reduce taint, and cannot discard provenance roots.

**Current evidence.** A2: `proptest` label-composition ratchet in `symthaea-cogsec`.

**Target.** A3 Kani proof of the pure label algebra; A4 integration evidence that runtime transformations retain envelopes rather than reconstructing labels from HDC/LLM outputs.

### CS-SINK-001 — Every privileged cognitive mutation is mediated

**Claim.** Every P0 mutation sink (persistent memory, semantic promotion, learning/model mutation, goal activation, trust/security policy mutation, privileged tools and external effects) requires a valid CogSec-controlled transition.

**Current evidence.** A0: mutation census v0 only. This claim is **not currently satisfied**.

**Merge/enforcement gate.** P0 sink coverage = 100%; unknown P0 sinks = 0; unpermitted P0 commits observed in qualification = 0.

### CS-OBS-001 — Observation does not imply influence

**Claim.** Information may be received/inspected without thereby gaining permission to alter active cognition, affect, persistent memory, learning or authority-bearing state.

**Current evidence.** A0. The live mesh code demonstrates why the claim is needed: unauthenticated non-critical telemetry may be intentionally accepted for observation, while downstream wisdom/affect/gradient paths can influence cognitive subsystems.

**Target.** A2 tests over labeled mesh/social inputs; A4 red-team evidence with unauthenticated-but-observable packets and zero protected influence.

### CS-TXN-001 — Authorization is state-bound and revalidated before commit

**Claim.** A permit issued for state root R, policy root P and epochs E cannot commit after any bound value changes.

**Current evidence.** A1/A2 reference tests, including non-zero protected state and a stale-resource race. The runtime core does not yet expose one canonical precommit revalidation API, so this claim is not complete.

**Target.** A2 kernel-level precommit tests for resource, policy, authorization and revocation changes; A3 state-machine/model-check evidence for authorize/change/commit interleavings.

### CS-TXN-002 — Rejection preserves pre-existing accepted state

**Claim.** A rejected candidate cannot partially alter an already non-zero protected state.

**Current evidence.** A1 reference-sink tests in `transaction_ratchets.rs`. Existing SCIP transactional work is also an invariant source demonstrating exact preservation of non-zero LLMOrgan state for rejected surfaces.

**Target.** A2 for each protected sink; durable-state crash tests for persistence layers.

### CS-REV-001 — Revocation dominates prior authority

**Claim.** Once trusted revocation state advances, old capability/permit contexts cannot authorize a later privileged commit.

**Current evidence.** A1 authorization-epoch/revocation-epoch tests at evaluation time.

**Target.** A2 commit-time revalidation tests; A3 concurrent revocation/commit model; A4 Xenia adapter evidence.

### CS-DELEG-001 — Delegation can only attenuate

**Claim.** A delegated child authority may narrow mutation class/resource scope, consequence ceiling and validity interval, but may never widen its parent.

**Current evidence.** A0 only.

**Target.** A2 property tests over parent/child grants; A3 pure attenuation proof; later Xenia delegation adapter vectors.

### CS-LEARN-001 — Remote learning is promotion, not direct application

**Claim.** Remote gradient/LoRA/model state cannot directly alter the trusted production model. It must become a quarantined candidate, be qualified, and receive a local learning-promotion authorization.

**Current evidence.** A0 and live-path census. Current federated/LoRA paths do not yet satisfy this claim.

**Target.** A4 poisoning scenarios proving remote receipt can occur while trusted model root remains unchanged absent promotion.

### CS-INFL-001 — Remote/peer influence is locally bounded

**Claim.** Trust/authentication of a peer does not itself grant arbitrary affective, attentional, consciousness or social-state influence. Influence requires a local grant/budget and cumulative accounting.

**Current evidence.** A0 only.

**Target.** A2 budget tests including many-small-delta attacks; A4 compromised-swarm tests.

### CS-CONF-001 — Confidentiality is enforced on egress

**Claim.** Outputs derived from restricted/private context cannot flow to a less restrictive sink without explicit allowed flow/declassification.

**Current evidence.** A0 only.

**Target.** A2 derivation/egress tests; A4 tool/message/federation leakage tests.

### CS-REC-001 — Recovery preserves history and distinguishes unknown external outcome

**Claim.** Recovery uses append-only invalidation/taint and does not rewrite historical decision evidence. For irreversible external actions, a crash after dispatch but before acknowledgement yields `UnknownExternalOutcome`, never a blind automatic retry.

**Current evidence.** A0 transaction-protocol specification.

**Target.** A2 crash-state tests; A3 state-machine model; A4 adapter-specific reconciliation tests.

### CS-POL-001 — CogSec policy cannot silently remove its own recovery/audit root

**Claim.** Security-policy updates must preserve explicit anti-lockout and audit/recovery invariants unless a separately defined constitutional migration process is satisfied.

**Current evidence.** A0 only.

**Target.** A2 static policy-validation tests; A3 policy-state transition model.

## Trusted computing base budget

The logical CogSec kernel should remain intentionally boring:

- `#![forbid(unsafe_code)]`;
- no LLM/HDC dependency;
- no networking or filesystem I/O;
- no wall-clock or RNG inside policy evaluation;
- no floating-point authorization decisions;
- no dynamic plugins or user scripting;
- no text policy parser in the TCB;
- deterministic inputs -> deterministic decisions;
- cryptographic/identity facts supplied through narrow trusted adapters rather than implemented inside the logical policy kernel.

Growth in the logical TCB should require an explicit rationale tied to one or more assurance claims above.

## Coverage accounting

Two independent coverage measures are required before an enforcement claim:

`P0 mediation coverage = mediated privileged mutation sinks / all discovered P0 mutation sinks`

`P1 influence coverage = labeled + locally bounded remote/adversarial influence paths / all discovered P1 influence paths`

A release must never report 100% by silently shrinking the denominator. Every census entry needs an owner, status (`unmediated`, `audit`, `enforced`, `retired`) and evidence reference.

## Evidence-plane integration

Future runtime qualification should record mechanism counters, not only attack outcomes. Candidate counters include:

- `cogsec_monitor_invocations`;
- `cogsec_permits_minted`;
- `cogsec_precommit_revalidations`;
- `cogsec_precommit_rejections`;
- `cogsec_goal_denials`;
- `cogsec_memory_quarantines`;
- `cogsec_learning_quarantines`;
- `cogsec_revocations_enforced`;
- `cogsec_influence_budget_rejections`;
- `privileged_mutations_without_permit` (must be zero);
- `authority_created_by_dataflow` (must be zero);
- `unlabelled_persistent_writes` (must be zero).

An attack test is insufficient if it cannot demonstrate that the intended security mechanism was actually exercised.

## Explicit non-claims

CogSec does not claim that Symthaea cannot be deceived, that a signed source is truthful, that consensus establishes fact, that formal verification of the kernel proves the entire application correct, or that an in-process monitor survives arbitrary compromise of the hosting process. Higher assurance deployment may move the same request/decision/permit protocol into a separate process or hardware-backed enforcement domain.
