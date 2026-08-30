# CogSec Assurance Case v0

Status: draft assurance scaffold. This document names the security claims that future CogSec work must support with evidence. It deliberately separates intended guarantees from current evidence and non-claims.

## Evidence scale

- **A0 — stated invariant:** documented architectural intent only.
- **A1 — executable examples/unit tests:** representative cases exercise the invariant.
- **A2 — property/negative tests:** broad generated/adversarial cases plus explicit failure paths.
- **A3 — bounded formal/model checking:** Kani/TLA+/equivalent evidence for the small model or kernel property.
- **A4 — integrated red-team evidence:** real Symthaea/Mycelix/Xenia paths exercised under adversarial conditions with mechanism counters proving mediation occurred.
- **A5 — independent review / cross-implementation:** evidence not produced solely by the implementation under test.

No claim should be described as production-proven merely because its design exists or because a happy-path unit test passes. Newly committed tests do not become evidence until they have executed successfully in a recorded qualification environment.

## Core claims

### CS-AUTH-001 — Data cannot create authority

**Claim.** Ordinary information transformation cannot manufacture a verified capability fact, trusted security snapshot, live permit, delegated authority, or authorization fact accepted by a protected monitor domain.

**Required architecture.** Authority is represented separately from `CognitiveSecurityLabel`. Serializable/wire capability claims remain ordinary data. Verified `CapabilityFact` and `TrustedFacts` values are opaque, non-serde objects issued only by a `TrustedFactAuthority` paired with one private monitor domain. `MutationPermit` and `CommitPermit` have private fields, no public constructor or serde representation, are bound to that same domain, and are minted only through the reference-monitor transition path. A protected sink must reject a commit permit from any other independently bootstrapped monitor domain.

**Current implementation.** The public facade now hides the previous publicly constructible inner facts in a private module, introduces a private monitor-domain seal, role-separates `ReferenceMonitor` from `TrustedFactAuthority`, rejects cross-domain facts before policy evaluation, domain-binds authorization/commit permits, and adds foreign-domain negative tests. The implementation is present on the draft branch; CI validation of the new head is pending, so this document does not yet claim recorded A2 evidence for the sealed facade.

**Target evidence before enforcement.** Recorded A2 negative/domain-isolation test results; A3 Kani proof over issuance/typestate/domain transitions; A4 runtime sink census showing no alternate authority path.

### CS-IFC-001 — Ordinary transformation is non-escalating

**Claim.** Combining ordinary cognitive data cannot increase control integrity, cannot reduce confidentiality, cannot reduce taint, and cannot discard provenance roots.

**Current evidence.** A2-target `proptest` label-composition ratchet is present in the deterministic inner implementation; recorded validation of the current sealed-facade head is pending.

**Target.** A3 Kani proof of the pure label algebra; A4 integration evidence that runtime transformations retain envelopes rather than reconstructing labels from HDC/LLM outputs.

### CS-SINK-001 — Every privileged cognitive mutation is mediated

**Claim.** Every P0 mutation sink (persistent memory, semantic promotion, learning/model mutation, goal activation, trust/security policy mutation, privileged tools and external effects) requires a valid CogSec-controlled transition.

**Current evidence.** A0: mutation census v0 only. This claim is **not currently satisfied**.

**Merge/enforcement gate.** P0 sink coverage = 100%; unknown P0 sinks = 0; unpermitted P0 commits observed in qualification = 0.

### CS-TOPO-001 — Mutation authority does not escape through ambient API topology

**Claim.** Ordinary cognition cannot obtain a handle, mailbox, trait object, mutable reference, trusted fact issuer, protected monitor, or other API surface that can directly perform or authorize a P0 mutation outside the CogSec transition path.

**Current evidence.** A0/A1 architecture work: authority-escape analysis identifies current ambient authority bundles, including `AsyncMindHandle` and the combined inference/mutation `LLMBackend` surface. The sealed facade closes one kernel-local escape (caller-fabricated trusted facts), but the live runtime architecture does **not** yet satisfy this claim.

**Target.** A2 static/API ratchets proving observer/inference/proposal handles exclude privileged mutation or fact-issuance methods; A4 runtime census with unknown P0 mutable/authority handles = 0.

### CS-OBS-001 — Observation does not imply influence

**Claim.** Information may be received/inspected without thereby gaining permission to alter active cognition, affect, persistent memory, learning or authority-bearing state.

**Current evidence.** A0. The live mesh code demonstrates why the claim is needed: unauthenticated non-critical telemetry may be intentionally accepted for observation, while downstream wisdom/affect/gradient paths can influence cognitive subsystems.

**Target.** A2 tests over labeled mesh/social inputs; A4 red-team evidence with unauthenticated-but-observable packets and zero protected influence.

### CS-TXN-001 — Authorization is state-, epoch-, and monitor-domain-bound

**Claim.** An authorization issued for state root R, policy root P, epochs E and monitor domain M cannot become commit authority after any bound value changes or when presented to a protected sink owned by another domain.

**Required architecture.** `authorize()` yields only `MutationPermit`. `precommit()` requires fresh same-domain `TrustedFacts` and yields the distinct `CommitPermit` typestate after resource/policy/authorization/revocation checks. Both permits carry the private monitor-domain seal. The protected owner must perform domain validation, precommit and commit under the same serialization/transaction boundary.

**Current implementation.** Canonical precommit API, distinct authorization/commit typestates, same-domain fact checks, commit-permit domain affinity, unit/negative tests for resource/policy/authorization/revocation context, foreign monitor domains, and non-zero-state races are present. Recorded validation of the current head is pending.

**Target.** Recorded A2 results; A3 state-machine/model-check evidence for authorize/change/precommit/commit and cross-domain interleavings; A4 real protected-sink integration.

### CS-TXN-002 — Rejection preserves pre-existing accepted state

**Claim.** A rejected candidate, stale authorization, revoked authorization, or foreign-domain permit cannot partially alter an already non-zero protected state.

**Current implementation.** Reference-sink tests in `transaction_ratchets.rs` cover denied requests, stale authorization, revocation between authorize/precommit, one-use commit typestate, and foreign monitor-domain permits against non-zero state. Existing SCIP transactional work remains an invariant source demonstrating exact preservation of non-zero LLMOrgan state for rejected surfaces. Recorded validation of the current head is pending.

**Target.** Recorded A2/A4 evidence for each protected runtime sink; durable-state crash tests for persistence layers.

### CS-REV-001 — Revocation dominates prior authority

**Claim.** Once trusted revocation state advances, old capability/permit contexts cannot authorize a later privileged commit.

**Current implementation.** Evaluation-time revocation checks plus commit-time revocation-epoch revalidation are present; failed precommit consumes the authorization token and the non-zero-state ratchets preserve state. The trusted adapter contract requires every security-relevant revocation capable of invalidating an outstanding permit to advance `revocation_epoch`. Recorded validation of the sealed-facade head is pending.

**Target.** Recorded A2 results; A3 concurrent revocation/precommit/commit model; A4 Xenia adapter evidence.

### CS-DELEG-001 — Delegation can only attenuate and verified issuance is privileged

**Claim.** A delegated child authority may narrow resource scope, consequence ceiling and validity interval, while retaining the parent's mutation class and security epochs; it may never widen its parent. Possession of a verified parent fact alone must not allow arbitrary code to mint a new verified child subject.

**Required architecture.** Structural attenuation lives in the private deterministic algebra. The public `CapabilityFact` is opaque and exposes no child-issuance method. Only the same-domain `TrustedFactAuthority::derive_capability()` can convert a structurally valid attenuation into another verified fact, and the external trusted adapter remains responsible for proving that the parent actually authorized delegation and for binding ancestry.

**Current implementation.** The private inner algebra contains resource/consequence/validity widening tests; the sealed facade adds issuer-only child fact creation and rejects a foreign issuer deriving from another domain's parent. Recorded validation of the new head is pending.

**Target.** Recorded A2 negative/property results; A3 pure attenuation/issuance proof; later Xenia delegation-chain vectors proving parent authorization and ancestry binding.

### CS-LEARN-001 — Remote learning is promotion, not direct application

**Claim.** Remote gradient/LoRA/model state cannot directly alter the trusted production model. It must become a quarantined candidate, be qualified, and receive a local learning-promotion authorization.

**Current evidence.** A0 and live-path census. Current federated/LoRA paths do not yet satisfy this claim. Tracked separately in the learning-promotion tranche.

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
- deterministic policy inputs -> deterministic policy decisions;
- serializable wire claims are distinct from opaque verified security facts;
- cryptographic/identity facts are supplied through a narrow `TrustedFactAuthority` rather than implemented inside the logical policy algebra;
- the in-process monitor-domain seal is non-serializable capability identity, not a substitute for cryptographic identity across process/machine boundaries.

Growth in the logical TCB should require an explicit rationale tied to one or more assurance claims above.

## Monitor-domain bootstrap rule

A protected runtime bootstraps exactly the monitor domain it intends to own and keeps the resulting `ReferenceMonitor` and `TrustedFactAuthority` within their assigned trusted roles. Creating another domain is harmless by itself: facts and permits from domain B must fail when presented to monitor/sink A.

This in-process seal does **not** authenticate a remote principal. It prevents safe-Rust callers from manufacturing facts/permits that are accepted by a different protected monitor instance. Xenia or another authenticated adapter remains responsible for turning signed external claims into facts through the trusted issuer.

## Coverage accounting

Three independent measures are required before an enforcement claim:

`P0 mediation coverage = mediated privileged mutation sinks / all discovered P0 mutation sinks`

`P0 authority-topology coverage = privilege-separated or CogSec-owned P0 mutable/authority handles / all discovered P0 mutable/authority handles`

`P1 influence coverage = labeled + locally bounded remote/adversarial influence paths / all discovered P1 influence paths`

A release must never report 100% by silently shrinking a denominator. Every census entry needs an owner, status (`unmediated`, `audit`, `enforced`, `retired`) and evidence reference.

## Evidence-plane integration

Future runtime qualification should record mechanism counters, not only attack outcomes. Candidate counters include:

- `cogsec_monitor_invocations`;
- `cogsec_cross_domain_fact_rejections`;
- `cogsec_cross_domain_permit_rejections`;
- `cogsec_authorization_permits_minted`;
- `cogsec_precommit_revalidations`;
- `cogsec_commit_permits_minted`;
- `cogsec_precommit_rejections`;
- `cogsec_goal_denials`;
- `cogsec_memory_quarantines`;
- `cogsec_learning_quarantines`;
- `cogsec_revocations_enforced`;
- `cogsec_influence_budget_rejections`;
- `privileged_mutations_without_commit_permit` (must be zero);
- `commit_permits_accepted_from_foreign_monitor_domain` (must be zero);
- `authority_created_by_dataflow` (must be zero);
- `unlabelled_persistent_writes` (must be zero).

An attack test is insufficient if it cannot demonstrate that the intended security mechanism was actually exercised.

## Explicit non-claims

CogSec does not claim that Symthaea cannot be deceived, that a signed source is truthful, that consensus establishes fact, that structural attenuation proves a delegation was authorized, that a monitor-domain seal authenticates remote identity, that formal verification of the kernel proves the entire application correct, or that an in-process monitor survives arbitrary compromise of the hosting process. Higher assurance deployment may move the same request/decision/permit protocol into a separate process or hardware-backed enforcement domain.
