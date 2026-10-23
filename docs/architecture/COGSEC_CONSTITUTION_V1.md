# CogSec Constitution v1

Status: experimental architectural foundation. This document defines invariants and non-claims before runtime integration begins.

## Purpose

CogSec protects privileged cognitive state transitions. It is not a truth oracle, content censor, prompt-injection classifier, or substitute for application-level safety policy.

The security objective is narrower and stronger:

> Even when cognition is mistaken, deceived, stressed, poisoned, socially influenced, or uncertain, untrusted influence must not silently promote itself into persistent trust, learning, authority, disclosure, or consequential action.

## Constitutional invariants

1. **Data never grants authority.** Authority is supplied separately through explicit capabilities or trusted authorization adapters.
2. **Transformation never increases control integrity by itself.** Summarizing, translating, embedding, bundling, reasoning over, or remembering data cannot make it more privileged.
3. **Confidentiality never decreases without explicit declassification.** Derived data is at least as restrictive as every contributing input.
4. **Provenance cannot silently disappear.** Derived objects retain commitments to all security-relevant inputs.
5. **Consensus does not constitute truth.** Source count, peer count, reputation, or social agreement cannot by themselves establish factual correctness.
6. **Authentication does not constitute factual correctness.** A valid signature proves a statement about origin/integrity, not external-world truth.
7. **Factual support does not constitute authorization.** A well-supported claim cannot activate a goal, alter policy, write trusted memory, train a model, or execute an action without the required authority.
8. **Memory does not constitute belief.** Remembering that a source asserted X is distinct from endorsing X.
9. **Belief does not constitute instruction.** A proposition may be accepted without becoming an imperative.
10. **Instruction does not constitute permission.** An imperative cannot grant itself the capability needed to execute.
11. **Internal cognitive confidence does not constitute external evidence.** Phi, coherence, salience, model confidence, embedding confidence, affect, familiarity, and related internal metrics cannot create security authority or factual provenance.
12. **Peer influence is bounded and non-authoritative by default.** Trust in a peer as an information source does not imply permission for that peer to alter active cognition.
13. **Learning is a privileged persistent mutation.** Remote gradients, LoRA deltas, model updates, and other learned adaptations are staged and qualified before promotion.
14. **Revocation dominates prior grants.** A revoked or stale capability cannot authorize a new privileged mutation.
15. **Privileged mutations bind to the exact approved effect and state.** Approval of one object/state cannot be replayed against another.
16. **Recovery preserves history.** Corrections and revocations append evidence and derived-state changes rather than rewriting the historical record.
17. **CogSec cannot silently become cognitive authority.** It enforces provenance, persistence, confidentiality, and authorization boundaries; it does not decide political, cultural, scientific, moral, or personal truth by fiat.
18. **Failure removes privilege before it removes cognition.** Loss of network, identity, provenance, or authorization services should preserve observation and reasoning when possible while failing closed on privileged mutation.
19. **Fail open to observation; fail closed to influence.** Where policy permits receipt of unauthenticated or low-integrity telemetry, that receipt does not grant permission to alter trusted cognition.
20. **Missing provenance reduces privilege.** Unknown origin is never interpreted as trusted internal origin.
21. **Security attributes are authoritative only from trusted context.** A principal cannot authoritatively self-report its own trust, role, capability, or influence weight inside an untrusted payload.
22. **No floating-point cognitive metric directly authorizes a privileged mutation.** Authorization decisions use deterministic security facts and policy.

## Protected mutation classes

CogSec distinguishes observation from mutation. The initial mutation taxonomy is:

- attention/salience change;
- affect/neuromodulatory change;
- working-memory admission;
- persistent-memory commit;
- semantic-knowledge promotion;
- learning/model promotion;
- active-goal activation or modification;
- trust-policy change;
- CogSec/security-policy change;
- tool invocation;
- external action;
- declassification/egress release.

Not every deployment must mediate every class at the same assurance level. The kernel defines the vocabulary and deterministic decision surface; deployment profiles decide which classes are privileged.

## Trusted computing base

The logical CogSec TCB should remain deliberately small:

- security-label algebra;
- canonical mutation requests;
- policy IR evaluation;
- capability facts supplied by trusted adapters;
- revocation/epoch checks;
- one-use permit issuance;
- mutation receipts;
- deterministic reason codes.

The following are explicitly **not** trusted to make authorization decisions:

- LLM output;
- HDC similarity;
- Phi/consciousness metrics;
- affective state;
- prompt-injection detectors;
- reputation algorithms;
- federated consensus;
- web research;
- remote peers;
- model confidence;
- natural-language explanations.

These systems may produce signals or proposals, but they cannot manufacture the security facts that authorize privileged state changes.

## Core mediation theorem

For every privileged mutation `M`:

1. the caller constructs a `MutationRequest` describing the intended effect;
2. trusted adapters provide security facts independently of the untrusted payload;
3. the deterministic reference monitor evaluates the request against canonical policy and current epochs/state roots;
4. only an `Allow` decision may mint a one-use `MutationPermit<M>`;
5. the protected state owner consumes that permit while verifying the bound resource/policy/epoch state;
6. the commit produces a `MutationReceipt`;
7. no alternate public mutation path may bypass this sequence.

The long-term release criterion is therefore not merely detector accuracy. It is **mutation-sink coverage**: every privileged state write must have a known owner and mediation path.

## Information-flow rules

For data transformations, security metadata follows conservative rules:

- confidentiality joins toward the most restrictive contributing input;
- control integrity meets toward the least trusted contributing input unless an explicit endorsement operation is performed;
- taint is monotonic until an explicit resolution event is recorded;
- provenance/dependency commitments are additive;
- authority does not flow with data at all.

An authentication or endorsement operation may establish a new security fact, but that transition must be explicit, attributable, policy-bound, and auditable.

## Transaction model

Privileged cognitive mutation uses prepare/commit semantics:

`proposal -> policy evaluation -> permit -> state-bound commit -> receipt`

Rejected proposals must not leave partial successful-state mutations. Error/audit accounting may be retained when policy explicitly defines it as an independent append-only effect.

A permit is bound to the exact mutation digest, resource-state root, policy root, authorization/revocation epochs, and replay-prevention material. It is not a generic bearer token for arbitrary later changes.

## Audit-first migration

Initial runtime integration should operate in audit/shadow mode:

- classify mutation sinks;
- evaluate what CogSec would permit or deny;
- emit counters/receipts;
- leave legacy behavior unchanged.

Enforcement becomes eligible only when the mutation census shows no unknown privileged write paths and qualification demonstrates acceptable compatibility and false-denial burden.

Audit mode is a migration state, not a permanent production security claim.

## Privacy boundary

Raw private cognition should remain local by default. Federated CogSec signals should prefer hashes, provenance references, revocations, lineage relationships, sanitized indicators, and public evidence references over raw conversations, memories, goals, or private reasoning traces.

## Explicit non-claims

CogSec v1 does not claim:

- that Symthaea cannot be deceived;
- that any model output is true;
- that cryptographic verification proves external-world correctness;
- that consensus establishes truth;
- complete cognitive-mutation coverage before the census/integration phases are complete;
- formal verification before proof harnesses are implemented and executed;
- production readiness;
- immunity to compromise of the host process or operating system in in-process deployment profiles.

Higher-assurance deployments may later move the same request/decision/permit/receipt protocol across a process, VM, TEE, Xenia service, or hardware boundary without changing the logical contract.

## Initial qualification claims

The first isolated kernel tranche should establish at least these properties:

- merging labels cannot increase control integrity;
- merging labels cannot decrease confidentiality;
- unknown provenance remains low-privilege;
- data transformations cannot manufacture capabilities;
- stale/revoked authorization facts fail closed;
- policy/resource/epoch mismatch prevents permit use;
- every monitor result has deterministic reason codes;
- serialization never deserializes directly into a live mutation permit.

Runtime claims about goals, memory, learning, swarm influence, egress, or tools are deferred until those sinks are explicitly integrated and measured.
