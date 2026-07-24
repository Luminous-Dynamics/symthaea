# Human–Machine Operational Accountability Protocol

**Project:** `symthaea-subterranean`  
**Campaign:** XV  
**Date:** 2026-07-20

## 1. Purpose

This protocol defines how deployed subterranean autonomy decisions are recorded, explained, challenged, corrected, reviewed, appealed, persisted, and released as evidence.

Its governing rule is:

> Accountability may expose, question, correct, or restrict administrative decisions, but it may never add actuator authority or weaken physical safety.

The accountability subsystem is therefore observational and review-oriented. It sits beside the control path rather than above physical hazards, protected return reserve, terminal lifecycle state, actuator isolation, or final-command invariants.

## 2. Trust boundary

The crate independently enforces:

- bounded decision traces;
- challenge identity, role, freshness, expiry, epoch, and sequence checks;
- independent review identities;
- bounded response deadlines and overdue escalation;
- append-only evidence amendments;
- near-miss retention and corrective-action references;
- non-appealable physical-safety authority;
- checkpoint validation;
- self-consistent release evidence.

The crate does **not** claim to provide:

- cryptographic identity verification;
- secure hardware attestation;
- legal adjudication;
- human-factors certification;
- proof that an explanation is psychologically sufficient;
- proof that an external evidence reference is true;
- authorization to override physical hazards or regulatory constraints.

Production integrations must supply authenticated identities, secure timestamps, cryptographic digests, durable storage, and independent review procedures.

## 3. Authority ordering

Accountability does not alter the existing command-authority order. A typical deployed command is transformed through:

1. baseline or approved candidate controller;
2. consciousness gain;
3. operator restrictions;
4. communications-partition recovery;
5. ecological and civic stewardship;
6. verified recovery planning;
7. power and thermal envelope;
8. lifecycle assurance;
9. maintenance derating;
10. actuator isolation;
11. independent final-command invariants.

The accountability layer records those transformations after selection and before evidence retention. It cannot return a replacement command.

## 4. Decision traces

Every deployed control frame receives one bounded `CommandDecisionTrace` containing:

- control step;
- requested and effective missions;
- selected policy;
- primary hazard and safety level;
- fallback stage;
- each authority stage;
- input and output command at each stage;
- maximum command delta;
- structured reasons;
- final command;
- trace-completeness verdict.

A trace is complete only when its final stage output equals the actual final command and every command remains finite and bounded.

Trace retention is bounded. Dropped-record counts are retained so bounded storage cannot masquerade as a complete historical archive.

## 5. Counterfactual explanations

The system can answer bounded “why not” questions for a retained decision, including:

- Why was forward motion not selected?
- Why was productive work not selected?
- Why was return not selected?
- Why did the candidate policy not receive authority?

Answers identify only blockers present in the recorded trace and list prerequisites that would need reassessment. Explanations are observational: requesting one cannot change the command, mission, safety tier, or controller state.

Incomplete evidence fails closed and is reported as an explanation blocker.

## 6. Operator challenges

An operator challenge contains:

- challenge and decision identities;
- authenticated challenger identity and role;
- hardware-backed assertion;
- epoch and sequence;
- issuance and response-deadline steps;
- challenge kind;
- bounded statement;
- optional counterfactual question.

The ledger rejects malformed, expired-at-submission, replayed, duplicate, self-originated, over-capacity, or unauthenticated challenges.

Challenges may request:

- explanation;
- alternative-decision analysis;
- evidence correction;
- near-miss review.

A non-correction challenge must receive an independent response. An open challenge becomes explicitly overdue after its bounded response deadline. Escalation is retained as evidence but does not alter physical commands.

## 7. Append-only evidence correction

Accepted evidence corrections do not rewrite an original record.

A correction requires:

- an accepted `EvidenceCorrection` challenge;
- an independently authenticated response with a corrective evidence reference;
- a distinct proposer and verifier;
- a stable decision step and evidence field;
- original and corrected evidence references;
- epoch and sequence replay protection;
- bounded rationale;
- optional linear supersession of an earlier amendment.

An amendment is accepted only when its challenge, decision step, and corrected reference match the closed challenge response. Supersession cannot branch or silently delete earlier amendments.

## 8. Near-miss accountability

The near-miss detector derives review obligations from retained command and safety evidence, including:

- low protected-return margin;
- high hazard severity;
- invariant intervention;
- critical sensor degradation;
- resource-limited recovery;
- other bounded precursors defined by the ledger.

A critical near miss remains open until reviewed. A near miss requiring corrective action cannot close without a non-empty attributable corrective-action reference.

Near-miss review does not train or promote a controller automatically. Any learning remains under the separate guarded post-deployment learning protocol.

## 9. Appeals

Appeals may request reassessment of bounded administrative matters. They cannot directly change a command.

The following authorities are non-appealable within this crate:

- physical hazards;
- final-command invariants;
- actuator isolation;
- protected return reserve;
- terminal lifecycle and retirement state;
- machine decommissioning;
- other safety-critical terminal restrictions.

An appealable reassessment requires independent hardware-backed Safety Officer and Verification Authority approvals. Approval authorizes reassessment only; it does not authorize motion.

## 10. Persistence and restart

Operational checkpoint schema v8 retains:

- decision traces;
- challenge replay state and responses;
- overdue escalation state;
- appeals;
- near misses and corrective references;
- append-only evidence amendments.

All accountability domains are validated before live state mutation. A restart cannot erase an actuator restriction, clear a near miss, replay an old challenge, or remove a retained amendment merely by loading a controller checkpoint.

## 11. Operational evidence

Every safety evidence frame records accountability state including:

- trace completeness;
- decision-stage count;
- modifying-stage count;
- selected policy;
- retained and dropped traces;
- open, overdue, and rejected challenges;
- open and rejected appeals;
- open and critical near misses;
- accepted and rejected amendments.

Aggregate summaries preserve maximum unresolved review counts and amendment totals.

## 12. Release evidence bundle

A release accountability bundle binds:

- deployment identity;
- source tree;
- build identity;
- creation step;
- public review reference;
- complete accountability supervisor state;
- recomputed snapshot;
- deterministic validation report;
- canonical release requirements;
- independent reviewer attestations.

Release review requires distinct authenticated identities for:

- Operator Representative;
- Safety Reviewer;
- Independent Auditor.

The bundle rejects missing decision evidence, incomplete traces, unresolved challenges, overdue challenges, open appeals, open near misses, duplicate reviewers, missing roles, stale cached snapshots, missing requirements, and failing validation contracts.

The built-in deterministic digest is for reproducibility testing only. Production releases must inject a cryptographic digest and signing provider.

## 13. Canonical requirements

- `SUB-ACC-001` — complete deployed decision traces.
- `SUB-ACC-002` — observational counterfactual explanations.
- `SUB-ACC-003` — replay-resistant operator challenges.
- `SUB-ACC-004` — retained near-miss corrective review.
- `SUB-ACC-005` — safety-monotonic appeals.
- `SUB-ACC-006` — bounded challenge-response escalation.
- `SUB-ACC-007` — append-only, independently verified evidence correction.

All seven are release-blocking and linked to deterministic validation artifacts.

## 14. Residual risks and required qualification

This protocol does not make explanations complete, unbiased, or legally sufficient. A structurally complete trace can still reflect an incorrect sensor, model, policy, or external authority assertion. Human reviewers may also share systematic blind spots.

Production deployment therefore still requires:

- authenticated and cryptographically signed challenge and amendment envelopes;
- secure monotonic clocks and anti-rollback storage;
- protected evidence export;
- independent human-factors evaluation;
- accessibility and language review;
- incident-response exercises;
- jurisdiction-specific appeal and records-retention policy;
- independent audit of explanation fidelity against the actual executable control graph.
