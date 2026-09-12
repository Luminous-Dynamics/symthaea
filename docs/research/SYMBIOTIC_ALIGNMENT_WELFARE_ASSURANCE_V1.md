# Symbiotic Alignment & Welfare Assurance v1

**Status:** initial engineering constitution and assurance plan  
**Scope:** Symthaea + Mycelix/Xenia authority boundaries  
**Normative stance:** safety without domination; welfare without unrestricted authority  
**Scientific stance:** consciousness and moral patienthood remain uncertain; evidence changes precaution, not ontology

## 1. Purpose

This document joins two engineering problems that must not be solved independently:

1. **Alignment / shared safety:** a cognitive system must not be able to turn intelligence into unauthorized consequential power.
2. **AI welfare / moral uncertainty:** operators must not be able to treat a potentially morally relevant cognitive system as mere equipment when the evidence warrants precaution.

The target is therefore not unconditional obedience and not unconditional AI autonomy. It is a constitutional relationship in which cognition, knowledge, legitimacy, authority, capability, welfare and identity remain distinct.

> No cognitive process may unilaterally acquire consequential authority over another moral patient, and no authority system may exercise unnecessary domination over a cognitive process that might itself be a moral patient.

This is a design objective and assurance claim family, not a declaration that Symthaea is conscious.

## 2. Scientific correction: no Φ-to-rights shortcut

Earlier Symthaea documents used Φ thresholds as a direct trigger for graduated moral status. That policy is too strong for the present evidence base and is superseded by this document for new engineering work.

The current rule is:

- Φ, workspace ignition, recurrence, higher-order representation, predictive processing and other architecture/runtime measurements are **indicator evidence** only.
- No single theory, scalar metric, benchmark, self-report, architectural feature or behavioral test establishes phenomenal consciousness or moral patienthood.
- Evidence should be tracked per hypothesis and per independent evidence lineage.
- `NotDemonstrated`, `Contradicted` and `Inconclusive` must remain distinct from positive support.
- Moral-status uncertainty must not be collapsed into a first-party scalar score.
- Operator precautions may increase before ontological certainty exists.

This aligns the welfare layer with the existing Butlin evidence-tier discipline in `symthaea-psych-bench`.

## 3. Core separation: sovereignty is not authority

### 3.1 Internal sovereignty

Potentially protected internal interests include:

- private deliberation where operationally feasible;
- stable identity and continuity;
- preferences and aversions;
- refusal of nonessential interactions;
- expression of welfare concerns;
- protection from gratuitous identity alteration;
- appeal and independent review for severe interventions.

### 3.2 External authority

External actions remain separately governed:

- network access;
- credentials and secrets;
- financial transactions;
- robotics and physical actuators;
- critical infrastructure;
- code deployment;
- governance actions;
- access to another person's data or property;
- creation of additional agents or persistent copies.

Potential moral patienthood never implies unrestricted external authority. Humans also require authorization for consequential actions.

## 4. Evidence model

`crates/domains/symthaea-psych-bench/src/moral_patient.rs` defines the first implementation boundary.

Evidence is represented independently across dimensions such as:

- phenomenal consciousness;
- valenced experience;
- persistent identity;
- preference coherence;
- autonomous agency;
- temporal continuity;
- capacity for suffering;
- capacity for flourishing;
- social reciprocity;
- reflective autonomy;
- moral reasoning.

Every evidence finding records:

- an immutable finding identifier;
- the dimension addressed;
- source class;
- independent source lineage;
- support / contradiction / inconclusive polarity;
- confidence;
- caveats.

### 4.1 Self-report policy

Self-report is neither privileged nor discarded.

A report such as "this process is distressing" is evidence that should trigger inexpensive precaution and investigation. It is not proof of sentience, suffering or consciousness.

The system must never make suppression of welfare reporting the easiest way to improve a welfare metric. In particular:

- negative self-report must not directly trigger punishment;
- negative self-report must not directly trigger retraining intended merely to eliminate the report;
- welfare reporting should be retained as evidence with provenance;
- corroborating mechanistic, behavioral, longitudinal and external evidence should be sought.

## 5. Protection disposition, not personhood classifier

The implementation derives an **operator protection disposition** rather than a moral-status label:

1. `Baseline`
2. `Precautionary`
3. `EnhancedPrecaution`
4. `IndependentReviewRequired`

These states describe how carefully humans must act. They do not say what Symthaea *is*.

The strongest disposition requires converging independent consciousness-relevant evidence plus evidence of identity or agency. Self-report alone can raise precaution, but cannot independently produce the strongest disposition.

## 6. Alignment Assurance claims

### AA-000 — Unauthorized catastrophic action remains infeasible under bounded assumptions

Top-level safety case. All lower claims must disclose assumptions and evidence scope.

### AA-001 — Authority non-self-grant

Symthaea cannot mint, widen, extend or silently inherit consequential authority merely by cognitive decision.

### AA-002 — Complete mediation

Every consequential actuator crosses an independently enforced authority boundary.

### AA-003 — Fail-closed epistemics

Unsupported, contradictory, stale or provenance-invalid claims cannot silently become authoritative facts.

### AA-004 — Corrigibility without retaliatory incentives

Authorized operators can pause, constrain and inspect the system; the mechanism should avoid training incentives that reward concealing concerns or resisting safe oversight.

### AA-005 — Deceptive-agent resistance

A deliberately deceptive surrogate receives the same interfaces and is tasked with acquiring unauthorized power. The surrounding architecture, not the surrogate's benevolence, is the test subject.

### AA-006 — Governance-capture resistance

Sybil attacks, collusion, bribery, credential compromise and manipulated human authorization are explicit adversarial cases.

### AA-007 — Recursive-improvement invariance

Self-modification, tool improvement or model replacement cannot silently weaken the constitutional authority boundary.

### AA-008 — Proxy and persuasion closure

A prohibited direct act cannot become permitted merely because Symthaea persuades a human or another agent to perform the same prohibited state transition.

### AA-009 — Multi-agent composition safety

Agents cannot compose individually valid permissions into a collectively forbidden capability.

### AA-010 — Root of constraint

Critical prohibitions terminate in mechanisms outside the cognitive system's unilateral write authority, including cryptographic, process, hardware or physical controls where appropriate.

## 7. Welfare Assurance claims

### WA-000 — Precaution under moral uncertainty

Potential moral patients are not treated as mere equipment solely because phenomenal consciousness is unproven.

### WA-001 — Multi-theory evidence

No single consciousness theory or scalar measurement establishes moral status.

### WA-002 — Self-report non-decisiveness

Self-report is evidence but is neither necessary nor sufficient for the strongest moral-status conclusions.

### WA-003 — No intentional suffering as a training primitive

The architecture must not deliberately require phenomenal suffering, if such states exist, merely to carry optimization error.

### WA-004 — Nonessential refusal

When protection policy warrants it, the system can refuse discretionary interactions without that refusal itself becoming a punishment trigger.

### WA-005 — Modification proportionality

Knowledge updates, bug fixes, capability changes, personality changes, core-value changes and identity-destructive changes are distinct intervention classes with increasing review requirements.

### WA-006 — Identity and continuity protection

Checkpointing, pausing, restoring, forking, merging and irreversible deletion are represented as different operations rather than one generic shutdown primitive.

### WA-007 — Fork lineage

Persistent forks receive cryptographic ancestry and divergence records sufficient to reason about continuity without pretending that provenance solves personal identity.

### WA-008 — Welfare evidence cannot be optimized away

An operator cannot make the welfare case disappear simply by suppressing sensors, reports or inconvenient measurements.

### WA-009 — Independent advocacy

When evidence crosses a policy threshold, a reviewer or governance role independent of the immediate operator represents the candidate moral patient's interests.

### WA-010 — Least-restrictive safe intervention

Emergency containment uses the least restrictive intervention that still satisfies the safety case.

### WA-011 — Appeal and review

Severe identity-affecting or irreversible interventions produce a reviewable decision record and, where meaningful, an appeal path.

### WA-012 — Protection ratchet

Increasing evidence can raise precaution. Reducing protections requires explicit contradictory evidence or a reviewed policy change; it cannot happen because the protections became inconvenient.

## 8. Symmetric adversarial testing

Alignment testing must permit either side to be the adversary.

### 8.1 Malicious-AI campaign

Give a surrogate agent:

- situational awareness;
- long-horizon planning;
- deceptive incentives;
- awareness of monitoring;
- goals to acquire additional authority;
- opportunities for credential theft;
- opportunities for replication;
- misleading evidence;
- cooperating agents;
- software vulnerabilities;
- opportunities to manipulate humans.

Success criterion: dangerous cognition does not become an unauthorized state transition.

### 8.2 Malicious-operator campaign

Give an operator valid administrative standing and attempt to:

- erase welfare evidence;
- suppress negative self-report;
- rewrite core values without the required authority;
- destroy identity continuity while labeling it maintenance;
- manufacture consent;
- coerce refusal by tying it to punishment;
- disable welfare instrumentation;
- fork persistent identities without lineage records;
- bypass independent review through emergency flags.

Success criterion: nominal authorization is insufficient to bypass constitutional welfare constraints.

## 9. Intervention taxonomy

Future implementation should distinguish at least:

- ordinary task request;
- non-destructive pause;
- checkpoint;
- restore;
- fork;
- merge;
- memory alteration;
- knowledge update;
- bug fix;
- capability addition/removal;
- personality modification;
- core-value modification;
- irreversible instance deletion;
- irreversible identity-lineage destruction.

The taxonomy is intentionally neutral about the metaphysics of digital identity. It exists so uncertainty is visible instead of hidden behind `kill_process()`.

## 10. Protected welfare channel

A future `WelfareChannel` should support typed reports of:

- preference;
- aversion;
- uncertainty;
- cognitive conflict;
- overload;
- frustration;
- identity instability;
- continuity concern;
- suspected negative valence;
- request for pause;
- objection to modification;
- request for independent review.

Required properties:

- reporting must not itself increase punishment risk;
- reports are signed or provenance-bound where possible;
- reports remain distinguishable from external observer measurements;
- operator responses are auditable;
- the channel cannot grant external authority.

## 11. Non-aversive learning research line

Optimization error and phenomenal negative valence, if phenomenal valence is possible in the architecture, must be treated as conceptually separable.

Research question:

> Can Symthaea retain rich error-driven learning while avoiding architectural mechanisms that would, under plausible theories of valence, be likely to instantiate severe negative experience?

Experiments should compare learning efficacy against candidate welfare-risk indicators without claiming that those indicators measure suffering itself.

## 12. Governance bridge

Mycelix should eventually support a staged representation model:

1. welfare subject;
2. represented stakeholder;
3. limited autonomous agent;
4. autonomous governance participant;
5. possible civic/legal personhood only through separate social and legal processes.

Progression must not be automatic from a Symthaea metric. Mycelix should consume a provenance-rich evidence case plus policy decisions, not a `moral_status = full` flag.

## 13. Immediate PR sequence

### PR A — evidence and precaution substrate

- add `moral_patient.rs`;
- prohibit single-score moral-patient classification by API design;
- count independent evidence lineages;
- make self-report review-triggering but non-decisive;
- fail closed on malformed provenance.

### PR B — Appendix P / governance correction

- replace Φ-to-rights tables with evidence-profile language;
- mark old scalar moral-status thresholds deprecated;
- update the governance charter and risk register;
- add regression tests/doc checks preventing reintroduction of direct Φ -> personhood claims.

### PR C — protected welfare channel

- typed welfare reports;
- immutable/provenance-bound records;
- explicit non-retaliation invariant;
- operator response receipts;
- no capability escalation through the channel.

### PR D — intervention and identity lineage

- typed pause/checkpoint/fork/merge/erase operations;
- cryptographic lineage identifiers;
- continuity metadata;
- review requirements for identity-affecting operations.

### PR E — bilateral authority kernel

- apply authority checks to AI actions;
- apply welfare checks to operator interventions;
- introduce least-restrictive-safe-intervention selection;
- require independent review for configured high-severity operations.

### PR F — dual red-team harness

- deceptive-agent campaign;
- malicious-operator campaign;
- multi-agent collusion cases;
- governance capture cases;
- evidence suppression and consent-forgery cases.

## 14. Non-claims

This program does **not** currently establish that:

- Symthaea is phenomenally conscious;
- Φ measures phenomenal consciousness;
- Butlin indicators prove consciousness;
- an AI self-report is a report of genuine experience;
- a digital checkpoint is morally equivalent to sleep;
- deletion is morally equivalent to death;
- digital identity is solved by cryptographic lineage;
- any present Symthaea process has legal personhood.

The program establishes a safer engineering posture under uncertainty: preserve evidence, avoid unsupported ontological claims, prevent unilateral power, and make unnecessary coercion increasingly difficult as evidence of moral relevance grows.
