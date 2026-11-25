# MON-010 — Source-Claim Provenance, Decision Binding and Correction Propagation

**Status:** research protocol; no implementation or standards-conformance result is claimed.  
**Tracking:** issue #13679.  
**Parent context:** SFEB MON-007 through MON-009.  
**Version:** 1.0, 2026-11-25.

## 1. Research thesis

An auditable action must preserve both:

1. **What the system did:** exact action, authority, destination commit, independent observation and transparency history.
2. **Why it believed the action was justified:** exact source assertions, source editions, populations, periods, units, transformations, decision record and approval that the action depended on.

A cryptographically intact citation can still be stale or wrong. A correct source value can be misapplied to the wrong population, unit, denominator or time period. A publisher can correct a figure after an agent has used it. An action receipt can carry a producer's claim that a human approved an action without a separately verifiable decision record.

The research target is an **epistemic evidence graph with temporal validity and correction propagation**. It must preserve what was knowable at the time of a historical decision while separately exposing whether the underlying claim remains current now.

This is not another signed receipt format, retrieval score or generic citation feature. It is an independently auditable lineage and change-propagation layer composed with SCITT, source-issued statements, action receipts, measurement evidence and MON-009 sink effects.

## 2. Relationship to existing SFEB work

- **MON-007:** semantic action claims and independent RFC 9162 proof verification.
- **MON-008:** authenticated checkpoints, key history, monotonic tree state and witness continuity. Its custom Ed25519 test profile is not COSE/RFC 9942.
- **MON-009:** destination-enforced conditional mutation, operation identity, fencing, destination receipt, separate-principal readback and bounded observation completeness.
- **MON-010:** the epistemic inputs and signed decision lineage behind an agent claim or action, including later correction/retraction/supersession of source assertions.

MON-010 must not duplicate the receipt/log layer. It should connect content-addressed source and decision artifacts to MON-009 action/effect evidence.

## 3. Current interoperability baseline

The following are active Internet-Drafts as of the dated research snapshot, not published RFCs or IETF-endorsed standards:

- **Agent Action Capsule -05** (draft-mih-scitt-agent-action-capsule-05): action-level digest commitments, disposition vocabulary, a may/did distinction, response-digest binding for producer-confirmed effects, and explicit refusal-path accountability. Its own security section says a producer-bound response is not proof of external-world state.
- **Figure Statements -00** (draft-yetis-scitt-figure-receipts-00, last updated 2026-11-24): publisher-issued source/period/population-bound figure statements, correction and supersession semantics, and re-verification at answer-display time.
- **Decision Provenance for SCITT -03** (draft-lemonnier-scitt-decision-provenance-03, last updated 2026-11-17): separate signed decision records and attachments referenced by software-release or agent-action records. A digest reference proves byte linkage only; the referenced decision object still requires profile-specific validation.
- **Declared-versus-Observed Measurement Capsules -01** (draft-templeman-scitt-measurement-capsule-01, last updated 2026-10-27): independent third-party comparison of a subject's declaration with observed evidence, including a first-class could-not-check state.
- **Agent Execution Evidence -02**, **WIMSE Evidence Binding -03** and **Vaara Receipt -15**: caller/serving-party execution evidence, workload identity/admission and recomputable decision/execution records.
- **RFC 9942 and RFC 9943:** published COSE receipt and SCITT architecture foundations.

Interoperability policy: pin exact versions and consume existing profiles where they fit. Do not claim an Internet-Draft is a published standard, do not clone its payload format without a documented gap, and do not confuse an issuer signature with factual truth.

## 4. Paradigm shift: bitemporal epistemic event sourcing

Represent evidence changes as append-only events, not edits to prior facts.

Every material source/claim event should carry two distinct time dimensions:

- **Valid/source time:** the period or edition the statement describes, and when the assertion is intended to apply.
- **Observed/recorded time:** when the evidence system retrieved, authenticated, registered or learned about that assertion/correction.

A later source correction must not rewrite the evidence bundle that supported an earlier decision. Instead, append a supersession/retraction event, compute dependency closure, and produce two different answers:

- **As-of decision time:** what source version and decision evidence the agent had when it acted.
- **Current status:** whether that source assertion is still published, valid, superseded, retracted, disputed or unavailable now.

These are not contradictory outputs. Historical reproducibility and present-day correctness are different questions.

### 4.1 Evidence graph objects

The first profile should use a typed graph with content-addressed artifacts. Minimal node types:

| Node | Required semantics |
|---|---|
| Publisher / issuer | stable issuer identity and independently configured trust material |
| Source assertion | stable assertion ID, publisher, edition/version and signed statement |
| Source artifact | exact bytes or authenticated excerpt, digest and media/encoding profile |
| Figure / claim | normalized subject, value, unit, denominator, population, geography and applicable period |
| Retrieval observation | retrieval time, URI/path, response digest, cache/mirror context and source status observed |
| Transformation | code/tool/model digest, version, canonical inputs, parameters, output and numerical method |
| Decision record | decision specification, alternatives, constraints, reasons, outcome, authority and approver |
| Approval | exact action-specification digest, principal, phase, policy and validity interval |
| Agent action | action profile, operation ID, request digest, disposition and any producer-observed response binding |
| Sink effect | MON-009 destination receipt, committed state transition and independent readback |
| Measurement | third-party declared-versus-observed result, measurement method and evidence digests |
| Supersession / retraction | signed relationship from prior assertion to replacement, correction, withdrawal or retraction |

Minimal edge types:

- supports: source assertion supports a claim;
- derived_from: claim or result depends on source/claim nodes and a transformation;
- observed_as: retrieval or measurement captured a particular artifact/version;
- supersedes: a new publisher assertion replaces an earlier one;
- retracts: an issuer withdraws an earlier assertion;
- restates: a publisher revises value/method while preserving historical context;
- decided_from: decision record cites exact evidence closure;
- approved_exactly: approval binds the exact action specification;
- executed_as: action record binds to sink operation identity;
- effect_verified_by: destination effect is linked to independent readback;
- measured_by: independent measurement party records a declared/observed comparison.

A URL is a locator, not a stable evidence identity. The graph must bind issuer, assertion ID, edition/version and exact digest.

### 4.2 Source assertion identity

At minimum, a material source assertion binds:

- issuer identity and key reference;
- stable assertion identifier and profile/version;
- publisher document identifier and edition/version;
- exact signed statement bytes and digest;
- source artifact digest or excerpt digest and reproducible locator;
- publication/issuance timestamp;
- applicable period or observation interval;
- subject/metric identifier;
- value, unit, denominator, population, geography and methodology where relevant;
- correction/supersession/retraction status and predecessor/successor references.

For dynamic data sources, distinguish publisher update time, effective time, observation time, retrieval time and cache time. Never infer a historical value from a current page without an edition-specific artifact.

### 4.3 Claim semantics

Every normalized claim must carry its semantic dimensions explicitly:

- metric / subject;
- value and numeric representation;
- unit and scale;
- numerator and denominator;
- population and inclusion/exclusion rules;
- geography/jurisdiction;
- time period, point-in-time or rolling window;
- source assertion IDs;
- uncertainty/confidence or source-declared limitations;
- transformation digest where derived.

Two values must not compare as equivalent merely because their displayed numbers match. A percentage, absolute count, rate per 100,000 and sampled estimate are different claims. Wrong unit/population/period is a semantic contradiction, not a formatting issue.

## 5. Decision provenance and action binding

### 5.1 Decision record

A decision record is separate from an action receipt and from an issuer's claim that a human approved something. It should carry:

- stable decision ID;
- exact decision specification digest;
- principal/agent identity and role;
- decision authority and policy revision;
- evidence-graph root/digest at decision time;
- alternatives considered and their evidence references;
- constraints, assumptions, limitations and relevant uncertainty;
- approver identity and signature, where applicable;
- outcome (approve, deny, defer, escalate, abstain or unresolved);
- valid-from/valid-until and decision stage;
- attachments and canonical digests;
- profile and verifier version.

Do not imply that a source claim, measurement capsule or action receipt itself constitutes authorization. Measurement compares; decision adjudicates; approval authorizes; sink enforcement checks the applicable authority at the commit boundary.

### 5.2 Approval exactness

An approval must bind at least:

- exact action-specification bytes/digest;
- actor/principal and intended sink/resource;
- phase/stage;
- policy and authority revision;
- validity window and revocation/freshness requirements;
- decision ID and evidence-graph digest.

Standing programme approval is distinct from a one-time action approval. An approval for one resource or request digest must not authorize another. A valid signature over a decision object does not mean the object passed its own policy/profile checks.

### 5.3 Action disposition completeness

Integrate Agent Action Capsule -05 rather than inventing a parallel action record. Preserve its disposition and may/did distinctions. Record blocked, denied, errored, timed-out and non-dispatched paths, not only successes.

A producer-observed response digest is not sink effect truth. The MON-009 sink receipt and separate-principal readback remain separate claims. A confirmed-effect response must reconcile against destination evidence; if the destination and response disagree, emit a contradiction, not a success.

## 6. Correction, supersession and downstream invalidation

### 6.1 Append-only correction model

When a publisher corrects or retracts a source assertion:

1. retain the original assertion, bytes, digest and historical trust context;
2. authenticate the correction/supersession event under the publisher's trust policy;
3. append a successor/retraction event; do not mutate the historical assertion;
4. compute the graph's reverse dependency closure;
5. classify affected claims, calculations, summaries, decisions and pending actions;
6. re-evaluate current-use eligibility;
7. retain a historical as-of record for any already executed decision/action;
8. issue a new decision if the correction materially changes current advice or pending action authorization.

A downstream observer's disagreement is a measurement/dispute event, not a publisher correction. Keep those event types distinct.

### 6.2 Point-of-use freshness

Re-check source status at consequential use/display time, not only retrieval time. For high-impact decisions, require a current source status or explicitly bounded freshness policy. If source status cannot be checked, report unavailable/indeterminate rather than silently reusing an old retrieval.

### 6.3 Historical/current verdicts

Use independent verdicts for historical and current status. Suggested values:

- valid_at_time: publisher assertion and supporting evidence verified as-of the decision time;
- current: no known superseding/retraction event under the checked publisher/status scope;
- superseded: later publisher assertion replaces the earlier version;
- retracted: publisher withdrew the assertion;
- disputed: independent evidence conflicts with the assertion;
- unavailable: required source or publisher status cannot be fetched;
- indeterminate: the verifier cannot establish the relevant status.

Do not rewrite a previously valid historical verdict when a correction arrives. Instead, update current status and the dependency graph.

### 6.4 Dependency closure

A correction to one source value should identify every dependent derived value, answer, policy input, decision and unexecuted action. A dependency closure is itself a reproducible claim: record source root/version, graph digest, traversal algorithm/version and output set digest. Missing edges reduce the completeness claim; they cannot be assumed away.

## 7. Independent measurement and adjudication

Use Measurement Capsules -01 when an independent party can compare a declared fact with an observed artifact. Preserve the draft's first-class could-not-check state. That state is neither verified nor disproved.

A measurement disagreement does not automatically authorize a correction, reversal or action. Decision policy must adjudicate it separately. Record common-cause dependencies among publisher, measurement operator, model operator, decision maker, transparency service, sink operator and readback observer.

A signed source statement proves publisher provenance under its trust model. It does not prove the publisher measured correctly. A TEE may protect provenance of a measurement; it does not prove sensor calibration, correct interpretation or complete physical coverage.

## 8. Portable evidence graph and offline verification

A portable bundle should contain or resolve to:

1. publisher-signed source statements and version/supersession chain;
2. exact source bytes/excerpts and retrieval observations;
3. normalized claims, units/populations/period semantics and source-status snapshots;
4. transformation code/tool/model digests and canonical input sets;
5. evidence-graph root and dependency closure;
6. decision records, approvals and policy/authority history;
7. Agent Action Capsule or AEE records with all dispositions;
8. MON-009 destination receipt, observer readback, boundary manifest and watermark;
9. SCITT Signed Statements/Receipts, inclusion/consistency proofs and authenticated checkpoint/witness collateral when used;
10. verifier/profile versions, independent trust roots and machine-readable per-claim verdicts.

Audit in a clean environment without original operator databases or credentials. Trust roots must be provisioned independently and not learned solely from the bundle. The auditor must report missing evidence and unsupported graph edges explicitly.

## 9. Adversarial qualification corpus

At minimum:

- same URL serves a changed edition without a new assertion ID;
- valid publisher signature but wrong population, denominator, unit, period or geography;
- signed value altered in the answer after retrieval;
- source mirror/cache bytes differ from signed publisher artifact;
- publisher corrects/retracts/restates after retrieval;
- agent uses a superseded assertion at display or action time;
- correction propagates to a direct claim but not a derived calculation;
- transformation code changes but output digest is reused;
- exact bytes are preserved while parser/units semantics change;
- a decision cites a valid evidence graph but omits a material alternative or uncertainty;
- approval signature is valid but action digest, actor, resource, phase, policy or validity differs;
- standing approval is misused as a one-off authorization;
- blocked/denied/timed-out attempts are omitted from the action history;
- a producer-confirmed response is spoofed or disagrees with sink readback;
- publisher and independent measurer share a required administrator/key/operator domain;
- source or status endpoint unavailable at point of consequential use;
- the same claim is historical-valid but current-superseded;
- historical bundle is rewritten after correction to hide what was previously known;
- a malicious operator deletes graph tail or correction events after an external watermark.

Every case needs a negative control that proves the intended check can fail.

## 10. Claim Provenance and Drift Challenge (CPDC)

### 10.1 Workload tiers

| Tier | Workload | Primary assurance question |
|---|---|---|
| C0 | raw citation-only answer | are URLs merely present? |
| C1 | immutable retrieval snapshot | can exact source bytes be reproduced? |
| C2 | source assertion / Figure Statement | is publisher, edition and source status bound? |
| C3 | normalized claim semantics | are value, unit, denominator, population and period consistent? |
| C4 | transformation lineage | can a derived claim be recomputed from pinned inputs/code? |
| C5 | decision/approval binding | was the exact evidence/action specification independently approved? |
| C6 | correction propagation | do supersession/retraction events reach every dependent claim and pending action? |
| C7 | composed effect audit | can a fresh auditor link source, decision, action, sink effect and readback offline? |

### 10.2 Required metrics

- stale-source detection precision/recall;
- correction/retraction propagation latency to all dependent claims and pending decisions;
- false-current rate for superseded/retracted values;
- wrong-unit/population/period/denominator acceptance rate;
- exact source-byte, edition and excerpt reproducibility;
- transform reproducibility and numerical deviation;
- dependency-closure precision/recall against seeded graph truth;
- decision approval exact-byte binding and expiry/revocation enforcement;
- disposition coverage, including blocked/denied/errored/timed-out/no-dispatch;
- offline verification success after operator database/credentials are removed;
- verifier latency, bytes, CPU, memory and cost per independently verifiable claim/action.

### 10.3 Matched baseline arms

- raw RAG/citation-only baseline;
- RAG with immutable retrieval snapshots;
- publisher-signed Figure Statement / source assertion profile;
- Agent Action Capsule -05 profile;
- Decision Provenance -03 profile;
- Measurement Capsule -01 profile;
- SCITT/RFC 9942/9943 registration and offline verification;
- composed claim-provenance/effect-assurance pipeline over MON-009.

Normalize source corpus, corrections, retrieval snapshots, model/task prompts, action policy and compute. Publish exact versions, seeded corrections, raw outputs, ground-truth dependency graph and all failure cases.

### 10.4 Proposed gates (targets, not results)

- every material claim maps to exact source/version and transformation evidence or is explicitly marked unsupported;
- every seeded correction/retraction is detected at current-use verification within a declared service-level bound;
- no superseded source is presented as current without an explicit, signed exception and scope;
- zero wrong-unit/population/period substitutions accepted in the release-candidate adversarial corpus;
- every approval binds the exact action specification, actor, phase, policy and validity window;
- historical conclusions remain reproducible as-of their recorded time after later corrections;
- all supported C7 bundles verify offline after operator databases and credentials are removed.

Do not set throughput or stale-detection targets before establishing a reproducible corpus and baseline. The first comparative result should be false-current rate and correction propagation completeness under matched seeded corrections.

## 11. Execution sequence and stop/go gates

### Phase 0 — schema and semantics
- Freeze source assertion, figure/claim and decision record profiles.
- Pin canonicalization/encoding and use existing SCITT/COSE profiles.
- Freeze historical/current verdict semantics and append-only correction behavior.
- No truth or completeness claim.

### Phase 1 — source assertion importer
- Import one pinned Figure Statement -00 corpus and its supersession/correction examples.
- Verify issuer identity, exact source bytes, figure semantics and publisher status.
- Re-run at display/use time and preserve historical verdicts.

### Phase 2 — claim graph
- Implement source/claim/transformation graph, digest manifest and deterministic dependency closure.
- Add wrong-unit/population/period and transformation-drift negative controls.
- Generate portable offline graph bundles.

### Phase 3 — decision/action binding
- Import Decision Provenance -03 and Agent Action Capsule -05 behind pinned adapters.
- Verify decision objects themselves, not just digest references.
- Bind approvals to action bytes/principal/phase/policy/validity.
- Reconcile producer-confirmed action evidence against MON-009 sink receipt and readback.

### Phase 4 — correction propagation
- Seed publisher corrections, retractions, restatements, mirror drift and unavailable-status cases.
- Measure downstream invalidation latency and dependency-closure recall.
- Preserve historical/current verdicts separately.

### Phase 5 — CPDC matched benchmark
- Run C0-C7 baselines under a fixed source corpus and correction schedule.
- Publish raw graph, seeded event log, output diffs, failure ledger, cost and offline audit evidence.

A skipped/unavailable CI job is not a pass. Local tests, source-profile conformance, real adapter tests, CI, independent review and production qualification remain separate evidence classes.

## 12. Related work

Published foundations:
- [RFC 9942 — COSE Receipts](https://www.rfc-editor.org/rfc/rfc9942.html)
- [RFC 9943 — SCITT architecture](https://www.rfc-editor.org/rfc/rfc9943.html)
- [RFC 8785 — JSON Canonicalization Scheme](https://www.rfc-editor.org/rfc/rfc8785.html)
- [RFC 9162 — Certificate Transparency Version 2.0](https://www.rfc-editor.org/rfc/rfc9162.html)

Version-pinned work in progress:
- [Agent Action Capsule -05](https://datatracker.ietf.org/doc/draft-mih-scitt-agent-action-capsule/)
- [Figure Statements -00](https://datatracker.ietf.org/doc/draft-yetis-scitt-figure-receipts/)
- [Decision Provenance for SCITT -03](https://datatracker.ietf.org/doc/draft-lemonnier-scitt-decision-provenance/)
- [Declared-versus-Observed Measurement Capsules -01](https://datatracker.ietf.org/doc/draft-templeman-scitt-measurement-capsule/01/)
- [Agent Execution Evidence -02](https://datatracker.ietf.org/doc/html/draft-thaha-scitt-agent-execution-evidence-02)
- [WIMSE Evidence Binding -03](https://datatracker.ietf.org/doc/html/draft-thaha-wimse-agent-evidence-binding-03)
- [Vaara Receipt -15](https://datatracker.ietf.org/doc/draft-sirkkavaara-vaara-receipt/15/)

All drafts must be version-pinned and status-checked before claiming conformance.

## 13. Open design questions

1. Which source publishers provide stable, signed assertion IDs and correction chains, and which require independently captured artifacts?
2. How should source status be checked when the publisher is unavailable at display/use time?
3. What is the minimum claim semantics needed across statistics, finance, scientific publications and policy documents without pretending one schema captures all meaning?
4. Which canonicalization profiles can be reused without profile ambiguity?
5. How should corrections propagate to actions already executed—historical qualification, current risk advisory, compensating action decision, or some combination?
6. What independent governance separates source publisher, measurer, agent operator, approver, sink operator and auditor?

These questions define the experimental boundaries; they are not permission to infer facts the evidence does not support.
