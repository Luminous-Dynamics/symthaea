# MON-009 interoperability addendum — action, effect and decision evidence

**Status:** profile comparison and adapter requirements; no implementation of these Internet-Draft profiles is claimed.  
**Version:** 1.0, 2026-11-25.

## 1. Why the new comparison matters

MON-009 should not be described as inventing a confirmed-effect field or the may/did distinction. The active Internet-Draft **An Agent Action Capsule Profile for SCITT -05** already specifies action-level digest commitments, disposition classes, a confirmed-effect binding and a chain/assurance model.

Its strongest rule is that a producer cannot say an effect is confirmed without binding the actually observed response digest. Its own security considerations are equally important: a confirmed response is producer-observed bytes, not proof that the external world reached the claimed state. A spoofed response can still induce an honest-looking confirmed record.

This creates a precise composition point:

- Agent Action Capsule records the action attempt and the producer's observed response.
- MON-009 sink evidence records the destination's own operation outcome.
- A separate principal reads back the exact committed revision/digest.
- Measurement Capsule / SCITT transparency records independent observation and historical registration evidence.
- An independent auditor reconciles all of the above.

Treat the profile as an active Internet-Draft, not an approved RFC. Datatracker currently shows no RFC stream or intended status metadata, even though the draft text itself labels its intended status Standards Track.

## 2. Preserve each profile's semantics

| Evidence layer | What it can establish | What MON-009 adds or composes |
|---|---|---|
| Agent Action Capsule -05 | A digest-bound action record, producer disposition, constraints and producer-observed response binding | Compare producer response with destination-issued outcome and separate readback |
| Agent Execution Evidence -02 | Caller/serving-party records, admission evidence, call linkage and offline bundles | Compare both parties' claims to the actual sink commit and independent readback |
| WIMSE Evidence Binding -03 | Workload identity to logical-agent binding and pre-action admission checks | Preserve the separation between admission authority and actual sink enforcement |
| Vaara Receipt -15 | Recomputable decision/execution evidence pairs and declared completeness | Check those claims against the destination's committed state and independently observed revision |
| Decision Provenance -03 | A separate signed decision record and attachment, with exact references from an action/release record | Bind approval to the exact operation specification, authority revision, principal, phase and validity window |
| Measurement Capsule -01 | Third-party declared-versus-observed measurement, including could-not-check | Apply to the bounded sink observation manifest/watermark without creating a second capsule format |
| RFC 9942/9943 | COSE receipts and the SCITT statement/transparency architecture | Use as the wire/interoperability foundation, not replace it |

No one of these layers implies the next. A valid receipt can be based on false source data; an approval can be signed for the wrong bytes; a producer can observe a spoofed response; a sink can commit while readback is stale; and a registered statement can still omit unobserved actions.

## 3. Required MON-009 adapter behavior

### 3.1 Agent Action Capsule import/export

- Pin the exact draft version and profile identifiers.
- Preserve the draft's digest and chain semantics; do not invent a parallel action-receipt schema.
- Round-trip the full disposition vocabulary, including executed, blocked, denied, errored and timed out.
- Preserve the draft's distinction between not-dispatched/not-applicable, dispatched-unconfirmed and confirmed.
- Never upgrade a producer-confirmed effect to destination-verified merely because response_digest validates.
- Link the action record to the sink operation by stable operation ID and exact canonical request digest.
- Treat producer response binding as an input assertion to reconcile, not as the destination oracle.

### 3.2 Destination effect binding

A sink-qualified effect requires:

1. sink-enforced tenant/resource, stable operation ID, request digest, authority revision, fence and expected resource revision;
2. committed destination ledger record;
3. destination-issued receipt bound to prior and resulting revision/digest;
4. separately authorized readback at the exact resulting revision/digest;
5. freshness and observer-independence verdicts;
6. transparency and historical policy/checkpoint evidence where the selected tier requires it.

If any layer is unavailable, retain the relevant claim as missing, stale or indeterminate. Do not infer that no effect occurred.

### 3.3 Decision provenance

A decision record is not the same as a producer's claim that a human approved an action.

The adapter should support a separate portable decision object with:
- exact action-specification digest;
- decision/approval identifier and issuer;
- approving principal and credential;
- policy/authority revision;
- decision outcome, constraints and validity interval;
- stage/phase (shown, approved, requested, rechecked, executed where the chosen profile uses them);
- evidence/attachment digests and typed references.

The executor must re-check that the approval is valid for the actual bytes, principal, phase and policy at the sink boundary. A standing programme approval is not automatically a per-action approval. A valid decision digest only proves byte linkage until the referenced object is independently parsed and its profile-specific rules verified.

### 3.4 Completeness and refusal paths

The attempt ledger and offline bundle should retain blocked, denied, errored, timed-out and non-dispatched outcomes, not only successful effects. A receipt sequence that contains only successes is survivorship-biased and cannot show which refusals occurred.

Completeness is always scoped to the declared observation boundary. A log sequence can expose missing records inside its own capture domain, but it cannot establish that every real action entered that domain. Measure and publish bypass paths separately.

## 4. Adversarial conformance vectors

Add cross-profile test cases for:

- confirmed effect with missing response digest;
- valid response digest but sink reports rejected;
- producer-confirmed response whose revision/digest disagrees with sink readback;
- response spoofed at the serving boundary;
- blocked/denied/errored/timed-out disposition with a contradictory claimed effect mode;
- only successful action records provided while refusal/timeout attempts are omitted;
- caller and serving-party call digests differ;
- approval signature is valid but action bytes or policy revision differ;
- programme-level approval used as if it were per-action approval;
- approval expires or is revoked between admission and sink commit;
- decision record digest points to bytes that do not pass the decision profile;
- a Measurement Capsule says could-not-check but the consumer turns it into a verified observation;
- action receipt and SCITT registration are valid while the destination receipt is missing;
- sink receipt is valid but separate reader shares a required administrator/key/storage domain;
- sink receipt/readback are valid but a publisher later supersedes the source figure used in the decision.

Each vector must specify expected per-claim verdicts; there is no single aggregate pass bit.

## 5. Benchmark extension

Add these measures to the Assurance Gap Challenge:

- percentage of action records preserving all dispositions and refusal paths;
- false-confirmed rate: producer-confirmed actions with no matching sink commit/readback;
- false-dispatch/false-effect distinction errors;
- approval-to-exact-action binding failure rate;
- post-approval revocation race outcomes;
- time to detect mismatched producer response, sink outcome and independent readback;
- completeness within the declared attempt/capture boundary;
- fraction of action and decision bundles verifiable offline after producer/operator credentials and databases are removed.

Compare the Action Capsule, AEE, Vaara, Decision Provenance, Measurement Capsule and MON-009 adapters only on the claims each profile actually defines. Report native, adapter-derived, missing and outside-threat-boundary functionality separately.

## 6. Related documents

- [Agent Action Capsule -05](https://datatracker.ietf.org/doc/draft-mih-scitt-agent-action-capsule/)
- [Figure Statements -00](https://datatracker.ietf.org/doc/draft-yetis-scitt-figure-receipts/)
- [Decision Provenance for SCITT -03](https://datatracker.ietf.org/doc/draft-lemonnier-scitt-decision-provenance/)
- [Declared-versus-Observed Measurement Capsules -01](https://datatracker.ietf.org/doc/draft-templeman-scitt-measurement-capsule/01/)
- [Agent Execution Evidence -02](https://datatracker.ietf.org/doc/html/draft-thaha-scitt-agent-execution-evidence-02)
- [WIMSE Evidence Binding -03](https://datatracker.ietf.org/doc/html/draft-thaha-wimse-agent-evidence-binding-03)
- [Vaara Receipt -15](https://datatracker.ietf.org/doc/draft-sirkkavaara-vaara-receipt/15/)
- [RFC 9942](https://www.rfc-editor.org/rfc/rfc9942.html)
- [RFC 9943](https://www.rfc-editor.org/rfc/rfc9943.html)

All draft profiles are work in progress and must be version-pinned and rechecked before conformance claims.
