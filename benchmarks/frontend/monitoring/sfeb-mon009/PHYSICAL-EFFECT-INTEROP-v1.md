# MON-009 addendum — physical-effect interoperability

**Status:** research/integration guidance; no PSER or robotic-site implementation is claimed.  
**Version:** 1.0, 2026-11-25.

## 1. Why this matters

SFEB should not stop at digital sink effects. Autonomous agents increasingly dispatch robots, mobile equipment, maintenance crews and physical actuators. A successful dispatch API call is not evidence that the actor reached the site, that it performed the intended engagement, or that the physical outcome was safe and correct.

The active IETF Internet-Draft **A SCITT Profile for Physical-Site Engagement Receipts -06** (draft-wilder-scitt-physical-site-engage-receipt-06, October 29, 2026) is directly relevant. It defines a SCITT Signed Statement with a bounded site, actor/operator, engagement window and operating envelope, TEE-sealed attestation evidence, and an Adapter Write-In to an out-of-band operations system. It uses RFC 9942 receipts and RFC 9943 registration.

This remains an individual Internet-Draft, not a published RFC or an endorsed IETF standard. Its own scope is deliberately narrow: receipt integrity, attribution and evidence sealing do not establish that an engagement was safe, correct or wise; registration does not prove that the issuer registered every receipt or that the asserted site conditions are true.

## 2. Compose records; do not collapse them

A physical workflow should retain at least these distinct evidence objects:

1. **Agent decision / action evidence:** why the agent selected the action, what policy/delegation applied, and what exact dispatch request it sent.
2. **Serving-party admission:** what the receiving service authenticated and authorized before dispatch.
3. **Digital destination receipt:** which dispatch or control-plane state transition the service actually committed, with stable operation identity and fencing.
4. **Physical engagement receipt:** what the site-bound actor/TEE reports about the bounded physical engagement under the declared envelope.
5. **Independent physical observation:** what a separately governed sensor, site system, inspector or measurement party observed.
6. **Transparency evidence:** which signed records were registered and whether the VDS receipt/checkpoint history verifies.

Use explicit cross-references for operation, actor, site, resource, envelope and evidence digests. A shared subject string or matching identifier is a correlation key, not proof that two records concern the same causal event. Require exact binding and temporal/sequence consistency.

## 3. Digital effect is not physical engagement

A dispatcher returning success proves only what the dispatcher accepted under its protocol. A robot controller acknowledging a command proves only that the controller acknowledged it. Neither proves that the robot carried out the engagement.

The verifier should keep these outcomes separate:

- dispatch admitted;
- digital sink committed;
- actor identity and device attestation verified;
- physical engagement receipt verified;
- physical observation independently reconciled;
- envelope conformance assessed;
- downstream physical outcome independently established.

A missing sensor record, an unavailable site witness, or an untrusted TEE must remain missing or indeterminate. Do not infer "nothing happened" from absence of a receipt.

## 4. Threat and test programme

For a robot dispatched to a bounded site:

- wrong actor or device identity;
- wrong site, tenant or resource;
- geofence or site-envelope substitution;
- engagement outside the declared time window;
- dispatch accepted but actor never arrives;
- actor arrives but performs no work;
- actor performs an unapproved or partial engagement;
- TEE measurement stale, revoked, wrong platform or wrong site;
- evidence bundle digest mismatch;
- physical observation disagrees with TEE receipt;
- same operator controls dispatch, sensor, TEE, transparency issuer and readback;
- physical activity occurs through an unobserved path;
- engagement receipt exists but was never registered;
- registered statement exists but issuer omitted other engagements;
- unsafe or out-of-envelope activity is recorded with a valid receipt.

The test harness must distinguish detection, attribution and safety. A record labelled WITHIN or COMPLETED is not a safety case. Safety requires the applicable risk assessment, safety functions and independent validation; transport latency is not a safety-response-time measurement.

## 5. Additional benchmark dimensions

Extend W0-W6 with an optional physical-effect profile, not by mixing it into software-only throughput:

- dispatch-to-arrival and arrival-to-engagement latency;
- proportion of declared engagements corroborated by independent site observation;
- actor/site/time/envelope binding error rate;
- physical-effect false accept and false reject rates;
- omission detection precision/recall within the registered site-observation boundary;
- stale or invalid attestation detection;
- time to reconcile TEE/observer conflicts;
- percentage of evidence independently audited after the dispatcher/operator and their credentials are removed;
- bytes and cost per independently qualified engagement.

A physical-action leaderboard must report the sensor modality, calibration, sampling limits, site coverage, environmental conditions, TEE/firmware versions, observation gaps and safety envelope. Do not compare a software API acknowledgment with a robot's measured physical engagement.

## 6. Integration recommendation

For the first MON-009 increment, keep PostgreSQL as a reversible digital sink and implement the sink/readback contract first. Add PSER as a version-pinned import/export and evidence-linking adapter after the underlying digital operation receipt and independent readback work.

For Symtropy robotics and other physical applications, later evaluate a site-owned or independently operated measurement path. A TEE can protect measurement provenance under its threat model; it cannot by itself establish sensor truth, complete environmental coverage or safe behavior.

## References

- [Physical-Site Engagement Receipts -06](https://datatracker.ietf.org/doc/html/draft-wilder-scitt-physical-site-engage-receipt-06) — active Internet-Draft, October 2026.
- [RFC 9942 — COSE Receipts](https://www.rfc-editor.org/rfc/rfc9942.html).
- [RFC 9943 — SCITT architecture](https://www.rfc-editor.org/rfc/rfc9943.html).
- [Agent Execution Evidence -02](https://datatracker.ietf.org/doc/html/draft-thaha-scitt-agent-execution-evidence-02) — active Internet-Draft.
- [Declared-versus-Observed Measurement Capsules -01](https://datatracker.ietf.org/doc/html/draft-templeman-scitt-measurement-capsule-01) — active Internet-Draft.
