# Assurance Gap Challenge — benchmark protocol v2

**Status:** proposed benchmark extension; not yet run against industry systems. Complements MON-007 W0–W6 and explicitly includes SCITT agent-evidence and decision/execution receipt drafts.

## Primary outcome

**Offline-audited external effects per second** under a declared threat model, not workflow records per second.

Keep separate verdicts for intent, authority, sink execution, independent observation, transparency inclusion, append-only consistency, witness agreement, historical policy/key context and coverage completeness. Each claim is `verified`, `missing`, `contradicted`, `stale` or `indeterminate`.

## Matched baseline arms

Run identical semantics, destination API, data, operation IDs, durability, fault schedule, hardware and concurrency across:

1. PostgreSQL transactional outbox + idempotent sink adapter.
2. DBOS durable transaction/step + documented external-effect pattern.
3. Temporal Activity + stable idempotency key and workflow history.
4. RFC 9942 / RFC 9943 SCITT implementation with a qualified VDS.
5. Sigstore Rekor with available independent monitor/witness path.
6. AEE profile, draft-thaha-scitt-agent-execution-evidence-02.
7. SCITT AI Agent Execution profile, draft-emirdag-scitt-ai-agent-execution-01.
8. Vaara Receipt, draft-sirkkavaara-vaara-receipt-15.
9. WIMSE Evidence Binding -03, draft-thaha-wimse-agent-evidence-binding-03 (credential-to-agent identity, per-call caller reference and admission/freshness profile).
10. SFEB effect-assurance bridge over applicable systems above.
11. Native sink control: direct conditional write, sink receipt and separate-principal readback.

These profiles differ. Normalize the workload contract and compare only claims each can actually prove.

## Identity and admission sub-campaign

For the WIMSE arm, test valid and revoked workload credentials, runtime-identity-to-logical-agent mapping, stale assurance status, wrong Evidence Service, token reuse across two calls, caller-reference replay at another serving party, caller-supplied call-digest substitution, host/gateway ambiguity, actor-chain disagreement and a mismatch between admission and subsequent linked verification. Inject evidence-service outage, unknown identity, stale status and failed proof-of-possession; the serving party must refuse before the external effect, while still retaining a refusal record. Verify that prior actor-chain entries cannot expand authority and that only token claims/current actor drive admission. Post-hoc linked audit must not be treated as the same thing as the pre-action authorization decision.

## Workload tiers

| Tier | Operation | Primary metric |
|---|---|---|
| W0 | No-op durable workflow | Raw orchestration ceiling only |
| W1 | One durable DB mutation | DB transaction/outbox cost |
| W2 | Sink-enforced conditional mutation; stable operation ID, tenant/resource, authority revision and monotonic fence | Real effect and replay safety |
| W3 | W2 + caller receipt + serving-party admission/response receipt, linked by identical call digest | Two-sided observation/linkage |
| W4 | W3 + sink-issued receipt + independent-principal readback | Effect truth and separate observation |
| W5 | W4 + RFC 9942 receipt, RFC 9943 historical registration-policy collateral, signed checkpoint chain and independent witnesses | Transparency and trust continuity |
| W6 | W5 + original operator DB/credentials unavailable or malicious + fresh offline audit | Operator-independent verifiability |

Keep W0 separate from W2–W6. No-op workflow throughput is not evidence throughput.

## Adversarial sub-campaign: coverage and omission

Run across W2–W6:
- caller records a call but receiver lacks a matching admission receipt;
- receiver records an admission with no caller-side record;
- request payload, resource, tenant, principal or operation ID differs between sides;
- action occurs through a path not mediated by the declared harness;
- compromised caller capture suppresses an action record;
- sink commits but acknowledgement and caller receipt are suppressed;
- sink rejects but operator reports success;
- sink receipt names a different resource revision or operation ID;
- readback principal colludes, is unauthorized, is stale, or sees a lagging replica;
- events are omitted from a claimed complete sequence, count, epoch or sealed set;
- a declared set is presented as covering actions outside its observation boundary.

Measure omission detection recall and precision only within the declared coverage boundary. Show the blind spot when every observer on a path is compromised or bypassed. Absence of a receipt must not be interpreted as proof that no action occurred.

## Sink contract (minimum W2 qualification)

The destination atomically enforces tenant/resource identity, authority revision or capability digest, monotonic fencing token, stable operation ID and request digest, and deduplication/conditional-write semantics. It returns an immutable receipt binding operation ID, canonical request digest, tenant/resource, prior and new revision/digest, outcome, fence, authority revision, server identity and destination sequence. A different principal must read back the exact resulting resource revision. Unknown commit outcome means reconcile the same operation ID before any new mutation.

Client prechecks are not sink-side enforcement. A generic HTTP 200 or workflow completion record is not a sink receipt.

## Metrics

Performance: intent admissions/s, destination-accepted effects/s, two-sided linked actions/s, readback-verified actions/s, transparency-anchored actions/s, offline-audited actions/s, p50/p95/p99 per tier, reconciliation latency, recovery time, proof/bundle bytes, CPU-seconds, memory, IOPS, network and total cost per independently qualified effect.

Assurance: unauthorized/stale/fenced/replayed writes, duplicate effects, caller/receiver mismatch, unpaired receipts, missing destination receipts, readback disagreement, false accepts/rejects, indeterminate rate/age, offline verification success, root/policy replay success, equivocation detection latency, omission detection recall/precision within the declared boundary and unobserved-path residual risk.

Publish raw attempt ledgers, fault seeds, hardware/software, durability settings, sink, exact draft/profile versions, logs, verifiers, confidence intervals and analysis scripts. Use at least five independent performance runs per point.

## Proposed qualification gates (targets, not results)

- Zero unauthorized, stale-authority, stale-fence, tenant-substitution and replayed effects accepted by the real sink in at least 1,000,000 seeded negative attempts per release candidate.
- Zero duplicate destination mutations for repeated delivery of the same stable operation ID.
- Every W6-qualified action has a destination-issued receipt and separately authorized readback, unless the profile explicitly marks it unavailable/indeterminate.
- Every linked caller/receiver record has the same canonical call digest; mismatches are detected.
- 100% of W6-qualified bundles audit offline without original operator database or credentials.
- Full applicable policy/key history replays for the declared retention interval.
- Zero false accepts in the published negative-control corpus; publish exact corpus size/distribution. This campaign does not prove zero risk.
- Omission claims are restricted to independently attested coverage boundaries; no universal completeness claim.
- E2E performance is compared only within matched semantics. Microbenchmarks cannot satisfy tier gates.

## Reporting

Publish a per-claim capability matrix, not one composite green badge. For each arm and tier classify whether evidence is native, adapter-derived, explicitly missing, or outside the threat boundary. Report throughput together with evidence completeness and residual uncertainty.

**Research hypothesis:** the useful advantage will not be a faster signature or a new receipt schema. It will be lower false-accept and unresolved-outcome rates, with less operator trust needed to qualify actual external state transitions.
