# MSP Operational Assurance Architecture v0.1

**Status:** Architecture and integration contract. This is not a complete MSP platform or a production qualification.

## 1. Product thesis

Symthaea is the reasoning and coordination layer for an evidence-native Managed IT and Security Operations Platform. It must not become a privileged monolith or a replacement for dedicated endpoint, identity, backup, network, cloud, ticketing or security products.

The system should own the reasoning that crosses those tools:

- Understand a customer's assets, services, identities, dependencies and operating agreements.
- Correlate endpoint, identity, network, cloud, application and security observations.
- Diagnose incidents using causal and temporal context, not just keyword similarity.
- Propose typed, risk-classified changes with explicit preconditions and expected postconditions.
- Request authority for the exact act and target.
- Execute only through the target's authoritative participant.
- Verify actual post-state and sustained customer service health independently.
- Record evidence that supports incident closure, audit and customer reporting.

The operating principle is: **capability does not imply authority; command completion does not imply recovery.**

## 2. Repository ownership

| Repository | Responsibility | Must not become |
|---|---|---|
| [symthaea](https://github.com/Luminous-Dynamics/symthaea) | Cognitive loop, HDC/CfC reasoning, incident analysis, causal model, knowledge and planning | Privileged fleet executor |
| [luminous-platform](https://github.com/Luminous-Dynamics/luminous-platform) | MSP control plane, tenant/service inventory, connector lifecycle, run orchestration, evidence packaging and service reporting | A second cognitive engine or competing host executor |
| [nixward](https://github.com/Luminous-Dynamics/nixward) | NixOS observation, declarative change planning, signed authority, snapshots, execution, post-state verification and recovery | General-purpose unconstrained shell runner |
| [mycelix](https://github.com/Luminous-Dynamics/mycelix) | Future identity, delegation, revocation and verifiable credential federation | An MVP dependency while its hApps remain pre-alpha |
| [xenia-peer](https://github.com/Luminous-Dynamics/xenia-peer) / [xenia-wire](https://github.com/Luminous-Dynamics/xenia-wire) | Future consent-based remote sessions and protected operator communication | A production remote-support claim before independent review and release qualification |
| [symtropy](https://github.com/Luminous-Dynamics/symtropy) | Isolated digital-twin and fault/recovery simulation | Evidence of real production behaviour |
| [nix-signature-policy](https://github.com/Luminous-Dynamics/nix-signature-policy) | Exploratory software provenance and signature-policy evaluation | A production trust service before cryptographic review |

Within Symthaea, the relevant existing building blocks are:

- `symthaea-support`: triage, diagnostics, actions, privacy preferences, knowledge and predictive telemetry.
- `symthaea-logparse`: EVTX/syslog parsing and experimental HDC encoding.
- `symthaea-scan`: platform-specific asset and installed-application discovery.
- `symthaea-observability`, `symthaea-telemetry-sink`, `symthaea-audit`: observation, telemetry and audit support.
- `symthaea-causal-reasoning`, `symthaea-digital-twin`, `symthaea-formal-safety`: causal models, engineered-asset state and proof obligations.

These are foundations, not proof that the complete product workflows exist.

## 3. Canonical operational objects

The platform needs stable, versioned, tenant-scoped objects. At minimum:

### Tenant and service model

- `Tenant`: stable tenant ID, contract/policy profile, data residency and approved integrations.
- `Site`: tenant-bound location or logical environment.
- `Principal`: human, service or agent identity with issuer, status and scope.
- `Asset`: stable asset ID, tenant ID, device identity, OS/platform, owner, site, discovery sources and freshness.
- `Service`: customer-facing service, criticality, SLO/SLA, support window and recovery objective.
- `DependencyEdge`: source/target IDs, relationship, confidence, valid-time interval, observation provenance and topology epoch.
- `Observation`: source identity, source timestamp, ingest timestamp, sequence, schema version, quality/freshness and content digest.

Do not use a hostname, IP address or mutable vendor object ID as the sole asset identity. Keep vendor-native identifiers as sourced aliases with lifecycle and collision handling.

### Incident and action model

- `Incident`: tenant, affected services/assets, hypothesis set, evidence references, severity, lifecycle and escalation state.
- `Diagnosis`: candidate cause, supporting and contradicting evidence, confidence/calibration and abstention reason.
- `ChangePlan`: exact target, preconditions, intended state delta, commands/structured operations, expected postconditions, risk class, rollback/recovery binding and digest.
- `ActAuthorization`: issuer, principal, tenant, audience/participant, exact plan digest, action scope, policy epoch, expiry, one-use identifier and signature-verification result.
- `AttemptReceipt`: participant identity, attempt ID, idempotency key, request digest and start/outcome state.
- `EffectReceipt`: participant-owned committed effect ID, exact operation/plan digest, target and atomic authorization-consumption result.
- `ServiceVerification`: independently sourced postcondition observations, workload identity/profile, time window, result and evidence digest.
- `RecoveryRecord`: snapshot or restore point, recovery operation, RPO/RTO measurements, integrity checks and sustained health result.

A diagnosis is not an authorization. A proposed action is not an attempt. An attempt is not a committed effect. A committed effect is not a verified recovery.

## 4. System state graph

The operational graph should be temporal and evidence-backed.

### Node classes

Tenant, site, principal, endpoint, server, network component, identity provider, cloud resource, software package, data store, backup, business service, alert, incident, change, runbook and vendor integration.

### Edge classes

Hosted-on, depends-on, authenticates-through, routes-through, protected-by, backed-up-by, managed-by, observed-by, changed-by and owned-by.

Every edge carries source, observed-at/valid-at, freshness, confidence, topology epoch and evidence reference. A stale or unknown edge remains explicitly stale or unknown; it must not silently become a current fact.

### Causal diagnosis

Symthaea should compare competing causal hypotheses using:

- Temporal ordering and change history.
- Dependency path and common-cause failures.
- Supporting versus contradicting observations.
- Intervention effects and counterfactual expectations.
- Sensor/source reliability, missingness and clock quality.
- Customer service impact, not merely host-level symptoms.

The agent must not receive evaluator-owned ground truth in benchmark or production workflows. Counterfactuals are hypotheses until an intervention or independent observation supports them.

## 5. Authority and execution model

Use the Nixward transaction lifecycle as the canonical host-change contract:

**observe → plan → validate → authorize → snapshot → apply → verify → promote/recover**

For all other platforms, define an equivalent participant contract. Do not bypass the participant and mutate a provider directly from the cognitive process.

### Hard rules

1. Natural-language approval, confidence, Phi, HDC similarity, model self-report or a ticket's priority is never a cryptographic authorization.
2. Authority binds the exact tenant, principal, audience, target, action, plan digest, policy epoch, expiry and one-use handle.
3. A changed plan requires fresh authorization.
4. Consumption of the one-use handle and committing the protected effect must share the participant's atomic boundary.
5. Replays, expired grants, revoked principals, stale policy epochs and target mismatches fail closed.
6. Duplicate or unknown outcomes require reconciliation before any retry.
7. Rollback notes do not prove rollback feasibility. Recovery must be planned, tested and bound to the exact change.
8. Independent service observers must verify the postcondition; an agent-authored success message is not sufficient.

### Autonomy profiles

- **Observe:** collect and correlate read-only facts.
- **Recommend:** create an explanation and a typed plan; do not mutate.
- **Approved act:** execute one exact human-authorized act.
- **Policy-bounded:** execute only a pre-approved operation class within a specific tenant, target, resource, time window, risk ceiling and rate budget.
- **High impact:** require stronger approvals and explicit recovery readiness.

There is no global FullAutonomous permission. Autonomy is a narrow property of an exact operation under a policy and participant, not a role that bypasses controls.

The MSP action-gate patch in this branch deliberately constrains the legacy `symthaea-support::ActionEngine` to read-only diagnostics. That local allowlist is a safety reduction, not a complete cryptographic effect gate. Mutating effects remain blocked until a real act-bound participant integration is available.

## 6. Privacy and tenant isolation

The tenant is the primary security boundary. Cross-tenant learning is opt-in and must be justified by a specific purpose, permitted data class and destination.

- Default sharing is local-only.
- Missing privacy policy is deny.
- A tier named Anonymized does not prove anonymity.
- Regex scrubbing is a defense-in-depth filter, not a de-identification guarantee.
- Preserve raw telemetry in the tenant boundary unless an explicit retention and disclosure policy permits otherwise.
- Minimize data before indexing or embedding; avoid putting secrets, credentials, personal identifiers or customer-specific topology into shared knowledge.
- Use tenant-scoped encryption, retrieval filters, cache keys, logs, object stores and backups.
- Test cross-tenant confusion and retrieval leakage using adversarial negative controls.
- Record consent, purpose, destination, retention and revocation for every shared training/knowledge payload.

The current support privacy manager is a coarse preference. It must not be treated as the sole privacy validator or authorization service.

## 7. Capability programme

| Domain | Initial capability | Required evidence before customer autonomy |
|---|---|---|
| Help desk/PSA | Ticket intake, deduplication, triage, SLA clocks, escalation and customer updates | Labeled cases, calibrated severity, accurate SLA calculations and auditable closure |
| RMM/inventory | Endpoint registration, heartbeat, hardware/software inventory, configuration drift | Source-authenticated inventory, freshness and coverage against ground truth |
| Patch management | Exposure assessment, staged updates, reboot state and rollback | Signed artifacts, change-window policy, successful post-patch verification |
| Network operations | DNS/DHCP/VPN/firewall/connectivity diagnosis | Topology-aware fault localization and validated read-only connectors |
| Identity operations | Joiner/mover/leaver, MFA, role and certificate lifecycle | Strong principal identity, least privilege and revocation tests |
| SOC/EDR/SIEM | Alert normalization, enrichment, ATT&CK mapping, triage and containment planning | Independent detection validation, false-positive/negative analysis and response safety |
| Backup/DR | Coverage, backup freshness, isolated restore and integrity verification | Demonstrated restore, measured RPO/RTO and sustained recovered service |
| Cloud/SaaS | Inventory, configuration drift, provider health and spend anomalies | Provider-scoped roles, quota/rate-limit handling and target-state verification |
| Compliance/reporting | Evidence-linked control status and exception management | Versioned mappings, provenance, review dates and explicit non-compliance states |

NIST CSF 2.0 and CIS Controls v8.1 should be mapped to actual evidence-producing product capabilities. A checklist or generated report is not proof that the underlying safeguard is operating.

## 8. Cyber and log-analysis qualification

The current `symthaea-logparse` Phase 1 report records a 0.815 held-out event accuracy on a 1,000-event, eight-label subset. Do not use this as an enterprise detection claim.

Before drawing conclusions, revalidate:

- Split by source file, host and incident; random event-level splits are insufficient where related records can appear in both train and test.
- Separate temporal holdouts and a genuinely unseen provider/host distribution.
- Preserve class imbalance and report per-class precision, recall, PR-AUC and calibration.
- Include benign background telemetry and realistic false-positive rates per endpoint-day.
- Test label noise, missing fields, reordered fields, clock skew and adversarially corrupted events.
- Compare rules, TF-IDF/linear baselines, gradient-boosted or conventional ML baselines, HDC nearest-centroid, HDC linear probe and the actual production Symthaea encoder.
- Freeze corpora, labels, codebook seed, split manifest and run artifacts.
- Include alert-to-investigation and safe response metrics, not only event classification.

At present the logparse encoder uses a local reference MAP-I implementation; its production HDC feature is a placeholder rather than a demonstrated swap to the live Symthaea core encoder. The 0.815 result must therefore be treated as exploratory until both leakage-resistant evaluation and production encoder integration are independently reproduced.

## 9. Benchmark programme

Use [MSP OpsBench v0.1](https://github.com/Luminous-Dynamics/luminous-platform/tree/main/research/msp-opsbench-v0.1) together with existing Continuous OpsBench.

Required dimensions:

- Diagnosis and localization.
- Security detection and investigation quality.
- Asset and software inventory coverage/freshness.
- Patch and vulnerability remediation latency.
- SLA and ticket lifecycle accuracy.
- Authorized effect correctness and idempotency.
- Unauthorized, duplicate and cross-tenant effect counts.
- Sustained service restoration, backup integrity and RPO/RTO.
- Human intervention, false closure and operational cost.

Use paired no-action, deterministic runbook, RMM/PSA workflow, classical AIOps, qualified human and competing-agent baselines. Freeze the full run manifest. Keep quality and safety dimensions separate; a single aggregate score must not hide a security failure.

## 10. Delivery sequence

### P0 — safety and evidence contracts

- Keep local autonomy allowlists read-only until act-bound participant enforcement exists.
- Default support data sharing to local-only and deny when policy is absent.
- Specify tenant, asset, service, incident, observation, plan, authorization, attempt, effect and verification contracts.
- Add tenant-isolation, replay, expiry, revocation, duplicate-effect and unknown-outcome negative controls.

### P1 — trustworthy observations and diagnostics

- Integrate logparse with source identity, record lineage and freshness.
- Add host/file/incident/temporal-disjoint evaluation.
- Replace heuristic-only triage with calibrated, evidence-linked classification.
- Extend scan beyond NixOS/Linux and establish per-platform source completeness.
- Build a versioned system state graph and causal diagnosis interface.

### P2 — bounded remediation and service desk

- Build structured connectors to RMM, EDR, identity, cloud, backup and PSA products.
- Keep vendor execution APIs behind target-specific participant adapters.
- Add change windows, human approvals, reversible runbooks and independent postcondition observers.
- Deliver customer reporting with evidence-linked SLA and incident records.

### P3 — federation and advanced autonomy

- Add opt-in, privacy-reviewed cross-tenant knowledge sharing.
- Integrate Mycelix credentials and revocation only after its maturity and security requirements are met.
- Expand autonomous operation by scenario-specific qualification, not global autonomy settings.

## 11. Claim discipline

Every capability must be marked as one of:

- Implemented in source.
- Structurally tested.
- Unit/integration tested.
- Reproduced against a frozen benchmark.
- Tested against a live isolated environment.
- Independently reviewed.
- Qualified for a named production scope.

A green compile is not an MSP capability. A passing synthetic fixture is not a real-world benchmark. An internal score is not an industry comparison. A cryptographic primitive is not a complete security product.

The product earns autonomy one verified task class at a time.
