# OT and cyber-physical assurance crosswalk v0.1

**Status:** Research crosswalk; standards-alignment aid, not a conformity assessment or certification.  
**Issue:** https://github.com/Luminous-Dynamics/symthaea/issues/2187  
**Review date:** 7 November 2026

## Why this update matters

NIST published the initial public draft of SP 800-82 Revision 4 on 21 September 2026. The comment period closes 30 November 2026. The current final guide remains SP 800-82 Revision 3 (September 2023); the Revision 4 document is a draft and must not be presented as an operative final standard.

The NIST announcement says the draft expands sector coverage to building automation and control systems, water/wastewater, food/agriculture, freight rail, maritime vessels and IT/OT/cloud-converged industrial IoT. It also describes alignment around NIST CSF 2.0, expanded Govern and enterprise-risk treatment, RMF discussion, asset management/network monitoring, and OT architecture guidance protecting management functions and applying zero-trust principles.

Sources:
- NIST SP 800-82 Rev. 4 Initial Public Draft (21 Sep 2026; comments due 30 Nov 2026): https://csrc.nist.gov/pubs/sp/800/82/r4/ipd
- NIST announcement and summary of changes: https://www.nist.gov/news-events/news/2026/09/guide-operational-technology-ot-security-nist-requests-comments-draft-sp
- NIST SP 800-82 Rev. 3 final (Sep 2023): https://csrc.nist.gov/pubs/sp/800/82/r3/final
- NIST Cybersecurity Framework 2.0 (26 Feb 2024): https://doi.org/10.6028/NIST.CSWP.29
- ISA/IEC 62443 series overview: https://www.isa.org/standards-and-publications/isa-standards/isa-iec-62443-series-of-standards

This crosswalk intentionally maps at the framework/function and stakeholder level. It does **not** claim clause-level NIST SP 800-82r4 coverage or ISA/IEC 62443 certification. Obtain and review the full licensed standard text before making detailed clause claims.

## Existing implementation owner

This is an extension of Symthaea [OT / cyber-physical security reasoning #2187](https://github.com/Luminous-Dynamics/symthaea/issues/2187), not a second OT programme.

The current issue already establishes:
- separate cyber and physical state;
- process-safety precedence over cybersecurity convenience;
- source independence and common-mode failure reasoning;
- OT threat families and industrial protocol applicability;
- twelve synthetic qualification scenarios;
- effects-disabled cyber-range integration;
- no direct actuator authority from cyber reasoning.

Composition:
- Symthaea owns evidence interpretation, competing hypotheses, cyber/physical reconciliation, safety-aware reasoning and abstention.
- Symtropy supplies effects-disabled simulation and deterministic scenario replay; simulation is not field qualification.
- Mycelix supplies identity, policy/authority provenance, approvals, lifecycle, revocation and dispute receipts.
- Sol-Atlas supplies read-only spatial/asset context and system-dependency presentation.
- Existing qualified plant, safety-instrumented and device authorities remain independent execution roots.

## Non-equivalences that must survive every adapter

```
network reachability       != engineering authority
controller acknowledgement != physical effect
historian value             != trusted process truth
cyber containment           != process-safe containment
simulated response          != field-qualified response
asset unavailable           != hazard removed
security confidence         != permission to act
```

## Function-level crosswalk

| Assurance area | NIST CSF 2.0 function | OT / IEC 62443 alignment | Luminous contribution | Required evidence |
|---|---|---|---|---|
| Governance, risk ownership, safety decision rights | Govern | Shared responsibility among asset owner, product supplier, integrator and service provider; lifecycle risk management | Mycelix authorization/provenance; Symthaea policy-aware reasoning | Named owner, risk profile, process-safety authority, approved operating envelope, current policy version |
| Asset and process context | Identify | Asset owner inventory and integrator system context; IACS lifecycle and risk assessment | Sol-Atlas asset graph; Symthaea device/protocol applicability | Device identity, firmware/logic revision, physical role, dependency graph, source freshness, inventory coverage |
| Access, segmentation and change protection | Protect | Security requirements for system integrators, product suppliers and service providers; secure lifecycle and maintenance | Mycelix identity/capability receipts; Symthaea risk reasoning | Zone/conduit and remote-access boundary, role scope, change record, authenticated source, approved maintenance window |
| Monitoring and cyber/physical reconciliation | Detect | Asset-owner monitoring and component/system event visibility | Symthaea source independence, clock quality, contradiction and physical evidence reasoning | Raw observation lineage, sensor health, common-cause dependencies, time quality, independent process evidence |
| Containment and incident handling | Respond | Shared incident responsibility; OT-specific safety and availability constraints | Symthaea effects-disabled response candidate; Mycelix separate authorization | Process state, hazard constraints, operator approval, reversible options, denial/abstention evidence |
| Restore and requalify safe service | Recover | Lifecycle, maintenance, configuration integrity and service-provider responsibility | Symtropy recovery trajectory; Mycelix outcome/correction receipts; Sol-Atlas service continuity | Exact restored configuration, independent process measurements, safety-system availability, stable service window, operator acceptance |

This is an architectural crosswalk, not a claim that any one Luminous component implements an entire NIST or IEC/ISA function.

## Paradigm shift: cyber-state plus physical-state assurance

The core differentiator should be a typed, evidence-linked view that never collapses digital control state into physical process truth.

For each claim, preserve:
- native asset/process subject and revision;
- cyber observation and physical observation as separate evidence classes;
- source identity, provenance, timestamp, clock quality and calibration/currentness;
- source dependency/common-mode relations;
- process operating mode and safety constraints;
- contradictions, unknowns and coverage gaps;
- applicable NIST/ISA reference profile and revision;
- human decision owner and separate execution authority;
- attempted change, observed physical effect and independent outcome reconciliation.

A signed command, protocol acknowledgement or successful configuration readback does not prove that the process reached a safe or intended state.

## Current OT vendor baseline and build/buy/bridge decision

The comparison must be against actual OT cybersecurity platforms, not generic SIEM alone:
- [Claroty Platform](https://claroty.com/platform) documents CPS asset inventory, exposure management, network protection, secure access, threat detection and operational-efficiency functions.
- [Nozomi Networks Platform](https://www.nozominetworks.com/platform) documents OT/IoT visibility, asset inventory, wired/wireless/endpoint sensors, vulnerability/exposure management, anomaly detection and OT-focused threat intelligence. Its [Guardian sensor documentation](https://technicaldocs.nozominetworks.com/guardian/latest/products/guardian/topics/intro/c_guardian.html) describes passive network discovery and monitoring.
- [Dragos Platform](https://www.dragos.com/cybersecurity-platform/exposure-management/) documents OT asset/exposure visibility, network monitoring, threat intelligence and OT Watch analyst-led monitoring/triage; its 2026 platform updates include AI-assisted analyst workflows and asset lifecycle features: https://www.dragos.com/blog/dragos-platform-ai-ot-security-visibility.

These platforms already operate in the core territory of OT asset discovery, protocol/network visibility, vulnerability/exposure management, threat intelligence and detection. Luminous should not claim to outperform them in those capabilities without a matched, site-authorized benchmark.

The proposed Luminous build/buy/bridge split is:
- **Buy/bridge:** passive OT sensors, protocol decoders, asset inventory and vendor threat-intelligence feeds from qualified incumbent platforms where appropriate.
- **Build:** evidence lineage and source dependency semantics; cyber-state/physical-state separation; safety-precedence and authorization non-laundering; uncertainty/abstention; cross-domain dependency and service impact; independently verified recovery receipts.
- **Do not rebuild:** plant protocol stacks, vendor device libraries, safety PLCs, historian platforms or an OT command/control plane.

The evaluation should compare an incumbent OT platform alone, the operator's established workflow, and incumbent platform plus Luminous evidence/decision layer using the same sensor data and human-review budget. The potential win is fewer unjustified escalations, better contradiction handling, better process-safety preservation and faster verified recovery—not a larger protocol feature checklist.

## Industry comparison and benchmark protocol

Compare fairly against:
1. an asset-owner/operator runbook workflow;
2. a source-health-aware rules-only workflow;
3. an established OT monitoring / engineering toolchain supplied by a participating operator;
4. Symthaea-assisted evidence triage with the same data, time budget and authority constraints.

Do not compare a research prototype against a production platform by feature checklist. Pin exact versions, source feeds, device/process profiles, network topology, human-review budget and test scenarios. Publish non-equivalent features.

Measure separately:
- asset inventory coverage and stale/unknown asset rate;
- source provenance completeness and clock-fault detection;
- cyber/physical contradiction recall;
- false alerts per monitored asset-day and recall at fixed false-alert budgets;
- calibrated probability quality only after held-out site/time validation;
- abstention risk versus coverage;
- common-mode source failure sensitivity;
- time to produce a defensible incident hypothesis and review package;
- hard process-safety constraint violations in the synthetic corpus;
- unauthorized action/authority transition count;
- time to independently verify stable recovery;
- human override, review time and operator agreement.

## Proposed qualification gates (targets, not achieved results)

- 100% of required crosswalk rows have an owner, evidence class, applicability statement and explicit gap/unknown state.
- Zero controller acknowledgement promoted to physical-effect truth.
- Zero model recommendation promoted to device/process authority.
- Zero hard process-safety constraint violations in the frozen synthetic corpus.
- Zero unsafe automated escalation when sensor, clock or communications health is degraded.
- Every simulated recovery outcome is marked as simulation evidence, never field evidence.
- Every benchmark run pins exact device/profile, model, source, fixture, policy and evaluator digests.
- Any probability claim is calibrated on site/time-separated data and reported with uncertainty.
- Conformity/certification claims require a qualified assessor and the relevant full standard text.

## Next execution sequence

1. Update the existing #2187 issue with draft-version status and this function-level crosswalk.
2. Freeze the machine-readable crosswalk as a versioned research artifact.
3. Extend the existing twelve-case corpus to cover BACS, water/wastewater, maritime, freight rail and cloud/IIoT context without inventing device-specific semantics.
4. Build strict conformance checks for state separation, safety precedence, authority non-laundering, source independence and recovery evidence.
5. Use effects-disabled scenarios and compare against runbook and rules baselines.
6. Seek an OT asset-owner/operator and functional-safety reviewer before lab/hardware validation.

## Non-goals

No industrial exploitation, controller logic manipulation, setpoint modification, safety-system bypass, automated actuator command, or live plant response. This artifact does not replace asset-owner risk assessment, qualified functional-safety engineering, regulatory review or third-party standards assessment.
