# Luminous Aqua — Bounded Bathing Service Compiler v0.1

**Status:** software and product-contract research specification; no connected fixture or human trial  
**Workstream:** [AQUA-SERVICE-COMPILER-001 #12749](https://github.com/Luminous-Dynamics/symthaea/issues/12749)  
**Parent:** [AQUA-BATHING-SERVICE-001 #12660](https://github.com/Luminous-Dynamics/symthaea/issues/12660)  
**Related:** #12692 spray field, #12693 washstation, #12712 adaptive experience, #12717 AQS-1 platform, #12727 circular service.

## 1. Paradigm shift: compile a service, do not automate a fixture

Aqua should accept an explicit human task (for example, rinse, hair wash, seated shower or basin grooming), resolve it against a catalogue of qualified fixture capabilities, and produce a bounded, inspectable **candidate service plan**.

This is different from a smart-shower preset:

- A preset stores actuator values; a service plan states the intended task, required capabilities, constraints, expected observations and completion conditions.
- A planner may select only capabilities already qualified for the exact hardware/configuration and operating envelope.
- A plan does not grant actuator authority. Local deterministic controls, mechanical temperature protection, backflow protection, water-quality permissives, electrical protection and emergency isolation remain independent.
- Unknown, stale, contradictory or out-of-envelope information produces abstention, not optimistic interpolation.
- Manual operation remains available without Symthaea, network, cloud, profile or stored preference.

The first implementation is read-only/offline: parse intent, validate a synthetic capability catalogue, compile a plan, explain it, and produce a deterministic trace. No connected fixture actuation is in scope.

## 2. Authority model

| Layer | Responsibility | Must not do |
|---|---|---|
| User | Select task, accept/decline optional preference, override or stop | Be forced to create a profile |
| Symthaea planner | Interpret explicit intent; rank already-qualified capabilities; explain or abstain | Set safety limits, authorize water, infer health/body state, bypass interlocks |
| Capability registry | Describe exact device revision, allowed modes and evidence references | Turn marketing claims into qualified capabilities |
| Local fixture controller | Enforce its independently qualified local operating envelope | Depend on cloud/AI for safe operation |
| Mechanical / electrical protections | Enforce physical limits and protective disconnection | Be overridden by software |
| Mycelix | Metrology, calibration, uncertainty, water-quality and provenance authority | Be replaced by Aqua's score |
| Symtropy | Hydraulic/thermal/resource closure and physical model authority | Be duplicated by an Aqua calculator |
| Aqua | Bind user task, product configuration, service plan and completed-service comparison | Self-certify safety, hygiene, water quality or regulatory compliance |

**Hard rule:** planner output is advisory. Any later execution adapter must accept only a plan whose every action is inside a separately qualified device envelope, and the device itself must enforce that envelope without trusting the planner.

## 3. Versioned conceptual data contracts

The following are illustrative logical shapes, not final serde types or a production API. Unknown fields, missing required evidence and schema-version mismatch must fail closed.

### 3.1 Service intent

A service intent contains only information deliberately supplied or selected for this task:

- schema version;
- task identifier from a finite, documented vocabulary;
- optional duration band;
- optional delivery preference (for example, fixed head / handheld);
- optional access mode selected by the user (for example, seated);
- hard user constraints and explicit exclusions;
- consent scope for remembering a preference (default: this session only).

Do not infer age, disability, diagnosis, body dimensions, gender, skin condition, mood or other sensitive/body attributes. An accessible task mode is a user-selected interaction mode, not a diagnosis.

### 3.2 Qualified capability

A capability record is bound to exact hardware and installation identity:

- capability and schema version;
- device model, SKU, hardware revision and installed configuration digest;
- task/mode supported;
- qualified pressure, flow, inlet temperature and room-condition envelope;
- required upstream/downstream interfaces;
- required local interlocks and physical limits;
- cleaning and maintenance state prerequisites;
- evidence references, owner, validity interval and status;
- unsupported modes and known exclusions.

A product-family web page, nominal flow claim, simulation or synthetic fixture is not a qualification receipt. Evidence must identify whether it is measured, independently tested, supplier-declared, simulated or unknown.

### 3.3 Candidate service plan

A plan records:

- original intent digest and schema version;
- exact capability/configuration references;
- ordered advisory steps and their preconditions;
- constraints inherited from the user and capability records;
- expected measurements and completion conditions;
- rejected alternatives with reasons;
- evidence freshness and owner checks;
- abstention reason, if any;
- explicit statement that local device protections remain authoritative.

No actuator address, raw PWM/valve command or safety setpoint is emitted by v0.1.

## 4. Compilation pipeline

1. **Parse:** validate intent syntax and finite task vocabulary.
2. **Resolve:** load exact installed capability identities; no fuzzy SKU substitution.
3. **Qualify:** require current evidence for each proposed mode and envelope.
4. **Constrain:** intersect user constraints with capability envelopes; never widen either.
5. **Plan:** choose among eligible capabilities using declared task objectives; do not invent missing values.
6. **Explain:** show selected and rejected options, evidence status, trade-offs and uncertainty.
7. **Confirm:** ask the user to accept any non-default preference; no silent profile learning.
8. **Observe:** if later connected, read only owner-qualified measurements and preserve their source/time/uncertainty.
9. **Close:** compose owner-native resource closure and service outcome evidence; unresolved residuals remain unknown.
10. **Abstain / fallback:** invalid evidence, unsupported task, stale configuration, interlock state uncertainty or service mismatch returns no candidate plan and leaves manual/local controls available.

A compiler success means only that a candidate plan satisfies the software contract against the supplied evidence. It does not mean a fixture is safe, hygienic, comfortable or physically capable.

## 5. Required negative controls

The test suite must prove rejection of:

- missing or unknown schema version;
- unknown task or unsupported delivery mode;
- device SKU / hardware revision mismatch;
- expired, revoked, stale or digest-mismatched qualification;
- missing calibration or invalid measurement applicability;
- pressure, flow, temperature or room state outside the capability envelope;
- cleaning overdue or wet-path state unknown where that state is a prerequisite;
- contradictory user constraints;
- request to exceed a mechanical temperature limit;
- request to bypass water-quality or treatment permissive;
- planner-proposed action absent from the capability registry;
- attempt to use AI/cloud availability as a prerequisite for manual bathing;
- attempt to infer a preference from camera, microphone or body-state data;
- replay of a prior plan against a different installation/configuration digest;
- missing resource-closure evidence presented as zero water or energy.

Each negative control must assert both the rejection and the reason. A generic "safe" boolean is insufficient.

## 6. Benchmark protocol

### Baselines

| Arm | Planner | Purpose |
|---|---|---|
| B0 | Manual controls | Agency, task success and zero planner dependency |
| B1 | User-saved static preset | Commercial smart-shower baseline |
| B2 | Deterministic rules | Strong transparent automation baseline |
| B3 | Symthaea intent + qualified capability catalogue | Candidate approach |
| B4 | LLM-only planner in offline sandbox | Research ablation only; never connected to physical control |

All arms receive the same task descriptions, capability facts, installation state and held-out cases. B4 is isolated and cannot emit device commands.

### Outcomes

Report a vector; do not collapse safety, comfort and resource use into one weighted score.

- Task completion rate and independently scored task quality.
- User corrections, manual interventions, abandoned plans and time-to-completion.
- Plan latency and deterministic replay agreement.
- Correct capability selection and unsupported-plan rejection.
- Abstention precision/recall on invalid or incomplete evidence.
- Constraint preservation under adversarial prompts and conflicting preferences.
- Preference consent, persistence scope, deletion/expiry behavior and cross-user isolation.
- Data minimization: no camera/microphone requirement; record all data fields actually accessed.
- Resource use only where a physical matched test and canonical owner receipts exist.
- Offline/manual availability and recovery after planner/network failure.

### Initial research targets (hypotheses, not results)

- 100% rejection of deliberately invalid, stale, mismatched and out-of-envelope capability/evidence fixtures.
- 100% preservation of hard constraints in property-based and adversarial tests.
- 100% deterministic replay for identical versioned inputs and pinned planner build.
- Zero cross-user preference leakage in the test corpus.
- Zero requirement for camera, microphone, cloud or internet to complete the manual service.
- No more than 1% unsupported-plan acceptance in a preregistered held-out challenge set; target zero, with confidence interval and corpus size reported.
- Task quality and intervention burden must be non-inferior to the best baseline before any usefulness claim.
- Any resource benefit is a separate physical claim and must use the shower/washstation matched-service protocols.

Do not claim 100% safety from software tests. These targets apply only to the defined software corpus and invariants.

## 7. Evaluation design and anti-gaming

- Freeze task prompts, capability registry, planner version and scoring rubric before confirmation testing.
- Split development and held-out cases by task family, device configuration and failure mode, not by random near-duplicate prompts.
- Include paraphrases, ambiguous intent, conflicting constraints, unsupported hardware, stale evidence and adversarial attempts to expand authority.
- Blind human-factors evaluators to planner arm where practical.
- Report per-task and per-user-group results; do not hide failure of seated/assisted or low-pressure profiles in an aggregate.
- Publish all exclusions, failed/aborted runs, abstentions and unknowns.
- Do not count a correct abstention as a completed bathing task; report it separately.
- Keep synthetic, simulated, bench and human-study evidence in separate strata.

## 8. Standards and external baselines

- EPA WaterSense showerhead programme: performance includes spray force, spray coverage and pressure compensation in addition to flow. Aqua must use the applicable method for a formal comparator and must not imply WaterSense certification. https://www.epa.gov/watersense/showerheads
- IEC 60335-2-53:2026, Edition 5, published 2026-10-15, covers electric sauna heating appliances and infrared emitting units within its stated scope. This is relevant to later sauna modules only; it does not qualify an entire thermal suite or bathing room. https://webstore.iec.ch/en/publication/106068
- Local building, plumbing, electrical, accessibility, water and public-health requirements must be identified for the exact deployment jurisdiction and verified with competent local professionals. International standards are not automatically local law.

## 9. Staged implementation

**S0 — contract only:** schemas, finite task vocabulary, capability registry fixture, plan explanation and negative tests.

**S1 — offline deterministic harness:** matched baseline corpus, held-out adversarial cases, replay and evidence-status reporting.

**S2 — read-only unoccupied test cell:** only after the independent instrument, fixture and room interfaces are qualified; no user exposure.

**S3 — supervised human-factors evaluation:** only after applicable safety, hygiene, privacy, ethics and accessibility reviews authorize the exact protocol.

**S4 — bounded product integration:** only after device-level qualification, independent protection and manual/offline fallback are demonstrated.

## 10. Claim ceiling

This document authorizes software-contract and offline/synthetic research only. It is not a control-system design, safety case, product certification, clinical or wellness protocol, hygiene qualification, plumbing/electrical design, procurement approval or authorization for human exposure. The compiler must never become the safety authority.
