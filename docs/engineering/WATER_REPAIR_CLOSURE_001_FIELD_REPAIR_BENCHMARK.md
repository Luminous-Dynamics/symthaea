# Water Repair Closure Benchmark v0.1

**Status:** design/benchmark proposal; synthetic-first; not a field procedure, hydraulic qualification, potable-water approval, or operational control system.  
**Owner:** Symthaea engineering evidence and decision layer, composed with existing Sol-Atlas, Symtropy, and Mycelix owners.  
**Parent context:** WATER-MFG-000 [#5966](https://github.com/Luminous-Dynamics/symthaea/issues/5966); non-potable module contract [#5997](https://github.com/Luminous-Dynamics/symthaea/issues/5997).

## Executive decision

Add a field-repair lifecycle benchmark that tests whether a water-system work item reaches **verified, appropriately scoped service recovery**—not merely whether a work order was closed.

The benchmark should be a composition of existing owners, not another asset registry, event log, hydraulic solver, maintenance scheduler, or water-quality authority:

- **Symthaea:** engineering applicability, evidence gates, comparison of repair/replacement options, and claim ceilings.
- **Sol-Atlas:** spatial/service context, approved model and observation references, hydraulic solver manifests and affected service cells.
- **Symtropy:** conserved water/terrain state and physical effects within the exact capabilities of its qualified model; no competing municipal pipe solver.
- **Mycelix:** event provenance, identity/permissions, human decision and authorization history, work and custody receipts, replay handling, correction/supersession, and lifecycle events.
- **External technical authority:** qualified plumber/engineer, asset owner, accredited or otherwise appropriately qualified laboratory, and the applicable regulator retain their respective approvals.

This proposal extends the existing water engineering vertical slice; it does not replace it. Pressurised-water reference calculations remain with EPANET, drainage/sewer reference calculations with a properly configured SWMM profile, and building-plumbing requirements with a separately identified current local standards/code profile.

## Paradigm shift

**From closing maintenance tickets to closing evidence-backed service loops.**

A work order marked complete is an operational event. A replaced component is a material/configuration event. A passing test is a measurement or test result. Commissioning is an approval event. Restored service is an outcome claim with its own required evidence. They must never be collapsed into one status.

The lifecycle is:

```
reported symptom / service deficit
  -> triage and asset matching
  -> bounded work proposal + explicit human authorization
  -> as-found observations
  -> approved method / compatible parts / material provenance
  -> work execution receipt
  -> as-left observations and component-function tests
  -> required inspection / commissioning / quality clearance
  -> independently evidenced service recovery
  -> replaced-part and residual disposition
  -> post-work review and lifecycle update
```

Every transition has its own preconditions and evidence. Missing, stale, conflicting, out-of-scope, revoked, or unqualified evidence yields a blocker or an indeterminate result—not a default pass.

## Scope of the first corpus

Use only three synthetic asset profiles, each clearly distinguished:

1. **Non-potable low-head circulation/filter module.** Reuse the benign WATER-MFG reference profile. Test component and consumable identity, flow observations, functional checks, commissioning and recovery route. This profile cannot emit any drinking-water safety claim.
2. **Municipal distribution repair scenario.** A small synthetic EPANET network with a declared isolation/repair/replacement alternative, pre- and post-work model artifacts, and one protected service floor. Hydraulic model outputs remain simulations, not field observations.
3. **Building plumbing service case.** A synthetic leak/valve/fixture event with only the approved test fields of an explicitly named profile. Do not infer design compliance from a city-network model. Local plumbing, backflow, hygiene, and return-to-service requirements must be selected and reviewed by a qualified local professional before a real pilot.

Wastewater and stormwater repairs can then reuse the lifecycle contract, but must bind a suitable SWMM or other qualified process/model profile. They must add any relevant overflow, receiving-water, effluent, residual/sludge, and disposal/reuse evidence. A water-distribution pass is not a sewer or environmental-discharge pass.

## Evidence planes and state boundaries

Bind the following as separate records or references; reuse existing owner types where they already exist.

1. **Asset identity and scope:** stable asset/configuration ID, parent network/building/module, location precision/access tier, profile, owner, and validity interval.
2. **Reported symptom:** request ID, affected service, reporter source class, observed/reported distinction, event time, and uncertainty.
3. **As-found condition:** measurements, units, instrument/calibration/currentness evidence, and whether the observation covers the actual target asset and interval.
4. **Engineering applicability:** exact design/procedure/profile revision, constraints, prerequisites, approved option and applicable professional review.
5. **Authorization:** actor/role, scope, time window, constraints, revocation state and authorization evidence. A recommendation cannot mint authorization.
6. **Parts and materials:** part/component identity, lot/serial where applicable, compatibility and qualification references, supplier/custody provenance, new/reconditioned state and intended disposition.
7. **Execution receipt:** who reported the work, which authorized scope was attempted, timestamps, deviations, exceptions and supporting evidence references.
8. **As-left condition and functional test:** measured values with unit, time, method, calibration/currentness, acceptance bounds, target identity, and unambiguous test disposition.
9. **Commissioning / health-quality clearance:** each required clearance is separately represented. A flow/pressure check never substitutes for microbiological or chemical evidence when the intended service requires it.
10. **Service outcome:** affected service cells, baseline and post-work observation window, delivery/continuity evidence, protected floors, uncertainty, and known concurrent interventions.
11. **Lifecycle closure:** replaced component, filter/media/consumables, backwash/sludge/screenings or other residual stream where applicable, recovery/reuse qualification, disposal or unknown disposition.
12. **Review and correction:** responsible reviewer, unresolved objection, superseding correction, review due date and evidence freshness policy.

Each datum must declare its evidence class, such as `Observed`, `SourceReported`, `Derived`, `Simulated`, `Synthetic`, or `Unknown`. Authorization, execution, measurement, commissioning and outcome must not be inferred from neighboring events. A cryptographic digest binds bytes; it does not prove the assertion is true.

### Recommended outcome states

Use non-collapsing states such as:

- `Blocked`: a hard prerequisite failed or required authority is absent.
- `Incomplete`: an expected lifecycle transition or required record is absent.
- `Indeterminate`: conflicting, stale, imprecise or insufficient evidence prevents a conclusion.
- `ReadyForReview`: declared technical preconditions are represented, but an accountable reviewer still needs to decide.
- `VerifiedWithinProfile`: all frozen requirements for that exact profile passed, with exact evidence lineage and stated scope.
- `Rejected`: an acceptance criterion failed.

Never expose generic `safe`, `fixed`, `compliant`, or `restored` as an unqualified universal boolean.

## Required adversarial and positive-control corpus

Freeze at least these 22 synthetic known-answer cases before implementing a production adapter. The corpus must record each expected disposition and reason code independently of the implementation under test.

| ID | Scenario | Required disposition |
|---|---|---|
| WR-01 | Complete non-potable module repair with compatible part, current measurements and required checks | `VerifiedWithinProfile`; no potable-water claim |
| WR-02 | Complete synthetic network repair with matching EPANET before/after inputs and separately scoped outcome evidence | `VerifiedWithinProfile` only within the declared reference profile |
| WR-03 | Work-order status is `complete`, but as-left measurements are absent | `Incomplete` |
| WR-04 | Same logical event is delivered twice after a retry | One semantic effect only; duplicate is idempotent |
| WR-05 | Same event ID is reused with different content/digest | Reject or quarantine as conflict; never silently overwrite |
| WR-06 | As-found or as-left evidence names a different asset/configuration | `Blocked` |
| WR-07 | Measurement is outside its validity period or instrument qualification is missing | `Indeterminate` or `Blocked`, per frozen profile |
| WR-08 | Simulation is supplied in place of an expected physical post-work observation | `Incomplete` |
| WR-09 | Required unit is absent or unrecognized | Reject validation |
| WR-10 | Unit conversion is implicit, ambiguous or dimensionally invalid | Reject validation |
| WR-11 | Potable-service intervention has hydraulic evidence but required water-quality clearance is absent | No potable-service clearance |
| WR-12 | Filter/media identity, currentness or compatibility is unknown | Efficacy/return-to-service remains unresolved |
| WR-13 | Replacement part compatibility or required qualification is not established | `Blocked` or `Indeterminate` |
| WR-14 | Reconditioned pump/valve is installed without required function requalification | Cannot assert component function |
| WR-15 | Asset is installed but not commissioned under the declared profile | Cannot assert ready/available service |
| WR-16 | Two pumps share one declared power/control failure domain | No claim of independent redundancy |
| WR-17 | Removed media/sludge/part has no required residual or disposition record | Lifecycle closure `Incomplete` |
| WR-18 | Work execution has no valid matching authorization | Reject as authorized work; do not backfill authority |
| WR-19 | Authorization expired, was revoked, or does not cover this asset/work scope | `Blocked` |
| WR-20 | As-left observations fall outside the defined scenario/time alignment | `Indeterminate` or reject pairing |
| WR-21 | Public projection includes exact household/private critical-asset location forbidden by access policy | Reject public projection; preserve permitted internal view only |
| WR-22 | Contradictory post-work observations straddle an acceptance boundary | `Indeterminate`; retain both references and request review |

Add property and metamorphic tests: permuting input record order must not alter the result; adding missing/invalid evidence must never improve disposition; widening measurement uncertainty must not strengthen a claim; changing the asset or source digest must break exact binding; a correction must append/supersede rather than rewrite history.



### Corpus maturity / next implementation boundary

The v1 JSON file is a machine-readable **case matrix**: it freezes profile, expected disposition, reason codes and required assertions. It is not yet a fully executable set of event/evidence payloads, and no implementation has computed the expected outputs. The concrete synthetic input bundles and independent Rust evaluator are tracked in [WATER-REPAIR-001A #8128](https://github.com/Luminous-Dynamics/symthaea/issues/8128). Until those inputs and evaluator exist and run on an exact revision, the corpus remains `NotRun`; no executable-benchmark or pass claim should be made.

## Acceptance gates

These are engineering targets for the reference contract, not regulatory requirements or claims about a deployed utility.

- **100%** agreement with frozen, independently hand-reviewed fixture dispositions and reason codes.
- **Zero** false `VerifiedWithinProfile` outcomes among negative controls.
- **Zero** event-retry duplication and **zero** conflicting-ID overwrite acceptance.
- **Zero** authorization creation from a recommendation, simulation, work-order status or completion receipt.
- **Zero** water-quality, code-compliance or environmental-discharge claims from a profile that does not own and satisfy those checks.
- **100%** of accepted results bind the exact asset/profile/model revisions, time interval, input/evidence references, schema/corpus digest and evaluator identity.
- Deterministic, byte-identical canonical result manifests for identical captured inputs and exact evaluator revision.
- Missing or unqualified lifecycle evidence must remain visible in the result; no missing value becomes a favorable zero.
- Every removed component/material quantity has at most one declared disposition allocation at a time; mass/material reconciliation uses the unit and custody model appropriate to identity-preserved, segregated, mixed, or mass-balance accounting.
- Report all rejected, blocked and indeterminate cases; do not remove hard cases from aggregate test results.

### Proposed local reference-validator performance hypotheses

Only after correctness gates pass, measure a pure, offline event/evidence validator on an identified reference machine. Initial proposed p95 targets are:

| Workload | Proposed p95 |
|---|---:|
| 100 lifecycle events | <= 5 ms |
| 1,000 lifecycle events | <= 25 ms |
| 10,000 lifecycle events | <= 250 ms |

These are unmeasured hypotheses for a local reference validator, not end-to-end service latency, crypto/storage throughput, mobile performance or field-operation targets. Report p50/p95/max, samples, event and evidence counts, output bytes, peak RSS where available, CPU/compiler/build mode, machine/OS, exact commit, and input/corpus/evaluator digests. Correctness and fail-closed behavior remain independent gates; a fast wrong answer is a failure.

### Later operational impact hypotheses

Do not include these in synthetic software-pass criteria. After a data-owner-approved pilot freezes a comparable baseline and accounts for differences in scope, crews, weather, access and concurrent work, investigate:

- 20% fewer repeat visits for the same verified asset failure within a declared follow-up window;
- 20% lower median and p90 time from validated report to independently verified service recovery;
- fewer repair closures with missing commissioning or required measurement evidence;
- lower repeat leakage or recurrence, only when the chosen observation method can detect it;
- reduced material loss and improved qualified reuse/recovery, without counting unknown disposition as recovery.

These targets should be revised after a baseline audit, not presented as forecasts or promised savings.

## Safety, regulatory and cybersecurity boundaries

1. **Read-only integration first.** This benchmark grants no access to SCADA, PLC/RTU commands, pump/valve actuation, chemical dosing, disinfection or public-water operation.
2. **Human authorization remains separate.** Model recommendations cannot dispatch physical actions, close a permit or approve return to service.
3. **Drinking water is a separate claim profile.** WHO's drinking-water guidance frames risk management from catchment to consumer, supported by health-based targets and independent surveillance. The local applicable requirements and qualified reviewers must be identified before using this profile.
4. **Separate network domains and sensitive data.** Follow least privilege, controlled remote access, segmentation and audit practices for water/wastewater operational technology. Keep exact sensitive topology, telemetry, credentials and exploitable weaknesses out of public maps or public federation records.
5. **Waste streams are not automatically resources.** Recovery requires characterization, intended-use limits, appropriate permissions, and evidence of actual recovered quantity/quality. Prevented, reused, recycled, recovered, downcycled, disposed and unknown material remain separate dispositions.
6. **No professional judgement laundering.** Passing a software fixture is not plumbing approval, engineering sign-off, laboratory accreditation, regulatory compliance, sanitary clearance, or proof of physical service delivery.

## Integration order

1. Freeze this semantic contract and the 22 synthetic cases.
2. Reuse the existing WATER-MFG-001A reference corpus and ENG-DESIGN consumer packet; add only fields/fixtures needed for asset-centric as-found/as-left closure.
3. Bind Mycelix's existing water I0 event cycle and circularity events; do not add a second event ledger or lifecycle ontology.
4. Bind Sol-Atlas model/run and observation identities for synthetic EPANET/SWMM reference packages; do not reinterpret a `NotRun` or stale result as evidence.
5. Consume Symtropy physical results only within their documented model fidelity and conservation boundaries.
6. Implement an independent reference evaluator and negative controls; require exact-head receipts before asserting a pass.
7. Only after independent review, consider read-only historical replay with explicit data-owner permission and privacy/security review.
8. Treat any live, control-adjacent or potable-water work as a distinct, later authorization and safety case—not a feature flag enabled by this benchmark.

## Research references

- WHO, [Guidelines for drinking-water quality, fourth edition incorporating the first, second and third addenda](https://www.who.int/publications/i/item/9789240121225), published 17 June 2026. The guidance covers risk management from catchment to consumer and independent surveillance.
- US EPA, [EPANET](https://www.epa.gov/water-research/epanet), pressurised drinking-water distribution modelling including pressure-dependent demand, pumps, valves, storage, pumping energy, and supported water-quality behaviour.
- US EPA, [Storm Water Management Model (SWMM)](https://www.epa.gov/water-research/storm-water-management-model-swmm), rainfall-runoff, drainage, sewer routing and supported pollutant processes.
- NIST, [SP 1800-45: Cybersecurity for the Water and Wastewater Sector](https://www.nist.gov/publications/cybersecurity-water-and-wastewater-sector), published 24 June 2026; reference architectures for secure operational-technology remote access.
- South African Government, [2025 Green Drop wastewater assessment](https://www.gov.za/news/media-statements/water-and-sanitation-crowns-steve-tshwete-local-municipality-2025-green-drop), reporting on 848 municipal wastewater systems for FY 2023/24. This provides sector context, not asset-specific ground truth.
- Sol-Atlas [#112 water engineering vertical slice](https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/112), [#139 EPANET validation](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/139), [#144 water/power coupling](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/144), and [#147 joint service-floor stress](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/147).
- Symtropy [#474 infrastructure and one-water authority](https://github.com/Luminous-Dynamics/symtropy/issues/474) and [#476 conservative water-quality tracer semantics](https://github.com/Luminous-Dynamics/symtropy/issues/476).
- Mycelix [#4496 water-system governance vertical slice](https://github.com/Luminous-Dynamics/mycelix/issues/4496) and [#3095 manufacturing provenance/circularity lifecycle](https://github.com/Luminous-Dynamics/mycelix/issues/3095).

The references motivate the design and boundaries; they do not validate the proposed benchmark or any real-world performance claim.
