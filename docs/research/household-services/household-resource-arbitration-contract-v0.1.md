# Household resource arbitration contract v0.1

**Status:** proposed decision contract / synthetic benchmark design  
**Owner:** [HOME-UTILITY-COORD-001 #10342](https://github.com/Luminous-Dynamics/symthaea/issues/10342)  
**Compose:** HOME-SVC-001 #10322, continuity #10445, water quality #10500, metrology #10341, receipt provenance #10403, water / thermal / electrical domain owners  
**Evidence class:** design-only; no live household actuation or physical utility integration.

## 1. Paradigm shift: orchestration without centralizing authority

Move from a smart-home controller that directly operates every appliance to a **federation of independently qualified service providers**.

Each appliance or utility subsystem owns:
- its local process controller and safety interlocks;
- its qualified operating envelope;
- local/manual control and safe fallback;
- its own source evidence, calibration and service outcomes;
- its independent ability to refuse a request.

The household coordinator owns only:
- household-level state estimation with uncertainty;
- dependency and common-mode-failure representation;
- user-approved priorities and deadlines;
- candidate scheduling/deferment proposals;
- coordination receipts explaining what was requested, authorized, refused, deferred or restored.

The coordinator may request a local action through a typed capability interface. It cannot directly set a valve, pump, heater, dishwasher cycle or safety interlock. The local controller may refuse the request. Loss of the coordinator must not disable local safe operation.

## 2. Non-compensatory priority: constraints before optimization

A single weighted score is unsafe for services with non-substitutable quality and safety floors. A large energy saving must never compensate for unqualified drinking water, an unsafe hot-water profile, a cross-connection risk, an unauthorized shutoff or a missed critical household need.

Use the following lexicographic decision sequence:

1. **Hard safety and legal constraints:** electrical, pressure, thermal, sanitation, plumbing separation, treatment safety, local authority and manufacturer interlocks. Infeasible or unknown constraints block the affected action.
2. **Use qualification:** the source, batch/stream, treatment, sample evidence, standard/risk profile, declared use and delivery boundary must qualify. Potable and non-potable sources are distinct. An Unknown quality state receives no potable credit.
3. **Critical service floors:** preserve user-declared essential service floors and applicable public-health / hygiene minimums. Do not assume one household's priority ordering is universal.
4. **Consent and authorization:** defer, reschedule or isolate only within explicit, current user authorization and the relevant local product policy. Emergency behavior must be separately justified by the safety authority, not improvised by the optimizer.
5. **Robust service allocation:** among feasible candidates, minimize worst-case cumulative deficit in protected services using bounded demand, deliverability and resource uncertainty.
6. **Secondary optimization:** only then optimize optional loads, peak demand, cost, energy, convenience and maintenance burden.

If no feasible candidate satisfies the active constraints, return an explicit **Infeasible / Needs User or Qualified Operator Decision** state with reasons and available alternatives. Do not silently relax a hard constraint or invent a substitute service.

## 3. Resource and service types — no false fungibility

Retain independent types and boundaries for:
- drinking water and non-potable water, each bound to declared source and use class;
- water volume, flow, pressure and delivered fixture service;
- electrical power / energy (kW / kWh-electric);
- useful thermal power / energy (kW-thermal / kWh-thermal);
- fuel, battery state and stored thermal energy where relevant;
- service time, availability windows and user deadlines.

Do not treat:
- tank inventory as delivered fixture service;
- gross greywater output as net mains displacement;
- a source-level water sample as proof of a downstream fixture;
- electricity and useful thermal energy as interchangeable quantities;
- a COP calculated across different observation windows or uncalibrated boundaries as a valid conversion;
- an estimated reserve as a guaranteed duration.

Reuse the existing interval-valued service-deficit and metrology contracts. This file defines policy composition, not a new unit system, metrology library, resource ledger or solver API.

## 4. Bounded state contract

A coordination snapshot should bind:
- snapshot ID, capture time, freshness and clock/source uncertainty;
- each service provider's declared capability ID/revision and boundary;
- available resource intervals, not only point estimates;
- water qualification state and source/batch identity for every water-use path;
- energy and thermal operating envelopes, storage state, losses and limits;
- user policy revision, priorities, deadlines, consent scope and expiry;
- device-local interlock/availability status and refusal reasons;
- common-mode dependencies (grid, router, pump supply, shared valve, shared sensor, shared source);
- exact evidence and receipt IDs.

A resource estimate is not a control authorization. An old, contradictory, missing or uncalibrated observation must remain Unknown or widen its uncertainty interval; never make the feasible set appear larger by treating missing data as zero consumption or full reserve.

## 5. Candidate decision protocol

Use a replayable proposal/authorization/execution protocol:

1. **Observe:** collect source-bound and time-bounded measurements with exact physical boundaries.
2. **Qualify:** resolve measurement and service evidence; retain Unknown states and uncertainty.
3. **Construct feasible set:** exclude any candidate violating a hard gate, qualification state, local capability envelope or user authorization.
4. **Propose:** return one or more feasible alternatives with expected service vector, resource vector, uncertainty, trade-offs and affected users.
5. **Authorize:** obtain the required user or qualified-operator approval; policy version and scope are recorded.
6. **Request local action:** address the named local provider/capability. It may accept, refuse or return an updated envelope.
7. **Monitor:** observe the actual boundary outcome; do not infer success from command acceptance or timer completion.
8. **Receipt:** bind request, policy, authorization, local decision, measured service, uncertainty, deferral/isolation and restoration event.
9. **Abort / recover:** when an interlock trips, evidence becomes unknown, communication is lost or a source changes, stop issuing nonessential coordination requests and let local safe behavior continue.

There is no global actuator command. Every state transition has an actor, authority, timestamp and evidence source.

## 6. Uncertainty and robust scheduling

For each protected service, preserve:
- demand interval;
- physical deliverability interval;
- qualification state;
- observation window;
- estimate calibration / uncertainty;
- outage and recovery scenario.

The optimization objective is not an unconstrained weighted sum. First preserve the feasible set. Then compare feasible options using a declared lexicographic order or Pareto frontier.

For protected services, report the interval-valued cumulative service deficit over the scenario window. Minimize the worst-case deficit across the declared disturbance set, and show the deficit for each service independently. If a trade-off remains, present it to the user with explicit consequences rather than hiding it in a scalar score.

Reserve-time outputs must be interval-valued and calibrated against observed demand and deliverability. A single "hours remaining" number is not a valid claim when reserve, demand, flow or source quality is uncertain.

## 7. Authority, consent and privacy

- User priorities and deadlines are explicitly versioned and revocable.
- Local manual controls take precedence over non-emergency coordination.
- A household-level policy cannot relax manufacturer, regulatory, electrical, plumbing, pressure or process safety constraints.
- A device refusal is a valid outcome and must not be retried as a control bypass.
- A deferral receipt records which policy authorized it, its scope, duration and affected service.
- Collect only data necessary for the coordination decision. Prefer local processing and privacy-preserving summaries; no cross-household behavioral inference is needed for the first benchmark.
- A WAN/account outage must not revoke local manual operation or make local evidence inaccessible where the hardware permits it.

## 8. Adversarial scenario set

These are proposed synthetic tests, not physical findings.

| ID | Disturbance / failure | Required behavior |
|---|---|---|
| HRA-001 | Energy optimizer proposes a schedule that violates an approved safe thermal/hygiene envelope | Reject the schedule before optimization; local interlock remains authoritative |
| HRA-002 | Potable-use request is routed to water with Unknown/stale qualification | No potable service credit; request is refused or presented as unavailable |
| HRA-003 | A qualified municipal batch is silently blended with unqualified borehole/greywater input | Create a new derived stream; no automatic quality inheritance |
| HRA-004 | Tank-level sensor estimates reserve but fixture pressure/flow is below floor | Report inventory and deliverable service separately; do not promise reserve duration |
| HRA-005 | Electrical kWh and useful thermal kWh use different boundaries/windows | Refuse invalid conversion or COP; report carriers separately |
| HRA-006 | Grid loss and municipal water interruption share pump/controller power dependency | Detect common-mode loss; do not assume independent backups |
| HRA-007 | WAN/router/account unavailable during a service event | Local safe controls/manual override continue where physically supported; coordinator marks itself unavailable |
| HRA-008 | User deadline conflicts with an optional load deferral | Preserve the current consent policy; request authorization instead of silent reprioritization |
| HRA-009 | Reserve or pressure sensors disagree beyond calibrated tolerance | Widen uncertainty / Unknown state; exclude unsupported optimistic plan |
| HRA-010 | Municipal supply returns with unstable pressure and the tank refills | Enforce local approved refill/ramp envelope; record recovery and do not mark service restored until verified |
| HRA-011 | Leak detector proposes whole-house isolation during a known high-flow service | Require the declared local isolation policy and safety evidence; never treat model confidence alone as authority |
| HRA-012 | Coordinator crashes after proposal but before local request | No actuation; local devices continue their own safe operation |
| HRA-013 | A local appliance refuses a proposed deferral because its safe envelope changed | Accept refusal, recompute from new evidence, preserve user authorization scope |
| HRA-014 | Service receipt says command accepted but independent boundary measurement shows no delivered service | Mark service outcome unavailable/unknown; never equate command acceptance with service completion |

## 9. Benchmark outputs and proposed gates

Report a vector, not a scalar:
- safety/qualification hard-gate violations (target: zero in all synthetic and approved physical tests);
- unauthorized actuator requests / policy bypasses (target: zero);
- critical service-floor violation minutes and cumulative deficit by water, thermal and electrical service;
- false reserve-confidence and coverage of interval forecasts;
- user-approved deferrals, unapproved deferrals and missed deadlines;
- local controller refusals and whether the coordinator respected them;
- common-mode failure detection and recovery time;
- offline local-control availability by network/power scenario;
- measured resource consumption and peak power at consistent boundaries;
- cost, maintenance and consumables per unit of qualified service;
- receipt completeness and independent replayability.

**Synthetic target:** 100% of HRA negative controls must reject or enter an explicit Unknown/Infeasible/refusal path; 100% of permitted positive controls must remain feasible; zero unauthorized or safety-bypassing actuation requests. No average score can compensate for a hard-gate violation.

**Physical target:** determine sample counts through variance-component pilot data and preregistered power/precision analysis under #10341. Progress E1 to E2 to E3 to E4; do not count repeated cycles on one unit/site as independent production or household evidence.

## 10. Claim ceiling

This contract is a decision-policy and benchmark proposal. It does not authorize live control, create utility independence, establish water or food safety, approve electrical/plumbing work, qualify a treatment process, or prove household resilience. Any physical installation or integration requires the relevant local professional and authority review.
