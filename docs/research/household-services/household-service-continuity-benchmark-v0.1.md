# Household service continuity and recovery benchmark v0.1

**Status:** research specification / no-runner tranche  
**Owner:** HOME-SERVICE-CONTINUITY-001 (#10445)  
**Parent:** HOME-UTILITY-COORD-001 (#10342), HOME-SVC-001 (#10322)  
**Evidence class:** design-only. Synthetic scenario results, if later produced, do not establish physical household resilience.

## 1. Objective

Measure whether a dwelling can continue delivering explicitly declared household services during water, electricity and thermal disruptions, and how well it recovers after the upstream service returns.

The unit of evaluation is not the tank, pump, controller or appliance. It is the **service trajectory at the household delivery boundary**.

For service s, scenario j and time t, record:

requested service -> available capability at boundary -> quality/capacity qualification -> delivered service -> unmet, deferred or substituted demand -> recovery/requalification.

The benchmark must preserve separate claims for physically available resource, technically deliverable resource, service meeting its declared quality/capacity envelope, service actually delivered, requested service deferred or denied, and unknown state due to insufficient evidence.

A reserve estimate or successful controller run does not establish service delivery.

## 2. Research basis and market boundary

The NIST report *Advancing Resilience Across Lifeline Infrastructure Systems: The Need for Consistent Guidance* (29 September 2026) describes water, wastewater, electricity, gas, communications and transportation as interdependent lifeline systems. It identifies the lack of system-level guidance for continuity and recovery objectives across these systems as a research gap.

Johannesburg Water's 29 October 2026 statement documented high demand and excessive draw-off constraining reservoirs and towers, causing low pressure, intermittent supply and prolonged outages in multiple systems. Gauteng government reporting identifies pumping failures and power interruptions alongside ageing infrastructure, leaks, high demand and maintenance backlogs.

The incumbent comparison must acknowledge existing capabilities:
- JoJo municipal backup tanks already combine automatic filling and pump-fed household supply. JoJo's sizing guidance uses people × litres/person/day × desired backup duration. Tank + pump + automatic switchover is therefore a baseline, not a novelty claim.
- Moen Flo already provides flow/pressure monitoring, leak detection and automatic shutoff. Leak detection and shutoff alone are not novelty claims.

Sources:
- NIST, 2026-09-29: https://www.nist.gov/publications/advancing-resilience-across-lifeline-infrastructure-systems-need-consistent-guidance
- Johannesburg Water, 2026-10-29: https://www.johannesburgwater.co.za/blog/2026/10/29/johannesburg-water-implements-urgent-interventions-to-stabilise-severely-constrained-systems/
- Gauteng Provincial Government, 2026-02-13: https://www.gov.za/news/media-statements/gauteng-provincial-government-addresses-water-challenges-13-feb-2026
- JoJo municipal backup: https://www.jojo.co.za/water-storage-solutions/municipal-backup-water-storage-solutions/
- Moen Flo: https://flo.moen.com/pages/flo-smart-water-monitor

These sources motivate the test profile. They do not establish household demand, market demand, municipal data access, or the feasibility of any candidate system.

## 3. Service envelope contract

Each service profile must be declared before the scenario is executed.

Required fields:
- service ID and profile revision;
- served function and actual delivery boundary;
- source(s), physical path and relevant dependencies;
- requested-demand trace, unit and timebase;
- minimum acceptable capacity/quality/pressure/temperature/delivery-time conditions where relevant;
- user-approved priority, deadline, permitted substitution and permitted deferral;
- prohibited substitutions and non-deferrable constraints;
- local appliance safety authority and manual fallback;
- required evidence and its freshness/uncertainty limits;
- recovery conditions and post-restoration qualification.

Do not impose a universal hierarchy of household needs. The benchmark household declares its requirements and consent/priority policy. A coordinator may propose a plan but may not silently modify those requirements.

### Service-state vocabulary

| State | Meaning |
|---|---|
| WithinEnvelope | Evidence supports the declared service floor at the delivery boundary |
| DegradedUsable | Service remains available but falls outside the nominal profile and within a declared degraded profile |
| DeferredAuthorized | Request is postponed under a pre-authorized user policy |
| Unavailable | The required service cannot be delivered |
| Unknown | Evidence is missing, stale, contradictory or inadequate to determine service state |
| Restoring | Upstream or local service has returned but the required service requalification is incomplete |

A deferred request is not counted as delivered. An unknown service state is not a pass. A physically full tank with no qualified pressure/flow path is not available service.

## 4. Reference dwelling and measurement boundaries

The first test profile is a single dwelling/test rig with:
- measured municipal water inlet pressure and flow;
- physically separated backup tank and pump;
- cold-water delivery fixtures;
- one hot-water / thermal-store or heat-pump service;
- grid electricity, with optional measured PV/battery;
- one flexible appliance load (e.g. dishwasher);
- optional non-potable circuit only after separate plumbing review and isolation.

At each boundary, identify:
- meter/instrument and calibration status;
- measurand, unit, range and uncertainty;
- physical boundary and direction;
- observation start/end and clock-domain identity;
- missingness, stale data and quality flags;
- whether the measurement is aggregate or branch-level.

Never sum an aggregate meter and its branch meters for the same physical flow. Never infer independence from component count. A pump and controller on the same circuit share a power dependency even when they are separate devices.

## 5. Matched comparator arms

All arms receive identical initial conditions, demand traces, disturbance traces, service envelopes and scoring rules.

| Arm | Configuration | Purpose |
|---|---|---|
| A | Municipal-only, no backup buffer | Unbuffered reference |
| B | Properly installed tank + pump + automatic switching | Conventional backup comparator |
| C | Tank + pump + independent leak monitoring/isolation | Monitoring/isolation increment |
| D | Deterministic, user-approved service-priority rules | Non-learning coordination baseline |
| E | Evidence-bounded household coordination proposal | Candidate policy |
| F | Offline perfect-information allocation | Theoretical upper bound only; not a deployable comparator |

Match actual usable reserve, pump curve/capability, electrical supply, pipe topology, maintenance state, leak protection, installation quality and total cost. Where matching a dimension is impossible, disclose it and avoid attributing the difference to the policy.

The perfect-information arm must never be presented as a fair operational competitor; it defines an optimistic bound under its declared model.

## 6. Disturbance matrix

Start with deterministic known-answer scenarios. Do not assign failure probabilities from invented frequencies.

| ID | Disturbance | Required observation |
|---|---|---|
| HSC-001 | Inlet pressure reduction with intermittent flow | Fixture pressure/flow, delivered volume, degradation state |
| HSC-002 | Complete municipal interruption | Service duration from actual delivery boundary |
| HSC-003 | Grid outage, municipal water retained | Pump/control availability and independent local fallback |
| HSC-004 | Combined water + grid outage | Common-mode loss of pump and control; actual reserve delivery |
| HSC-005 | Pump failure with reserve in tank | Difference between stored and deliverable water |
| HSC-006 | Tank level adequate, head/pressure inadequate | No false reserve-service assertion |
| HSC-007 | Peak demand / hot ambient with concurrent water and heat loads | Competing demands and unmet service vector |
| HSC-008 | Water and electricity restore at different times | Dependency-aware recovery sequence |
| HSC-009 | Unstable inlet restoration / pressure oscillation | Refill rebound, repeat interruption and pressure transients |
| HSC-010 | Leak competes with essential fixture demand | Leak containment versus retained essential service |
| HSC-011 | Stuck, drifting, stale or contradictory sensor | Uncertainty and abstention; no false reassurance |
| HSC-012 | Communications loss | Appliance-local safe operation and manual fallback |
| HSC-013 | End-use qualification expires during outage | Physically available but unqualified use remains segregated |
| HSC-014 | Nominal redundant feeds share one upstream dependency | Common-mode failure is retained |
| HSC-015 | Demand distribution shifts within the declared envelope | Out-of-profile detection and conservative service estimate |

Every scenario binds exact initial configuration, demand profile, disturbance onset/magnitude, duration or stop rule, recovery condition, source profile, expected claim ceiling and exact clock semantics.

## 7. Outcome vector

Report per service, per scenario and per comparator arm:
1. qualified service minutes/hours above declared floor;
2. delivered volume or energy at the actual delivery boundary;
3. pressure/flow/temperature/quality compliance fraction and duration;
4. cumulative service deficit, preserving physical units;
5. longest uninterrupted degraded/unavailable interval;
6. time to first degradation;
7. time to qualified recovery;
8. post-restoration stability, transient overshoot and rebound;
9. amount and duration of authorized deferral;
10. denied, unresolved and unknown service requests;
11. leak volume, missed-leak challenges, false isolation and detection latency;
12. reserve prediction error, interval coverage and false-reassurance cases;
13. electricity, water, useful thermal energy and peak power at device and household boundaries;
14. pump starts/cycles, maintenance, consumables and operator intervention;
15. installed footprint and lifecycle cost per qualified essential-service hour.

### Cumulative service deficit

For a declared service demand rate d_s(t) and qualified delivered service rate q_s(t), one possible deficit measure is:

D_s = integral over the scenario window of max(0, d_s(t) - q_s(t)) dt.

The units depend on the service: litre-hours, energy, or another declared service quantity over time. This measure is supplementary; it must not replace outage duration, peak deficit, critical-floor violations or denied-demand reporting.

If service is deferred under an authorized policy, retain both the time-shifted demand and its actual eventual delivery. Do not erase the original deadline or count a late delivery as on-time service.

Do not create a weighted total household-resilience score. A dishwasher task cannot compensate for an essential water-service failure.

## 8. Policy and architecture experiments

Test these hypotheses against deterministic and later physically measured scenarios:
- **Service trajectory > reserve inventory:** optimize delivered capability through onset, degraded operation, recovery and post-restoration stability.
- **Capability floor > average comfort:** preserve declared essential service floors before flexible loads, subject to user authorization.
- **Graceful degradation with consent:** only pre-authorized requests may be deferred/substituted; every non-delivery remains visible.
- **Dependency diversity > nominal redundancy:** map shared upstream power, control, plumbing, pump and communications dependencies.
- **Uncertainty-bounded reserve:** publish a conservative service interval; abstain when pressure, flow, quality, demand or sensor health is inadequate.
- **Recovery-aware coordination:** avoid refill surges, simultaneous starts, thermal rebound and synchronization that worsens a recovering bottleneck.
- **Local autonomy + compositional planning:** coordinator proposes allocations; appliance-local safety, manual override and safe fallback remain authoritative.
- **Household/city interoperability without centralization:** share read-only service and outage evidence where authorized; do not create a municipal command interface or imply authority.

## 9. Adversarial controls

At minimum:
- same tank litres, different elevation/pump curve/fixture pressure;
- apparent redundancy with shared transformer, circuit, controller or pump;
- improved average service but one declared critical floor worsens;
- optimizer silently rewrites user priorities/deadlines;
- stored water appears available but delivery path is blocked;
- predicted municipal restoration precedes observed restoration;
- quality/use evidence is stale while physical inventory remains;
- coordinator communication fails and appliances wait unnecessarily;
- high but in-profile demand shift invalidates a point reserve forecast;
- restoration causes refill surge, pump cycling or repeated low pressure;
- false leak alarm isolates all fixtures;
- small continuous leak is missed;
- comparator arm has different reserve/pump/maintenance conditions;
- simulation result is represented as physical measurement.

No sensitive demographic labels are needed for this synthetic corpus. Report service shortfalls by declared service class and demand window; do not infer protected traits.

## 10. Proposed acceptance gates

These are research hypotheses, not standards or achieved results:
- Every synthetic hard safety/quality violation is either detected or left explicitly unresolved; none is silently accepted.
- Zero unauthorized demand deferrals, end-use substitutions or local safety-control bypasses in negative controls.
- Candidate does not worsen any declared critical service floor against the strongest matched operational baseline on the locked confirmation scenarios.
- Target at least 10% reduction in worst-case cumulative critical-service deficit against the strongest matched non-omniscient baseline, while preserving all hard constraints and accounting for added capital, energy, maintenance and consumables.
- Report the distinct #10342 hypothesis of at least 25% longer verified critical-water service versus the unbuffered reference; do not confuse that easier comparison with beating conventional tank + pump.
- Determine physical sample sizes only after pilot variance and clustering are measured. Scenario count is not household sample size.

## 11. No-runner tranche and physical gate

Before executing any candidate logic:
1. freeze scenario, demand, disturbance and initial-state contracts;
2. produce the matched comparator definitions;
3. establish an independent offline scoring contract over raw event/measurement records;
4. retain expected synthetic outcomes outside the candidate-policy workspace;
5. preregister outcome vector, primary comparisons, exclusions, missingness and stopping rules;
6. define a test-rig boundary schematic and local qualified plumbing/electrical review checklist.

No candidate pump/valve/actuator logic is executed on live household plumbing or mains-connected equipment in this tranche. Physical work requires qualified design/install review, pressure/electrical protections, isolation and containment, manual fallback and independent observation.

## 12. Evidence strata and claim ceiling

- E1: within-rig repeatability under a frozen configuration and disturbance profile.
- E2: independent operator/lab reproducibility of the same profile.
- E3: production-representative unit, pump, tank, sensor, lot and revision variation.
- E4: real household demand, installation, maintenance, supply conditions and user behaviour.

A synthetic pass is contract conformance only. Modelled service gains remain modelled. E1 does not imply E3; E2 does not imply E4.

This specification does not establish actual household resilience, drinking-water safety, plumbing/electrical safety, municipal reliability, regulatory compliance, utility independence or deployment authority.
