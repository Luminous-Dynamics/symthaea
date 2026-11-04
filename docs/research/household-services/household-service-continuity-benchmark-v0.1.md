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

### Cumulative service deficit — interval-valued and quality-qualified

Use this integral only for rate-based services whose requested and delivered rates have the same declared unit and timebase.

Let:
- d_s(t) be requested demand rate, with uncertainty interval [d_s^-(t), d_s^+(t)];
- q_phys_s(t) be physically measured delivery rate at the service boundary;
- q_qual_s(t) be the rate that is both physically delivered and qualified against every mandatory service-envelope gate, with interval [q_s^-(t), q_s^+(t)].

Never substitute q_phys for q_qual when quality, pressure, end-use or other mandatory qualification is missing or failed. Preserve physical delivery separately even when it cannot be credited as qualified service.

The instantaneous deficit interval is:

Ddot_s^-(t) = max(0, d_s^-(t) - q_s^+(t))
Ddot_s^+(t) = max(0, d_s^+(t) - q_s^-(t))

and the cumulative bound over [0,T] is:

D_s in [ integral_0^T Ddot_s^-(t) dt,
         integral_0^T Ddot_s^+(t) dt ].

This yields the declared service-quantity unit, not a unit multiplied by time: for example, L/hour integrated over hours yields L; kW integrated over hours yields kWh. The previous "litre-hours" wording would be dimensionally incorrect.

Qualification semantics:
- all mandatory evidence current and passing: q_qual is bounded by the measured physical delivery and its uncertainty;
- a mandatory gate is known to fail: q_qual = [0,0] for the affected use class, while q_phys remains reported;
- a mandatory gate is unknown, stale or contradictory: retain the state as Unknown and use a conservative lower qualified-delivery bound of zero; do not rewrite Unknown as a physical failure or a pass;
- quality/pressure/temperature failures should also be reported as distinct hard-constraint violations, not hidden inside the integral.

For discrete services (such as an accepted dishwasher request), do not force a continuous-rate integral. Report request count, acceptance, deadline, latency, failure/deferral and rework directly.

This measure is supplementary; it must not replace outage duration, peak deficit, critical-floor violations, denied-demand reporting, unknown-state duration or recovery stability.

If service is deferred under an authorized policy, retain both the original request/deadline and its later completion. Do not erase the original deadline or count late delivery as on-time service.

Do not create a weighted total household-resilience score. A dishwasher task cannot compensate for an essential water-service failure.

## 8. Affordability, accessibility and public-service complementarity

The household capability floor must not assume every household can buy, install and maintain a private tank, pump, battery or treatment system. Nor should a household product normalize permanent inferior service or transfer public-service accountability to residents.

The South African Human Rights Commission's Gauteng inquiry report (released 2 November 2026) identifies failures across the water-service value chain, including ageing infrastructure, losses, maintenance, storage/pumping capacity, electricity interruptions, technical skills and financial constraints. It emphasizes that responsibility must remain attributable to the relevant public/utility institutions and cautions against normalising recurring road-based water distribution as a permanent inferior service:
https://sahrc.org.za/index.php/sahrc-media/news-2/item/4567-media-statement-the-south-african-human-rights-commission-releases-an-inquiry-report-highlighting-neglected-water-and-sanitation-infrastructure-and-heavy-reliance-on-road-based-water-distribution-water-tankers-in-gauteng

SAnews reported on 3 November 2026 that Johannesburg's overall reservoir storage had improved to 41% from 37% the previous day, while some households had experienced dry taps for more than 10 days and recovery remained uneven:
https://www.sanews.gov.za/node/85387

These are problem and scenario-design signals, not product demand estimates or permission to substitute private equipment for municipal infrastructure.

### Required affordability and access reporting

For each matched configuration, report separately:
- purchase and installation cost;
- financing/lease cost where applicable;
- plumbing/electrical modification and reinstatement cost;
- energy, water, consumables, filters and replacement parts;
- maintenance labour, service travel, downtime and disposal;
- total annualized and lifecycle cost under declared usage/outage profiles;
- cost per delivered qualified litre, useful kWh or accepted service request, plus separately reported continuity hours where relevant;
- voluntary, privacy-preserving affordability bands (e.g. share of disposable household resources), not exact income or identity data;
- sensitivity to tariffs, interest rates, parts availability and outage frequency.

Do not collapse affordability into one average. Report outcome/cost distributions by voluntarily supplied, anonymized affordability bands and installation constraints. Do not infer or proxy protected traits.

### Installability and reversibility

Test representative constraints separately:
- owner-occupied dwelling with suitable utility space;
- rental dwelling where equipment must be removable and installation reversible;
- small or space-constrained dwelling;
- dwelling with limited electrical capacity;
- shared or multi-unit property with a common utility boundary.

Record area/volume footprint, access, required professional modifications, landlord/body-corporate/utility permissions where applicable, reinstatement burden and safe maintenance access. Do not assume permission or code compliance.

### Public-service complementarity

WaterLoop is a temporary, bounded resilience buffer and observability/maintenance aid. It is not a replacement for reliable municipal supply, a permanent tanker service, an alternative public utility, or a reason to reduce public investment and accountability.

Benchmark:
- continuity during a declared interruption;
- return to normal piped service after utility restoration;
- dependence on upstream public/utility service;
- affordability and availability for households unable to purchase private equipment;
- compatibility with shared/community resilience options, where locally reviewed and feasible.

Keep public-system restoration metrics and household-appliance metrics separate. A household buffer may reduce a bounded service interruption; it cannot claim that the municipal service has been restored.


## 9. Policy and architecture experiments

Test these hypotheses against deterministic and later physically measured scenarios:
- **Service trajectory > reserve inventory:** optimize delivered capability through onset, degraded operation, recovery and post-restoration stability.
- **Capability floor > average comfort:** preserve declared essential service floors before flexible loads, subject to user authorization.
- **Graceful degradation with consent:** only pre-authorized requests may be deferred/substituted; every non-delivery remains visible.
- **Dependency diversity > nominal redundancy:** map shared upstream power, control, plumbing, pump and communications dependencies.
- **Uncertainty-bounded reserve:** publish a conservative service interval; abstain when pressure, flow, quality, demand or sensor health is inadequate.
- **Recovery-aware coordination:** avoid refill surges, simultaneous starts, thermal rebound and synchronization that worsens a recovering bottleneck.
- **Local autonomy + compositional planning:** coordinator proposes allocations; appliance-local safety, manual override and safe fallback remain authoritative.
- **Household/city interoperability without centralization:** share read-only service and outage evidence where authorized; do not create a municipal command interface or imply authority.

## 10. Adversarial controls

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

## 11. Proposed acceptance gates

These are research hypotheses, not standards or achieved results:
- Every synthetic hard safety/quality violation is either detected or left explicitly unresolved; none is silently accepted.
- Zero unauthorized demand deferrals, end-use substitutions or local safety-control bypasses in negative controls.
- Candidate does not worsen any declared critical service floor against the strongest matched operational baseline on the locked confirmation scenarios.
- Target at least 10% reduction in worst-case cumulative critical-service deficit against the strongest matched non-omniscient baseline, while preserving all hard constraints and accounting for added capital, energy, maintenance and consumables.
- Report the distinct #10342 hypothesis of at least 25% longer verified critical-water service versus the unbuffered reference; do not confuse that easier comparison with beating conventional tank + pump.
- Determine physical sample sizes only after pilot variance and clustering are measured. Scenario count is not household sample size.

## 12. No-runner tranche and physical gate

Before executing any candidate logic:
1. freeze scenario, demand, disturbance and initial-state contracts;
2. produce the matched comparator definitions;
3. establish an independent offline scoring contract over raw event/measurement records;
4. retain expected synthetic outcomes outside the candidate-policy workspace;
5. preregister outcome vector, primary comparisons, exclusions, missingness and stopping rules;
6. define a test-rig boundary schematic and local qualified plumbing/electrical review checklist.

No candidate pump/valve/actuator logic is executed on live household plumbing or mains-connected equipment in this tranche. Physical work requires qualified design/install review, pressure/electrical protections, isolation and containment, manual fallback and independent observation.

## 13. Evidence strata and claim ceiling

- E1: within-rig repeatability under a frozen configuration and disturbance profile.
- E2: independent operator/lab reproducibility of the same profile.
- E3: production-representative unit, pump, tank, sensor, lot and revision variation.
- E4: real household demand, installation, maintenance, supply conditions and user behaviour.

A synthetic pass is contract conformance only. Modelled service gains remain modelled. E1 does not imply E3; E2 does not imply E4.

This specification does not establish actual household resilience, drinking-water safety, plumbing/electrical safety, municipal reliability, regulatory compliance, utility independence or deployment authority.
