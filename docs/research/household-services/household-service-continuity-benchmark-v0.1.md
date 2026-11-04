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

### Reuse the existing water-continuity fixture; do not fork it

The existing synthetic-only water corpus is already present on the household charter branch in [PR #10386](https://github.com/Luminous-Dynamics/symthaea/pull/10386) (original fixture work also appears in [PR #10344](https://github.com/Luminous-Dynamics/symthaea/pull/10344)): `docs/research/household-services/fixtures/water-continuity-scenarios-v0.1.json`, corpus ID `HOME-UTILITY-COORD-001-WATER-CONTINUITY-SYNTHETIC-001`. It contains 10 frozen WCONT cases. This HSC profile must compose those cases rather than create a second, overlapping 10-case water fixture.

| HSC profile | Existing WCONT coverage | Remaining gap / rule |
|---|---|---|
| HSC-001 low/intermittent inlet pressure | WCONT-002, WCONT-003 | Extend only if a distinct pressure/demand trace is needed |
| HSC-002 complete municipal interruption | WCONT-002, WCONT-006 | Confirm zero-inlet and outage-duration semantics before reusing |
| HSC-003 grid outage with municipal water | None exact | New case required; do not infer from combined outage |
| HSC-004 combined water + grid outage | WCONT-006 | Reuse |
| HSC-005 pump failure with water reserve present | None exact | New case required; separate pump fault from grid failure |
| HSC-006 stored volume but inadequate fixture head/pressure | None exact | New case required; delivery-boundary pressure must be measured |
| HSC-007 peak concurrent water/thermal demand | None exact | New demand-trace case required |
| HSC-008 staggered water/electricity restoration | WCONT-003, WCONT-010 | Existing cases cover staggered water recovery / unstable pressure, not two independently timed utilities |
| HSC-009 unstable pressure restoration | WCONT-010 | Reuse |
| HSC-010 leak under essential demand | WCONT-005 | Reuse; demand-interaction details may need an extension |
| HSC-011 sensor stuck/drift/stale/conflict | WCONT-004, WCONT-007, WCONT-008, WCONT-009 | Reuse the applicable fault-specific control; do not merge their dispositions |
| HSC-012 communications outage | None exact | New case required; local appliance operation must remain independently specified |
| HSC-013 end-use evidence expires | WCONT-007 | Reuse |
| HSC-014 apparent redundant feeds share upstream dependency | None exact | New topology/common-mode case required; compose CIV-PLACE #6600 |
| HSC-015 in-envelope demand shift | None exact | New demand-shift case required; no invented occurrence probabilities |

The WCONT corpus' `expected_disposition` values are synthetic semantic expectations, not measured physical outcomes. A future candidate-policy comparison must keep confirmatory expected outcomes under independent evaluator custody; a public conformance fixture is not a hidden or independent policy-scoring oracle.

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

For example (illustrative synthetic values only), suppose demand is bounded by [1.8, 2.2] L/min and physical fixture flow by [1.7, 1.9] L/min for 30 minutes. If the mandatory end-use qualification is Unknown, use q_qual = [0, 1.9] L/min: the deficit bound is [0, 66] L over that interval. If that end-use gate is known to fail, q_qual = [0,0] and the deficit bound is [54, 66] L. The physical flow remains separately reported in both cases. These examples assume no independence between the bound components and do not convert Unknown into a known failure.

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

## 10. Energy-carrier and thermal-boundary accounting

A household water/thermal benchmark must not collapse electrical input and useful thermal output merely because both can be expressed in kWh.

Record separately:
- wall electrical energy input (kWh-electric) by appliance, pump, heat pump, controls and standby;
- useful thermal energy delivered at the declared service boundary (kWh-thermal);
- thermal storage state change, charge/discharge loss, ambient loss and rejected heat;
- upstream hot-water or externally supplied thermal input, if present;
- pump/fan/control parasitic electricity and duty profile.

Do not subtract kWh-electric from kWh-thermal or combine them as if they were the same carrier. A COP or coefficient-of-performance ratio is a derived metric, not an accounting identity and not a validation of either meter. It must bind exact input/output measurement boundaries, a common observation window, operating mode/thermal lift, instrument calibration/uncertainty, storage state change and relevant boundary heat losses.

For an electrical service deficit, compare electrical kW demand against electrical kW delivery. For a useful-heat service deficit, compare useful thermal kW demand against useful thermal kW delivered. Cross-carrier comparisons are invalid even when the numeric unit label is kW. The service-deficit implementation tracked by [#10486](https://github.com/Luminous-Dynamics/symthaea/issues/10486) follows this type separation.

## 11. Adversarial controls

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

## 12. Proposed acceptance gates

These are research hypotheses, not standards or achieved results:
- Every synthetic hard safety/quality violation is either detected or left explicitly unresolved; none is silently accepted.
- Zero unauthorized demand deferrals, end-use substitutions or local safety-control bypasses in negative controls.
- Candidate does not worsen any declared critical service floor against the strongest matched operational baseline on the locked confirmation scenarios.
- Target at least 10% reduction in worst-case cumulative critical-service deficit against the strongest matched non-omniscient baseline, while preserving all hard constraints and accounting for added capital, energy, maintenance and consumables.
- Report the distinct #10342 hypothesis of at least 25% longer verified critical-water service versus the unbuffered reference; do not confuse that easier comparison with beating conventional tank + pump.
- Determine physical sample sizes only after pilot variance and clustering are measured. Scenario count is not household sample size.

## 13. No-runner tranche and physical gate

Before executing any candidate logic:
1. freeze scenario, demand, disturbance and initial-state contracts;
2. produce the matched comparator definitions;
3. establish an independent offline scoring contract over raw event/measurement records;
4. retain expected synthetic outcomes outside the candidate-policy workspace;
5. preregister outcome vector, primary comparisons, exclusions, missingness and stopping rules;
6. define a test-rig boundary schematic and local qualified plumbing/electrical review checklist.

No candidate pump/valve/actuator logic is executed on live household plumbing or mains-connected equipment in this tranche. Physical work requires qualified design/install review, pressure/electrical protections, isolation and containment, manual fallback and independent observation.

## 14. Evidence strata and claim ceiling

- E1: within-rig repeatability under a frozen configuration and disturbance profile.
- E2: independent operator/lab reproducibility of the same profile.
- E3: production-representative unit, pump, tank, sensor, lot and revision variation.
- E4: real household demand, installation, maintenance, supply conditions and user behaviour.

A synthetic pass is contract conformance only. Modelled service gains remain modelled. E1 does not imply E3; E2 does not imply E4.

This specification does not establish actual household resilience, drinking-water safety, plumbing/electrical safety, municipal reliability, regulatory compliance, utility independence or deployment authority.


## 15. South African national service-standard crosswalk (2025)

This benchmark now records the **Revised Compulsory National Water and Sanitation Services Standards**, Government Gazette 52814, Government Notice 6292, published 6 June 2025. The standards were prescribed under section 9(1) of the Water Services Act. Regulation 29 expressly repeals the 2001 Regulations (GNR 509). Do not use the older 25 L/person/day and seven-days-per-year wording as the current basic-service threshold without a separate legal basis.

### 15.1 Basic water service — formal settlements

Regulation 2(2) states a minimum basic-service profile comprising each of the following independent requirements:

| Dimension | Published requirement | Benchmark treatment |
|---|---|---|
| Household quantity | 6 kL per household per month | Compare measured monthly delivery at the declared user-connection boundary; retain household occupancy separately |
| Minimum flow | At least 10 L/min | Measure flow at the applicable service boundary under the declared test condition; do not infer it from tank volume or pump nameplate |
| Annual availability | Water made available for at least 358 days per year | Report available days and the exact observation window; do not replace the condition with a favourable average alone |
| Maximum continuous interruption | Not interrupted for longer than 48 consecutive hours | Preserve event-level start/end times and every interruption; a monthly volume pass cannot compensate for a >48-hour interruption |
| Drinking-water quality | Must comply with SANS 241 | Quality is a hard eligibility gate for drinking-water service, not a model estimate or a reuse-water proxy |

Regulation 2(2)(a) also specifies that, within two years of promulgation, an access/delivery point is to be at least at the end boundary of the yard (user connection point) of an existing settlement. The two-year date from publication is 6 June 2027; do not treat the future transition date as already elapsed on this benchmark's 4 November 2026 reference date.

Regulation 2(3) separates maintenance responsibility: the Water Services Institution is responsible up to the user connection; maintenance within the property boundary is the owner's responsibility. The benchmark therefore needs both a user-connection measurement boundary and an internal fixture boundary. A low-flow fixture result must not automatically be attributed to the utility, and a utility-side delivery result must not prove internal fixture service.

### 15.2 Interim service in informal settlements

Regulation 3(5) has a distinct interim-service path:
- a communal standpipe / delivery point within a reasonable walking distance of no more than 200 m from the furthest household;
- at least 6 kL/household/month;
- at least 10 L/min;
- availability at least 358 days/year;
- no interruption longer than 48 consecutive hours; and
- drinking water compliant with SANS 241.

Regulation 3(7) states that interim water services provided through tankers must not exceed 12 consecutive months. Preserve this as an interim-service / public-authority condition, not a product feature or an assumption that household storage is a compliant substitute.

### 15.3 Alternative drinking water when an interruption exceeds 48 hours

Regulation 4(1) requires the Water Services Authority, in agreement with relevant Water Services Institutions, to take reasonable measures to ensure access to alternative water supply where an interruption lasts more than 48 hours. The prescribed minimum is:
- at least 15 L of **drinking water per person per day**; and
- delivery at strategically determined points that are relatively convenient and safe.

Regulation 4(2) requires emergency/alternative water to be fit for human consumption under SANS 241. Where alternative drinking-water services are used for longer than a week, Regulation 4(3) requires a registered monitoring programme that records the alternative source and compliance results for disinfectant residuals, turbidity, *E. coli*, heterotrophic plate count, conductivity and pH until tankering is discontinued. The tanker record requirements include vehicle/driver, volume per trip, schedule, delivery point and appropriate cleanliness evidence; Regulation 4(3)(viii) says alternatives to tankering for mid- and longer-term supply must be prioritised.

The 15 L/person/day quantity is an alternative-supply obligation at designated delivery points; do not silently reinterpret it as a guarantee of 15 L/person/day at each household fixture. The benchmark should record the offered quantity, actual availability, collection/delivery location, user receipt, access burden and quality evidence separately.

### 15.4 Benchmark profile and attribution rules

Add the following crosswalk cases to the HSC test plan (they are benchmark cases, not additional claims about current municipal compliance):

- **HSC-SA-001 — basic formal-settlement service:** independently assess household-month quantity, minimum flow, annual availability, maximum consecutive interruption and SANS 241 status. Keep each dimension as its own pass/fail/unknown outcome.
- **HSC-SA-002 — extended-interruption alternative supply:** trigger at a >48-hour interruption; assess the 15 L/person/day drinking-water quantity at designated delivery points, convenience/safety evidence, actual collection and SANS 241 quality.
- **HSC-SA-003 — interim informal-settlement service:** separately assess the <=200 m access condition, quantity, flow, availability, interruption duration, drinking-water quality and the 12-consecutive-month tanker ceiling.
- **HSC-SA-004 — boundary attribution:** inject the same restriction at (a) the utility/user-connection boundary and (b) the internal household plumbing/fixture boundary. The report must identify the observed boundary and not assign responsibility beyond the measurement evidence.
- **HSC-SA-005 — unknown quality:** keep delivered physical litres visible but credit zero confirmed drinking-water service when required SANS 241 evidence is absent, stale or contradictory; record the state as Unknown rather than asserting a physical contaminant failure.

The benchmark shall distinguish:
1. public/utility basic service at the statutory access or user-connection boundary;
2. emergency alternative supply at designated public delivery points;
3. private household storage and actual fixture delivery; and
4. private non-potable reuse, which is not drinking water and cannot satisfy a drinking-water quantity/quality gate.

A household tank, WaterLoop device, private borehole, treated-greywater output or water-level sensor does not by itself establish SANS 241 conformity. Any drinking-water credit needs appropriate source identity, sampling, method, result, time validity and qualified interpretation. A household buffer may improve a measured service trajectory, but it does not discharge the Water Services Authority's duties or demonstrate restoration of the public network.

### 15.5 WHO service-level context — not South African legal thresholds

WHO's 2020 *Domestic water quantity, service level and health* guidance distinguishes access levels and associated health concerns. Its summary table describes basic access as unlikely to exceed about 20 L/person/day on average, intermediate access as approximately 50 L/person/day, and optimal access as exceeding 100 L/person/day, with access distance/time and reliability also relevant.

Keep WHO's access-level framework separate from the South African legal standard. The 6 kL/month South African basic-service quantity is defined per household, not per person. Any per-capita normalization must use declared household occupancy and the actual calendar-month length. For illustration only, 6,000 L over a 30-day month is 50 L/person/day for a four-person household; that arithmetic does not turn the legal household allocation into a universal per-capita threshold or a WHO service-level classification.

### 15.6 Competitive and product implications

The benchmark should compare the best matched operational household backup baseline (including properly sized tank + pump + automatic switching) against any WaterLoop configuration, while preserving the independent public-service reference. The national standard is a public service duty and an adequacy/accountability reference, not a product-performance certification.

A candidate product may claim only the service actually demonstrated at its own boundary. It must not:
- describe stored volume as qualified drinking water without SANS 241 evidence;
- count greywater reuse as potable or emergency drinking-water supply;
- count public standpipe/tanker availability as household fixture delivery;
- claim that meeting a private reserve target makes a municipality compliant; or
- use a 6 kL/month volume result to hide a >48-hour service interruption, a flow shortfall or a quality failure.

Regulation 27 permits a Water Services Authority to identify provisions where immediate compliance cannot be achieved and submit a progressive compliance plan to the Department for consideration and approval. The benchmark must therefore distinguish the published national standard, an authority's approved implementation plan (if evidenced), and observed service. The existence of a plan is not proof that a service requirement has been met.

**Primary sources**
- Department of Water and Sanitation, Government Gazette 52814, GoN 6292, 6 June 2025, especially Regulations 2–5, 27 and 29: https://www.gov.za/sites/default/files/gcis_document/202506/52814gon6292.pdf
- WHO, *Domestic water quantity, service level and health*, second edition, 23 December 2020: https://www.who.int/publications/i/item/9789240015241
- WHO access-level table: https://www.ncbi.nlm.nih.gov/books/NBK579449/table/ch5.tab1/?report=objectonly

This crosswalk is a research benchmark interpretation, not legal advice, a municipal compliance finding, a plumbing approval or a product certification. Confirm current legal interpretation and any municipal implementation detail with qualified South African water-services counsel / authorities before a deployment or compliance claim.


**Machine-readable national profile:** [`south-africa-water-service-standard-profile-v0.1.json`](south-africa-water-service-standard-profile-v0.1.json). It preserves the formal-settlement, interim informal-settlement and >48-hour alternative drinking-water profiles as distinct records, with regulatory clauses, boundaries, requirements and interpretation guards. JSON was parsed and fetched back after creation; this is not a legal-compliance validator.


## 15.7 SANS 241:2025 — risk-driven, time-bounded quality evidence

The legal standard names "SANS 241" as the drinking-water standard "as may be updated". The current technical migration must therefore be version-aware: a benchmark or product must not pin its interpretation to SANS 241:2015 merely because an older municipal report or archived lab result used that edition.

Rand Water's 2026 technical briefing on migrating from SANS 241:2015 to SANS 241:2025 describes a substantive shift from a result-only checklist toward an integrated risk-management and evidence process:
- water-quality risk assessment is ongoing and specific to the supply system;
- mandatory process-risk parameters are monitored routinely in all relevant supply systems;
- system-specific risk-defined parameters are selected through risk assessment and become part of the programme when risks are confirmed;
- monitoring locations and frequency are driven by system risks and representative source / final-water / distribution sample points;
- the standard includes distinct sections for water-supply and borehole systems, mobile tankers and static tanks, plus management of non-compliant results and verification.

This is a technical interpretation from a Rand Water water-quality specialist's migration briefing, not a substitute for the licensed current SANS text. The detailed parameter limits, sampling frequencies and applicability conditions must be taken from the current SANS 241 edition and the qualified risk assessment for the actual system. Do not copy numerical tables from the presentation into product firmware or treat the presentation itself as a certification criterion.

**Benchmark consequences:**
1. A historic lab result is a time- and sample-point-bound observation, not a perpetual approval for a source, tank, treatment unit or household outlet.
2. The sample point must match the claimed delivery boundary. A raw-source result does not qualify treated output; a treatment outlet does not automatically qualify a downstream storage tank or a household fixture.
3. Source changes, treatment bypass/faults, storage/stagnation events, material changes in risk, and non-compliance/corrective-action states must be visible in the currentness model. The qualified monitoring programme—not an arbitrary universal TTL—determines required sampling and review timing.
4. "Continuous monitoring" telemetry may support process control but does not replace required laboratory or qualified verification methods unless the governing standard and approved programme explicitly permit it.
5. A non-compliant result requires an explicit incident/corrective-action path. The service state remains restricted or Unknown until the relevant requalification evidence is accepted.
6. Where the current standard has distinct supply-system, borehole, tanker or static-tank regimes, the source/system classification must be preserved. Do not apply municipal-network assumptions blindly to private boreholes or stored water.

The machine-readable profile captures the standard revision and evidence-boundary requirements but intentionally contains no SANS numerical parameter limits. Its role is to prevent a false "quality pass" from stale, wrong-source, wrong-boundary or out-of-context evidence; it is not a substitute for water-quality testing.

**Technical reference:** Rand Water, *Migrating from SANS 241:2015 to SANS 241:2025 — Practical implications*, presentation by a Rand Water Water Quality Specialist (2026): https://www.randwater.co.za/media/forums/presentations/RW%20-%20Migrating%20from%20SANS%20241%202015%20to%20SANS%20241%202025.pdf. Supporting SABS-referenced update: https://plumbingafrica.co.za/update-to-sans-241/ (2026-05-12). Treat the current SANS publication as controlling; these explanatory sources are not the standard itself.


The machine-readable profile now also carries synthetic benchmark templates HSC-SA-001 through HSC-SA-005, each with independent gates and adversarial variants for volume/continuity trade-offs, emergency delivery-point accounting, 200 m access, public/private boundary attribution and stale/mis-scoped quality evidence. These are test designs, not observations of municipal compliance.
