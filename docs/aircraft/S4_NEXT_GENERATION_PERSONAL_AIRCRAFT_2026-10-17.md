# S4: Next-Generation Skyhawk-Class Personal Aircraft — Concept Requirements Card

**Date:** 2026-10-17  
**Status:** pre-design requirements hypothesis; no validated geometry, weight statement, performance estimate, or certification claim.  
**Parent strategy:** [Owner-maintainable aircraft family](OWNER_MAINTAINABLE_AIRCRAFT_FAMILY_2026-10-17.md)  
**Tracking:** [AIRCRAFT-LINE-001 #7761](https://github.com/Luminous-Dynamics/symthaea/issues/7761)

## 1. Design intent

S4 is the anchor four-seat, owner-pilot general-aviation concept: a modern successor to the *mission class* served by the Cessna 172 Skyhawk, not a Cessna product and not a copy of the existing approved design.

The goal is to preserve the qualities that make a simple personal airplane useful—straightforward handling, fixed-gear simplicity, broad training/touring utility and service familiarity—while engineering a better ownership experience: more honest payload planning, better inspection access, predictable repairs, good cabin ergonomics, clear diagnostics, locally available records and explicit lifecycle-cost accounting.

Use a conservative conventional single-engine fixed-wing design as the control case. Any hybrid, electric, fly-by-wire, retractable-gear, unconventional propulsion, pressurization, or other complexity must justify itself against that baseline, including the maintenance burden and certification impact.

This card sets requirements and experiments. It does not specify a flyable aircraft.

## 2. Reference point: current published Skyhawk

Textron's current page (reviewed 2026-10-17) lists for the Skyhawk:
- four maximum occupants;
- 1,157 kg maximum takeoff weight;
- 762 kg basic empty weight;
- 398 kg useful load;
- 254 kg published payload with full fuel;
- 180 hp Lycoming IO-360-L2A;
- 124 ktas maximum cruise speed;
- 640 nm maximum range;
- 497 m takeoff distance and 407 m landing distance.

Source: https://cessna.txtav.com/en/piston/cessna-skyhawk

The manufacturer's range figure is a headline tied to its stated ferry mission and reserve/operating conditions, not a general guarantee for a four-person trip. See the [Skyhawk reference data card](REFERENCE_BASELINE_SKYHAWK_2026-10-17.md) for weight-definition cautions, the 45%-power/IFR-reserve source correction and derived arithmetic metrics.

**Reference rule:** never compare S4 against a single headline number. Freeze candidate-specific payload, route, weather, reserve policy, takeoff/landing conditions, flight time, energy/fuel, and annual utilization. Retain all assumptions and failed/aborted results.

## 3. Primary mission cards to evaluate

These are starting requirements to test, not promised S4 performance.

### S4-M1 — Family / personal trip

- Four installed seats.
- Candidate design payload: **300 kg occupants plus baggage** as an ambitious target to test, reported separately from fuel and any mission equipment.
- Planned route: **150 nautical miles (278 km)**, with a declared weather scenario, diversion/alternate policy and applicable reserve requirements.
- The fuel/energy carried must cover the declared route, climb/descent/ground phases, reserve and any required diversion under the frozen conditions.
- Report payload, baggage volume, takeoff mass, centre-of-gravity limits, fuel/energy, uncertainty and completion margin. Do not assume 300 kg plus full tanks fits until the mass statement closes.
- Compare two configurations separately: (a) 300 kg target payload on the 150 nm mission, and (b) a full-fuel loading case with at least the incumbent Skyhawk's published 254 kg full-fuel payload as the reference floor. These are distinct loading cases.

**Decision test:** if 300 kg on S4-M1 cannot close on mass, energy/power, and relevant performance limits, show the trade-off explicitly. Do not hide it by dropping reserve, using an optimistic empty weight without a subsystem breakdown, or mixing a ferry range into the family mission.

### S4-M2 — Two-person touring

- Two adults plus baggage/cargo as an explicit mass and volume manifest.
- Planned 300 nautical mile route candidate, with weather, diversion and reserve assumptions frozen before comparing designs.
- Report total flight time, fuel/energy, payload, total cost, expected service interval/task burden, and the model's validity limits.
- Compare maximum speed only as a secondary metric. A faster aircraft that costs much more to own, is frequently unavailable, or cannot carry the declared load has not automatically won.

### S4-M3 — Training / high-utilization day

- Repeated local training sorties with start/stop/cool-down/turnaround profiles, not just one cruise calculation.
- Include daily inspection, turnaround time, fuel/energy service, heat management, wear drivers, maintenance interval requirements and student workload.
- Use representative task and dispatch counts to measure annual availability and maintenance burden.
- Do not extend an approved inspection/overhaul interval simply to improve an ownership cost number.

### S4-M4 — Emergency and degraded conditions

- Define non-normal conditions with the appropriate qualified specialists, including instrument/electrical degradation, sensor disagreement, powerplant anomalies, and loss of optional connected services.
- The design should make approved emergency procedures, physical controls, independent indications and maintenance limits clear.
- Symthaea may explain evidence and detect consistency problems; it is not a safety-critical flight-control or emergency-authority substitute.
- Define test methods under the applicable certification basis before implementation.

## 4. Candidate architecture to evaluate first

The first candidate is a **conventional high-wing, fixed-tricycle-gear, single-engine piston airplane**, with four seats and a cabin optimized around the S4-M1 load case. This is a trade-study starting point because it keeps the early comparison near the established Skyhawk mission class; it is not a final selection.

### 4.1 Airframe/material choice

- Freeze an aluminum-airframe reference because its inspection/repair ecosystem is widely understood in general aviation; compare other production approaches only with explicit damage detection, field-repair, training, tooling, corrosion and lifecycle cost evidence.
- Do not select composite simply to reduce part count or shape-drag. Account for impact damage inspection, repair supply chain, process control and field qualification.
- No wing, spar, structure, aerodynamic coefficient or dimension is specified until a documented mass/mission model and qualified aeronautical engineering trade study exist.
- Optimize for inspectability and corrosion control along with structural mass. Do not remove access or inspection provisions to improve a headline empty-weight number.

### 4.2 Powerplant and propulsion

- Use a currently supported, appropriately certificated piston-engine/propeller installation as the control baseline.
- Evaluate candidate power classes around the incumbent 180 hp class as a screening starting point; do not set a final rated power until mass, climb, cruise, hot/high, runway and reserve requirements have been checked.
- Treat diesel, alternate-fuel, hybrid-electric and all-electric choices as separate architecture cases with their own mass, cooling, fire protection, service, supply-chain and certification analysis.
- No unproven powertrain becomes the base design because its energy cost per unit looks attractive. Compare the installed system, overhaul/replacement reserve and dispatch support.

### 4.3 Landing gear and handling intent

- Keep fixed gear as the reference because it avoids the complexity of a retractable gear system; validate drag, handling, structural loads, ground clearance, propeller clearance, taxi/ground inspection, tire/brake service and landing-energy requirements before treating it as superior.
- No short-field, crosswind, stall or handling claim is made without a defined flight-test/certification basis.
- If a rough-strip capability is required, keep it as a separately qualified configuration or a different family member rather than silently loading S4 with utility-aircraft trade-offs.

### 4.4 Cabin and daily delight

- Four occupants must be defined by realistic body-size/seat-position and loading envelopes, not only seat count.
- Prioritize visible runway and traffic, easy boarding, usable baggage loading, intuitive ventilation/heat, low vibration/noise, comfortable seating and surfaces that can be cleaned/repaired.
- Design controls around predictable workload and guard against accidental switch/lever use. Modern avionics are acceptable only with a clear backup/abnormal-operation plan and supportable upgrades.
- Provide practical tie-downs, charging/storage, weather gear placement and simple cabin access to the extent compatible with approved structure and safety.
- Conduct supervised user tests on ingress/egress, reaching, sight lines, glare, sound, vibration, task load and comprehension; report individual measures rather than reducing comfort to an opaque single score.

## 5. S4 maintenance architecture

Make each task a design input before packaging freezes.

### Candidate requirements

- Every defined routine-service task has a tool/access envelope, safe isolation requirements, task-card owner and documented post-task check.
- Aim for **80% of eligible non-structural service tasks to take 60 minutes or less**, including required checks, after measuring the baseline task list. Publish the included/excluded tasks and 50th/90th percentile.
- Aim for at least **90% of eligible routine tasks to use a documented common toolkit**; any special fixture or calibrated tool must be listed, justified and available.
- Do not require removal of unrelated major assemblies for routine tasks unless engineering/safety review shows no better qualified method.
- Provide inspection access and explicit limits for control systems, engine/propeller interfaces, fuel system, electrical harnesses, landing gear, drain/corrosion zones and structural attachment areas.
- Standardize fastener families, connector identification, labels, harness drawings, approved consumables and part nomenclature when that genuinely reduces errors. All hardware and processes remain within the approved design/maintenance data.
- Use quick replacement boundaries only for equipment whose interchangeability, attachment, electrical/mechanical interface, functional check and return-to-service procedure are defined and validated.
- Every service item has an identity/revision, authorized task class, parts/tool list, hazard/isolation step, acceptance criterion, record requirement and applicable skill/certificate boundary.
- Owner self-service is a product feature only where the specific task is legally permitted for that owner and aircraft category. It is not a blanket invitation to carry out repairs, inspections or alterations.

### Owner/technician maintenance boundary

For the United States, owner-performed preventive maintenance is bounded by 14 CFR 43.3 and Appendix A; inspection, major repair/alteration, and other maintenance require the applicable authorized persons, records and approvals. For South Africa, map the applicable SACAA rules and aircraft category separately and consult the Engineering/airworthiness authority early. Keep three lists in the design and operating documentation:
1. owner-permitted checks and tasks for the defined category;
2. tasks for appropriately authorized maintenance personnel;
3. inspections/approvals and tasks that must never be represented as owner-permitted.

Sources:
- https://www.ecfr.gov/current/title-14/chapter-I/subchapter-C/part-43/section-43.3
- https://www.ecfr.gov/current/title-14/chapter-I/subchapter-C/part-43/appendix-Appendix%20A%20to%20Part%2043
- https://www.caa.co.za/information-for-the-public/frequently-asked-questions/

## 6. S4 acceptance benchmark plan

| Metric | Candidate threshold or method | Gate |
|---|---|---|
| Payload / mass | Demonstrate mass closure for S4-M1 300 kg payload target and separate incumbent 254 kg full-fuel reference case; state all limits | Before choosing geometry |
| Mission performance | Candidate-specific 150 nm and 300 nm mission cards, with declared reserves, environment and uncertainty; no extrapolated marketing range | Before selecting propulsion |
| Energy/fuel | Meter/compute at declared system boundary; include start/taxi/climb/cruise/descent and reserves; distinguish fuel flow from rated power | Before economic comparison |
| Routine maintenance | 80% eligible non-structural tasks ≤60 minutes including verification; median/p90, tools and rework published | Serviceability mock-up, then prototype |
| Common-tool access | 90% eligible routine task coverage with documented common toolkit; exceptions stated | Serviceability mock-up |
| First-time fix | Establish baseline, then target improvement only with repeat-discrepancy observation window and comparable tasks | Fleet trial |
| Parts | High-turn part fill rate, order-to-dispatch, order-to-owner and AOG days; separate in-stock from all catalog parts | Support-system design and fleet trial |
| Lifecycle cost | Model at 50/100/200 flight hours/year. A 20% lower cost per completed mission is a hypothesis, not a promise | After independent model review and measured service data |
| Comfort | Instrumented noise/vibration plus defined human-factor studies for ingress, visibility, heat, glare and workload | Mock-up and flight test as authorized |
| Manufacturing | 10% rework-labor/unit improvement vs measured baseline as a later target; trace serial units and configuration | Pilot production |
| Documentation | Supervised task comprehension and correct completion; records traceable, revision controlled and available offline | Before customer use |
| Service safety | No missing approved step, no invalid task authority, and no return-to-service without proper sign-off | Every maintenance workflow |

All numerical thresholds are design targets to evaluate. If a target conflicts with compliance or a safety margin, reject/modify the target with a documented decision. Do not declare success from a simulation-only result.

## 7. Next-gen design workflow

1. **Freeze mission + persona:** S4-M1 and S4-M2, occupant/payload mass distribution, luggage geometry, route conditions, reserve, and annual utilization.
2. **Make baseline data reproducible:** capture Skyhawk manufacturer sources/revisions and, where available, controlled POH/engine/propeller/performance and maintenance data. Mark missing information as missing.
3. **Model candidate mass boundaries:** airframe, engine/propeller, fuel, avionics, interior, installed optional equipment, payload and margins counted exactly once.
4. **Build the serviceability demonstrator:** physical cowl/engine bay/harness/panel/gear access mock-up; have qualified maintainers execute selected tasks and measure time/errors.
5. **Screen aircraft candidates:** only then compare aerodynamic/material/powertrain alternatives with sourceable coefficients and candidate-specific mission profiles.
6. **Run independent reviews:** structure, flight sciences, powerplant, electrical/fire, maintainability/human factors, production quality and certification basis.
7. **Define production/service business model:** parts availability, approved documents, training, warranties, diagnostic updates, continued operational safety, and fleet feedback.
8. **Select prototype gates:** use qualified test plans and the relevant authority path before any flight/human-carrying operation.

## 8. How Symthaea participates

- Energy model: reject candidates that miss declared energy/power/mass bounds, but keep its arithmetic limitations visible.
- Mission manifest: bind a candidate-specific model run to configuration, source data, route, mass, environment and phase uncertainty. Hash validation is not truth validation.
- Factory operations research: schedule only after task dependencies, skills and measured times are available; its current scheduling method is heuristic.
- Mycelix: record parts/lot provenance, as-built revision, inspection and service history with correction/revocation semantics. Provenance must not make incorrect evidence look true.
- Maintenance intelligence: show likely cause and confidence, evidence behind a suggestion, severity, next authorized action and uncertainty; never suppress a warning because the system's confidence score is high.
- Human authority: approved flight/maintenance manuals, qualified maintainers and the competent authority stay authoritative.

## 9. Stage gates

- **S0 — requirements baseline:** user interviews, mission cards, mass/payload definitions, operating jurisdiction and certification option scan.
- **S1 — traceable reference baseline:** sourced Skyhawk data plus controlled performance/maintenance records adequate for the declared comparison. The published summary card alone does not pass S1.
- **S2 — maintenance mock-up:** task trials, access metrics, hazard review, and documented owner/technician task boundaries.
- **S3 — candidate screening:** mass/energy/aerodynamic model with traceable inputs, uncertainty and independent review; rejected cases remain published.
- **S4 — preliminary design review:** independent discipline review and certification-basis discussions with the relevant authority.
- **S5 — approved prototype/testing:** use the applicable permits, conformity and safety process; no ad hoc flight-test work from this concept card.
- **S6 — production/field learning:** serial configuration control, service metrics, parts support, approved changes and continuing-operational-safety process.

## Nonclaims

S4 is a concept requirements card, not an aircraft specification or approved design. It does not establish weight, range, cruise performance, power rating, safety, flight characteristics, cost advantage, or owner maintenance permissions. Any physical implementation needs qualified aerospace engineering, substantiated data, appropriate testing, and the competent authority's certification/approval process.
