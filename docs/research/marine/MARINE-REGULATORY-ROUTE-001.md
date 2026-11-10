# MARINE-REGULATORY-ROUTE-001
## Route-aware regulatory accounting and compliance scenario benchmark

**Status:** research specification v0.1.0  
**Created:** 2026-11-10  
**Parent:** MARINE-WAPS-CO-DESIGN-001 (#11406)  
**Related:** MARINE-SERVICE-ROBUSTNESS-001, #11429; MARINE-COASTAL-HYBRID-REPRODUCTION-001, #11431; MARINE-CYBER-PHYSICAL-RESILIENCE-001, #11432

> This is a research/accounting contract, not legal advice, a compliance determination, a verified ship emissions report or regulatory approval. Applicability must be determined by qualified operators, verifiers, class/flag and legal specialists.

## 1. Paradigm shift: the route is also a regulatory exposure

A ship's least-cost route is not necessarily its lowest-cost or lowest-impact route after fuel pathway, emissions reporting, carbon pricing, port rules, emission-control areas, vessel scope and reporting periods are accounted for.

The vessel design and voyage policy should therefore be evaluated against two independent but connected maps:

1. **Physical service map:** route geometry, metocean conditions, cargo work, time, fuel, electricity, equipment state and reliability.
2. **Regulatory exposure map:** jurisdiction, port call, emissions-control area, applicable rule, vessel scope, gas, fuel pathway, reporting period, rule status and effective date.

The regulatory map is not a replacement for physical energy accounting. It changes the cost and constraints associated with physical outcomes. Keep physical energy, delivered service, lifecycle GHG, statutory indicators and compliance liabilities as separate output dimensions.

## 2. Current regulatory snapshot (10 November 2026)

### 2.1 IMO EEXI/CII and wind-propulsion accounting

The IMO's phase-two review of the short-term GHG measures (EEXI, SEEMP and CII) was initiated at MEPC 84. The work plan includes possible revisions to CII metrics and consideration of how wind propulsion is accounted for in EEDI/EEXI calculations. This is directly relevant to WAPS design, but it is a review activity—not a new adopted wind-propulsion credit formula.

Keep these distinct:
- **EEXI:** technical/design energy-efficiency index for applicable existing ships.
- **CII:** annual operational carbon-intensity indicator, with vessel-type-dependent calculation and A–E rating.
- **EEDI:** design index for applicable new ships.

Do not use TEU-nautical-mile or tonne-nautical-mile as a substitute for the applicable statutory denominator. Keep the commercial service metric alongside the statutory metric, not in place of it.

### 2.2 Non-CO2 marine diesel emissions

MEPC 84 adopted/issued three 2026 guideline resolutions relevant to methane and nitrous oxide measurement:
- MEPC.414(84): test-bed and onboard measurements of CH4 and/or N2O from marine diesel engines;
- MEPC.415(84): engine-load monitoring and calculation of emission values;
- MEPC.416(84): continuous emission monitoring systems for CH4 and/or N2O.

The IMO has described actual tank-to-wake CH4/N2O quantification using engine-load monitoring with measurement guidelines and/or CEMS as a voluntary approach, without prejudice to future mandatory frameworks. This is an opportunity to make the evidence chain gas-resolved rather than treat CO2e as one opaque number.

### 2.3 EU ETS maritime

The EU ETS has applied to maritime transport since 2024. For applicable voyages:
- 100% of emissions are covered for voyages between two EU ports and emissions within EU ports;
- 50% of emissions are covered for voyages with one end outside the EU;
- the maritime ETS includes CH4 and N2O from 2026, in addition to CO2;
- the phase-in reaches 70% surrender for 2025 emissions reported in 2026 and 100% from 2027.

Applicability, gas coverage and surrender fractions must be evaluated per reporting year and route, not baked into one timeless coefficient.

### 2.4 FuelEU Maritime

FuelEU Maritime has applied since 1 January 2025. Regulation (EU) 2023/1805 sets yearly-average GHG-intensity limits for energy used on board, based on a 91.16 gCO2e/MJ reference value and reduction factors:
- 2% from 2025;
- 6% from 2030;
- 14.5% from 2035;
- 31% from 2040;
- 62% from 2045;
- 80% from 2050.

The Regulation includes well-to-wake methodology and conditional RFNBO reward provisions. Model its applicability, actual energy use, fuel pathway, gas factors, allowed pooling/banking/borrowing and verification state only under the current legal text and delegated/implementing acts. Do not reduce it to a generic tank-to-wake CO2 price.

### 2.5 Geographic air-pollution constraints

MEPC 84 adopted a North-East Atlantic Emission Control Area with implementation in 2028 and a 0.10% sulphur limit within the area. Represent ECA boundaries and effective dates as versioned geographic rules. A route segment that crosses an ECA boundary must not inherit one uniform fuel assumption for the whole voyage.

### 2.6 IMO Net-Zero Framework status

As of 10 November 2026, the IMO Net-Zero Framework had not yet completed formal adoption. MEPC 84 established intersessional work ahead of MEPC 85 (30 November–3 December 2026), with a resumed extraordinary session provisionally scheduled for 4 December, subject to confirmation. Treat proposals and pending amendments as scenarios, not binding rules.

## 3. Core architecture: a versioned regulatory evidence graph

## New ship manoeuvrability standards under development

SDC 12 (19–23 January 2026) continued development of goal-based SOLAS Chapter II-1 (Part C) and Chapter V steering/propulsion provisions intended to cover traditional and non-traditional systems. It also discussed draft International Ship Manoeuvrability Standards (ISSM) for new ships subject to the future goal-based requirements. SDC 12 agreed the principles in principle, but technical content remains under development.

The current roadmap anticipates further work at SDC 13 in 2027, expected adoption around 2029 and target entry into force on 1 January 2032. These are planning milestones, not adopted rules. The existing MSC.137(76) standards remain the reference for existing ships under the current roadmap. The machine-readable rule register marks the new ISSM/SOLAS work as `under_review` without assigning a made-up future meeting date.


A first machine-readable snapshot is retained at [`marine-regulatory-rule-register-2026-11-10.json`](./marine-regulatory-rule-register-2026-11-10.json), with its v0.1.0 Draft 2020-12 contract. The corrected current snapshot is [`marine-regulatory-rule-register-2026-11-10-v0.2.0.json`](./marine-regulatory-rule-register-2026-11-10-v0.2.0.json), with schema [`marine-regulatory-rule-register-v0.2.0.schema.json`](./marine-regulatory-rule-register-v0.2.0.schema.json). The standalone `marine-regulatory-rule-audit` checks date/status consistency and source references. This is a register integrity check, not a legal applicability engine.

Each rule record should bind:

- stable rule ID, instrument/circular/resolution and exact revision;
- responsible authority and jurisdiction;
- status: proposed, under_review, approved, adopted, entered_into_force, applicable, superseded or withdrawn;
- published/adoption/entry-into-force/application dates;
- ship types, size/GT/DWT thresholds, build-date and flag conditions;
- port-call, voyage-leg, geographic and fuel/consumer applicability;
- gases and pollutants covered;
- calculation method, units, functional denominator and reporting period;
- phase-in/reduction factors, price or liability inputs, exemptions and boundary rules;
- authoritative source URI, source digest, rights/access status, reviewer and limitations.

No rule can be promoted from draft to effective by model inference. The engine should preserve the status and effective date that applied to each simulated voyage/reporting period.

## 4. Route and voyage segmentation

The route representation must support:
- port calls with jurisdiction, coordinates and arrival/departure times;
- voyage legs between ports;
- intersections with ECA and other geographic compliance areas;
- emissions occurring at berth, within port limits, on the high seas and in jurisdictional waters;
- dates crossing rule effective dates or reporting-year boundaries;
- deviations, waiting, anchorage and weather re-routing.

A route-level rule result must explain which segments were included, their fraction of covered emissions, and why. When a geographic rule boundary or voyage leg is unresolved, return UNKNOWN/INCOMPLETE rather than assuming full or zero coverage.

## 5. Emissions and fuel-chain data contract

Maintain separate ledgers for:
- fuel mass/energy by fuel batch, pathway and consumer;
- shore electricity by location, provider, mix and accounting method;
- CO2, CH4 and N2O tank-to-wake mass by engine/consumer;
- upstream well-to-tank emissions and source pathway;
- well-to-wake GHG intensity and global-warming-potential revision;
- NOx, SOx and particulate matter where relevant to geographic/engine requirements;
- measurement method (default factor, engine load monitoring, CEMS, supplier certificate or other declared method);
- fuel origin, chain of custody, sustainability certification, book-and-claim or mass-balance treatment as applicable;
- verification status, uncertainty, exclusions, source rights and data quality.

Never substitute a CO2-only factor for CO2e where CH4/N2O are in scope. Never treat a supplier's nominal fuel pathway as verified chain of custody. A hash identifies the supplied document/record, not its truth.

## 6. Keep commercial and statutory indicators distinct

The reporting layer must expose separate outputs:

| Dimension | Measure | Boundary |
|---|---|---|
| Commercial service | delivered TEU-nm or tonne-nm; annual throughput; on-time rate | Declared service contract |
| Physical energy | primary energy per cargo work; shaft, electrical, auxiliary and fuel boundaries | Whole vessel |
| Lifecycle climate | WtT, TtW and WtW GHG by gas/pathway | Declared LCA method and fuel chain |
| CII | attained/required annual operational CII and rating | Applicable vessel type and IMO method |
| EEXI/EEDI | attained/required technical/design index | Applicable vessel and statutory calculation |
| EU ETS | covered CO2e tonnes, coverage fraction, allowances and scenario liability | Voyage/reporting-year/jurisdiction |
| FuelEU | annual GHG intensity, compliance balance and eligible flexibility | In-scope ship, energy and reporting period |
| ECA / local air pollution | fuel sulfur and other required pollutant limits | Geographic segment and effective date |

Do not combine these into a single score that allows an energy win to conceal a statutory failure or a cargo-service loss.

## 7. Benchmark and falsification suite

### Structural and applicability
- 100% agreement with hand-built positive/negative applicability cases.
- Zero false-applicable or false-exempt outcomes in the declared rule corpus.
- Every calculation carries the rule ID, version, applicability decision and effective-date basis.
- 100% voyage segment coverage; no silently unassigned emissions.

### Arithmetic and accounting
- 100% agreement with hand-calculated ETS coverage fractions, CO2e, FuelEU GHG intensity, and applicable CII/EEXI fixtures within frozen tolerances.
- Zero unknown fuel/energy/emissions values silently coerced to zero.
- No double counting between tank-to-wake, well-to-tank and well-to-wake.
- CH4/N2O reporting remains explicitly unknown when evidence is required but absent; no CO2-only silent fallback.

### Change management
- 100% tests for future-dated, superseded, withdrawn and amended rules.
- A pending IMO Net-Zero proposal never appears as an effective legal obligation before adoption/entry into force.
- Re-running a historic voyage with its historic rule snapshot produces the same result.
- Every regulatory update generates a machine-readable delta against prior versions.

### Decision usefulness
- Route alternatives are compared on the same delivered cargo service, while statutory measures and costs are shown separately.
- Sensitivity to carbon prices, fuel pathway, electricity mix, RFNBO eligibility, port calls and rule effective dates is visible.
- A regulatory-cost route recommendation is advisory only and never supersedes navigation/safety constraints or legal review.

## 8. Research opportunities beyond current SOTA

1. **Regulation-aware ship co-design:** make fuel/energy topology and WAPS choice sensitive to the actual route and effective rules while preserving the physical-service Pareto frontier.
2. **Wind-propulsion accounting evidence:** connect device operating state, wind/shaft measurements, fuel flow and the applicable EEDI/EEXI/CII calculation revision. This may help the industry produce auditable WAPS credit evidence as IMO methods evolve.
3. **Non-CO2 measurement qualification:** integrate ELM/CEMS data, analyzer calibration, engine-load profiles, gas-specific uncertainty and measurement source into the same trial-evidence registry.
4. **Fuel chain-of-custody provenance:** connect supplier/batch certificates and lifecycle factors to energy consumed on each route; keep verified, declared and unknown pathways distinct.
5. **Policy uncertainty without legal confusion:** compare current rules, adopted future rules and pending proposals as separately labelled scenarios. This supports investment decisions without implying that proposed rules are already law.

## 9. Luminous system boundaries

- **Symthaea:** rule applicability, source/evidence admission, versioned calculation contract, scenario design and comparison eligibility.
- **Symtropy:** deterministic energy/emissions arithmetic and replay within declared calculation methods.
- **Sol-Atlas:** route geometry, port and ECA segment intersections, coordinate/system provenance.
- **Mycelix:** supplier, fuel-batch, certificate, verification, chain-of-custody and revision provenance.

Do not build another general GIS, emissions ledger or legal expert system. Integrate authoritative rule data and keep approvals with the responsible authorities and verifiers.

## 10. Sources

- IMO MEPC 84, 27 April–1 May 2026: https://www.imo.org/en/mediacentre/meetingsummaries/pages/temporary-mepc-84th-session.aspx
- IMO MEPC resolutions 2026–27, including MEPC.414(84), .415(84), .416(84): https://www.imo.org/en/knowledgecentre/indexofimoresolutions/pages/mepc-2026-27.aspx
- IMO EEXI/CII FAQ and phase-two review: https://www.imo.org/en/mediacentre/hottopics/pages/eexi-cii-faq.aspx
- IMO 2023 GHG Strategy and regulatory workplan: https://www.imo.org/en/ourwork/environment/pages/2023-imo-strategy-on-reduction-of-ghg-emissions-from-ships.aspx
- European Commission, maritime EU ETS: https://climate.ec.europa.eu/areas-action/transport-decarbonisation/reducing-emissions-shipping-sector_en
- European Commission, EU ETS maritime FAQ: https://climate.ec.europa.eu/areas-action/transport-decarbonisation/reducing-emissions-shipping-sector/faq-maritime-transport-eu-emissions-trading-system-ets_en
- Regulation (EU) 2023/1805 (FuelEU Maritime): https://eur-lex.europa.eu/eli/reg/2023/1805/oj/eng

## 11. Claim ceiling

A rule-engine pass means only that supplied data satisfy the encoded rule version and declared calculation method. It is not a legal opinion, statutory verifier statement, class/flag approval or proof of complete emissions data. All outputs must show applicable rule editions, route segments, vessel scope, reporting period, fuel pathway, gas coverage, data quality and exclusions.
