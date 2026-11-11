# MARINE-PORT-ENERGY-CO-DESIGN-001
## Ship–port shore-power, berth-window and energy-interface qualification

**Status:** research specification v0.1.0  
**Created:** 11 November 2026  
**Parent:** MARINE-WAPS-CO-DESIGN-001 (#11406)  
**Interfaces:** #11558 service flexibility; #11507 source ledger; #11501 WAPS MRV; #11028 shipboard transient-power assurance; #11469 regulatory route rules.

> This specification is an evidence and deterministic accounting contract. It is not a port scheduler, grid study, OPS installation approval or statutory compliance determination.

## 1. Paradigm shift: the ship's energy system extends across the quay

Energy system design commonly ends at the ship's shore-connection plug or at the ship's electrical bus. This misses the actual operational service: a vessel can only use shore electricity when a compatible berth, safe connection, reserved port capacity, grid energy and sufficient time are simultaneously available.

The unit of design should therefore include:
- vessel electrical architecture and shore interface;
- port/berth connection equipment and technical compatibility;
- berth and turnaround window;
- reserved grid power and energy availability;
- shore electricity tariff, carbon intensity and temporal profile;
- alternative onboard generation and battery constraints;
- cargo work, departure schedule and fleet rotation.

A ship with a larger battery is not automatically more flexible. The port may be the limiting resource. Conversely, adding shore connection capacity may reduce auxiliary generator runtime without increasing onboard battery mass. Evaluate both sides under the same service contract.

## 2. Industry and regulatory baseline

### 2.1 Published optimization research

Wang et al., *An integrated robust framework for berth, yard, and shore power planning considering vessel arrival uncertainty*, Ocean Engineering 358 (15 June 2026), article 125850, DOI 10.1016/j.oceaneng.2026.125850, studies robust coordination of berth, yard and shore-power planning under vessel-arrival uncertainty and reports 4–13% cost savings in its studied cases.

This is a methodological SOTA comparator, not a transferable savings forecast. Reproduce its published assumptions and baseline before comparing; the present work addresses a more bounded auditable ship–port connection and energy-delivery contract, not a replacement for its multi-agent optimization.

### 2.2 Shore-side electricity engineering

EMSA's *Guidance on Shore-Side Electricity: Guidance to Port Authorities and Administrations* discusses infrastructure planning, responsibility frameworks, construction and operating controls. It identifies technical/safety challenges across OPS, battery charging, battery swapping, power banking and microgeneration.

Our contract must preserve actual connector identity, voltage/frequency, ratings, protection boundary, reserved capacity, validity and responsible-party evidence. A text flag such as "OPS available" is not sufficient evidence of a compatible energized berth.

### 2.3 FuelEU Article 6 scope must not be over-generalized

The European Commission's FuelEU Q&A states that the zero-emission-at-berth obligation applies to passenger ships and ships carrying exclusively containers within the Regulation's defined scope. It identifies 1 January 2030 for applicable AFIR-covered ports and 1 January 2035 for other ports equipped with OPS, with additional Member-State options in the interim.

Do not silently apply this obligation to general cargo or mixed-use vessels. Rule applicability is owned by #11469 and must be provided as a resolved, versioned input; this audit is not a legal classifier.

## 3. Boundary with existing Luminous work

| Existing owner | Input / decision consumed here | What this project must not duplicate |
|---|---|---|
| #11558 Service Flexibility | Exact port-call, arrival/departure and hard-window contract | Route, fleet rotation or schedule optimizer |
| #11507 Energy Source Ledger | Integer source-input and conversion-flow accounting | A competing energy ledger or flat summation of all meters |
| #11501 WAPS MRV | WAPS evidence/source identity and substitute-energy distinction | A statutory wind-energy method |
| #11028 Shipboard Transients | Ship-side electrical ratings, start/step limits, connection/reconnection test envelope | Electrical protection or transient solver |
| #11469 Regulatory Route | Port/vessel legal status and applicable OPS rules | Legal interpretation or statutory verifier |
| This project | Ship/berth compatibility, usable connection time, power/energy cap and bounded OPS delivery | Grid planning/berth allocation optimization |

## 4. V0.1 deterministic ship–port energy contract

Each scenario binds:
- scenario ID/revision and service fingerprint;
- port, terminal, berth, vessel and shore connection identity;
- berth start/end, planned vessel arrival/departure, connection time and disconnection time;
- ship/port OPS compatibility disposition: compatible, incompatible or unknown;
- operational availability: available, unavailable or unknown;
- connector standard/identity, voltage, frequency and phase where relevant;
- ship maximum input power, berth connection limit, reserved power, port energy availability and requested energy;
- grid and displaced onboard-generator GHG factors, revision, source ID, accounting boundary ID and method limitations.

For V0.1 use integer quantities only:
- power: watts (W);
- time: seconds (s);
- energy: watt-hours (Wh);
- emissions factors: micro-kilograms CO2e per kWh;
- emissions: micro-kilograms CO2e.

No floating-point accumulation. Round delivered Wh down, never up. Preserve unserved requested energy.

### 4.1 Join to the #11558 service call

The port case embeds the exact parent service-call identity and timing contract. The port-call ID must match the root call ID, the service fingerprint must be the same value as the parent service, and the copied service-call timing must resolve to an evidence-source record. The planned arrival must be within the contractual earliest/latest window; departure must satisfy minimum turnaround and latest departure. This is a consistency check, not a signature or independent re-computation of the parent's service fingerprint.

### 4.2 Usable connection time

`usable_seconds = max(0, planned_departure - planned_arrival - connect_seconds - disconnect_seconds)`

The arrival and departure must lie inside the declared berth interval. If another schedule owner supplies a stricter port-call/turnaround window, use that resolved window; this audit does not relax it.

### 4.3 Power and energy ceiling

`connection_power_w = min(ship_max_input_power_w, berth_connection_limit_w, port_reserved_power_w)`

`power_time_energy_wh = floor(connection_power_w × usable_seconds / 3600)`

`delivered_energy_wh = min(requested_energy_wh, port_energy_available_wh, power_time_energy_wh)`

`unserved_energy_wh = requested_energy_wh - delivered_energy_wh`

If compatibility or availability is unknown, disposition is INCOMPLETE and no eligibility claim is made. If known incompatible or unavailable, the case is BLOCKED and delivered energy is zero. A zero reservation is not interpreted as unknown.

### 4.4 GHG comparison

Only calculate the grid-versus-generator GHG delta when:
- both emission factors are present and positive/known as quantities;
- the factor source IDs resolve;
- both factors share the exact same declared accounting boundary ID;
- both factors use micro-kg CO2e/kWh delivered electrical energy at the declared ship/port boundary.

`grid_ghg_micro_kg = delivered_energy_wh × grid_factor_micro_kg_per_kwh / 1000`

`genset_ghg_avoided_micro_kg = delivered_energy_wh × genset_factor_micro_kg_per_kwh / 1000`

`net_delta_micro_kg = grid_ghg_micro_kg - genset_ghg_avoided_micro_kg`

Negative delta means lower modelled GHG under the declared common boundary; positive delta means higher. This is a scenario comparison, not metered avoided emissions or an official FuelEU/EU ETS figure. The factor method must include the stated upstream/downstream components. If boundaries differ, return INCOMPARABLE and preserve the two gross values separately.

## 5. Test matrix

Positive:
1. Compatible 400/50 Hz berth with reserved capacity and sufficient time.
2. Power-limited berth; energy capped by the smallest of ship/berth/reserved limits.
3. Port-energy-limited call; energy cap independent of peak power.
4. Connection/disconnection consumes real berth time.
5. Port call identity, service fingerprint, earliest/latest arrival, minimum turnaround and departure cutoff join exactly to the service flexibility contract.
5. Grid and generator factors share a boundary and yield exact integer GHG arithmetic.

Negative/adversarial:
1. Unknown or incompatible voltage/frequency/connector.
2. OPS unavailable, de-energized or out of maintenance.
3. No grid reservation or reservation revoked.
4. Vessel arrives after berth start/last compatible connection time or departs before minimum connection duration.
5. Requested energy exceeds the physically deliverable cap.
6. Grid and genset emissions factor boundary IDs differ.
7. Missing/unknown factor, stale factor revision or unresolved source.
8. Service fingerprint/port-call identity mismatch with #11558.
9. Connector rating or port reservation is changed after scenario fingerprinting.
10. FuelEU zero-emission-at-berth rule incorrectly assumed for general cargo or a non-eligible port.

## 6. Benchmarks and proposed targets

Targets are research goals:
- 100% known-answer integer arithmetic agreement.
- 0 false-eligible OPS calls in adversarial controls.
- 0 over-delivery beyond power, energy, time or berth limits.
- 100% provenance for connector, capacity, energy and emissions-factor inputs.
- 0 GHG delta calculations with mismatched boundaries or missing sources.
- 100% join coverage to the exact #11558 port-call identity and service fingerprint.
- Report served/unserved energy, useable connection time, average/peak power, reserved-power utilization, OPS availability, grid-versus-generator GHG and tariff cost separately.

The 2026 Ocean Engineering paper's 4–13% case savings are a reproduction target only. Luminous must not claim equivalent savings until an independent matched case is completed.

## 7. New research questions beyond current SOTA

1. **Port capacity as a design variable:** does a vessel architecture optimized for the expected berth-energy distribution require a different battery/generator design than one optimized for nameplate OPS at every port?
2. **Flexibility value:** can time-window slack from #11558 reduce peak port reservation or battery capacity without violating cargo/arrival service?
3. **Grid carbon and marginal energy:** how does time-varying marginal electricity intensity alter the best berth charging window? Average annual grid factor is insufficient for dispatch unless explicitly chosen.
4. **Resilient connection:** what is the service penalty when OPS is unavailable, derated, disconnected or incompatible at one port in a multi-port rotation?
5. **Interoperability:** can a versioned interface contract support multiple connector standards and voltage/frequency systems without pretending compatibility from labels alone?
6. **Regulatory specificity:** how do FuelEU Article 6 and AFIR scope differ by vessel class and port status? Treat general cargo separately from ships carrying exclusively containers.

## 8. Execution sequence

1. Commit v0.1 JSON Schema and synthetic fixture.
2. Implement the standalone Rust audit with integer arithmetic and source/boundary resolution.
3. Bind a fixture to #11558's exact service/port-call fingerprint without recomputing schedule decisions.
4. Add adversarial cases for insufficient connection time, connector mismatch, no reservation and inconsistent GHG boundaries.
5. Reproduce the Wang et al. published benchmark from its stated inputs if data/access permit; otherwise report a published-number reconstruction only.
6. Obtain port authority/terminal and vessel electrical integrator input before any physical connection test.

## 9. Sources

- Wang et al. (15 June 2026), Ocean Engineering 358, 125850, DOI 10.1016/j.oceaneng.2026.125850: https://www.sciencedirect.com/science/article/abs/pii/S0029801826016847
- EMSA, *Guidance on Shore-Side Electricity: Guidance to Port Authorities and Administrations*, updated 11 November 2025: https://www.emsa.europa.eu/electrification/sse.html
- European Commission, FuelEU Maritime Regulation Q&A, Article 6 zero-emission-at-berth scope and dates: https://transport.ec.europa.eu/transport-modes/maritime/decarbonising-maritime-transport-fueleu-maritime/questions-and-answers-regulation-eu-20231805-use-renewable-and-low-carbon-fuels-maritime-transport_en
- Regulation (EU) 2023/1804 (AFIR): https://eur-lex.europa.eu/eli/reg/2023/1804/oj

## 10. Claim ceiling

A schema/audit pass verifies only the declared scenario contract and deterministic caps. It does not prove a real berth is energized, a ship's connector is compatible, grid capacity is reserved, electricity factors are accurate, a regulation applies or a physical shore connection is safe. All physical connections require port/ship electrical engineering review, approved procedures and responsible-authority permissions.
