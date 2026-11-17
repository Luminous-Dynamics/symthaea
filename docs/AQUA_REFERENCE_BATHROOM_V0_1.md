# Luminous Aqua — Reference Bathroom Architecture v0.1

**Status:** conceptual system architecture; not a construction drawing, P&ID, plumbing design or building approval  
**Parent:** [AQUA-BATHING-SERVICE-001 #12660](https://github.com/Luminous-Dynamics/symthaea/issues/12660)

## 1. Design intent

Design a modular, beautiful, accessible and maintainable bathroom / bathing room around the user's complete tasks and the physical flows of water, energy, heat, moisture, air, people and maintenance.

The reference layout is a design family, not one fixed floor plan. It can be composed as:
- compact home bathroom;
- accessible home bathroom;
- premium residential bathing suite;
- apartment / multifamily wet-room module;
- hospitality or community bathing suite;
- specialist rehabilitation / care environment (with separate clinical and accessibility review).

A single luxury layout is not the reference for all settings. The same service and safety contracts should scale down to affordable and maintainable versions.

## 2. Spatial zones

### Zone A — entry, changing and dry support
- Clear step-free entry where site constraints permit.
- Dry storage, towel and garment provision.
- Reachable local controls; high-contrast/manual controls.
- Safe lighting at entry and nighttime navigation.
- Space and turning/reach envelopes based on the declared user profile and applicable accessibility standard.
- Drainage and wet/dry separation designed to prevent water migration to adjacent occupied spaces.

### Zone B — shower and wash
- Direct-flow shower as initial qualified baseline.
- Fixed overhead and handheld shower options, each with independent shutoff where practicable.
- Thermostatic mixing plus independent mechanical/physical scald protection designed by qualified professionals.
- Reachable control and handheld support for seated/assisted use.
- Seat, grab-bar and transfer support positions derived from anthropometric/accessibility design, not decorative placement.
- Floor falls, drainage, slip resistance and cleaning access defined by professional design.
- Service panels for valves, strainers, meters and replaceable wet-side parts; avoid inaccessible concealed cavities.

### Zone C — bath / soak
- Optional compact, ergonomic, deep-soak or accessible bath modules.
- Actual fill volume, heat loss, time-to-ready, cleaning and entry/exit requirements measured.
- Handheld rinse and accessible fill/drain controls.
- No assumption that a deep bath is intrinsically more restorative or more efficient.
- If hydrotherapy jets are considered, treat as a separate recirculating aerosol / water-quality exposure system requiring dedicated design and hygiene qualification.

### Zone D — basin / grooming / bidet
- Basin geometry tested for splash, reach, task performance, cleanability and water flow.
- Replaceable aerator, trap, cartridge and service parts.
- Bidet uses its own qualified potable supply and backflow protection; it is not connected to greywater or pool lines.
- Controls and nozzles accessible for inspection, cleaning and replacement.
- Electrical outlets and appliances kept within professionally determined wet-location zones and protected by required devices.

### Zone E — technical service spine
- Accessible service cabinet or plant niche, sized from equipment and maintenance envelope rather than arbitrary aesthetic dimensions.
- Isolatable valves, labelled circuits, drain points, sampling ports and meters.
- Separate potable treatment / distribution; no common pressurized path with non-potable systems.
- Optional indirect shower-drain heat exchanger.
- WaterLoop greywater module is physically separate, with source-labeled shower/bath/basin capture, its own treatment and hold/release authority, and approved non-potable end use only.
- Within-session shower recirculation is a distinct local exposure path and may not connect to WaterLoop storage.
- Electrical isolation, earthing/bonding, enclosure, ventilation and spill protection require specialist design.

### Zone F — air, drying and moisture control
- Local exhaust and make-up air paths.
- Humidity / dew-point-aware operation and safe post-shower drying.
- Materials and junctions selected for moisture cycling, cleaning chemistry, corrosion and service access.
- Steam room is a separate sealed high-moisture zone; do not simply add a steam generator to the standard shower without envelope, drainage, control and condensation design.

## 3. Water circuits — must remain physically distinct

1. **POTABLE_SUPPLY:** qualified incoming supply to shower, bath, basin, bidet and handwashing fixtures.
2. **SHOWER_IN_SESSION_LOOP:** an optional within-one-user-session recirculation circuit; not admitted by default and not stored for subsequent users.
3. **GREYWATER_RAW:** source-separated shower/bath/approved basin effluent routed to WaterLoop's own intake/hold treatment boundary.
4. **GREYWATER_TREATED_NONPOTABLE:** separately qualified and labelled distribution for only the approved end-use class. Not drinking, handwashing, bathing or bidet supply.
5. **POOL_OR_SPA_WATER:** dedicated pool/spa recirculation and treatment; not connected to shower or greywater.
6. **WASTEWATER_AND_REJECT:** sanitary drain, cleaning purge, treatment reject and other waste streams with distinct accounting and lawful disposal.
7. **RAINWATER / OTHER NONPOTABLE:** only if separately captured, assessed and permitted for a named purpose.

No single software valve, user setting or control algorithm is a substitute for physical separation, required air gaps / backflow protection and qualified installation.

## 4. Heat and energy architecture

- Hot-water source may be heat pump, solar thermal, electric or another qualified system; compare local energy costs, source temperature, recovery time, storage loss and maintenance.
- Thermal storage and heater sizing must use actual demand profiles, not fixture count alone.
- Shower-drain heat recovery is indirect. Meter incoming/outgoing temperatures, flow, pressure loss, recovered useful heat and parasitic electricity.
- Heat recovery cannot create potable/non-potable mixing.
- Sauna, steam and pool heat loads are separate profiles; share heat only through qualified indirect interfaces with a full first-law balance.
- Do not treat a pool as an infinite heat sink. Consume Sol-Atlas pool thermal / evaporation / microclimate evidence rather than creating a new pool solver.
- Count ventilation and dehumidification energy when the room configuration changes those loads.

## 5. Materials, cleaning and repair

Every wet-side module must declare:
- exact wetted material and surface finish;
- compatibility evidence for temperature, pressure, hardness, disinfectants, descalers, cleaning chemicals and expected water chemistry;
- cleanout/drain path and inspectable surfaces;
- replaceable seals, strainers, cartridges, valves and sensors;
- service tool and access envelope;
- component mass, lifting/handling and safe isolation;
- failure modes, inspection interval, replacement procedure and end-of-life route.

Avoid inaccessible dead legs and hidden stagnant reservoirs. Prefer common, replaceable service components where compatibility and safety are proven. Do not make antimicrobial, food-contact, potable-contact or pathogen-control claims from material marketing language.

## 6. Accessibility and dignity requirements

- Provide a step-free/low-threshold configuration where feasible.
- Offer seated and standing shower paths.
- Make controls, showerheads, storage, towel and soap within declared reach envelopes.
- Design grab-bar anchorage and shower seating for structural loads and appropriate professional review.
- Provide clear space for assistants and transfer devices where the declared service profile requires them.
- Reduce glare, acoustic harshness, abrupt temperature changes and confusing controls.
- Permit mechanical shutoff and operation without a smartphone, internet connection, camera or voice assistant.
- Offer privacy-preserving assistance modes; do not infer disability or health status from observed behavior.
- Validate with varied bodies and task profiles before claiming accessibility.

## 7. Luminous system architecture and control authority

- **Symthaea:** local, privacy-preserving contextual assistant; read-only observation and bounded preference adaptation. It can recommend an already qualified shower/bath profile; it cannot override mechanical temperature limits, water-quality gates, disinfectant controls or physical isolations.
- **Symtropy:** supplies thermal/fluid/moisture/energy simulations through the existing specialist model owners; every result carries exact model version, boundary, assumptions and evidence class.
- **Mycelix:** admits measurement sources, calibration, uncertainty, sample records, standards/use-specific assessment and component/service provenance; it does not let a product controller self-certify its output.
- **Sol-Atlas:** site/climate/material/heritage/landscape and pool thermal integration; retains authority over its own environmental models and spatial evidence.
- **Luminous Platform:** local interface, versioned configuration, device interoperability, service records and evidence receipt composition; not a safety authority.

Core function must remain locally available during internet/cloud failure. A loss of AI, network, analytics or remote maintenance service must not defeat physical shutoff, safe temperature control, drainage or safe fallback.

## 8. Reference module boundary table

| Module | Inputs | Outputs | Primary owner / boundary |
|---|---|---|---|
| Shower fixture | Qualified potable supply, user profile, thermal supply | Bathing spray, drain effluent, measured service record | Aqua product contract; qualified plumbing/fixture engineering |
| Bath | Qualified potable supply, thermal supply | Soak service, drain effluent, measured fill/heat loss | Aqua product contract; separate fixture qualification |
| Basin / wash | Qualified potable supply | Wash service, basin effluent | Aqua product contract |
| Bidet | Qualified potable supply, power if fitted | Hygiene service, waste stream | Separate fixture and backflow qualification |
| Shower recirculation | Same-session captured shower water, independent treatment and measurement | Recirculated delivery or isolated reject | Unoccupied research only pending independent exposure qualification |
| WaterLoop | Source-separated eligible greywater | Named non-potable service or hold/reject | Symtropy WaterLoop; Mycelix quality authority |
| Heat recovery | Indirect drain heat + incoming supply | Useful preheat, measured parasitic burden | Symtropy thermal/energy owners |
| Ventilation / drying | Room air, outdoor/make-up air, electrical/thermal input | Moisture and contaminant control | Existing building/IAQ engineering owner |
| Sauna / steam | Thermal/electrical supply, water where applicable | Declared thermal environment, condensate | Separate sauna/steam engineering and safety profile |
| Pool / spa | Qualified makeup water, treatment, heat, electricity | Aquatic service, backwash, evaporation and heat flows | Sol-Atlas pool thermal owner + applicable aquatic health authority |

## 9. Design release gates

- G0: service profiles and accessibility intent frozen.
- G1: concept layout reviewed against site constraints and local professional requirements.
- G2: water, electrical, structural, drainage, ventilation and moisture interfaces reviewed.
- G3: hazard analysis and physical circuit separation reviewed.
- G4: unoccupied bench and measurement system qualified.
- G5: water-quality, recirculation exposure and sanitation claims independently qualified where applicable.
- G6: human-factors validation and local approvals complete before occupied pilot.

This document does not authorize a construction drawing, product manufacture, procurement, installation, occupied use or regulatory claim.
