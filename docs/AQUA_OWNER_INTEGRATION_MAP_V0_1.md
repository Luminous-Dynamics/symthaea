# Luminous Aqua — Owner Integration Map v0.1

**Parent:** [AQUA-BATHING-SERVICE-001 #12660](https://github.com/Luminous-Dynamics/symthaea/issues/12660)  
**Status:** ownership and interface map; no new physical subsystem is qualified.

## Aqua product-level ownership

Aqua owns:
- fixture/room product requirements and task profiles;
- matched-comparator design;
- user-facing bathing-service outcome definitions;
- the comparison context that binds each run to canonical subsystem receipts;
- integrated product architecture and customer/service experience.

Aqua does not own canonical water accounting, cost accounting, physical water/heat flow, water quality, instrument calibration, or pool thermal claims.

## Existing authoritative owners

| Concern | Existing owner | Aqua consumes |
|---|---|---|
| Water-service arithmetic and synthetic T0 cases | Symtropy #4978 / PR #4962; WaterLoop #5335 | Exact water-service T0 receipt and case/evidence ceiling |
| Household and community water-service continuity | Symtropy #5364 / PR #5368 | Delivered/unmet service and outage/recovery results |
| Water-service lifecycle economics | Symtropy #5425 / PR #5429 | Cost per verified useful service with tariff and payer boundaries |
| Physical stock/flow/energy/residual closure | Symtropy #2152 | Exact conservation and resource-closure receipt |
| WaterLoop treatment / greywater / reusable-water design | Symtropy #5713/#5749 and #5335 | Separate source and use boundaries only; no reuse credit from a shower fixture claim |
| Measurement identity and capability | Mycelix #6025 | Device/channel admission evidence |
| Water, pressure, temperature, level and electrical measurands | Mycelix #6029 and #6034 | Exact channel definitions, unit and instrument applicability |
| Uncertainty, timebase and field condition | Mycelix #6048/#6054/#6057 | Measurement uncertainty and applicability receipts |
| Water observations vs use-specific quality assessment | Mycelix #6100 | Exact source/sample/use/time-scoped assessment; no generic safe-water boolean |
| Pool thermal balance, evaporation and microclimate | Sol-Atlas #765/#766/#770/#772 | Site-bounded pool/water-feature thermal and water budget results |
| Cross-service test cell | Symtropy #4531 | Unoccupied room-scale integration and disturbance traces |
| Whole-household outcome and affordability context | Sol-Atlas #694/#700/#816 | Resident-authorized service and place-cost outcomes |

## Adapter contract

The new Symthaea crate `symthaea-aqua-bench` performs only:
1. exact task, duration, inlet-water, delivery-condition, room-condition, measurement-protocol and accounting-boundary matching;
2. syntax/owner checks on the canonical evidence references;
3. enforcement that an energy comparison carries the Symtropy resource-closure receipt;
4. rejection of missing or context-mismatched runs.

It does not compute litres, kWh, cost, water quality, temperature, pressure, spray force, comfort or safety. The authoritative subsystem receipts carry those results.

## Open interfaces / future work

1. Confirm the exact receipt field paths and digest format from Symtropy #4978/#5364/#5425 and Mycelix #6034/#6100 before implementing cross-repo deserialization.
2. Add a result composer only after the owner schemas are pinned; it may join owner-native outputs but must not re-evaluate their arithmetic.
3. Add human-factors endpoint binding after the non-inferiority margin, assessor protocol and sampling design are preregistered.
4. Add pool/steam/sauna room compositions only through their existing physical and safety owners.
5. Keep the physical demonstrator unoccupied until qualified plumbing, electrical, water-quality, exposure and building reviews admit the exact setup.

## Nonclaims

This integration map and adapter do not establish matched physical experiments, product superiority, recirculation hygiene, accessible design certification, regulatory compliance, water savings, heat recovery or affordability.


## Wet-room and room-mode integration

- Symthaea [AQUA-WETROOM-ENVELOPE-001 #12702](https://github.com/Luminous-Dynamics/symthaea/issues/12702) owns the Aqua product-level zone/mode matrix, envelope/ventilation/pressure interfaces, commissioning requirements, accessible circulation and room heat-cascade boundary.
- Aqua shower aerosol exposure and wet-side component research consumes the independent 2025/2026 microbial/aerosol literature and competent laboratory assessments; Mycelix #6100 remains water assessment authority.
- ASHRAE natatorium guidance and Sol-Atlas #765/#766/#770/#772 remain the method and owner references for pool rooms. Aqua does not extend pool thermal/evaporation results to showers or steam rooms.
- Symtropy #4531 remains the owner of the unoccupied room-scale integrated test-cell architecture and cross-service event traces.
- ISO 21542:2021 is an accessibility/usability design reference; local building/accessibility professionals determine applicability and approval.

The room-mode contract records the required evidence and return-to-ready state for shower, bath, accessible bathing, sauna, steam, plunge/whirlpool, pool, cleaning and maintenance. It does not become a new scheduler or safety controller.


## Wet-path cleanability and exposure integration

- Symthaea [AQUA-WETPATH-HYGIENE-001 #12703](https://github.com/Luminous-Dynamics/symthaea/issues/12703) owns product-side wetted-path topology, retained-volume mapping, inspectability, drainability, cleaning-cycle design, maintenance burden and the sampling/exposure protocol proposal for independent specialist review.
- Mycelix #6100 remains authoritative for source/sample/method/intended-use/time-scoped water assessment. Mycelix metrology owners remain authoritative for instrument admission and uncertainty.
- Independent laboratory/exposure specialists decide biological endpoints, sampling media, detection limits, methods and acceptance criteria. Aqua must not convert internal process signals or visual clarity into a hygiene pass.
- Symtropy #2152/#4978 remain resource closure and deterministic water-service accounting owners; all cleaning/purge resource burden is included through owner-native receipts.



## Washstation benchmark protocol

- Symthaea [AQUA-WASHSTATION-BENCHMARK-001](https://github.com/Luminous-Dynamics/symthaea/blob/research/aqua-competitor-ledger-v0.1-20261117/docs/AQUA_WASHSTATION_BENCHMARK_V0_1.md) defines separate bath, bidet and basin task comparators and evidence boundaries.
- EPA WaterSense published private-lavatory reference is 1.5 gpm max at 60 psi / 414 kPa and 0.8 gpm min at 20 psi / 138 kPa. The draft 1.2 gpm v2.0 revision is not approved: EPA paused work in February 2025.
- ASME A112.4.2-2021 / CSA B45.16:21 supplies relevant bidet backflow, temperature, spray-pressure, lifecycle and self-clean test references; local applicability remains for qualified South African review.


## Whole-system DHW / heat-cascade integration

- Symthaea [AQUA-DHW-CASCADE-001 #12709](https://github.com/Luminous-Dynamics/symthaea/issues/12709) compares hot-water source, heat-pump/storage, shower drain-water recovery topology, mixer placement and standby/distribution effects against a matched bathing service.
- The benchmark consumes Symtropy's native thermofluid/heat-pump/resource-closure outputs and Mycelix metrology; it does not implement a parallel heat-pump or thermal solver.
- Key comparator insight: the 2026 DWHR/HP coupling paper reports study-specific global COP results up to 6.3 experimentally / 8.0 simulated, while some topologies incurred static heat loss and reduced HP COP; the unbalanced mixer-side configuration performed best in that study. Aqua must reproduce the result rather than assuming transferability.
- See `docs/AQUA_DHW_CASCADE_BENCHMARK_V0_1.md`.


## Adaptive experience and profile authority

- Symthaea [AQUA-ADAPTIVE-EXPERIENCE-001 #12712](https://github.com/Luminous-Dynamics/symthaea/issues/12712) owns task-aware user-experience requirements, explicit preference/profile semantics, guest/shared-user handling, recommendation evaluation and human-factors protocol.
- Symthaea may recommend or recall only device-qualified settings. Fixture-local deterministic constraints and independent physical safety interlocks remain authoritative; profile, model and network loss must fall back to manual operation.
- Mycelix consent/identity/authorization evidence is consumed where required; no identity/profile record becomes physical safety authority.
- KOHLER Anthem / Anthem+ / DTV already provide multi-outlet control, presets and app/voice integration. Aqua's research claim must therefore be about privacy, agency, task outcomes and evidence-qualified bounded adaptation, not "AI presets" alone.


## Bathing fountains / aquatic feature integration

- Aqua differentiates decorative non-contact fountains, interactive spray/bathing fountains, and immersive bathing/hydrotherapy pools.
- CDC MAHC 2024 defines interactive water-play venues as spray/jetted features that contact bathers whether single-pass or recirculated; this is a public aquatic venue model code, not automatically local law.
- CDC warns that decorative fountains entered by the public must be managed as a splash-pad venue or entry should be prevented.
- Sol-Atlas #765/#766/#770/#772 remains authority for water-feature/pool thermal, evaporation, regeneration and microclimate; Aqua owns user-contact fixture/task/access and room integration; Mycelix #6100 remains water-use assessment authority.
- The archetype and safety interface is documented in `docs/AQUA_REFERENCE_ROOM_ARCHETYPES_V0_1.md` and `AQUA-THERMAL-SUITE-001 #12694`.
