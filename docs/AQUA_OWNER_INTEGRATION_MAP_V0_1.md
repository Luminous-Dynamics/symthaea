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
