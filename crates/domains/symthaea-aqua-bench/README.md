# symthaea-aqua-bench

Product-level comparator-context adapter for Luminous Aqua.

## Scope

This crate does **not** implement a water ledger, energy ledger, hydraulic/thermal solver, water-quality assessment, metrology passport, safety interlock, hygiene assessor or comfort model.

It checks that a pair of Aqua benchmark runs use matching:
- task profile and duration;
- inlet-water profile;
- supply/delivery conditions;
- room conditions;
- measurement protocol;
- complete-service accounting boundary.

Each run must point to canonical owner evidence:
- Symtropy water-service T0 receipt;
- Mycelix metrology admission;
- Mycelix intended-use water-quality assessment;
- independent task-outcome assessment;
- Symtropy resource-closure receipt when energy is compared.

Device configuration and cleaning protocol are allowed to differ between comparators, but their exact digests must be recorded. That is essential: different filter/cleaning requirements are part of the product's real resource burden and must remain visible in the upstream receipts.

## Non-duplication

- Symtropy #4978 owns deterministic water-service T0 accounting/evaluation.
- Symtropy #5364 owns household/community water-service continuity.
- Symtropy #5425 owns water-specific lifecycle cost per verified service.
- Symtropy #2152 owns physical resource stock/flow/energy/residual closure.
- Mycelix #6025/#6029/#6034/#6048/#6054/#6057 own metrology, measurand, calibration, uncertainty, timebase and applicability.
- Mycelix #6100 owns raw-water observation versus use-specific quality assessment.
- Sol-Atlas #765/#766/#770/#772 own pool/water-feature thermal, evaporation and microclimate qualification.

The Aqua adapter only binds the product comparison to those owner outputs. It cannot validate or replace them.

## Run

```sh
cargo test -p symthaea-aqua-bench
```


## Aqua programme interfaces

Product and benchmark contexts are defined in:
- `docs/AQUA_SHOWER_SERVICE_CONTRACT_V0_1.md`
- `docs/AQUA_SHOWER_BENCHMARK_V0_1.md`
- `docs/AQUA_WASHSTATION_BENCHMARK_V0_1.md`
- `docs/AQUA_THERMAL_SUITE_BENCHMARK_V0_1.md`
- `docs/AQUA_DHW_CASCADE_BENCHMARK_V0_1.md`
- `docs/AQUA_REFERENCE_BATHROOM_V0_1.md`
- `docs/AQUA_REFERENCE_ROOM_ARCHETYPES_V0_1.md`
- `docs/AQUA_OWNER_INTEGRATION_MAP_V0_1.md`

The adapter remains deliberately small: it binds paired Aqua test runs to equivalent conditions and canonical owner evidence. New fixture classes and room modes must not add a second water, energy, hydraulic, water-quality, metrology, thermal, pool or safety authority.
