# ENV-MOBILITY-008: SPM-0 initial sourcing evidence snapshot

**Evidence date:** 2026-10-17.  
**Status:** Catalog listing snapshot for a low-voltage bench article; not a purchase order, supplier quote, confirmed stock reservation, or qualified instrument BOM.  
**Parent work:** [ENV-MOBILITY PR #7870](https://github.com/Luminous-Dynamics/symthaea/pull/7870) and [real BOM / quote follow-up #7901](https://github.com/Luminous-Dynamics/symthaea/issues/7901).

## 1. Candidate first article

SPM-0 is limited to a dry-side low-voltage sensor/logger module to exercise serviceable electronics, harness design, configuration identity, observation metadata and repair/test records. The first demonstrator can record temperature, relative humidity and barometric pressure from a BME280 breakout using an ESP32-C3 development board.

This is a minimum *bench and workflow demonstrator*, not an accredited weather instrument, flight-qualified payload, marine system, waterproof enclosure, altitude reference or underwater sensor. Its measurements must be compared with an identified reference; catalog specifications alone are not achieved system accuracy.

## 2. Snapshot of publicly displayed listings

The following are public catalog listings discovered on 2026-10-17. Inventory/price can change; the pages are not formal quotations and do not prove Johannesburg delivery time, shipping price, bulk break, return conditions or procurement terms.

| Item | Listed SKU / variant | Displayed price incl. VAT | Availability shown by page | Fit / caveat |
|---|---|---:|---|---|
| BME280 sensor breakout | Make Electronics Type 1, SKU 0ME1655 | R159 | 14 in stock (Durban stock shown) | Module contains Bosch BME280; 3.3 V interface, I²C/SPI and a six-way header supplied. Header/harness assembly still required. |
| ESP32-C3 controller board | Make Electronics ESP32-C3 SuperMini 4 MB – Soldered, SKU 3ME3079 | R89 | 16 in stock (Durban stock shown) | ESP32-C3, USB-C and 3.3 V system; confirm exact header and board revision at order time. |
| Alternative controller board | Make Electronics ESP32-C3 Dev Board – CH340, SKU 3ME3029 | R115 | 10 in stock (Durban stock shown) | Larger 46 × 29.5 mm class board with Micro-USB, I²C and through-hole-friendly 2.54 mm pin pitch; costs R26 more than the SuperMini listing. |
| Alternative sensor source | DigiKey South Africa, bare Bosch BME280 IC | R64.55 each at quantity 1 (listing) | 0 in stock; lead-time notice shown | Not equivalent to the R159 assembled breakout: requires PCB, assembly, passives and manufacturing/test capability. Listed as a future sourcing option, not a cheaper drop-in. |
| Lower-measurand alternative | Communica, GY-BMP280-3.3 pressure module | R13 incl. VAT | In stock shown; 2–5 working-day South Africa delivery stated | BMP280 profile is temperature + pressure only; it does not supply the BME280's humidity measurement. Component authenticity/revision and actual errors must be checked. Not an equivalent substitute if relative humidity is required. |

### Source links
- [Make Electronics — BME280 3-in-1 Sensor Module Type 1, SKU 0ME1655](https://make.net.za/product/bme280-3-in-1-sensor-module-type1/)
- [Make Electronics — ESP32-C3 SuperMini 4 MB – Soldered, SKU 3ME3079](https://make.net.za/product/3me3079/)
- [Make Electronics — ESP32-C3 Dev Board – CH340, SKU 3ME3029](https://make.net.za/product/esp32-c3-dev-board-ch340/)
- [DigiKey South Africa — Bosch BME280 bare component](https://www.digikey.co.za/en/products/detail/bosch-sensortec/BME280/828-1063-1-ND/6136314)
- [Communica — GY-BMP280-3.3 pressure module](https://www.communica.co.za/products/hkd-baromtrc-sensr-bmp280-3-3)
- [Make Electronics — USB-C data/power cable, 25 cm](https://make.net.za/product/4me2515/)

The Make Electronics pages show the listed prices as VAT-inclusive and inventory at its Durban location. The snapshot treats those as catalog claims only; the stock status was not verified through checkout or a written supplier quote. The DigiKey bare part listing is explicitly not a comparable completed sensor module and showed no stock at the time of discovery.

## 3. Minimal priced electronics subtotals and mission-first variants

Two initial *candidate profiles* show why the mission's measurands must be frozen before fixing a BOM:

- **Pressure + temperature profile:** ESP32-C3 SuperMini board R89 + Communica GY-BMP280-3.3 listing R13 = **R102 displayed electronics subtotal including VAT**.
- **Temperature + relative humidity + pressure profile:** ESP32-C3 SuperMini board R89 + BME280 breakout R159 = **R248 displayed electronics subtotal including VAT**.

These are current catalog arithmetic snapshots only. The R13 pressure board's true component identity and metrology performance are not independently established; verify the delivered article before using it in a data-quality claim. It is *not* a substitute for the BME280 when humidity matters. The cheaper profile is only appropriate if the declared scientific question does not require humidity.

The ESP32-C3 SuperMini's listed R89 board and the sensor prices exclude shipping to the intended build site, headers/crimp terminals, keyed harness, enclosure, adapter, fixture/material, power source, bench time, failed/reworked parts, reference instrument, calibration service, taxes not already included, maintenance and any product qualification. A programming cable is a reusable bench tool, not automatically one cable per sensor unit. These amounts are not total unit costs or purchase recommendations.

For a serviceable article, compare the R89 SuperMini with the R115 larger controller board against the same exact mounting/harness/test requirement. The lower sticker price may lose if it causes more assembly time, fragile connections or slower repair. A connector-specific cable/adapter and a non-destructive removable mounting interface remain required design decisions, not free parts.

## 4. Calibration and data-quality boundary

The BME280 breakout seller lists datasheet-derived operating limits and nominal sensor tolerances (temperature, humidity and pressure) for its module. Those are not observed SPM-0 performance. Before any scientific or operational claim:

1. freeze the required measurands, units, range, update rate, environmental conditions and maximum acceptable uncertainty;
2. bind exact sensor/controller/module revisions and the sensor-to-environment mounting arrangement;
3. record raw readings and a reference instrument's identity, calibration status, range and uncertainty;
4. test multiple points and repeatability appropriate to the application, including stabilization and mounting/environment effects;
5. report bias, spread, drift and out-of-range behavior; do not apply a correction without preserving its evidence and version;
6. keep “device has a calibration certificate,” “device passed this check,” and “measurement is suitable for this mission” as separate claims.

The prototype need not immediately buy an accredited calibration service for every low-cost sensor board. It does need an honest, repeatable reference-check method. Whether a professional or accredited calibration is required depends on the eventual measurement requirement and downstream decision, and must be confirmed with the specific laboratory and instrument.

## 5. Local factory implications

At this stage, buy standard sensor/controller modules and locally fabricate or outsource only the non-structural mounting adapter and assembly/check fixtures. Choose the sensor profile by the mission's minimum required measurands; do not bundle humidity into every unit without a measurement requirement. Do not invest in a custom PCB, CNC machine or automated assembly cell for a single module. First compare:
- purchase of preassembled boards vs bare IC + a qualified PCB assembly route;
- supplier-made adapter vs an appropriate local print/CNC service;
- hand-built wiring vs a standardized, testable removable harness;
- the smallest common tool set that achieves repeatable assembly and inspection.

For the Johannesburg/Gauteng delivery context, obtain a dated delivered quote from at least one supplier serving Gauteng and a second-source quote for each component before converting this snapshot into the real-BOM dataset. The Durban catalog's posted stock does not establish Johannesburg lead time or delivered cost.

## 6. Required follow-up evidence for #7901

- exact, released SPM-0 requirement/profile and acceptance tolerance;
- frozen part numbers, datasheet revisions, approved substitutions and full harness/connector BOM;
- delivered quotations with currency, validity date, VAT/freight/duty treatment, stock/lead time and minimum order;
- confirmation of local stock by written quote rather than public web stock count;
- versioned CAD/drawing and print-service quote for the adapter, including material and tolerances;
- observed build times from at least five physical units, with setup/touch/queue/inspection/rework separated;
- a declared reference/check approach with reference instrument and uncertainty record;
- post-repair checks and configuration identity for sensor/controller replacement.

## 7. Machine-readable evidence register

The exact catalog snapshot and dependency states are also frozen in [env-mobility-spm0-source-register-v1.json](fixtures/env-mobility-spm0-source-register-v1.json). It contains eight records: five public catalog listings with source URLs and explicit VAT/availability statements, and three unresolved dependencies with null prices and no supplier identity. The two arithmetic-only sensor/controller profiles are R102 (pressure + temperature) and R248 (temperature + relative humidity + pressure), excluding all unresolved/uncosted dependencies.

A source-register audit is available in source via `cargo run -p xtask -- env-mobility-sourcing-audit` and its tests via `cargo test -p xtask env_mobility_sourcing`. It checks evidence classes, unique IDs, source URLs, null handling for unknown prices, and arithmetic/profile consistency. Its presence is not a passing test claim until exact-head Rust compilation and test execution have been observed.

## 7. Stop conditions

Do not promote this snapshot to “frozen BOM,” “local supplier qualified,” “calibrated instrument,” “local production capability,” or “low-cost field-ready payload” without the additional evidence above. These listings merely support a small and affordable-looking first *bench article* and help identify what must be quoted, assembled and measured next.
