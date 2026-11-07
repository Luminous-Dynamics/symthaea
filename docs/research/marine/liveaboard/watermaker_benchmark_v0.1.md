# Onboard seawater reverse-osmosis watermaker benchmark — V0.1

**Status:** source-pinned screening protocol; not an independently tested ranking or potable-water approval.  
**Parent:** [MARINE-WATERMAKER-BENCH-001 #10994](https://github.com/Luminous-Dynamics/symthaea/issues/10994)

## 1. Why this subsystem first

The liveaboard needs qualified usable water, not a nominal watermaker L/h number. Water production couples to electrical power, intake condition, membrane condition, filter supply, storage hygiene, brine handling and the crew's maintenance capacity.

This is a bounded physical subsystem with measurable outcomes. It can be tested on a bench before a full vessel exists and then integrated into the vessel's energy and resource budgets.

## 2. Manufacturer-published discovery baselines

The companion CSV contains the machine-readable source register. The present numbers are manufacturer-published product data, not independent measurements.

| System | Rated output | Electrical input | Derived nameplate specific input |
|---|---:|---:|---:|
| Schenker Zen 30 | 30 L/h | 110 W | ~3.67 kWh/m³ |
| Spectra LB 800 | 128 L/h | 300–600 W | ~2.34–4.69 kWh/m³ |
| Rainman AC-140L | 100–154 L/h | 1,250 W | ~8.12–12.50 kWh/m³ |

Calculation: electrical power in kW divided by rated production in m³/h. This yields a nameplate-derived energy intensity only if the quoted power and production apply simultaneously and to the same measurement boundary. That must be tested.

### Evidence and comparability limitations

- **Schenker Zen 30:** official product page reports 30 L/h, 110 W, 12/24 V DC and an energy-recovery system. The public page does not expose all feed conditions and test method in the extracted product summary.
- **Spectra LB 800:** manufacturer fact sheet reports 128 L/h, approximately 300–600 W depending on feed-pump selection, with baseline feed water 33,000 ppm at 25 °C. It does not establish identical measurement boundaries to the other products.
- **Rainman AC-140L:** manufacturer page reports 100–154 L/h at 1,250 W, 115/230 V AC, with a 2 kVA inverter/generator starting/running concept. Actual output depends on configuration and operating conditions.

These nameplate ratios are not sufficient to rank performance. We still need matched feed salinity/temperature/pressure, recovery, product-water acceptance, full auxiliary/flush energy and net output over the duty cycle.

Sources:
- Schenker Zen range: https://schenkerwatermakers.com/products/zen/
- Spectra LB 800 manufacturer fact sheet: https://www.spectrawatermakers.com/downloads/all-downloads/fact-sheets/lb-800-fact-sheet.pdf
- Rainman modular watermaker: https://www.rainmandesal.com/modular-watermaker/

## 3. Membrane-element standard reference: ISO 25175:2026

ISO 25175:2026, Edition 1 (published 25 June 2026), defines marine RO/NF membrane-element test methods for air tightness and separation performance, including permeation and salt rejection. Apply it to qualified **membrane-element** characterization where the exact licensed method and applicability fit.

Important boundary: ISO 25175 is not a complete installed watermaker system test. It does not, by its published scope, qualify vessel intake/pre-treatment, electrical energy for the complete service, flushing/preservation, intermittency, product storage/distribution, potable-water safety or brine discharge. Keep those as separate installed-system/service tests in this protocol.

Official ISO record: https://www.iso.org/standard/89252.html

## 4. Standardized test article and system boundary

Treat the watermaker as an installed subsystem:

`seawater intake -> strainer/pre-filter -> feed pump -> high-pressure pump -> membrane -> quality check/divert -> product tank`

Record concentrate/brine, flushing, preservation, cleaning, standby, restart and failures. Include every powered device that is part of the installed watermaking service.

At minimum, instrument:
- feed, accepted product and concentrate flow;
- feed/product/concentrate salinity or conductivity and water temperature;
- feed and membrane pressure, plus pressure differential where available;
- DC or AC voltage, current, real power and integrated energy at a declared boundary;
- filter differential pressure, membrane operating hours and cleaning history;
- valve/process state, pump state, flush / diversion / cleaning events;
- device orientation, vessel motion / heel / trim when relevant;
- ambient temperature and installation environment.

Use calibrated meters and independently reviewed water-quality sampling. TDS or conductivity alone is not a potable-water determination.

## 4. Primary outputs

1. Net accepted product volume after all start-up, flush and off-spec water is diverted.
2. Specific electrical input in kWh/m³ of accepted product at a named meter boundary.
3. Feed/product/concentrate mass or volume balance and recovery ratio.
4. Product-water quality under the declared intended-use criteria, assessed by the appropriate competent laboratory / authority.
5. Concentrate volume and composition, including the environmental handling path.
6. Availability and continuity under the declared daily operating policy.

Secondary outputs: peak power, startup energy, flush/preservation energy, filter/membrane life, consumables, operator labour, MTTR, noise/vibration, mass/volume, spare requirements and installed service access.

## 5. Fair comparison protocol

Freeze the operating conditions before opening results. The minimum first bench matrix should include:
- feed salinity at declared reference and boundary conditions;
- feed temperature at declared reference and boundary conditions;
- product setpoints and accepted quality criteria;
- electrical supply voltage and waveform/quality;
- pre-filter type, loading and replacement state;
- steady-state production and intermittent daily duty;
- startup, automatic/manual flush, shutdown preservation and restart;
- low-power / constrained-generation operation;
- end-of-service membrane and filter condition;
- sensor drift, low inlet pressure, power interruption and abnormal feed conditions.

Measure the full duty cycle. A 2-hour steady-state nameplate test must not hide 10 minutes of flushing, failed starts, maintenance downtime or off-spec output. Do not compare a 25 °C / 33,000 ppm tested system to another with unknown feed conditions as if they were equal.

## 6. Mission integration

The watermaker's measured qualified daily output and actual energy enter separate downstream owners:
- Symtropy Habitat Metabolism owns water stocks, qualities, process flows, residuals and coupled resource state.
- WATER-MFG-000 #5966 owns water/feedstream and treatment equipment engineering/maintainability.
- MARINE-AUTONOMY-VECTOR-001 #10956 consumes qualified stock and depletion intervals to evaluate target duration.
- MARINE-ENERGY-BUDGET-001 #8092 / MARINE-TRACE-001 #8102 consume measured watermaker electrical energy within the full vessel energy boundary.
- MARINE-LIVEABOARD-EVIDENCE-JOIN-001 #10988 ensures energy, water, habitat and other evidence all apply to the same exact vessel configuration and operating case.

This avoids creating a parallel water-treatment simulator, desalination physics engine, energy calculator or potable-water authority.

## 7. Benchmark acceptance principles

- Report median, p95 and worst-condition output and specific energy across independent units / test days.
- Hold out complete hardware units or independent test days; individual time samples from one test are not independent test samples.
- Count failed, off-spec, censored and maintenance runs; do not delete them.
- Separate product-water chemistry, microbiology, taste/odour, and legal potable-water determinations. The appropriate authority defines acceptable water quality for intended use.
- Keep feed/product/concentrate balances; unknown brine handling blocks environmental claims.
- Set a performance-improvement percentage only after a matched baseline and uncertainty budget exist.
- A superior kWh/m³ number cannot compensate for product quality failure, poor availability, shorter membrane life, excessive consumables or unsafe maintenance.

## 8. Claim ceiling

Manufacturer data establish only what a supplier publishes. The derived kWh/m³ figures in the CSV are arithmetic transformations of those published figures, not measurement. A bench test qualifies only the specific device/configuration, feed envelope, meter boundary and method tested. No watermaker result alone establishes potable-water safety, full vessel autonomy, legal brine disposal, whole-vessel seaworthiness or regulatory approval.


## 10. Synthetic mission-screen example (not a demand standard)

A companion JSON fixture uses a deliberately hypothetical case: four permanent occupants, 25 L/person/day, therefore 100 L/day of qualified product demand over a 30-day target. This is a design scenario assumption, not an asserted normal consumption rate or a potable-water requirement.

Using published nameplate ratings only, the arithmetic estimates:
- Schenker Zen 30: about 3.33 h/day and 0.37 kWh/day.
- Spectra LB 800: about 0.78 h/day and 0.23–0.47 kWh/day.
- Rainman AC-140L: about 0.65–1.00 h/day and 0.81–1.25 kWh/day.

These numbers deliberately exclude startup, flush, preservation, standby, post-treatment, product pump, off-spec diversion, environmental derating, maintenance, and conversion losses. They are not net usable water nor full-system energy. The fixture exists to provide a reproducible arithmetic bridge from vendor nameplate figures to a mission-service question; it does not compare or rank products.

Before this becomes a mission input, replace the scenario demand with an approved project requirement and replace all vendor nameplate inputs with matched, instrumented bench results and independent product-water quality evidence.
