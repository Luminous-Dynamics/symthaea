# Open mobility telemetry reference design — v0.1

**Status:** open specification and planned session manifest; no hardware prototype or physical capture has yet been built.  
**Tracker:** [TYRE-DATA-001 #8076](https://github.com/Luminous-Dynamics/symthaea/issues/8076)  
**Code:** `xtask mobility-telemetry-manifest` validates the run manifest; it is not yet firmware or a logger.

## Decision

**Build our own cross-vehicle measurement contract and acquisition pipeline; reuse open-source drivers and viewers where their licenses permit.** Do not wait for any one manufacturer's data-access request, and do not fork a restrictive license into a commercial-capable product.

Existing starting points, with boundaries:

| Project / standard | Useful role | Decision |
|---|---|---|
| [MotoNav](https://github.com/shbmx/motonav) | Low-cost ESP32-C3 IMU/GNSS motorcycle/bicycle logger and offline ride report; roughly 10 Hz in its README | Excellent architectural reference for mounting, raw logging and honest gap handling, but its PolyForm Noncommercial + Share-Alike license does not grant unrestricted commercial use. Do not copy code into a commercial path without separate permission. |
| [MotoMonkey](https://github.com/Cr33ch3r/Motorcycle-datalogger) | Earlier low-cost track motorcycle logger concept with IMU/GPS, ECU and generic I/O | GitHub reports MIT, but README still has TODO installation/usage/license sections and the design names obsolete Intel Edison hardware. Check the actual LICENSE file and lineage before reusing source; use as historical reference for now. |
| [SavvyCAN](https://github.com/collin80/SavvyCAN) | Mature open-source CAN capture/analysis and DBC inspection | MIT-licensed; optional host-side CAN inspection tool, not our cross-sensor data standard. |
| [Open Vehicle Monitoring System (OVMS)](https://github.com/openvehicles/Open-Vehicle-Monitoring-System-3) | MIT-licensed EV telemetry/device/server ecosystem | Reuse APIs or adapters only when they fit; it is primarily an EV monitoring system, not a lab measurement/qualification recorder. Third-party components keep their own licenses. |
| [Rust SocketCAN](https://docs.rs/socketcan/latest/socketcan/) | Linux CAN/CAN-FD receive adapter | Good candidate for the optional host/edge adapter; its API exposes frames and timestamps but leaves the data format to the caller. Pin a known dependency version in Cargo.lock. |
| [ASAM MDF 4.3](https://www.asam.net/standards/detail/mdf/) | Interoperable binary measurement format for ECU values, bus traffic and multi-rate data | Preserve this option for export. Start with transparent raw observations plus a JSON manifest so the prototype is inspectable with basic tools; add MDF export after the exact reader/writer library and MDF subset are tested. ASAM currently lists version 4.3.0 (released 2025-09-23). |

Relevant sources: MotoNav describes its low-cost offline logger and restrictive license; SavvyCAN declares MIT; OVMS declares MIT for project code with component-level exceptions; Rust SocketCAN supports Linux CAN/CAN-FD; ASAM lists MDF 4.3.0 as its current measurement format. See the links above and [Rust MDF reader/writer](https://github.com/sigmatactical-org/mdf4-rs). MDF4 output and our JSON telemetry manifest are complementary, not competing standards.

## System architecture

### Data path

```text
Passive sensors / isolated electrical probes / approved read-only bus tap
                         |
                         v
      logger clock + per-sensor timestamp capture
                         |
                         v
  raw append-only file + loss/overflow counters + health events
                         |
             power-loss-safe close / index
                         |
                         v
 JSON manifest + raw file digests + calibration / rights receipts
                         |
                         v
       Rust intake validator and reproducible report
                         |
                         v
 Symthaea road-load model / Symtropy scenarios / Mycelix asset history
```

The acquisition box must not control traction, brakes, battery management or safety-critical vehicle functions. Start with **listen-only / no-transmit** bus operation and external sensors. Any bus decoding follows vehicle-owner permission and an approved interface; unknown IDs are raw frames, not inferred signals. No active ECU interrogation or writes for v0.1.

### Hardware prototype: two distinct instruments

**A. Ride logger (initial low-cost reference device)**

- MCU / timing: ESP32-class MCU or equivalent with a monotonic hardware timer; the board is a prototype choice, not a qualification decision.
- Storage: microSD with power-loss-aware append records, rotating files and explicit buffer-overflow counters.
- Position/time: GNSS module with PPS output where available; log fix validity, reported uncertainty, UTC anchor and clock metadata.
- Attitude/motion: a 6-axis IMU sampled at a target 200 Hz for inertial channels; log sensor range, output data rate/filter settings, axes and mounting transform.
- Ground/wheel motion: independent front and rear wheel-speed sensors at a target 100 Hz; GNSS speed remains a separate independent observation.
- Electric power: externally calibrated, appropriately rated voltage measurement and isolated current transducer or calibrated shunt; target 100 Hz for the first reference tests. Never attach low-voltage MCU inputs directly to a traction pack. Isolation, fusing, creepage/clearance and enclosures require qualified electrical review.
- Temperatures and tire health: battery/motor/inverter/ambient temperature as appropriate; tire pressure and temperature are low-rate health channels (target 1–10 Hz), not tire-force measurements.
- Optional passive CAN: suitably rated interface, hardware listen-only/receive-only configuration, bus voltage protection and separate isolation review. Keep original frames, hardware timestamps and bus state/errors. Do not assume every motorcycle exposes CAN; other diagnostic links or proprietary buses may exist.

**B. Bench characterization instruments (separate session class)**

A ride logger does not produce motor efficiency maps or tire-force curves by itself. Those need additional calibrated equipment: motor dynamometer torque and speed, DC input voltage/current, temperatures and torque-map protocol; or a tire test rig with normal load, longitudinal/lateral force, speed, slip ratio, slip angle, camber, inflation and temperature. Fixture design, guarding and emergency stops require specialist review.

## Clock and loss handling

Use a logger-wide monotonic nanosecond clock for every sample and record GNSS UTC/PPS anchors separately. Record measured clock skew and synchronization uncertainty rather than treating GNSS time as exact. Keep original sensor timestamps when available. Never silently interpolate through data loss; emit an explicit gap/overflow event with duration/count. Record raw values before filtering; derived/filter outputs get separate channel IDs and transform versions.

For a **physical claim**, v0.1 clock eligibility is fail-closed: `sync_method` must exactly match `gnss_pps_verified`, `hardware_timestamp_verified`, `ptp_verified` or `shared_clock_verified`; a `clock_sync_record` must be declared as a content-addressed artifact and pass normal artifact verification; and `max_alignment_uncertainty_ns` must be at most 2,000,000 ns. A word like `verified` in an arbitrary method label is not evidence. The 2 ms ceiling is the v0.1 proposal tied to the revision-A electrical capture target, not a universal timing tolerance; tighter claim-specific budgets may be required. For every physical claim, the manifest must also declare a claim-specific uncertainty budget with measurand/unit, standard and expanded uncertainty, coverage factor, an acceptance tolerance, an explicit evidence state, and a content-addressed uncertainty-budget record. The expanded uncertainty must be consistent with the stated coverage factor and must not exceed the pre-registered tolerance; `calibrated` plus a receipt ID alone is insufficient.

NIST’s [metrological-traceability policy](https://www.nist.gov/calibrations/traceability) is the rationale for the uncertainty gate: traceability concerns the measurement result and documented calibration chain, with each link contributing uncertainty; traceability alone does not establish fitness for purpose.

For v0.1, a single aggregate uncertainty number cannot cover an entire claim. Each required measurand needs its own unit-compatible uncertainty budget; the schema rejects unknown claim/measurand/unit combinations, and eligibility remains blocked if even one required budget is missing.

| Claim target | Required uncertainty measurands (v0.1) |
|---|---|
| `ride_energy` | DC bus voltage (V), DC bus current (A), net DC bus energy (J or Wh), ground speed (m/s) |
| `longitudinal_dynamics` | Ground speed (m/s), longitudinal acceleration (m/s²), front/rear wheel angular speed (rad/s), stopping distance (m) |
| `tire_health` | Front/rear tire pressure (Pa or kPa) and front/rear tire temperature (°C) |
| `motor_map` | Motor torque (N·m), motor angular speed (rad/s), DC voltage/current (V/A), motor/inverter temperature (°C), efficiency (unitless) |
| `tire_force_characterization` | Normal/longitudinal/lateral force (N), slip ratio (unitless), slip and camber angles (rad) |
| `ice_fuel_energy` | Fuel mass flow (kg/s), engine angular speed (rad/s), wheel torque (N·m), wheel angular speed (rad/s), fuel energy (J or Wh) |

These are completeness gates for evidence bookkeeping, not evidence that the stated quantities have been measured or that the proposed quantity set is sufficient for regulatory qualification.

## Open manifest contract

The planned example is [mobility_telemetry_manifest_v0_1.json](data/mobility_telemetry_manifest_v0_1.json). It declares vehicle identity, session state and evidence class; clock/synchronization plan; sensor IDs and calibration receipts; semantic channel IDs, SI units, rates, axes/sign conventions and sensor assignments; data rights; raw artifact hashes; and claim targets for ride energy, longitudinal dynamics, tire health, motor maps, tire-force rigs or combustion energy.

The Rust validator answers two separate questions: (1) is the manifest structurally valid? and (2) is a particular claim eligible for physical evaluation given its channels, calibration, rights, raw artifacts and closed session? A manifest can pass (1) and remain explicitly ineligible for (2); the planned example must do so.

### Record format

For the initial version:
- `manifest.json`: immutable session context and sensor/channel inventory;
- `raw.csv`: timestamp_ns, channel_id, raw_value, validity/status and optional raw frame fields, with stable units and no silent row deletion;
- `events.csv` or equivalent event records: start/stop, GNSS lock, calibration state, storage overflow, sensor reset, marks and manual notes;
- per-file byte size and SHA-256 in a final manifest/index;
- optional original bus capture saved unmodified, plus a separate decoded file with decoder revision and DBC/source provenance.

For CAN, record raw frame ID, extended/standard flag, DLC, byte payload, direction/listen-only mode, timestamp source and bus error state; decoding lives separately. A decoded value must cite a DBC or documented decoder revision and retain original raw frames. Do not make a vendor-specific CAN signal mandatory for generic capture.

Use a single SI unit table: m/s, m, rad/s, rad, N, N·m, V, A, W, J/Wh, kg, ns, °C and Pa/kPa. Include frame, axis and positive-direction conventions. The schema must not turn a missing signal into zero.

## Measurement programmes

### Electric motorcycle reference

1. Define pack-terminal input energy, auxiliary boundary and usable reserve before a test.
2. Capture voltage/current synchronously and integrate signed electrical energy with calibrated sensors; keep discharge and charge/regen separate.
3. Record GNSS speed and front/rear wheel speed, IMU, payload, tire setup, ambient conditions and route metadata.
4. Measure motor/inverter efficiency maps in a guarded dynamometer rig, including speed/load/temperature, positive torque and regeneration. Do not infer shaft torque from vehicle speed.
5. Run repeated coast-down tests and closed-course fixed cycles with calibration receipts, then reserve whole runs/day for validation.

### ICE motorcycle / vehicle reference

Use a lawful, approved chassis or engine test cell. Record fuel consumption with calibrated measurement, engine speed/load, wheel speed/torque, relevant temperatures and emissions if available. Keep fuel chemical energy, crank/shaft work, driveline loss and wheel work as separate boundaries. Do not claim a fuel-consumption map from RPM or throttle alone.

### Tire research reference

The road logger captures context (speed, wheel speed, pressure/temperature, IMU and braking signals where safely instrumented); it does not measure contact-patch forces. Use a tire test rig or qualified closed-course programme for rolling resistance, force/slip curves, wet braking, endurance, wear and impact. Results retain specimen, size, batch, rim, pressure, tread state, load and environment.

## Suggested gates

- **G0 schema:** valid manifest; unique IDs; finite positive rates; known SI units/clock domain; every channel-to-sensor reference resolves.
- **G1 raw data integrity:** every raw artifact is present, size/digest match, and its path cannot escape the session directory; gaps/overflows remain explicit.
- **G2 measurement quality:** calibrations/receipts, synchronized clocks, units, range checks, uncertainty and repeatability are documented for channels used by a claim.
- **G3 claim-specific coverage:** required channel set complete for that claim (a tire-health classifier does not qualify wet braking; wheel speed does not replace torque measurement).
- **G4 independent physical validation:** hold out complete runs/routes/test days/specimens; publish per-run residuals, uncertainty, every outlier, and source permission.
- **G5 safety/approval:** laboratory and jurisdiction-specific qualification is separate from software tests. A passing manifest is not road-use authorization.

Targets for the first software tranche: zero false accepts across missing-channel, uncalibrated-sensor, clock/rights, path-traversal and digest-mismatch controls; zero implicit gap filling; reproducible manifest/report digests. These test data integrity, not physical accuracy.

## Implementation order

1. Freeze and validate the JSON/Rust manifest contract (this change).
2. Write a Rust host recorder for replay/synthetic inputs and Linux SocketCAN receive-only frames; no MCU firmware until record semantics are stable.
3. Build the first MCU logger: GNSS + IMU + storage with timestamp and overflow observability.
4. Add isolated DC-bus instrumentation and wheel-speed sensors once calibrated electrical and mechanical fixtures exist.
5. Add manufacturer-authorized bus data as an optional adapter, not a dependency.
6. Instrument the dynamometer/tire-rig session class when hardware and lab access are available.
7. Add MDF 4.x export/interchange after an exact-head conformance suite exists.

### Current implementation

The first implementation is the **manifest, raw-trace validator and fault-matrix runner**, not pretend firmware. All three commands are wired into the existing `xtask` binary and introduce no new dependencies. The telemetry module has 34 authored unit tests covering schema/claim gates, trace failure modes, report-digest determinism, and all 28 frozen matrix cases; five further tests cover the vendored tire-monitoring intake. These tests have not been compiled or run in this environment.

From the workspace root:

```sh
cargo test -p xtask mobility_telemetry_manifest
cargo test -p xtask frozen_fault_matrix_cases_match_expected_outcomes
cargo run -p xtask -- mobility-telemetry-benchmark \
  --source-commit "$(git rev-parse HEAD)" \
  --out reports/mobility_telemetry_benchmark.json
cargo run -p xtask -- mobility-telemetry-manifest \
  --input docs/research/data/mobility_telemetry_manifest_v0_1.json \
  --out reports/mobility_telemetry_manifest.json

cargo run -p xtask -- mobility-telemetry-trace \
  --manifest docs/research/data/mobility_telemetry_manifest_v0_1.json \
  --input docs/research/data/mobility_telemetry_trace_v0_1.csv \
  --out reports/mobility_telemetry_trace.json
```

The planned manifest declares 12 sensors and 20 semantic channels. The five-row synthetic trace is deliberately small and includes one missing raw value explicitly marked `INVALID`; it is never silently replaced with zero. Source-side replay found no schema/channel/sequence/timestamp/value violations: four valid samples, one invalid-or-saturated sample, and one missing raw value. This replay is not execution of the Rust code.

The machine-readable fault matrix is `docs/research/data/mobility_telemetry_fault_matrix_v0_1.json` (23 cases). The runner applies each named mutation to the manifest/trace fixtures, checks observed vs expected outcomes and required diagnostics, hashes the matrix and manifest, writes a structured JSON report, and exits nonzero if any case mismatches. This runner is authored but not yet executed. The manifest is `session_state: planned` and `evidence_class: DesignOnly`; all three requested physical claim profiles remain ineligible. For `recorded`/`closed` sessions the Rust path verifies declared artifact sizes and SHA-256 and rejects paths outside the manifest directory. Cargo/rustc/rustfmt are unavailable in this environment, so compilation/tests remain unverified.

No hardware has been purchased or built in this increment, no physical measurements have been made, and this design is not a safety-qualified product.
