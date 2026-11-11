# Electric Rotorcraft Drive-Chain Reference Oracle

A dependency-free Rust analytical reference for one positive-power drive path:

rotor torque and RPM -> transmission -> motor -> inverter -> DC bus.

The rotor-side mechanical power is torque multiplied by angular velocity. The oracle then applies explicitly supplied transmission, motor and inverter efficiencies to compute the required power at each boundary and the resulting DC-bus current at an explicit operating voltage.

## Important boundaries

- No default or guessed efficiency values are provided.
- Direct drive must explicitly use transmission efficiency = 1.0.
- Each component conversion loss is reported separately.
- Loss energy is not assigned to a battery or thermal node by this crate; the physical thermal architecture must own that allocation.
- DC bus voltage is an explicit operating-point input. Battery OCV, SOC, SOH, internal resistance, voltage sag and battery current limits remain separate battery-model responsibilities.
- This v0.2 crate is motoring-only. Regeneration and reverse power flow are outside the model.
- The operating-point efficiencies must already be valid for the torque, RPM, temperature and configuration being evaluated. This crate does not create, interpolate or extrapolate motor/inverter maps.

This is not a motor electromagnetic model, inverter switching model, battery model, rotor aerodynamic model, aircraft solver, BMS or protection controller. Synthetic reference PASS is not a physical qualification or airworthiness result.

## Exact-path verification

cargo fmt --manifest-path tools/electric-rotorcraft-drive-oracle/Cargo.toml -- --check

cargo test --locked --manifest-path tools/electric-rotorcraft-drive-oracle/Cargo.toml

cargo clippy --locked --manifest-path tools/electric-rotorcraft-drive-oracle/Cargo.toml --all-targets -- -D warnings

All numerical cases are synthetic analytical fixtures. No physical motor, inverter, gearbox or rotor data is embedded.


## Gearbox kinematics (v0.2)

The v0.2 additive API keeps `evaluate_drive` and its rotor-side power contract intact. Use `evaluate_drive_with_kinematics` when the motor-side shaft operating point is needed.

The caller explicitly supplies `TransmissionKinematics { motor_to_rotor_speed_ratio }`, with the positive motoring convention:

`g = motor_speed / rotor_speed`

The function reports motor angular velocity, RPM and shaft output torque. Motor mechanical power remains the transmission-input power already computed by the drive-chain reference, while transmission efficiency remains a separate input in `DriveConfig`.

The speed ratio is not an efficiency. Neither value is inferred. This extension is still a motoring-only analytical reference and does not validate a motor torque-speed map, gearbox physical limits, regeneration, hardware behavior or airworthiness.

### Synthetic known-answer example

For 10 kW at the rotor shaft, 3,000 rotor RPM, a 5:1 motor-to-rotor speed ratio and 90% transmission efficiency:

- Rotor torque: approximately 31.831 N m.
- Motor speed: 15,000 RPM.
- Motor shaft mechanical power: approximately 11,111.111 W.
- Motor shaft output torque: approximately 7.074 N m.

With 95% motor efficiency, 98% inverter efficiency and a 400 V DC bus, the chain requires approximately 11,934.598 W and 29.836 A. These are synthetic analytical values, not measured hardware data.

Additional negative controls reject zero, negative, non-finite and numerically unrepresentable ratios.


## Map-qualified efficiency envelope (v0.5)

The v0.5 additive API connects a caller-supplied, already-qualified motor efficiency interval to the independent drive-chain calculation without introducing a dependency on a map library.

1. `derive_motor_shaft_operating_point` computes motor torque and speed from rotor demand, the explicit gearbox speed ratio and transmission efficiency. It does not require motor or inverter efficiency, so it can be used before a motor-map query.
2. The caller constructs `MotorEfficiencyEvidence` from the map lookup output, including map ID, exact map-content digest, configuration digest, context digest, evidence-manifest digest, exact motor torque/RPM query and lower/nominal/upper efficiency bounds.
3. `evaluate_drive_with_map_evidence` verifies the expected configuration/context, requires the map query point to match the current derived motor shaft point exactly, and independently evaluates all three efficiency values.

The result contains:
- **Adverse efficiency scenario:** lower motor efficiency, therefore highest DC-bus demand.
- **Nominal efficiency scenario:** nominal map value.
- **Favourable efficiency scenario:** upper motor efficiency, therefore lowest DC-bus demand.

Each scenario retains its own `Result<DriveReport, DriveError>`. A nominal PASS cannot hide an adverse-case bus power/current failure. The result carries map ID, map-content digest, map model version, cell indices, interpolation weights, configuration, context and evidence-manifest identities unchanged. The map-content digest is distinct from the evidence-manifest digest: one identifies the exact efficiency surface, the other identifies its underlying observations, calibration and transformations.

The efficiency interval is only as meaningful as the upstream map/evidence process. The current map oracle's weighted absolute bounds are not Gaussian confidence intervals. No joint uncertainty or correlation is inferred for transmission, inverter, battery, thermal system or rotor aerodynamics. These must be evaluated by their respective owners.

All fixtures remain synthetic analytical cases; no physical motor efficiency map is embedded. The envelope does not establish a continuous rating, motor limit, safe operating point, aircraft mission capability or airworthiness.


### Exact map-content identity (v0.5)

The drive consumer requires both `map_id` and `map_digest`, in addition to the source evidence-manifest digest. The map-content digest identifies the exact axes, nodes, uncertainty values and serialization used to create the efficiency surface. The evidence-manifest digest identifies the underlying measurements/calibration/transformations. Neither can substitute for the other. The caller must copy both from the exact map lookup result; this oracle checks non-empty identity and propagates it but does not recompute cryptographic hashes.


The consumer validates that all four interpolation weights are finite, each lies in [0, 1], and their sum is 1 within 1e-12. It preserves the selected torque/speed cell indices and weights in the resulting drive envelope. This checks the shape of the interpolation evidence but does not independently recompute the interpolation; the exact map-content digest and upstream map implementation version identify the artifact/algorithm that produced it.


## Mission-segment robust demand envelope (v0.6)

The mission composer evaluates a frozen, piecewise-constant propulsion demand profile for one drive path using the same gearbox-aware, map-evidence drive calculation as the pointwise API.

Each segment declares:
- unique segment ID;
- monotonic start/end timestamps;
- rotor shaft torque and speed;
- DC bus voltage;
- explicit motor-to-rotor speed ratio;
- exact operating-context digest;
- motor-map evidence at the derived motor shaft torque/RPM point.

The profile binds one exact aircraft configuration digest. The operating context may change by segment (for example, different measured temperature/cooling contexts), but each map lookup must match the segment's exact context and operating point.

### Temporal semantics

The operating point is treated as constant over its declared segment duration. There is no interpolation between segment endpoints. Segments must be supplied in chronological order and may not overlap. Gaps are preserved as **unmodelled intervals**, not assumed to have zero demand. The reported active propulsion duration is distinct from total elapsed profile duration.

For measured transient power traces, use the separate synchronized power-series oracle; this mission composer is a predicted segment-level calculation.

### Scenario outputs

For each segment, the composer evaluates adverse, nominal and favourable motor-efficiency cases. It retains the map/evidence envelope and each scenario's typed drive result. A failure in one segment appears with its segment ID and cause; it cannot be hidden by another segment's lower power.

Mission summaries are independently produced for all three scenarios only when every segment in that scenario evaluates successfully. Summaries report rotor-shaft energy, transmission/motor/inverter loss energy, DC-bus energy, average DC power, peak DC power and peak DC current.

A profile with time gaps is a partial propulsion profile. Its gap durations are reported explicitly and must be joined with other mission loads/operating policies before a complete aircraft mission claim.

### Claim boundary

All mission fixtures are synthetic. This is a deterministic composition of supplied map evidence and the independent drive-chain reference. It does not validate a real motor map, battery, aircraft, reserve policy, thermal state, rotor aerodynamics, safety or airworthiness.


## Simultaneous multi-propulsor DC-bus composition (v0.7)

The shared-bus reference composes multiple pointwise, gearbox-aware mission drive profiles already qualified by the v0.6 mission-segment API.

Each bus declares a source boundary, source-evidence digest, aircraft configuration and minimum-voltage / maximum-continuous-current / maximum-continuous-power limits. Each propulsion branch declares one drive path, one bus, a dedicated feeder, meter boundary, protection-device identity, common-cause group and branch continuous limits.

For each exact common mission interval, the composer requires:
- every connected propulsion path to provide the same segment ID and start/end times;
- every path on the ideal bus to declare the exact same DC voltage;
- every path profile to have no unmodelled intervals;
- every external TMS/auxiliary profile to use the same interval grid;
- unique branch, path, bus, feeder, load and meter-boundary identities.

The shared bus uses an ideal common-voltage assumption. Propulsion current is summed from the already evaluated drive results; external load current is calculated as P/V for that same declared voltage. Branch, bus and interval limits are checked independently. Results preserve per-branch map/configuration/context/evidence lineage and the protection/common-cause identifiers supplied by the topology.

### Scenario interpretation

Adverse demand combines the adverse drive-map result for every propulsion path with the upper bound of every external load. Nominal uses nominal values; favourable uses favourable drive-map results and lower external-load bounds.

These are rectangular conservative combinations when dependencies/covariances are unknown. They are not probability distributions or confidence levels. No correlation is inferred.

### v0.1 limitations

This is an accounting and static-limit reference, not an electrical network solver. It does not solve:
- cable resistance/inductance, voltage drop or contact resistance;
- battery source impedance, current limits or dynamic voltage sag;
- switching ripple, harmonics, arc faults or short-circuit current;
- contactor transients, protection selectivity, breaker/fuse coordination or bus-tie switching;
- dynamic load sharing, source dispatch or post-failure thrust/control allocation;
- N-1 capability or common-cause independence.

Every branch in a bus report is an independently declared path/feeder record, but common-cause groups are only preserved metadata. Multiple motors do not establish electrical or flight-control redundancy by themselves.

Measured raw channels and synchronized power traces remain owned by the observation/time-series oracles. This composer consumes predicted drive-path reports and separately evidenced TMS/auxiliary intervals; it does not infer zero demand for missing intervals or convert electrical loads into heat.
