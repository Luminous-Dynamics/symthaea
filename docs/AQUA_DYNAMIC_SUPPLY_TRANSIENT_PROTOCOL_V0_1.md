# Luminous Aqua — Dynamic Supply and Transient Service Protocol v0.1

**Status:** event-script and measurement preregistration; no dynamic supply test exists  
**Workstream:** [AQUA-DYNAMIC-SUPPLY-001 #12801](https://github.com/Luminous-Dynamics/symthaea/issues/12801)  
**Steady-state interface:** [AQUA-LOW-PRESSURE-001 #12784](https://github.com/Luminous-Dynamics/symthaea/issues/12784)  
**Product anchor:** AQS-1 #12717  
**Continuity semantics:** #12760

## 1. Research question

Does AQS-1 preserve qualified bathing service during realistic changes in supply pressure, hot/cold water availability, concurrent fixture demand and local utility state—or does a design that passes steady-state pressure points fail during the transient that occupants actually experience?

A steady flow point does not characterize a pressure step, start-up, second-fixture demand, regulator hysteresis, pump cycle or restoration event. Aqua therefore treats each event as a frozen, versioned experiment with time-synchronized observations.

## 2. Scope and safety boundary

This protocol authorizes event-script design, synthetic replay and unoccupied test-rig planning only.

It does not authorize:
- plumbing alteration or pressurized article fabrication;
- use of non-potable or unqualified stored water;
- booster pump or buffer tank procurement;
- occupied installation or human exposure;
- software control of valves or protective devices.

All physical work requires rated components, professionally reviewed apparatus, appropriate containment and independent protective systems. The test controller may inject a fault, but it does not replace the local device's qualified safety behavior.

## 3. Event-script contract

Each scenario is a versioned record containing:
- event ID and scenario family;
- precondition / initial stabilization state;
- exact pressure setpoint(s), step magnitude or ramp profile;
- event duration and hold period;
- second-fixture identity and demand profile, if present;
- hot/cold supply and inlet-water temperature profiles;
- room / ambient conditions;
- expected device state transitions from the qualified local design;
- test-article / apparatus rated envelope and source evidence;
- synchronized channel list, sample rate, uncertainty and timebase;
- abort conditions and the qualified local stop behavior;
- run order, randomization / blocking and recovery procedure.

Do not publish a generic pressure step or ramp rate as safe for all devices. The precise profile must stay inside the approved test fixture and component ratings. The pressure waveform, time window and acceptance bands are frozen before confirmation testing.

## 4. Pressure and thermal data channels

Required time-aligned channels:
- dynamic pressure at the terminal fitting inlet;
- upstream residual pressure at the declared service/manifold datum;
- flow at each active fixture branch;
- cold inlet, hot inlet and mixed fixture outlet temperatures;
- bather-plane surrogate temperature at fixed positions;
- local controller / protection state as reported by the qualified owner;
- fixture mode / manual input / user stop event;
- second-fixture activation state;
- fault injection, detection, local response and recovery timestamps;
- electricity / fuel / pump energy where the system uses active equipment;
- water admitted, discharged, purge and recovery resource volumes.

All channel identity, calibration, uncertainty, synchronization and applicability are Mycelix evidence. The actual pressure/thermal model and complete resource closure remain with Symtropy. Aqua consumes these records; it does not create another measurement or hydraulic authority.

## 5. Event families

### T0 — Cold start / hot-water warm-up

From a defined ready state, request the shower service and capture:
- time to first delivery;
- pressure/flow profile during startup;
- temperature ramp and overshoot/undershoot;
- purge/startup water and energy;
- transition into the stabilized service envelope;
- restart behavior after a user stop and re-request.

Do not claim that recirculation or a buffer is required to reduce warm-up losses. Compare direct-flow and any proposed alternative as separate arms with complete hygiene and resource review.

### T1 — Pressure step-down and step-up

While the shower is in stabilized operation, apply a frozen pressure step/ramp within the reviewed rig envelope. Repeat at each declared initial pressure. Record the full transient and the time spent outside the locally qualified operating envelope.

### T2 — Concurrent basin demand

Open and close a basin branch using a frozen demand profile while the shower is active. Record both branch flows, upstream residual pressure, showerhead-inlet dynamic pressure, mixed outlet and bather-plane temperature. Repeat for relevant orderings: basin-first, shower-first and simultaneous start.

### T3 — Bath-fill or additional fixture demand

Use a separate profile and fixture branch; do not scale the basin flow result or assume the bath is hydraulically equivalent. Record the same synchronized response channels and resource totals.

### T4 — Pressure source dynamics

Where a qualified test rig represents a pump, regulator, roof-tank or other source, capture cycling, regulator drift, drawdown or source transition using its exact approved configuration. Do not claim the synthetic source profile represents live Johannesburg utility behavior without a separately sourced and verified data set.

### T5 — Hot/cold supply disturbance

Inject an approved hot-side or cold-side interruption / variation profile within the test apparatus boundary. Record mixer response, local protective behavior and temperature excursion. Never ask an AI or planner to compensate outside independently qualified anti-scald and device protection limits.

### T6 — Restoration and recovery

After a disturbance, record:
- whether the device enters MANUAL_ONLY, ISOLATED_STOP or another exact qualified local state;
- whether recovery-pending remains active;
- inspection, flush, cleaning, metrology and water-quality reconciliation required;
- restart authorization source;
- time/resource burden to return to the qualified ready state.

Utility restoration or network reconnection alone is not a requalification event.

## 6. Analysis metrics — keep them separate

For each event and each exact product/specimen, report:
- peak and minimum dynamic pressure;
- peak and minimum flow;
- pressure and flow overshoot / undershoot relative to the frozen event profile;
- cumulative time and longest continuous interval outside the admitted envelope;
- time to settle within the preregistered band;
- mixed outlet and bather-plane temperature extrema;
- time outside the independent terminal-delivery envelope;
- unserved / interrupted service fraction;
- number of protective stops and manual interventions;
- water and energy consumed during startup, transient, purge and recovery;
- event-to-safe-state latency where the local qualified design requires a stop;
- time/resource burden for requalification and return to ready;
- missing, stale, contradictory, saturated or unsynchronized observations.

Do not collapse pressure stability, temperature stability, task completion and safety into one weighted resilience score. A protective stop that correctly prevents unsafe operation is not a safety failure; lost service is reported separately.

## 7. Matched comparator and statistical design

Compare an exact conventional installation and AQS candidate under the same event script, fixture task, inlet water, room and timebase. Freeze:
- exact model / market variant / mode and hardware configuration;
- installation and pipe/service geometry;
- event script and hash;
- test protocol and analysis revision;
- instrument identity / calibration / uncertainty;
- pressure-invariant context digest.

Randomize event order within practical blocks. Include repeated runs, independent specimens and a second day / water batch where feasible. A repeated time-series sample is not a new independent specimen. Preserve every invalid / aborted event and its reason.

The pressure-invariant context digest binds task, mode, inlet-water and temperature profile, room, mounting, specimen state and measurement setup; the run-specific context digest binds the pressure event. This is compatible with the paired exact-point flow-retention comparison in pressure_curve, but transient traces remain a separate data product.

## 8. Decision rules

- All pressure/time/temperature tolerance bands must be selected from the exact product and professional design envelope before confirmation testing.
- A design that exceeds its declared pressure/temperature envelope does not pass by having a good average value.
- Any false continuation after a local stop condition is a critical failure in the defined test corpus.
- Any automatic restart before required requalification is a failure.
- If the test script exceeds a rated apparatus boundary, the test is invalid and must not be run.
- Do not extrapolate across untested event magnitudes, durations, ramps or fixture combinations.

## 9. Claim ceiling

This is a preregistration and data-contract document. It does not establish transient-pressure resilience, safe temperature response, user comfort, building compliance, product superiority or live utility characteristics. It authorizes no physical construction, plumbing alteration, human exposure, occupied testing or procurement.
