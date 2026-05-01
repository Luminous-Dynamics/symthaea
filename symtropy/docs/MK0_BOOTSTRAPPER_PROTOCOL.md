# Mk0 Bootstrapper Protocol

The Mk0 bootstrapper is the minimum viable metabolism for starting Symthaea in the real world without waiting for the full city stack to exist. It assumes one room, off-the-shelf hardware, constrained power, and a small operator team.

This is not a separate permanent product line. It is the seed phase that should feed the real platform crates and operational patterns that appear later in the roadmap.

## Purpose

- close a minimal local material loop
- run local compute and coordination without permanent cloud dependency
- fabricate useful parts
- assemble repeatable subassemblies
- move materials between cells
- operate under explicit energy and safety constraints

## Room Assumptions

- one indoor room or warehouse bay
- stable floor space for print farm, bench arm, storage, and rover transit
- local microgrid or hybrid grid-tied power
- commodity cameras, sensors, and network switches
- at least one operator present for setup, maintenance, and emergency intervention

## Mk0 Stations

| Station | Physical hardware | Software role | Existing crate mapping |
|---|---|---|---|
| `mk0-seed-node` | workstation, rack server, SBC mesh | scenario orchestration, local DHT, telemetry, replay, policy distribution | `symtropy`, `symtropy-sim-bridge`, future `symthaea-archive` precursor |
| `mk0-helios` | solar, LiFePO4, inverter, smart metering | power scheduling, reserve awareness, energy logging | `symthaea-infrastructure`, future `symthaea-plexus` precursor |
| `mk0-detritivore` | shredder, extruder, feedstock bins | recycle plastic waste into printer feedstock | heavy-material profile of `symthaea-scavenger` |
| `mk0-fabricator` | FDM printer farm, light CNC | print parts, jigs, panels, brackets, fixtures | future `symthaea-fabricator` precursor |
| `mk0-manipulator` | 6-DOF bench arm / cobot | inserts, fastening, pick-place, tending | `symthaea-manipulator` |
| `mk0-vector` | low-speed rover | material movement between workcells | `symthaea-vehicle` logistics profile |

## First Demo Loop

The first honest Mk0 proof should be:

1. ingest failed prints or waste plastic
2. shred and extrude usable filament
3. print a known part set
4. move parts from printer to bench
5. assemble one repeatable subassembly
6. move finished subassembly to storage
7. recharge and log the full cycle

Good first subassemblies:
- rover bracket set
- sensor mast
- battery tray
- manipulator fixture
- cable guide / strain relief part

Avoid starting with a full robot body. The first loop should optimize for repeatability and instrumentation, not symbolic ambition.

## Control Architecture

### Local first

- orchestration should run locally on the seed node
- network dependency should be optional, not required for basic room operation
- all critical loops should degrade safely on comms loss

### Replayable by default

- every station action should emit timestamped events
- material movement, print start/stop, assembly success/failure, and power draw should be replayable
- success claims should be backed by scenario replays, not operator recollection

### Safety boundaries

- printer motion, heater control, shredder/extruder actuation, and rover motion all need explicit halt states
- a failed perception or policy signal should reduce authority, not increase it
- manual intervention must always outrank autonomy

## Mycelix Interaction Model

Mk0 is stronger if Mycelix is treated as the room's coordination substrate rather than a later optional integration.

### What Mycelix should do in Mk0

- provide identity for stations, operators, and work orders
- record material, energy, and assembly events as shareable operational facts
- carry local governance over scarce room resources
- expose a minimal economic/accounting surface for jobs, stock, and maintenance burden
- preserve history so a successful room procedure can be replayed, audited, and exported outward

### Recommended Mycelix responsibilities by station

| Station | Mycelix role |
|---|---|
| `mk0-seed-node` | hosts the local DHT-facing orchestration client, issues work orders, records scenario and telemetry summaries |
| `mk0-helios` | publishes reserve state, charge/discharge windows, and power-constrained scheduling advisories |
| `mk0-detritivore` | records feedstock intake, recovered filament batches, contamination flags, and yield per batch |
| `mk0-fabricator` | claims print jobs, reports success/failure, attaches recipe/version metadata to outputs |
| `mk0-manipulator` | consumes assembly tasks, logs completion/failure, emits quality-check and handoff events |
| `mk0-vector` | accepts delivery tasks, reports pickup/dropoff timestamps, and publishes route blockage or low-charge conditions |

### Minimum Mk0 Mycelix objects

- `work_order`
  - what part or subassembly should be produced
  - which station currently owns the task
  - what prerequisites must complete first
- `material_batch`
  - feedstock type, provenance, mass, and quality flags
- `energy_window`
  - whether noncritical fabrication is allowed under current reserve state
- `handoff_receipt`
  - proves a station completed its step and transferred responsibility onward
- `maintenance_flag`
  - records jams, nozzle fouling, low lubrication, motor wear, battery degradation, or sensor uncertainty

### Governance in the room

Mk0 does not need full city-scale governance, but it should still use Mycelix to govern:

- priority when energy is insufficient for all pending jobs
- whether recycled material is acceptable for a given print class
- whether a degraded machine can continue operating or must be quarantined
- whether human override is opening a one-off exception or changing normal policy

The point is not bureaucracy. The point is to make scarcity, exceptions, and responsibility explicit and replayable.

### Accounting / TEND stance

Mk0 does not need speculative market complexity on day one. Start with local accounting:

- internal cost tracking for watt-hours, feedstock mass, print time, and operator interventions
- optional TEND-denominated accounting only after the room loop is stable
- no economic layer should be allowed to bypass physical safety or maintenance constraints

### Offline-first rule

- the room must continue functioning if external connectivity disappears
- Mycelix interaction should prefer local-first records with later sync outward
- if consensus with the wider network is unavailable, Mk0 falls back to local room policy and logs the divergence for later reconciliation

### First useful Mycelix-backed demo

The first believable Mycelix interaction in Mk0 is:

1. seed node posts a `work_order` for a benchmark subassembly
2. detritivore publishes a usable `material_batch`
3. fabricator claims the job and attaches recipe/version metadata
4. vector receives a delivery task from printer to manipulator
5. manipulator emits a `handoff_receipt` for assembled output
6. helios publishes whether another cycle is allowed under current reserve

If that round-trip works locally and can be replayed, Mycelix is already doing real work in the bootstrapper.

## Metrics

Track these from the beginning:

- watt-hours per completed subassembly
- grams of recycled feedstock reused
- print success rate
- assembly success rate
- rover delivery latency
- work-order completion latency
- handoff success rate between stations
- operator interventions per cycle
- failure causes by station
- hours of autonomous uptime without unsafe event

## Success Criteria

Mk0 is credible when all of the following are true:

- one-room loop completes end to end without ad hoc manual carrying between every stage
- at least one subassembly can be reproduced reliably across multiple runs
- recycled material is actually reused, not just demonstrated once
- the microgrid / reserve model constrains behavior rather than being decorative
- failures are logged with enough fidelity to replay and diagnose

## What Mk0 Is Not

- not a claim of full self-replication
- not a substitute for the later city-scale platform stack
- not a new long-term namespace that duplicates the main platform taxonomy
- not a reason to delay the real platform crates; it should accelerate and ground them

## Recommended Immediate Work

1. define one benchmark subassembly and its bill of materials
2. map a single room layout with printer, manipulator, storage, and rover lanes
3. add telemetry events for print, pickup, assembly, delivery, and charge cycles
4. define a small scenario harness that replays one full bootstrapper loop
5. treat `mk0-helios` constraints as real scheduling input from the start

## Relationship To The Main Roadmap

- Mk0 is the practical on-ramp to the broader roadmap
- `mk0-detritivore` should mature into `symthaea-scavenger` profiles, not become a competing top-level platform
- `mk0-helios` should feed infrastructure / plexus semantics
- `mk0-fabricator` should become the proving ground for the later assembly platform
- `mk0-vector` should become the first logistics autonomy substrate for indoor civic movement

The point of Mk0 is simple: prove that a tiny metabolism can make useful things, move them, and sustain its own operation before claiming a city.
