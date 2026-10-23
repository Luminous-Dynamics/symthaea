# Lunar Elevator Transport Network — Phase 0 Execution Specification v0.1

Status: planning / research specification

Master tracker: #1542

## Phase 0 purpose

Phase 0 exists to prevent a preferred architecture from becoming an assumption.

Its output is not a selected lunar elevator design. Its output is a **reproducible comparison framework** that can tell us when an elevator, lander, mass driver, surface freight system, tug, tether, or hybrid is preferable for a declared demand scenario.

## Phase 0 exit statement

Phase 0 passes only when an independent reviewer can reproduce the same inputs and obtain materially consistent trade-space results without relying on undocumented constants, hidden cargo assumptions, or privileged cost models.

# Workstream P0-A — Frames, units, time, geometry

Freeze:
- SI engineering units at subsystem boundaries unless an existing domain has an explicit unit contract;
- canonical distance/velocity units for orbital APIs;
- Earth-centered, Moon-centered, barycentric, Earth-Moon rotating, body, tether-local, anchor-local, and surface frames;
- epoch/time scale conventions;
- Moon orientation / site coordinate conventions;
- transform provenance and numerical tolerance rules.

Deliverable:
`LETN_FRAME_CONTRACT_V0_1`.

Acceptance:
Round-trip transforms and independent ephemeris comparisons stay inside declared tolerances.

# Workstream P0-B — Evidence and authority contract

Use a common evidence ladder:

`Concept -> Analytical -> Simulation -> IndependentSimulation -> HardwareBench -> EnvironmentalTest -> OrbitalDemonstration -> LunarDemonstration -> OperationalCargoService -> HumanQualification`.

Every model/result records:
- evidence level;
- assumptions/version;
- source/provenance;
- validity domain;
- uncertainty;
- superseding evidence where applicable.

Authority contract:
- Symthaea proposes and predicts;
- Mycelix coordinates identity/resources/contracts/provenance;
- Xenia-style capability binding authenticates proposals/authorizations;
- deterministic local controllers accept/reject physical commands;
- no AI reasoning result directly actuates climbers, brakes, anchor hardware, deployment, counterweight control, or power switching.

Deliverable:
`LETN_AUTHORITY_EVIDENCE_ADR_V0_1`.

# Workstream P0-C — Network ontology

Canonical types:

`TransportNode`
- id
- location/frame
- interfaces
- storage
- power
- handling capability
- owner/operator
- authority requirements
- evidence

`TransportEdge`
- mode
- origin/destination
- cargo classes
- payload range
- capacity
- schedule/window
- duration
- energy/propellant
- cost model
- reliability
- availability
- infrastructure dependencies
- interface requirements
- evidence

`CargoClass`
- mass
- dimensions
- fragility
- hazardous-material constraints
- thermal constraints
- human/biological flag
- acceleration/vibration envelope
- time criticality
- contamination constraints
- custody requirements

`Service`
- handling
- power
- charging
- propellant
- storage
- maintenance
- inspection
- repair
- communications/PNT
- docking
- rescue

Deliverable:
`LETN-001` neutral transport-graph representation in/adjacent to `symthaea-infrastructure`.

# Workstream P0-D — Candidate nodes and corridors

Minimum initial graph:
- Earth surface
- LEO
- high Earth orbit / GEO service node
- NRHO/lunar orbit staging node
- EML1
- EML2
- South-Pole settlement/resource node
- Sinus-Medii-like nearside anchor reference node
- representative mine/power site
- orbital manufacturing node

Minimum candidate edges:
- Earth launch
- reusable lander
- surface robotic haul
- future guided freight/rail
- nearside elevator
- solar-electric tug
- chemical tug
- mass driver to catcher/hub
- momentum-exchange tether concept

Do not use exact site names as permanent architectural commitments.

# Workstream P0-E — Demand scenarios

Create explicit demand cases rather than one imagined mature lunar economy.

## D0 Scientific foothold
- tens of tonnes/year
- high-value instruments
- low return mass
- little/no bulk ISRU export

## D1 Sustained base
- hundreds of tonnes/year
- routine spares, food, equipment
- modest sample/material return

## D2 Early industry
- thousands of tonnes/year
- oxygen/water/material movement
- growing outbound cargo

## D3 Large lunar industry
- tens to hundreds of thousands of tonnes/year
- bulk commodities
- high utilization of depots/manufacturing

## D4 Cislunar industrial economy
- very high throughput
- orbital manufacturing and shipyards
- multiple hubs/providers

All architectures must be evaluated across D0-D4.

# Workstream P0-F — Architecture candidates

At minimum compare:

A. Lander-only + surface mobility
B. Lander + reusable orbital tug + depots
C. Elevator + lander + tug
D. Mass driver + catcher + tug + lander
E. Elevator + mass driver + lander + tug
F. Momentum-exchange tether hybrid where physically relevant
G. Delay elevator until traffic threshold, then deploy

The comparison must permit "do not build elevator yet" as a valid optimum.

# Workstream P0-G — Common cost and resource accounting

Every architecture uses the same accounting categories:

- Earth launch mass
- Earth launch cost range
- lunar construction mass
- lunar-produced mass
- electrical energy
- propellant
- infrastructure capex
- replacement/spares
- scheduled maintenance
- unscheduled repair
- staffing/operations
- communications/PNT
- storage/handling
- downtime
- cargo loss
- decommissioning / replacement

No candidate may hide infrastructure in an excluded category.

# Workstream P0-H — Common reliability model

At minimum model:
- component failure distributions
- correlated environmental events
- maintenance intervals
- spare availability
- repair time
- logistics delay
- provider outage
- comm outage
- power outage
- sensor disagreement
- bad cargo metadata
- cyber/authorization failure

Architecture-level metrics:
- probability of delivery
- expected downtime
- mean time to restore service
- annual cargo loss
- concentration risk
- graceful degradation modes

# Workstream P0-I — Elevator reference hypothesis

Define one deliberately conservative reference elevator only to make the trade study concrete.

It must include:
- anchor site as a variable, not a constant;
- total tether length;
- taper/tension distribution;
- counterweight;
- seed-ribbon mass;
- payload/climber mass range;
- climber spacing;
- ascent/descent speed range;
- acceleration limits;
- traction/contact assumptions;
- climber electrical efficiency;
- descending energy recovery;
- structural design factors;
- material degradation distributions;
- splices/crosslinks;
- inspection/repair policy;
- spare-ribbon policy;
- power availability;
- stationkeeping/control assumptions;
- deployment architecture.

All values start as evidence-tagged assumptions, not facts.

# Workstream P0-J — South-Pole-to-anchor trade study

This is mandatory.

Compare:
- direct lander movement between South Pole and EML1/lunar orbit;
- long-range robotic surface haul to anchor;
- staged depots;
- processing resources near source vs near anchor;
- future rail/guided freight;
- alternate anchor/site choices;
- whether a second/farside infrastructure corridor becomes preferable.

Metrics:
- tonne-km
- energy
- civil-works mass
- maintenance
- travel time
- vehicle count
- dust/terrain exposure
- cargo loss
- infrastructure utilization

# Workstream P0-K — Tether risk register

Minimum hazards:
- single-strand cut
- multi-strand correlated cut
- progressive unzip/crosslink failure
- climber jam
- climber runaway
- climber collision
- unexpected cargo mass
- thermal transient
- creep beyond model
- UV/radiation embrittlement
- micrometeoroid swarm
- anchor structural failure
- anchor regolith/foundation motion
- counterweight/control failure
- sensor drift
- false-positive damage alarm
- missed damage
- power loss
- deployment reel fault
- repair-robot failure
- communication blackout
- stale/forged authorization
- software rollback/version mismatch
- debris conjunction with upper tether/hub

Each hazard records detection, containment, recovery, evidence level, and residual uncertainty.

# Workstream P0-L — Benchmark protocols

Required benchmarks:

1. **Static equilibrium** — tension/taper/counterweight reproducibility.
2. **Climber passage** — transient loads and libration.
3. **Traffic schedule** — many climbers without resonance/load-envelope violation.
4. **Damage tolerance** — cuts/impact with redundant topology.
5. **Repair** — detection-to-repair time and remaining margin.
6. **Power** — ascent demand + descent regeneration + outage behavior.
7. **Deployment** — reel/shape/control sensitivity.
8. **Intermodal routing** — cargo chooses mode by evidence-bearing Pareto trade.
9. **South-Pole trunk** — anchor geography included rather than ignored.
10. **Economics** — break-even traffic threshold under uncertainty.
11. **Bootstrap** — contribution to Off-Earth Productive Multiplication.

# Workstream P0-M — Pareto objectives

Do not collapse into one arbitrary weighted score.

Primary dimensions:
- lifecycle cost/kg
- Earth mass/kg
- energy/kg
- propellant/kg
- throughput
- latency
- availability
- repairability
- construction mass
- structural margin
- cargo-risk
- concentration risk
- infrastructure reuse
- bootstrap multiplication

Symthaea may identify non-dominated frontiers and sensitivity regions.

# Workstream P0-N — Independent oracles

At minimum:
- one internal Rust reference implementation;
- one independent numerical implementation/tool for critical orbital/tether equations;
- high-fidelity structural solver adapter for later phases;
- hand-checkable analytical limits for simple cases.

No model may validate itself by comparing only to code sharing the same implementation.

# Workstream P0-O — External claim registry

Capture claims from companies, papers, NASA studies, and historical concepts as typed hypotheses:

- claimed material feasibility
- tether mass
- payload throughput
- cost/kg
- schedule
- anchor location
- lifter speed
- system lifetime
- repair assumptions

Each claim records source, date, exact wording/value, and validation status.

Company marketing and historical studies remain hypothesis/evidence inputs rather than baseline truth.

# Phase 0 PR plan

## LETN-001A — Architecture ADR + transport graph
Neutral types, evidence fields, authority boundaries, cargo classes, nodes/edges.

## LETN-001B — Reference scenarios + assumptions ledger
D0-D4 demand cases and external claim registry.

## LETN-001C — Common accounting + Pareto result schema
Cost/resource/reliability categories and comparison output.

## LETN-001D — Geometry/network reference cases
Reference nodes, surface trunk cases, anchor-as-variable.

## LETN-001E — Grand trade-study harness
Run architecture candidates A-G through common scenario/evidence interfaces.

# Phase 0 acceptance tests

- same cargo demand produces comparable output across all candidate architectures;
- changing one assumption updates every affected candidate through provenance, not hidden constants;
- unsupported/missing data remains unknown rather than silently defaulting favorable;
- elevator can lose the trade study;
- mass driver can lose the trade study;
- lander-only can win low-volume cases;
- mixed modes can emerge as Pareto-optimal;
- South-Pole-to-anchor burden is nonzero unless the chosen geometry actually removes it;
- simulation evidence cannot satisfy hardware/operational gates;
- optimization outputs never create actuator authority.

# Phase 0 completion artifact

Produce a frozen evidence capsule containing:
- exact code/tree/lock/toolchain lineage;
- reference assumptions;
- source registry;
- D0-D4 scenario definitions;
- benchmark definitions;
- all candidate architecture inputs;
- Pareto outputs;
- sensitivity analysis;
- unresolved unknowns;
- explicit recommendation: build / do not build / defer / demonstrate next.

The Phase 0 recommendation is allowed to be:

> The lunar elevator is physically promising but should not yet be built; execute Phase 1/2 pathfinders and revisit at traffic threshold X.

That is a valid successful outcome.