# Lunar Elevator Transport Network — Master Plan v0.1

Status: architecture / research plan only

Tracker: #1542

## Purpose

Design a lunar elevator as one mode in a multimodal lunar and cislunar transportation network. The program is successful only if it can determine, with explicit evidence and uncertainty, when an elevator is preferable to landers, surface mobility, electric tugs, mass drivers, momentum-exchange tethers, or hybrids.

The end state is not "a cable to the Moon." It is an interoperable transportation utility connecting lunar industry, settlements, surface logistics, EML1/EML2, depots, orbital manufacturing, Earth-orbit destinations, and eventually deeper-space logistics.

## Non-negotiable principles

1. Intelligence is not actuator authority.
2. Simulation evidence is not qualification evidence.
3. Cargo-first; humans are optional and separately qualified.
4. The mature system must not depend on one ribbon, one climber, one controller, one power source, one provider, or one network path.
5. Every cost, throughput, lifetime, and reliability claim carries assumptions, provenance, and uncertainty.
6. Use open standards and existing transport/network interfaces where possible.
7. The elevator must continuously compete against alternatives in the same trade space.
8. The transport network must solve lunar geography, not pretend all important sites lie under the elevator.

## Network topology

Reference nodes:

- Earth surface / launch sites
- LEO staging and depots
- high Earth orbit / GEO services where useful
- NRHO / lunar-orbit staging where useful
- EML1 logistics hub
- EML2 logistics / industrial hub
- lunar nearside elevator anchor corridor
- lunar South-Pole settlement / resource zone
- additional mines / power sites / science sites
- future farside anchor / radio-science corridor if justified
- orbital manufacturing / shipyard nodes

Reference transport edges:

- Earth launch
- reusable lunar lander
- lunar surface hauler
- future rail / guided freight
- lunar elevator
- mass driver / electromagnetic launcher
- solar-electric tug
- nuclear-electric tug if justified
- chemical tug for time-critical cargo
- momentum-exchange tether / rotovator if justified
- autonomous cargo handling / transshipment

The routing abstraction must not encode one preferred mode. A route advertises capacity, cargo classes, schedule, energy, latency, cost, risk, interface compatibility, availability, and evidence.

## Critical geography problem

Current sustained lunar-development planning is concentrated near the South Pole, while the canonical nearside EML1 elevator geometry places the surface anchor in the nearside/sub-Earth region. Current LiftPort material uses Sinus Medii as its anchor/hub concept.

Therefore the master network must explicitly trade:

- South Pole -> anchor robotic hauling
- South Pole -> anchor rail / guided freight
- South Pole -> lunar orbit / EML1 by lander
- processing resources near extraction vs near the elevator
- distributed depots
- alternate anchor geometries
- eventual farside / EML2 system

The elevator is not allowed to erase this distance in the model.

## Transport-mode roles to test, not assume

### Lunar elevator
Candidate strengths:
- electrically driven recurring cargo movement
- bidirectional freight
- low propellant consumption
- fragile / precision cargo
- controlled scheduled service
- connection to EML1 hub

Candidate weaknesses:
- very large fixed infrastructure
- anchor geography
- long structural lifetime / creep / impact problem
- capacity constrained by climber and tether loading
- deployment and repair complexity
- concentration risk until redundancy exists

### Reusable landers
Candidate strengths:
- geographic flexibility
- mature mission architecture
- time-sensitive cargo and crew
- no fixed global tether

Candidate weaknesses:
- propellant-intensive
- repeated engine/landing risk
- dust/plume effects
- recurring vehicle maintenance

### Mass driver / electromagnetic launcher
Candidate strengths:
- potentially exceptional bulk-export throughput
- well matched to rugged commodities
- little carried propellant

Candidate weaknesses:
- launch accuracy / catcher problem
- cargo packaging constraints
- surface installation
- likely one-way-biased service
- hazardous high-energy launch corridor

### Electric tugs
Candidate strengths:
- high propellant efficiency
- reusable orbital freight
- natural depot/hub integration

Candidate weaknesses:
- long transit times
- power and thruster lifetime

### Surface haul / rail
Candidate strengths:
- connects geographically separated mines, settlements, anchor, landing zones, and power sites
- can become a common utility independent of elevator timing

Candidate weaknesses:
- civil works over large distance
- dust / thermal / terrain maintenance
- energy distribution

## Common evidence levels

- Concept
- Analytical
- Simulation
- IndependentSimulation
- HardwareBench
- EnvironmentalTest
- OrbitalDemonstration
- LunarDemonstration
- OperationalCargoService
- HumanQualification

Promotion may only use evidence at or above the gate's declared minimum.

# Phase program

## Phase 0 — Program constitution and grand trade study

Objective: create a reproducible decision framework before designing around a preferred answer.

Deliverables:
- units / frames / epochs / time systems
- authority and evidence ADR
- network node/edge ontology
- cargo taxonomy
- candidate anchor/hub geometries
- common lifecycle-cost model
- common reliability model
- assumptions/provenance ledger
- technology-risk register
- benchmark protocols
- reference cargo-demand scenarios
- external comparison cases
- initial Pareto trade study

Exit gate G0:
- independent reviewers can reproduce the same comparison inputs and obtain materially consistent results;
- no architecture wins because it received easier assumptions than competitors;
- safety/authority boundaries are explicit.

## Phase 1 — Independent physics and materials feasibility

Objective: prove or falsify the basic structural/orbital case with independent models.

Physics:
- ephemeris-backed Earth/Moon/Sun states
- CR3BP and N-body cross-checks
- explicit rotating/inertial frames
- equilibrium tether tension and taper
- counterweight sensitivity
- anchor-load sensitivity
- climber-induced tension and libration
- thermal expansion/contraction
- power and regenerative descent

Materials:
- tensile distributions
- creep
- fatigue
- UV/radiation degradation
- thermal-cycle degradation
- manufacturing variation
- splice/crosslink efficiency
- impact/MMOD damage distributions

Requirement:
At least two independently implemented models must agree within declared tolerances over the design domain.

Exit gate G1:
No unresolved discrepancy large enough to reverse an architecture decision.

## Phase 2 — Materials and component environmental research

Objective: replace major material-life assumptions with measured distributions.

Test families:
- fiber and ribbon coupons
- full-width ribbon sections
- splices and terminations
- crosslinks / redundant load paths
- embedded strain / acoustic / thermal / continuity sensing
- power/data conductors where applicable
- climber contact / traction materials
- thermal-vacuum
- UV / ionizing radiation
- cyclic load
- long-duration creep
- abrasion / dust where relevant
- impact / hypervelocity representative work

Flight exposure demonstrator may be used where ground testing cannot reproduce the environment economically.

Exit gate G2:
Design factors are tied to distributions and confidence bounds rather than handbook maxima.

## Phase 3 — Integrated terrestrial tether/climber system

Objective: validate interfaces and failure behavior before orbital deployment.

Demonstrate:
- deployment reels
- full-scale cross-section ribbon
- high-cycle climber traction
- smooth acceleration profiles
- emergency braking
- climber passing / traffic scheduling
- damaged-segment localization
- robotic inspection
- segment reinforcement / replacement
- sensor degradation
- power interruption
- controller/watchdog failure
- recovery procedures

Exit gate G3:
Seeded component faults remain contained and recoverable under the declared operating envelope.

## Phase 4 — Orbital tether demonstrator

Objective: validate deployment and flexible-structure control in space.

Demonstrate:
- controlled deployment
- tether shape/tension estimation
- libration control
- translated payload / climber-like disturbance
- attitude/orbit coupling
- navigation
- autonomous safing
- long-duration materials exposure
- impact detection
- inspection/repair concept

No lunar anchoring required.

Exit gate G4:
Deployment/control/inspection objectives demonstrated in orbit with reproduced telemetry/evidence.

## Phase 5 — Lunar/deep-space tether pathfinder

Objective: prove lunar-environment tether operations before permanent infrastructure.

Candidate forms:
- rotating lunar flyby tether
- touch-and-go sample pathfinder
- partial lunar tether
- deep-space tether exposure mission

Demonstrate:
- lunar navigation and dynamics
- deployment near Moon
- tether control under lunar perturbations
- cargo/sampling interaction
- deep-space endurance
- fault recovery

Exit gate G5:
Lunar-specific dynamics no longer rest solely on simulation.

## Phase 6 — Surface and EML1 logistics prepositioning

Objective: build the infrastructure that makes elevator deployment serviceable rather than heroic.

Surface:
- anchor reconnaissance
- foundation/civil works
- power
- communications/PNT
- landing/handling area
- robotic warehousing
- standardized cargo interfaces
- spare ribbon / climber inventory
- surface route to major operations/resource sites

EML1:
- precursor hub
- power
- communications
- docking/cargo handling
- counterweight precursor
- inspection/maintenance capability

Hard rule:
The logistics fabric must function without the elevator.

Exit gate G6:
Deployment, maintenance, and recovery resources exist before seed-ribbon commitment.

## Phase 7 — Seed lunar elevator

Objective: first permanent cargo tether.

Characteristics:
- robotic only
- conservative payloads
- few climbers
- continuous structural state estimation
- independent inspection
- strict climber separation / load rules
- spare/replacement ribbon capacity
- deterministic local safing
- maintenance windows

No passenger operations.

Exit gate G7A:
Sustained low-rate freight with measured structural and maintenance behavior.

## Phase 8 — Early robotic freight service

Objective: operate as a transportation service rather than an experiment.

Add:
- standardized freight containers
- published schedules
- bidirectional cargo
- regenerative descent accounting
- EML1 transshipment
- tug/lander interchange
- reservation and custody records
- spare inventory optimization
- lifecycle-cost accounting

Exit gate G7B:
At least one real cargo class shows a measured operational advantage or strategically valuable complementarity versus alternatives.

## Phase 9 — Resilient elevator service

Objective: remove megastructure single-point fragility.

Target architecture:
- multiple load paths / ribbons
- sacrificial impact zones
- crosslinks where justified
- replaceable sections
- multiple independent climbers
- inspection robots
- repair robots
- segmented sensing
- independent power paths
- provider-independent operations
- cyber/authority isolation
- comm-outage local safing

Required fault campaign:
- one and multiple ribbon severances
- climber jam
- runaway climber
- bad cargo mass declaration
- sensor disagreement
- power loss
- counterweight-control fault
- anchor fault
- impact swarm
- thermal excursion
- command replay / stale authorization
- repair-robot failure
- correlated faults

Exit gate G8:
Bounded multiple faults do not produce uncontrolled network loss.

## Phase 10 — High-throughput multimodal lunar freight

Objective: stop optimizing the elevator in isolation.

Freight classes should be dynamically assigned among:
- elevator
- mass driver
- lander
- surface haul/rail
- SEP/NEP/chemical tugs
- future momentum-exchange systems

Candidate division:
- elevator: precision, bidirectional, fragile, high-value, scheduled cargo
- mass driver: very high-volume rugged outbound bulk
- lander: geographically flexible and time-critical delivery
- electric tug: slow orbital distribution

This is a hypothesis, not a rule; Symthaea must test it.

Exit gate G9:
Capacity expansion is justified by measured demand and lifecycle economics.

## Phase 11 — Industrial cislunar network

Objective: integrate transportation with production.

Add:
- lunar oxygen/water/material flows
- propellant depots
- orbital manufacturing
- shipyards
- large storage hubs
- EML1/EML2 interconnect
- resource/service markets
- maintenance/servicing network
- surface trunk expansion

Mycelix role:
identity, capability, reservation, custody, contracts, provenance, service discovery, accounting.

Symthaea role:
prediction, simulation, scheduling, routing, anomaly detection, trade-space optimization.

Xenia/local safety role:
authorization integrity, anti-replay, signed commands/receipts, deterministic actuator acceptance.

## Phase 12 — Optional human qualification

Objective: determine whether carrying humans on the elevator is actually superior to landers.

Hard separation:
Cargo operational reliability is not human-rating evidence.

Require separately:
- pressure/habitation
- rescue
- medical contingency
- life support
- fire
- radiation
- prolonged transit
- climber evacuation
- tether emergency
- docking/transfer
- independent abort strategy

The program remains successful if this phase is never entered.

## Phase 13 — End-state self-expanding network

Characteristics:
- multiple independent transport corridors
- mature nearside elevator service
- possible farside/EML2 elevator if justified
- high-throughput bulk launcher if justified
- large surface freight network
- cislunar depots and manufacturing
- increasing local manufacture of infrastructure components
- open service coordination across many owners/providers
- no single central controller required for network continuity

The end-state optimization target is productive multiplication, not simply tonnes moved.

# Promotion gates

G0 Architecture reproducibility
G1 Independent physics agreement
G2 Material-life evidence
G3 Component/failure containment
G4 Orbital deployment evidence
G5 Lunar tether evidence
G6 Logistics/deployment readiness
G7 Cargo service utility
G8 Multi-fault resilience
G9 Scale/economic justification
G10 Separate human qualification

No gate may be satisfied by redefining the evidence category after the result is known.

# Core metrics

Track a Pareto vector rather than one opaque score:

- lifecycle cost per kg
- Earth-imported mass per delivered kg
- electrical energy per kg
- propellant per kg
- throughput
- latency
- schedule reliability
- availability
- mean repair time
- spare inventory
- climber utilization
- tether structural margin
- accumulated damage
- cargo-loss probability
- construction mass
- surface-haul burden
- network concentration risk
- human exposure where applicable
- Off-Earth Productive Multiplication

# Software ownership

## `symthaea-infrastructure`
Neutral assets, services, resources, interfaces, reservations, observations, hazards, proposals, authorizations, receipts.

## `symthaea-orbital`
Cislunar frames, ephemerides, CR3BP/N-body, trajectories, proximity/orbital dynamics.

## `symthaea-engineering` / solver bridges
Trade studies and composition of OpenSees/other structural tools rather than duplicating high-fidelity FEM.

## Symtropy
Deterministic operational simulation: robots, cargo handling, surface logistics, maintenance, emergent infrastructure interactions. Do not turn it into the sole high-fidelity flexible-tether solver.

## Mycelix
Multi-owner coordination, identity, service discovery, reservations, custody, provenance, accounting, settlement.

## Xenia / local controllers
Command authentication, exact capability/authority binding, anti-replay, local fail-closed acceptance.

# Initial software PR sequence

LETN-001 — architecture, transport graph, assumptions/evidence ADR
LETN-002 — ephemeris/N-body/frame validation
LETN-003 — 1-D tether equilibrium/taper/tension reference model
LETN-004 — structural/FEM bridge + climber dynamic loads
LETN-005 — degradation/splice/redundancy model
LETN-006 — climber power/regen/thermal model
LETN-007 — impact/severance/repair fault campaign
LETN-008 — intermodal transport graph and routing
LETN-009 — South-Pole-to-anchor logistics trade study
LETN-010 — lifecycle economics / traffic-demand optimizer
LETN-011 — Mycelix/Xenia coordination-authority adapters
LETN-012 — full Phase-6→9 reference campaign

# Required reference scenarios

A. Small scientific outpost: tens of tonnes/year
B. Sustained South-Pole base: hundreds of tonnes/year
C. Early ISRU industry: thousands of tonnes/year
D. Large lunar industry: tens/hundreds of thousands of tonnes/year
E. Cislunar manufacturing economy: very high bulk flow

Every architecture is evaluated against all scenarios; it may legitimately win only at later traffic scales.

# Evidence discipline

Company projections—including schedules, costs, throughput, material claims, and first-lift dates—are treated as hypotheses until independently reproduced.

Historic NASA/NIAC and academic elevator/mass-driver work is used as prior art, not qualification evidence.

The system should be able to conclude any of the following without program failure:

- elevator is worthwhile now;
- elevator is worthwhile only beyond a traffic threshold;
- mass driver should come first;
- lander+tug network remains better for decades;
- elevator is technically feasible but economically premature;
- one architecture serves one cargo class while another serves the rest.

That ability to say "not yet" is part of the design.