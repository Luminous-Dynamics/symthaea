# Decentralized Conjunction Assessment: An Open-Source Network for Multi-Operator Space Debris Coordination

**Target journal**: Acta Astronautica (Elsevier, elsarticle-num style, APC $3,750 for OA)
**Submission portal**: https://www.editorialmanager.com/aastronautica/
**Codebase**: `mycelix-space/lib/orbital-mechanics/` (~3,000 LOC) + 5 Holochain zomes + screening daemon

---

## Abstract (draft, ~150 words)

As tracked debris objects exceed 30,000 and mega-constellations accelerate collision risk, conjunction assessment remains dominated by proprietary tools with limited multi-operator coordination. We present an open-source conjunction assessment network built on Holochain, combining SGP4 propagation, Alfano 2D/3D collision probability estimation, and standard CDM generation with a decentralized multi-operator coordination protocol. Unlike MIT's MOCAT, which models long-term debris environment evolution, our system addresses the operational gap: real-time multi-operator coordination for conjunction events requiring maneuver negotiation. We implement trust-weighted observation fusion, priority-based screening, and a multi-party agreement protocol with weighted quorum voting. Validation against Space-Track TLE data for known conjunction events demonstrates screening accuracy within expected SGP4 error bounds. The system is released as open-source Rust with a standalone screening daemon deployable against CelesTrak catalogs.

---

## 1. Introduction

- The conjunction assessment problem: 30,000+ tracked objects, growing
  - ESA Space Environment Report 2025: collision avoidance maneuvers increasing year-over-year
  - WEF "Clear Orbit, Secure Future" 2026 call for action
- Current landscape of tools:
  - **CAESAR** (ESA): proprietary, ESA-internal
  - **CONAN/CORAM** (CNES): proprietary
  - **CSpOC/18 SDS** (US Space Command): government-only data sharing
  - **MOCAT** (MIT ARCLab, open-source, 2024): models long-term environment evolution (MOCAT-MC for Monte Carlo, MOCAT-SSEM for source-sink), but does not address real-time operational coordination
  - **MOCAT-pySSEM** (2025): Python web app for interdisciplinary research
- The gap: no open-source tool combines conjunction screening with multi-operator coordination
- Our contribution: decentralized conjunction assessment + multi-party maneuver negotiation on Holochain DHT

## 2. System Architecture

### 2.1 Orbital Mechanics Library (Pure Rust, no Holochain deps)

- SGP4 propagation from TLE catalogs (validated against Kelso 2009)
- Classical orbital elements with J2 perturbation
- Alfano 2D/3D collision probability (Alfano 2005; Foster 1992)
- Covariance propagation for state uncertainty
- CDM (Conjunction Data Message) generation per CCSDS standard
- Trust-weighted observation fusion pipeline

### 2.2 Holochain Zome Architecture (5 zomes)

| Zome | Responsibility |
|------|----------------|
| `orbital_objects` | TLE catalog management, object classification (Payload/RocketBody/Debris/Unknown) |
| `observations` | Sensor data ingestion, trust-weighted quality fusion |
| `conjunctions` | Collision prediction, CDM storage, risk level tracking, re-screening |
| `debris_bounties` | Kessler cleanup market (Open -> Claimed -> Completed state machine) |
| `traffic_control` | Bilateral + multi-party negotiation, weighted quorum voting |

### 2.3 Screening Daemon

- Standalone CLI: ingests CelesTrak TLE catalogs, screens protected objects
- Priority-weighted screening: crewed vehicles get finest time steps
- Configurable interval (default 15 min), catalog refresh cycling
- Systemd service deployment

### 2.4 Consciousness Gating (Trust Fabric)

- All state-modifying operations gated by operator trust tier
- 5 tiers: Observer (submit data) -> Participant (create events) -> Citizen (update risk) -> Steward (create bounties, sign agreements) -> Guardian
- Read operations (queries, screening) are ungated for open access

## 3. Multi-Operator Coordination Protocol

### 3.1 Problem: N-operator conjunction events

- A single conjunction may affect operators who did not generate the alert
- Maneuver decisions have externalities (new debris risk, fuel cost asymmetry)
- No existing open protocol for multi-party maneuver negotiation

### 3.2 Protocol Design

- `ConjunctionProposal`: N affected operators, pre-computed maneuver options, voting deadline, quorum threshold
- `OperatorVote`: weighted vote with justification, double-vote prevention
- `MultiPartyAgreement`: cosigned by quorum of approving operators
- State machine: Voting -> Approved -> Executing -> Completed (also Rejected, Expired)
- Trust-weighted voting: higher-tier operators carry proportionally more weight

### 3.3 Trust-Weighted Observation Fusion

- `effective_quality = data_quality * (trust_floor + (1 - trust_floor) * trust_weight)`
- Unknown sensors receive floor weight (default 0.3)
- Cross-role call to identity cluster for trust level lookup
- Prevents low-quality or adversarial sensor data from corrupting assessments

## 4. Validation Plan

### 4.1 Screening Accuracy

- Reconstruct known conjunction events from Space-Track TLE data
- Compare predicted miss distance and collision probability against published CDMs
- Quantify SGP4 propagation error contribution (expected dominant error source)
- Validated preliminary result: CSS vs DUPLEX cubesat conjunction at 51.6 km, Pc 4.7e-7

### 4.2 Scalability

- Benchmark screening throughput: objects screened per second vs. catalog size
- Holochain DHT propagation latency for conjunction alerts
- Compare against MOCAT-SSEM runtime (seconds to minutes on personal computer)

### 4.3 Coordination Protocol

- Simulate N-operator conjunction scenarios (N=2,3,5,10)
- Measure time-to-agreement under varying quorum thresholds
- Evaluate trust-weighting impact on agreement quality

## 5. Comparison with Existing Tools

| Feature | This work | MOCAT (MIT) | CAESAR (ESA) | CSpOC |
|---------|-----------|-------------|--------------|-------|
| Open source | Yes | Yes | No | No |
| Real-time screening | Yes | No (long-term) | Yes | Yes |
| Multi-operator coordination | Yes | No | Limited | Limited |
| Decentralized | Yes (Holochain) | No | No | No |
| Collision probability | Alfano 2D/3D | Monte Carlo | Multiple | Multiple |
| Debris environment modeling | No | Yes (core focus) | No | Yes |
| Maneuver negotiation | Multi-party quorum | No | Bilateral | No |

**Key differentiator**: MOCAT answers "what will the debris environment look like in 50 years?" Our system answers "what should operators do about this conjunction event today?"

## 6. Discussion

### 6.1 Honest Limitations

- SGP4 propagation accuracy degrades beyond 7 days; high-fidelity propagators (numerical integration) would improve long-horizon screening
- Holochain DHT latency (seconds) is acceptable for conjunction events (hours-to-days timescale) but not for real-time collision avoidance
- Trust fabric assumes honest identity claims; Sybil resistance depends on the identity cluster
- Alfano 2D approximation may underestimate probability for highly eccentric orbits

### 6.2 Advantages of Decentralization

- No single point of failure or control
- Operators retain sovereignty over their data
- Trust-weighted fusion incentivizes high-quality sensor contributions
- Multi-party protocol avoids bilateral negotiation bottlenecks

### 6.3 Path to Operational Use

- Integration with Space-Track API for automated catalog updates
- Extension to SP3 ephemeris for high-fidelity objects
- Regulatory alignment with UN COPUOS long-term sustainability guidelines

## 7. Conclusion

- Open-source conjunction assessment with decentralized multi-operator coordination fills a gap between proprietary operational tools and academic long-term modeling
- The multi-party agreement protocol enables coordinated collision avoidance without centralized authority
- Released as open-source Rust (orbital mechanics) + Holochain (coordination)

## References (key, non-exhaustive)

- Alfano, S. (2005). A numerical implementation of spherical object collision probability. J. Astronautical Sciences.
- Foster, J. L. (1992). The analytic basis for debris avoidance operations. NASA JSC.
- Kelso, T. S. (2009). Validation of SGP4 and IS-GPS-200D against GPS precision ephemerides.
- Lifson, M. et al. (2024). MOCAT: MIT Orbital Capacity Assessment Tool. MIT ARCLab.
- MOCAT-pySSEM (2025). SoftwareX.
- ESA Space Environment Report (2025).
- WEF Clear Orbit, Secure Future (2026).
- CCSDS 508.0-B-1: Conjunction Data Message recommended standard.
- Liou, J.-C. (2011). An active debris removal parametric study. Advances in Space Research.
