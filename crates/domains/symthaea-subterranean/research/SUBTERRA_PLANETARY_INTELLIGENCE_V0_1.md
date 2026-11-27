# Subterra — Planetary Subsurface Intelligence for Symthaea v0.1

**Status:** Research programme and bounded reference-kernel proposal  
**Date:** 2026-11-16  
**Owner:** Luminous Dynamics / Symthaea  
**Scope:** Scientific belief formation, active measurement selection, mission-level reasoning and evidence-qualified interfaces. This document does not grant actuator authority or certify a planetary site.

## 1. Executive thesis

Subterra should have a first-class intelligence programme inside Symthaea, but it must compose with the existing `symthaea-subterranean` platform instead of creating a second robot controller.

The central research question is:

> Given incomplete, uncertain and sometimes conflicting observations of a planetary subsurface, what should a scientific robot team measure next, why is that measurement decision-relevant, what would change under each possible result, and what remains unknown afterwards?

The intended product is not a cave chatbot or a second SLAM stack. It is a bounded **evidence-to-science-policy layer** that turns a site evidence state into ranked, explainable measurement proposals while preserving uncertainty and remaining subordinate to the mission executive, independent safety supervisor and physical controller.

The key paradigm shift is from *map first, interpret later* to **decision-aware exploration**:

`observations → competing site hypotheses → decision uncertainty → value of measurement → admissible survey proposal → verified observation → belief revision`

The loop must be able to say that another measurement has little decision value, that two apparent witnesses are actually one sensor lineage, that a site is out of the validated domain, or that the evidence is not sufficient to recommend an operation.

## 2. Why the research frontier has moved

The 2026 literature strengthens the case for this layer, but also makes the competitive boundary clearer.

| Work | Current contribution | Consequence for Subterra |
|---|---|---|
| Swinton et al., *The Evolution of Autonomous Systems for Planetary Cave Exploration* (Journal of Field Robotics, first published 10 May 2026) | Reviews planetary cave robotics, sensing, multimodal localization, communication relays and heterogeneous teams | LiDAR/IMU/SLAM, breadcrumb relays and multi-robot teams are established research baselines, not our claimed invention |
| SKiD-SLAM (IEEE Transactions on Field Robotics, 2026; arXiv:2610.38141) | Distributed multi-robot LiDAR SLAM; 250 kbps inter-robot budget; 320-byte SOLiD descriptors; cave and planetary-analogue field deployments | Treat as a serious distributed-mapping baseline. Do not claim Symthaea replaces or beats its SLAM system |
| LunarLeaper (Acta Astronautica, March 2026) | 15 kg-class legged lunar-pit mission concept combining gravimetry, GPR, high-resolution imaging, spectroscopy and leg-based geomechanical tests | A useful example of science-question-driven sensor portfolios rather than camera-only cave scouting |
| Ape Cave morphometry (JGR: Planets, first published 7 November 2026) | 3.4 km terrestrial lava-tube model; 1,414 cross-sections; multiple recurring profile classes | Site models and priors must preserve non-uniform geometry and unobserved spans |
| Robotic lava-tube mapping dataset (Frontiers in Robotics and AI, 2026) | Open quadruped/LiDAR field dataset with navigation, video and environmental data | Enables real-data evaluation of observation selection and map/evidence interfaces |
| Mare Tranquillitatis radar conduit (Nature Astronomy, 15 July 2024) | Mini-RF interpretation supports a cave conduit extending tens of metres below the pit | This is a 2024 result, not a 2026 discovery; it is an evidence example, not a surveyed network |
| Nyx Mons Venus pyroduct (Nature Communications, 9 February 2026) | Magellan SAR interpretation proposes a large subsurface conduit with inferred dimensions | Keep the dimensions labelled radar-inferred, not directly measured or engineering-qualified |
| Elysium Mons Mars candidate (npj Space Exploration, 12 June 2026) | Multi-instrument candidate interpretation using optical, thermal, topographic and compositional context | Preserve native modality lineage and alternative interpretations; sensor products are not automatically independent confirmations |

References:
- Swinton et al. (2026): https://doi.org/10.1002/rob.70233
- SKiD-SLAM (2026): https://arxiv.org/abs/2610.38141 and https://github.com/sparolab/SKiD-SLAM
- LunarLeaper (2026): https://doi.org/10.1016/j.actaastro.2025.11.039
- Ape Cave morphometry (2026): https://doi.org/10.1029/2026JE010115
- Robotic lava-tube mapping dataset (2026): https://doi.org/10.3389/frobt.2026.1855443 and https://doi.org/10.4121/778253cc-3193-4f66-bc13-03b80380424e
- Mare Tranquillitatis (2024): https://www.nature.com/articles/s41550-024-02302-y
- Venus (2026): https://www.nature.com/articles/s41467-026-68643-6
- Mars Elysium Mons (2026): https://www.nature.com/articles/s44453-026-00039-8

## 3. Existing Symthaea capabilities to preserve and compose

Repository inspection on 2026-11-16 found several substantial foundations. The new programme should reuse them rather than duplicate them.

### 3.1 Existing subterranean embodiment

`crates/domains/symthaea-subterranean` already provides:
- HDC-LTC controller and a subterranean state encoder;
- deterministic geology/strata profiles, geotechnical lookahead and hazard variables;
- sensor redundancy, observation quality and delayed-observation handling;
- tunnel graph, shared map, return-path memory and relay mesh;
- work orders, mission executive, resource/logistics admission and maintenance;
- team operations, peer trust, partition recovery and rescue-consent boundaries;
- field survivability, hazard arbitration, operational checkpoints and a large hardening/test campaign.

This is valuable operational infrastructure. It should remain the owner of the existing boring/scout embodiment, local mobility, recovery and safety contracts.

### 3.2 A concrete intelligence gap

The current `ActiveInferenceSubterraneanAgent` in `src/fep_agent.rs` calls `perceive()` and exposes variational free energy and a precision input. Its own source comment explicitly states that `select_action`, `act` and `learn_from_outcome` are not used in this platform wiring; motor commands come from the separate HDC-LTC controller. Its FEP configuration has one unused action.

The larger Symthaea FEP module already contains ODE-based trajectory rollouts, expected-free-energy telemetry, pragmatic value and an information-gain field. But those generic capabilities are not yet a planetary site survey planner.

This creates a clear, bounded next step: implement a **measurement-selection oracle** that can later be compared with FEP policy rollouts. It must not become a motor-command path.

### 3.3 Geology and coordinate limits

The current subterranean `geology.rs` models depth-binned strata for the existing platform, including a 200 m simulated-depth limit and terrestrial material-property classes. That is not a universal lunar/Martian/Venusian/icy-world constitutive model.

Likewise, a geological profile or cave graph must not imply planetary coordinate correctness. The Sol Atlas integration specification in [Sol Atlas PR #893](https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/893) correctly separates body-fixed frames and site catalogue responsibilities from Symtropy's interior geometry.

### 3.4 Reusable evidence architecture already in the repository

The following Symthaea work is directly relevant; integrate only after checking each PR's exact head and qualification state:
- [#1223 — evidence-first domain awareness](https://github.com/Luminous-Dynamics/symthaea/pull/1223): observations, tracks, competing hypotheses, epistemic state and consequence-oriented risk.
- [#1226 — spatial safety and ODD](https://github.com/Luminous-Dynamics/symthaea/pull/1226): time-bounded spatial volumes, coordinate-frame agreement and reviewed operating domains.
- [#1628 — replay-resistant sensor trust](https://github.com/Luminous-Dynamics/symthaea/pull/1628): freshness, source lineage, anti-replay and independent-source quorum.
- [#1658 — model assurance](https://github.com/Luminous-Dynamics/symthaea/pull/1658): residual monitoring, evidence admission and explicit Aligned/Restricted/Unsafe/Incomplete states.
- Existing subterranean mission autonomy and team operations protocols: route truth, return requirements, resource admission and partition-aware team state.

These layers address evidence admission and operational constraints. Subterra adds planetary scientific hypothesis management and decision-focused measurement selection.

## 4. Proposed architecture

```text
SOL ATLAS
  body, epoch, reference frame, site catalogue, native orbital observations
       |
       v
EVIDENCE INTAKE
  observation identity, modality, calibration, source lineage,
  freshness, uncertainty, coordinate frame, provenance
       |
       v
SUBTERRA SITE BELIEF (Symthaea)
  morphology hypotheses
  origin hypotheses
  connectivity hypotheses
  material / environmental hypotheses
  access and egress evidence
  explicit unknown, contradiction and out-of-domain state
       |
       +-----------------------------+
       |                             |
       v                             v
PREDICTIVE WORLD MODEL          EVIDENCE CONFLICT
  CfC temporal predictions       preserve incompatible claims
  residuals and surprise         request discriminating evidence
       |                             |
       +--------------+--------------+
                      v
DECISION-AWARE MEASUREMENT PLANNER
  expected value of sample information
  decision regret reduction
  energy / time / data budget
  external admissibility and domain gate
                      |
                      v
MISSION-LEVEL PROPOSAL
  survey task, required instrument, expected outcomes,
  return / communication constraints, stopping condition
                      |
                      v
EXISTING MISSION EXECUTIVE + SAFETY SUPERVISORS
  work admission, route + retreat, logistics, hazards,
  team state, operator / authorization boundaries
                      |
                      v
SYMTROPY
  geometric scene, physical simulation, robot scenario,
  replay, benchmark execution and measured residuals
                      |
                      v
VERIFIED OBSERVATION + VERSIONED BELIEF REVISION
```

### Responsibility boundaries

| System | Subterra responsibility |
|---|---|
| Sol Atlas | Multi-body reference frames, site catalogue, original orbital observations and provenance |
| Symthaea | Competing hypotheses, belief updates, anomaly/conflict reasoning, measurement value, science-policy proposals and evidence-qualified abstention |
| Symtropy | Non-uniform 3D geometry, cave network, physical scenarios, robot mobility/communications simulation, replay and benchmark scoring |
| Mycelix | Cross-agent identity/attestation, sample custody, shared resource and access records; never a substitute for sensor truth |
| GeoBio Foundry | Later use-specific resource processing, biological/abiotic production and industrial qualification |

Cross-repository starting point: [Symtropy PR #5486](https://github.com/Luminous-Dynamics/symtropy/pull/5486) and [Sol Atlas PR #893](https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/893). Both are research drafts, not merged production integrations.

## 5. Five paradigm shifts

### A. From a single best map to an explicit belief state

Maintain competing, revisable hypotheses over:
- feature morphology;
- geological origin (volcanic, collapse, karst/dissolution, tectonic, ice-related, engineered or unknown);
- subsurface connectivity;
- traversability and directional egress;
- interior geometry and surveyed extent;
- environmental and structural properties;
- science value and use-specific qualification.

Every claim must carry evidence references, uncertainty, source lineage, time and coordinate frame. Contradiction is retained. The system must never transform an unknown interval into a measured wall or an inferred connection into a traversable edge.

### B. From entropy maximization to decision-focused value of information

Information gain alone can reward a measurement that reduces uncertainty but does not change the next decision. Prioritize expected value of sample information (EVSI): compare the best expected decision loss before a measurement with expected best decision loss after each possible measurement outcome.

Keep entropy reduction as a separate diagnostic. Use decision regret to choose between scans, assays and geophysical measurements when the mission question is whether to enter, retreat, resurvey, deploy a relay, sample, or leave the site unqualified.

The first reference kernel is intentionally single-step and finite-hypothesis. It does not model arbitrary correlated multi-sensor likelihoods, continuous belief filters or actuator control.

### C. From confidence averaging to evidence lineage

Ten processed products from one camera or radar acquisition are not ten independent witnesses. Group evidence by physical sensor, acquisition, calibration chain and processing lineage. Correlated products may add resolution or interpretation but may not satisfy an independent-source quorum.

The system must preserve raw observation identity and derived-product ancestry. Conflicting sources remain visible until independent evidence discriminates between them.

### D. From one robot to mission-level scientific teams

Task specialisation should follow actual platform capability and instrument complementarity:
- scout / mapper;
- geophysical surveyor;
- relay carrier;
- local anchor or reference platform;
- sampler or manipulator;
- surface coordinator.

Plan the observation, communication chain, payload, energy, return reserve and data exfiltration together. Existing Symthaea team and relay code should be reused; the planner must not invent transport guarantees.

### E. From Earth analogue transfer to declared-domain transfer

Represent each mission environment with a body-specific envelope: body and epoch, frame/shape model, gravity, thermal range, pressure/atmosphere, radiation, volatile/ice regime, dust/regolith, material assumptions, instrument limits and their evidence states.

Earth lava tubes are valuable for geometry, sensing and robot evaluation, but Earth geomechanics, gravity, weathering, gas and hydrology are not planetary truth. Icy-moon ice-shell voids, fractures, basal conduits and ocean-access concepts are distinct domains, not lava tubes with different labels. Small-body porosity and rubble-pile cavities require microgravity contact and anchoring models.

## 6. Measurement planning model

The first reference planner accepts:
1. a normalized finite hypothesis prior, including an explicit `unknown` hypothesis;
2. mission decisions and loss under each hypothesis;
3. candidate measurement actions with outcome likelihoods conditional on each hypothesis;
4. resource costs (energy, duration, data volume and exposure);
5. a resource budget and utility shadow prices;
6. an external admissibility result that has already passed the responsible reviewed gate.

For measurement action `a`, observation outcome `o`, hypothesis `h`, decision `d`, prior `p(h)`, likelihood `p(o|h,a)` and loss `L(d,h)`:

`p(o|a) = Σ_h p(h) p(o|h,a)`

`p(h|o,a) = p(h) p(o|h,a) / p(o|a)`

`R_before = min_d Σ_h p(h) L(d,h)`

`R_after(a) = Σ_o p(o|a) min_d Σ_h p(h|o,a) L(d,h)`

`EVSI(a) = R_before - R_after(a)`

The likelihood table must come from independently adjudicated site/action observations, with whole-site Training/Calibration/Evaluation splits and acquisition-lineage de-duplication. Qualitative claim states such as Candidate or DirectlyObserved are never converted directly into arbitrary probabilities. The new subterra-likelihood-calibration reference estimates categorical likelihoods on Training data with explicit Dirichlet smoothing, reports Brier/NLL on Calibration/Evaluation sites, and refuses to emit unsupported cells. A guarded converter builds EVSI input only when every cell is supported and the caller separately supplies priors, decision losses, resource costs and external gate results. This is empirical likelihood estimation, not proof of calibration under planetary shift.

A separately declared utility model prices energy, time, data and exposure. Hard safety, authorization, preservation and out-of-domain gates are not soft penalties and cannot be traded against EVSI. The planner only ranks a measurement proposal after those external gates have passed. It does not issue a robot command.

The `unknown` hypothesis prevents a closed-world classifier from forcing every site into a known geological class. It must have a nonzero prior in benchmark cases unless a preregistered case explicitly tests a closed-set mathematical control.

## 7. Priority work packages

| Priority | Work package | Concrete output | Gate |
|---|---|---|---|
| P0 | Measurement-selection reference | Standalone Rust EVSI kernel, deterministic tie-breaks, explicit unknown class, adversarial tests | Source authored; compilation and test execution still required |
| P1 | Site-belief schema adapter | Convert Subterra site-evidence and network records into Symthaea typed hypotheses without promoting evidence | Positive and negative cross-field fixtures |
| P1 | Evidence lineage bridge | Consume domain-awareness / sensor-trust evidence with exact source IDs and freshness | Replayed, stale, duplicate and correlated-source rejection |
| P1 | Site-disjoint likelihood calibration | Estimate outcome likelihoods from independently adjudicated Training sites; score Calibration/Evaluation sites; keep unsupported cells unknown | Whole-site split isolation, acquisition-lineage uniqueness, Brier/NLL, coverage and held-out calibration report |
| P1 | Calibrated-likelihood to EVSI adapter | Build planner inputs from supported likelihood cells and separately supplied priors/losses/costs/gates | Reject insufficient cells, hypothesis-order drift and missing/denied action policies |
| P1 | Observation task proposal | Convert selected measurement to a bounded survey work proposal, not a motor command | Existing logistics, retreat and safety layers remain mandatory |
| P2 | FEP comparison | Compare EVSI proposal with expected-free-energy policy rollouts; report disagreements | Matched baseline, calibration and ablation |
| P2 | Blind site transfer | Leave-site-out and leave-geology-class-out evaluation | Frozen train/calibration/evaluation split by site |
| P3 | Multi-agent science planning | Allocate measurements across heterogeneous agents under bandwidth and resource limits | Faulted communication, relay loss, robot-loss and evidence-return tests |
| P3 | Cross-body domain envelopes | Moon/Mars/Venus first; icy worlds and small bodies kept as separate environment families | No Earth-to-body physics promotion without independent qualification |

The new likelihood calibration package is a small reference estimator only. It does not yet implement hierarchical sensor models, temporal drift, conformal uncertainty, source authentication, clustered bootstrap intervals or planner-induced online adaptation. Those remain later work.

## 8. First terrestrial and orbital data programme

Use accessible, documented data before inventing another synthetic-only benchmark.

| Dataset | Why it matters | Initial use |
|---|---|---|
| Grotta di Monte Intraleo, Sicily — robotic quadruped/LiDAR and environmental dataset | Robot traversal, point clouds, FPV, navigation and environmental context | Sensor/evidence adapter; site-level blind partitions |
| Grotta di Monte Intraleo — LiDAR/photogrammetry comparison dataset | Registered and unregistered point clouds plus processing lineage and navigation mesh | Morphometry and registration uncertainty |
| Ape Cave, Washington — 2026 morphometry dataset | Longitudinal variability and non-uniform profile classes; includes a 67 m rockfall-obstructed scan gap | Profile priors and held-out geometry |
| Majang lava-tube cave / SKiD-SLAM public materials | Distributed multi-robot mapping under a severe bandwidth budget | Mapping baseline; do not duplicate its SLAM contribution |
| Mare Tranquillitatis pit and Mini-RF / LROC products | Real orbital evidence with a bounded radar interpretation | Open-world evidence and geophysical inference case |
| Elysium Mons candidate products | Multi-instrument site interpretation | Correlated modality and alternative-hypothesis case |
| Nyx Mons Magellan SAR analysis | Very large inferred conduit with extreme environment context | Radar-derived estimate versus direct observation distinction |

Dataset/source records:
- Robotic Grotta di Monte Intraleo: https://doi.org/10.4121/778253cc-3193-4f66-bc13-03b80380424e
- LiDAR/photogrammetry Grotta di Monte Intraleo: https://doi.org/10.4121/0CC49D43-8A5C-453E-9370-8E9088B00B48
- Ape Cave article/data/code: https://doi.org/10.1029/2026JE010115
- SKiD-SLAM code/data: https://github.com/sparolab/SKiD-SLAM and https://sparolab.github.io/research/skid_slam/
- Lunar Mini-RF supporting data: https://doi.org/10.5281/zenodo.11005458; MTP internal morphology point cloud: https://doi.org/10.5281/zenodo.6622042
- NASA PDS Geosciences Node: https://pds-geosciences.wustl.edu/
- LROC data: https://wms.lroc.asu.edu/

Dataset licensing, access, calibration and exact file manifests must be frozen before any benchmark run. The presence of a public DOI is not evidence that every file is immediately downloadable or that it carries the ground truth required for every benchmark.

## 9. Claim and authority boundaries

The following are non-negotiable:

- A candidate site is not a cave.
- A cave morphology is not proof of geological origin.
- A geophysical continuation is not a directly observed connection.
- A directly observed connection is not proof of robotic traversability.
- Demonstrated robot traversal is not proof of retreat or human access.
- A mapped interior is not a geotechnically stable or pressurizable space.
- A science-interest score is not a sampling permission.
- A biological hypothesis is not permission to introduce organisms.
- A sensor's confidence field is not an independent quality witness.
- An FEP or EVSI result is a proposal, not a command or authorization.
- Model improvement on a synthetic cave is not planetary performance.

Preservation and planetary-protection reviews remain external, explicit gates. Europa, Enceladus and other potential ocean-world targets should be preservation-first science domains, not assumed biological production sites.

## 10. Success criteria and non-claims

The first success is not "autonomous cave exploration." It is a reproducible planner that, on frozen data, can:
- maintain competing hypotheses including unknown;
- reject stale, duplicate, contradictory and correlated-evidence inflation;
- select decision-relevant measurements under matched budgets;
- abstain when no action has positive decision value or passes the external gate;
- provide an exact reason and evidence basis for each proposal;
- preserve all safety, return, resource and planetary-protection boundaries.

No planetary mission, flight readiness, autonomous subterranean safety, geotechnical certification, habitat suitability, biosignature detection, or superiority over SKiD-SLAM/CERBERUS is claimed by this programme.

---

## References

1. Swinton, S. et al. (2026), *The Evolution of Autonomous Systems for Planetary Cave Exploration: A Review*, Journal of Field Robotics. https://doi.org/10.1002/rob.70233
2. Kim, H. et al. (2026), *SKiD-SLAM: Resource-Aware Distributed Multi-Robot LiDAR SLAM for Planetary Missions with Field Deployments at Analog Sites*. https://arxiv.org/abs/2610.38141
3. Mittelholz, A. et al. (2026), *Lava Tube Exploration with LunarLeaper*. https://arxiv.org/abs/2609.11453
4. Montañez Muñoz, A. et al. (2026), *Quantitative Morphometry of the Ape Cave Lava Tube*. https://doi.org/10.1029/2026JE010115
5. Carrer, L. et al. (2024), *Radar evidence of an accessible cave conduit on the Moon below the Mare Tranquillitatis pit*. https://www.nature.com/articles/s41550-024-02302-y
6. Carrer, L., Diana, E. & Bruzzone, L. (2026), *Radar-based observation of a lava tube on Venus*. https://www.nature.com/articles/s41467-026-68643-6
7. Sharma et al. (2026), *A potential cave candidate in the Elysium Mons region of Mars*. https://www.nature.com/articles/s44453-026-00039-8
8. *Robotic lava tube mapping and multimodal data collection using quadruped and LiDAR* (2026), Frontiers in Robotics and AI. https://doi.org/10.3389/frobt.2026.1855443
9. *A Survey on World Models for Physical AI: From Interaction to Real-World Deployment* (2026). https://doi.org/10.1007/s44163-026-02122-1
10. Li et al. (2026), *Active Visual Perception: Opportunities and Challenges*, Sensors 26(20), 6544. https://doi.org/10.3390/s26206544
11. *A Sensor-Centric Survey of Autonomous Driving: Integrating Measurement Physics, Uncertainty Modeling, and Safety-Critical Multi-Sensor Fusion* (2026). https://doi.org/10.3390/s26123801
12. *Potential caves on Mars: From orbital evidence to subsurface exploration* (2026). https://www.sciencedirect.com/science/article/pii/S2950616626000410
