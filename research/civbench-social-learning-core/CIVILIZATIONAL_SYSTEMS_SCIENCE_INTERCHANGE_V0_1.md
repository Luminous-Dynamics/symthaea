# Civilizational Systems Science Interchange Contract v0.1

**Status:** cross-project research contract proposal.  
**Purpose:** provide one evidence-aware vocabulary for Symthaea, Symtropy, Mycelix and Sol-Atlas.  
**Non-goal:** this is not a universal civilizational ranking, a centralized authority over history, or a claim that extraterrestrial life/civilizations have been detected.

## 1. The scientific object

Civilization is not a single agent, country, culture, state, city or scalar complexity value. Model it as a time-varying, spatially situated, multi-level system comprising interacting:
- people, populations, communities and networks;
- households, firms, cooperatives, institutions and governance arrangements;
- knowledge, languages, practices, technologies and production capabilities;
- ecological systems, species, climate, land, water and material cycles;
- infrastructure, energy, food, health, transport and communication systems;
- exchanges, migration, conflict, cooperation, treaties and dependencies;
- planetary and astronomical environments.

A model may focus on one subsystem, but every result must declare the subsystem boundary and identify excluded interactions. Aggregation from agent → community → city → region → planet is a model choice, not a neutral database operation.

## 2. Evidence and hypothesis classes

Every factual assertion, measurement, inference or scenario must carry exactly one primary epistemic class:

| Class | Meaning | Example |
|---|---|---|
| `observed_measurement` | Instrumented or directly measured observation | Reservoir level on a dated observation |
| `archaeological_evidence` | Material evidence, with context and dating record | Settlement layer, artefact, isotope result |
| `primary_historical_source` | Contemporaneous written, visual or administrative source | Inscription, ledger, treaty |
| `oral_or_community_history` | Oral history or knowledge held by a community | Place-name tradition, oral chronology |
| `compiled_secondary_source` | Curated synthesis of prior evidence | Published historical synthesis |
| `model_inference` | Output or inference produced by a named model | Estimated population interval |
| `scenario_assumption` | Deliberate condition in a future or counterfactual scenario | Assumed storage capacity |
| `speculative_hypothesis` | Physically motivated but not observed hypothesis | Hypothetical non-terrestrial biosphere |
| `confirmed_detection` | Claim meeting a named, reproducible confirmation protocol | Reserved for independently confirmed detection |

These labels do not rank the value of cultures or knowledge traditions. A community's oral history is not demoted to a lower-truth category by default; its method, custodianship, scope and relationship to other evidence are recorded explicitly. Multiple sources may support, contradict or contextualize one claim.

An extraterrestrial inference must never silently advance through the ladder:
`astronomical_observation → candidate_anomaly → natural_explanations_tested → biosignature_hypothesis → technosignature_hypothesis → confirmed_detection`.
Each arrow requires a separate, pre-registered evidence and verification gate. A plausible model output is not an observation; an unexplained anomaly is not evidence of technology.

## 3. Canonical claim envelope

The first machine-readable draft is now provided in:

- `civilizational-claim-v0.1.schema.json` — immutable claim/source/time/space/missingness/rights envelope.
- `civilizational-model-run-v0.1.schema.json` — reproducibility receipt for model, forecast, hindcast, scenario and synthetic benchmark runs.

These schemas are cross-project data contracts, not runtime security or scientific-verification implementations. Each project should map the same semantics to its native types and add strict validators. Mycelix retains identity, provenance, rights, disputes and revocation; Symtropy binds model/parameter/solver/seed/data receipts; Sol-Atlas binds spatial-temporal geometry, units, layer source and uncertainty; Symthaea binds cognitive inference and skill/evaluation lineage.

## 3a. Canonical claim envelope

A claim is an immutable, addressable assertion linked to evidence, scope and interpretation. The following is the conceptual minimum; projects may serialize it in Rust structs, JSON, Holochain entries or columnar data, provided semantics remain equivalent.

```text
Claim {
  claim_id, schema_version,
  subject_ref, predicate, object_or_value,
  epistemic_class,
  valid_time, observation_time, publication_time,
  spatial_scope,
  measurement: { quantity, value, unit, uncertainty, method }?,
  sources: [SourceRef],
  derivation: [TransformationRef],
  model_ref?,
  alternative_interpretations: [ClaimRef],
  confidence_or_probability?,
  limitations,
  rights_and_access,
  supersedes?, disputes?
}
```

Required temporal fields:
- **valid time:** when the described condition/event applied;
- **observation time:** when it was observed or recorded;
- **publication time:** when the record became available.

Time may be an interval or probability distribution. Archaeological chronologies, uncertain historical dates and model-estimated intervals must not be silently rounded into exact timestamps.

Required spatial fields:
- geometry or geometry reference;
- coordinate reference system and units;
- spatial resolution and boundary source;
- whether scope is a point, site, catchment, network, region, planet or orbital frame.

Measurements require explicit units, uncertainty, method and source. Stocks and flows must remain distinct: stored water is not water throughput; installed power is not delivered energy; resource reserves are not annual production.

## 4. Missingness and uncertainty semantics

The following are distinct states and must not be collapsed:
- `measured_zero`;
- `not_observed`;
- `unknown`;
- `not_applicable`;
- `withheld_by_custodian`;
- `unavailable_from_source`;
- `invalid_or_rejected`.

Absence of a record is not evidence of absence unless the observation process had sufficient coverage and sensitivity. Every model result must preserve uncertainty, alternative explanations, measurement error and known blind spots. A single confidence scalar is insufficient where uncertainty is multidimensional.

## 5. Entity and relationship vocabulary

Minimum entity kinds:
`person`, `population`, `community`, `organization`, `institution`, `polity`, `settlement`, `city`, `region`, `ecosystem`, `species`, `technology`, `infrastructure_asset`, `resource`, `planetary_body`, `habitat`, `astronomical_source`, `hypothetical_society`.

Relationships are typed, directed where appropriate, time-bounded and sourced:
`trades_with`, `migrates_to`, `depends_on`, `governs`, `cooperates_with`, `conflicts_with`, `shares_knowledge_with`, `inherits_from`, `replaces`, `exchanges_resources_with`, `occupies`, `transforms_ecosystem`, `observed_by`.

Never infer a relationship merely from co-location or correlation. Record whether an edge is reported by a source, measured, inferred, simulated or assumed.

## 6. Model and scenario receipts

Every executable model, historical reconstruction, forecast, hindcast or simulation must declare:
- system boundary and excluded processes;
- model code/version and exact source digest;
- state variables, units, equations and update schedule;
- parameter values and prior/range, plus calibration data;
- initial and boundary conditions;
- stochastic seeds, solver/runtime and compute budget;
- data snapshots and all transformations;
- training/calibration/validation/held-out split identities;
- baselines and ablations;
- predictive distributions, calibration and residual diagnostics;
- known failures, out-of-domain cases and alternative models.

Forecasts are evaluated by their issue time and target horizon. No data published after the forecast origin may enter features, tuning, source selection or human adjudication. Hindcasts must distinguish genuine out-of-sample reconstruction from explanatory fit after the event.

Scenarios must expose their assumptions. They are not predictions unless a forecasting protocol has been declared and calibrated.

## 7. Ethical and governance requirements

1. **No civilizational league table.** Do not produce one scalar that ranks cultures, peoples, countries or species as more/less worthy, intelligent or evolved.
2. **Plural outcome reporting.** Report material sufficiency, ecological conditions, health, autonomy, knowledge, resilience, institutional adaptability and distributional outcomes separately.
3. **Community authority.** Oral histories, sacred or restricted knowledge, indigenous data and sensitive local records require custodial rights, consent/participation conditions and access controls. Public availability does not automatically grant ethical permission to repurpose.
4. **No deficit inference.** Do not infer intelligence, institutional quality or moral worth from poverty, energy use, GDP, language, religion, ethnicity or state capacity.
5. **Reversibility.** Retractions, corrections, disputes, key revocations and source restrictions must propagate to derived claims and scenario outputs without erasing the audit trail.
6. **No automated rule over people.** Simulation outputs do not authorize governance actions or speak for a population; affected communities retain decision rights.

## 8. Cross-project responsibility split

| Project | Primary responsibility | Must not claim |
|---|---|---|
| **Symthaea** | Cognitive learning, skill discovery, social learning, evidence-aware reasoning, uncertainty and human/agent decision models | That an embedding, consensus or packet receipt proves truth, understanding or collective consciousness |
| **Symtropy** | Reproducible multi-scale dynamics: agents, institutions, infrastructure, ecological networks, resources, energy and physical constraints | That an uncalibrated simulation predicts real civilizational outcomes |
| **Mycelix** | Source provenance, identity attestations, rights/consent, disputes, federated memory, governance receipts and inter-community agreements | That a signature proves a claim true or that a single organization represents humanity |
| **Sol-Atlas** | Spatial-temporal planetary/astronomical observations, resource and infrastructure layers, uncertainty visualization and scenario comparison | That a visual layer or map is an evidence verifier |

Cross-project data contracts should be versioned independently from any UI or physics implementation. Use existing Mycelix identity/evidence capabilities and existing Sol-Atlas geographic/resource layers rather than introducing duplicate ledgers or duplicate planetary catalogs.

## 9. Initial cross-project benchmark suite

### CSS-01: Historical evidence reconstruction
Select diverse cases across regions and social forms. Build claim graphs from source snapshots; predict a bounded outcome or state transition using only evidence available at a historical cutoff. Keep source, geography and period holdouts disjoint. Evaluate provenance coverage, chronology calibration, reconstruction error and contradictions found—not only narrative plausibility.

### CSS-02: Civilization–ecology hindcast
Couple population, food, water, energy, land, climate exposure, infrastructure and institutional response. Hindcast selected drought, flood, epidemic, conflict and energy-disruption episodes. Compare against persistence, stock-flow accounting, linear trend, system-dynamics and agent-based baselines. Validate conservation laws and uncertainty intervals before adding cognitive agents.

### CSS-03: Robust policy across futures
Evaluate policies across ensembles of climate, trade, demographic, energy, resource and governance conditions. Report regret, constraint violations, worst-decile outcomes, distributional effects and Pareto frontiers. Prefer policies robust across plausible futures over a single optimized forecast.

### CSS-04: Knowledge retention and institutional memory
Remove key agents, institutions or data centers; introduce contradictory records, migration, language change and memory pressure. Measure recovery time, false-memory rate, source traceability, lineage loss, redundancy and access by affected communities.

### CSS-05: Inter-community cooperation and conflict
Compare centralized, federated, polycentric and networked governance under shared-resource dilemmas, external shocks, asymmetrical information and treaty compliance. Measure negotiated outcomes, distribution, rights violations, defection recovery and concentration of decision power. Do not equate consensus with fairness.

### CSS-06: Planetary resource and infrastructure twin
Connect measured/estimated water, food, energy, materials, transport, housing, biodiversity and infrastructure data to explicit geographic and temporal scopes. Validate dimensional consistency, stocks/flows, conservation, source lineage and uncertainty before showing future build-outs.

### CSS-07: Extraterrestrial inference discipline
Create blinded synthetic signal suites with natural astrophysical sources, instrumental artifacts, human interference, biosignature candidates and technosignature injections. Measure sensitivity, false-positive rates, detection thresholds, independent reproducibility and alternative-explanation rejection. This is an inference benchmark, not a simulator that asserts alien societies exist.

## 10. Proposed engineering targets

These are proposed gates, not achieved results:
- 100% of published factual claims have a source or explicit `scenario_assumption`/`speculative_hypothesis` label.
- 0 temporal leakage in historical forecast/hindcast splits.
- 100% of numerical quantities include units and a known scope; unknown uncertainty remains explicitly unknown.
- 100% of simulation outputs bind to exact model, source snapshot, parameters, seed set, runtime and data-split identities.
- 100% of invalid dimensional/stock-flow fixtures are rejected by the verifier.
- 0 seeded natural/interference controls are accepted as technosignatures in the locked synthetic detection suite; report sensitivity separately.
- Every policy result includes a simple baseline, an ablation, uncertainty and distributional outcomes.
- No aggregate civilization score is used as a release gate.

## 11. First executable milestone

Use the Reward Network III CivBench as the first verified micro-benchmark for social skill acquisition, but do not treat it as a miniature universal civilization model. In parallel:
1. Symtropy specifies a small resource-constrained population model with conserved stock/flow accounting and reproducible shocks.
2. Mycelix defines a signed, disputable, rights-aware claim and evidence receipt, reusing existing identity/provenance work.
3. Sol-Atlas maps the same resource/assets to temporal and spatial scopes, with uncertainty and source links.
4. Symthaea supplies learning/coordination agents only after simple and non-cognitive baselines have been measured.

The first cross-project demo should answer one narrow question end-to-end: **under a specified water/energy shock, which resource-sharing and infrastructure policies keep essential services above declared floors, for whom, at what cost, and with what uncertainty?** It must expose assumptions and remain a scenario, not a prediction, until it passes hindcast calibration.
