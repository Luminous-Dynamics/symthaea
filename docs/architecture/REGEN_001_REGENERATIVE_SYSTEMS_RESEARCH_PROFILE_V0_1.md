# REGEN-001 — Regenerative Systems Research Profile v0.1

Status: research architecture draft; recommendation-only; no physical authority.

## 1. Purpose

This profile defines how Symthaea should reason about terrestrial regenerative systems without becoming another agricultural database, another climate authority, another resource ledger, another industrial-control system, or another Mycelix domain.

The first proving family is a community soil loop that composes:

```text
qualified biomass
    -> pyrolysis / compost / co-compost
    -> soil amendment
    -> crop / soil / water observations
    -> updated evidence
```

with optional useful-heat recovery and explicit waste/coproduct handling.

The profile is intentionally broader than biochar. The long-run objective is a compositional regenerative-resilience model spanning food, soil, water, biomass, nutrients, energy, repair, manufacturing, ecology, logistics, and continuity.

## 2. Governing theorem

```text
Symthaea model output
!= environmental observation
!= adopted policy
!= institutional authority
!= physical action
```

The research system may generate evidence-bearing predictions and recommendations. It does not gain execution authority from intelligence, confidence, model fit, prior success, or integration with Mycelix.

## 3. Compose; do not duplicate

REGEN MUST reuse or bridge existing Symthaea substrates where their semantics already apply.

Relevant existing foundations include:

- Planetary Industrial Ecology (PIE) process/material/utility/circularity semantics;
- generic resource model, hierarchy, resource quality, and routing;
- thermofluids and heat-grade reasoning;
- energy/grid/storage models;
- engineering habitat composition and failure laboratories;
- discovery contracts, uncertainty, experiment selection, and Pareto analysis;
- scenario-outcome vectors;
- continuity/civilization-capability architecture;
- operating envelopes / operating authority boundaries;
- world-interface / evidence ingress boundaries.

REGEN MUST NOT create parallel foundational concepts when a narrow adapter can preserve the stronger existing theorem.

## 4. Relationship to PIE

PIE already provides an important process-ecology separation between resource occurrence, actual material lots, processes, utility demands, equipment requirements, explicit output disposition, and recycle edges.

REGEN SHOULD begin as a terrestrial projection/adapter around those semantics rather than immediately refactoring PIE.

The first bridge should prove compatibility for terrestrial regenerative flows such as:

```text
biomass lot
    -> pyrolysis process
    -> biochar lot
    -> soil amendment process

organic residues
    -> compost process
    -> compost lot
    -> soil amendment process
```

and should preserve explicit non-product outputs.

Only after the adapter demonstrates repeated structural parity should a later PR consider extracting a body-neutral shared industrial-ecology kernel.

## 5. Relationship to resource-quality routing

Quantity conservation is insufficient for regenerative engineering.

Examples:

- heat at the wrong temperature cannot satisfy a process merely because joules exist;
- water of the wrong quality cannot satisfy potable or irrigation requirements merely because litres exist;
- biomass with unknown contamination cannot satisfy clean-feedstock requirements merely because kilograms exist;
- biochar with one quality profile does not automatically satisfy a different soil/application profile.

REGEN SHOULD therefore project physical flows into qualified resource semantics wherever the existing resource-quality graph can express the requirement.

## 6. Relationship to continuity

REGEN is a natural domain consumer of civilization-continuity semantics.

A regenerative capability should preserve:

```text
CapabilityDefinition
!= CapabilityRealization
!= CurrentAvailability
!= VerifiedSufficiency
!= Authority
```

Examples:

```text
knowledge of composting
!= locally realizable compost process
!= currently available compost process
!= enough compost capacity for adopted service needs
!= authority to operate a facility
```

A later continuity bridge may bind regenerative capabilities to exact local realizers such as machines, tools, operators, skills, materials, energy, water, knowledge, and safety evidence.

## 7. No universal resilience score

Symthaea MUST NOT collapse regenerative resilience into a canonical scalar fitness.

At minimum, analyses should be able to preserve distinct axes such as:

- essential nutritional service;
- potable water service;
- irrigation reserve;
- soil-condition trajectory;
- nutrient dependency;
- energy dependency;
- sustainable biomass supply;
- seed / propagation capability;
- equipment repairability;
- skill bottlenecks;
- external logistics dependency;
- ecological condition;
- recovery frontier;
- uncertainty.

Pareto or plural-outcome methods are preferred when alternatives trade off these axes.

## 8. No universal local-production objective

Symthaea MUST NOT treat locality as inherently better.

The model should be capable of concluding any of the following:

```text
local production improves resilience
local production is neutral
local production worsens resilience
external supply is currently superior
hybrid local/external supply is superior
```

The result depends on explicit evidence, service floors, failure domains, costs, ecological burden, skills, energy, repair, and recovery pathways.

## 9. Ecological-retention constraint

Candidate biomass availability is not equal to gross occurrence.

The research layer SHOULD model at least the conceptual separation:

```text
gross occurrence
- ecological retention
- erosion / soil-cover requirement
- habitat requirement
- existing priority use
- contamination exclusion
= candidate recoverable biomass
```

Where the quantities are unknown, the model should return uncertainty or infeasibility rather than assuming zero ecological retention.

Resource occurrence does not imply permission to harvest.

## 10. Research claim firewall

REGEN freezes these non-equivalences:

```text
biochar quality
!= soil suitability

soil suitability
!= expected crop response

expected crop response
!= observed field outcome

observed field outcome
!= causal attribution

agronomic benefit
!= carbon removal

carbon removal
!= whole-system ecological benefit
```

A later model may estimate relationships among these propositions, but must retain which proposition is actually evidenced.

## 11. Recommended first modeling objects

The initial Symthaea-side research vocabulary SHOULD stay narrow and reference Mycelix identities rather than duplicate their authoritative records.

Candidate objects include:

```text
RegenerativeStudyContext
RegenerativeMaterialRef
TerrestrialProcessProjection
BiomassAllocationCase
PyrolysisAccountingCase
CompostAccountingCase
NutrientBalanceCase
SoilResponsePrediction
WaterBalancePrediction
CropServicePrediction
RegenerativeScenario
RegenerativeRecommendation
```

Each object should state whether it is:

- descriptive;
- projected;
- predicted;
- scenario-only;
- recommendation-only.

None should contain execution authority.

## 12. Observation ingestion

Mycelix environmental/planetary observations should enter Symthaea through an explicit adapter/world-interface boundary.

Parsing or receiving an observation does not prove physical truth.

The adapter should preserve at least:

- observation identity;
- evidence class;
- units;
- uncertainty;
- spatial support;
- temporal support;
- provenance / lineage identity;
- freshness/currentness semantics where relevant.

Symthaea SHOULD NOT silently convert missing uncertainty into certainty or missing measurements into zero.

## 13. Biomass-allocation screen

The first analytical kernel after the terrestrial PIE projection should be a biomass-allocation screen.

It should accept explicit candidate biomass and competing obligations, and produce a conservative candidate allocation without assuming every residual stream should be pyrolyzed or composted.

Example obligations may include:

- retain on soil;
- animal bedding/feed where appropriate;
- habitat/ecology;
- existing industrial use;
- contamination exclusion;
- compost pathway;
- pyrolysis pathway;
- direct material reuse;
- unknown/undetermined.

The screen MUST NOT rank an ecologically prohibited allocation into admissibility through soft preferences.

## 14. Pyrolysis accounting

The first pyrolysis model should be an accounting/screening model, not a universal reactor simulator.

It should make explicit:

```text
feedstock mass / properties
+ process assumptions
+ energy input
-> char output
+ gas/vapor/liquid coproducts
+ useful heat candidate
+ losses / unknowns
```

Unknown coproduct disposition remains explicit.

Useful heat is not automatically useful; it must satisfy quantity, temperature/grade, timing, and route constraints.

The model must not claim equipment safety, emissions compliance, product certification, carbon permanence, or field efficacy.

## 15. Compost and co-compost accounting

Compost/co-compost models should distinguish:

- input material identity;
- water / aeration / process assumptions;
- mass changes;
- nutrient changes where modeled;
- maturation/stability evidence;
- output product identity;
- uncertainty.

Elapsed time alone should not become stability or safety evidence.

A biochar-compost mixture is a new process/product state; it should not silently inherit all claims from either input.

## 16. Nutrient accounting

A first nutrient kernel should make carbon, nitrogen, phosphorus, and potassium flows explicit where evidence supports them.

The primary theorem is conservation/accounting discipline, not detailed biogeochemistry.

```text
input nutrient stock
+ qualified transformations
- exported / lost / immobilized stock
= downstream accounted stock
```

A recovered nutrient unit cannot satisfy multiple simultaneous claims.

Unknown availability/bioavailability MUST remain distinct from total elemental presence.

## 17. Soil-response shadow model

The first soil-response model should be explicitly shadow / advisory.

It may estimate outcome distributions or intervals for dimensions such as:

- pH;
- water retention;
- soil carbon;
- nutrient availability;
- crop response;
- other measured local endpoints.

It MUST condition predictions on context and preserve uncertainty.

It MUST be possible for two soils, crops, climates, feedstocks, or process configurations to produce materially different predicted responses.

A model that always predicts `biochar positive` is not an acceptable target.

## 18. Water-balance model

The first water model should remain a transparent stock-and-flow screen around:

- starting soil water;
- rainfall / supplied water;
- irrigation;
- storage;
- evapotranspiration assumptions;
- drainage / runoff assumptions;
- soil-water retention parameters;
- crop demand where modeled.

It should expose uncertainty and missing terms rather than manufacture closure.

Detailed hydrology remains outside the first tranche.

## 19. Crop/nutrition service projection

Crop resilience is not only harvested mass.

A later service projection SHOULD be able to distinguish:

- harvested mass;
- edible fraction;
- storage loss;
- spoilage;
- nutritional contribution;
- seasonal availability;
- seed/propagation retention;
- water/energy/labor dependencies.

The output is a service projection, not a food-safety or medical-nutrition claim.

## 20. Heat-coupling bridge

REGEN should reuse the existing resource-quality/thermofluids stack for useful heat.

A pyrolysis process may produce a candidate thermal stream. The bridge must still establish that a destination can accept it under explicit grade and capacity constraints.

```text
heat exists
!= compatible sink exists
!= route exists
!= service delivered
```

This should compose with the same quality-aware routing used by Compute Commons rather than creating a regenerative-only heat network.

## 21. Settlement metabolism graph

Only after the individual kernels qualify should Symthaea compose them into a settlement-scale regenerative graph.

Candidate nodes may include:

- fields / greenhouses;
- water storage / treatment;
- compost;
- pyrolysis;
- food storage / processing;
- energy generation/storage;
- workshops;
- repair capability;
- waste/material handling;
- community-service loads.

The graph should preserve each subsystem's authoritative owner and use bridges for projections.

It must not become a second execution/control plane.

## 22. Resilience and shock campaigns

Regenerative resilience should reuse the semantics already demonstrated by Symthaea's resilience/shock work:

- finite reserves;
- explicit essential/nonessential services;
- common-mode failures;
- site interruptions;
- transport loss;
- real spares;
- repair delays;
- explicit service floors;
- worst-case summaries without invented probabilities.

Candidate regenerative shocks include:

- drought;
- grid outage;
- irrigation-pump failure;
- external fertilizer loss;
- crop failure;
- seed loss;
- compost-system failure;
- pyrolysis equipment failure;
- contaminated feedstock;
- transport interruption;
- simultaneous heat/water stress.

Scenario inputs are not predictions of frequency.

## 23. Failure-domain analysis

Nominal redundancy is not sufficient.

Two food systems may share the same irrigation pump. Two water systems may share the same grid feed. Two repair paths may depend on the same specialist. Two feedstock streams may depend on the same transport corridor.

REGEN analyses should preserve typed common-mode dependencies rather than counting raw component quantity as resilience.

## 24. Recovery frontier

A later continuity bridge should be able to ask:

```text
which essential regenerative services are currently below obligation?
which capability prerequisites block them?
which alternative realization paths exist?
which restoration actions unlock the largest downstream region?
which paths preserve ecological and human-rights floors?
```

The answer remains advisory.

A recovery frontier is not a work order, purchase authorization, labor assignment, or machine command.

## 25. Experiment intelligence

Only after deterministic baselines and evidence contracts qualify should REGEN consume Symthaea discovery machinery.

A candidate experiment loop is:

```text
qualified study context
    -> bounded candidate treatments
    -> predicted outcomes + uncertainty
    -> expected-information analysis
    -> recommendation-only trial proposal
    -> external/human authorization
    -> executed trial outside Symthaea authority
    -> observations
    -> updated evidence/model
```

The system should prefer uncertainty-reducing experiments where appropriate rather than maximizing one favored treatment outcome.

## 26. Treatment-generation boundary

Candidate generation must remain inside explicit bounds.

A treatment generator should not invent:

- unqualified chemicals;
- unbounded application rates;
- unsupported contamination assumptions;
- prohibited ecological extraction;
- authority to apply a candidate.

Candidate generation is not field authorization.

## 27. Heterogeneous-response analysis

A core research objective is detecting context dependence.

Analyses should preserve variation by factors such as:

- soil class;
- starting pH;
- climate;
- water regime;
- crop;
- feedstock;
- process configuration;
- amendment composition;
- application rate;
- time since application.

If results vary materially across contexts, the model should report that heterogeneity rather than promoting one global recipe.

## 28. Causal evidence boundary

Observed differences are not automatically causal.

The architecture should preserve:

```text
observed association
!= controlled comparison
!= replicated effect
!= generalizable causal claim
```

Where causal identification is weak, output language and types should remain correspondingly weak.

## 29. Negative-result retention

Symthaea research tooling MUST treat null/adverse results as evidence.

A model-selection or recommendation pipeline must not silently discard cases where a favored regenerative intervention:

- has no effect;
- worsens one axis;
- helps one soil and harms another;
- improves agronomy but worsens energy/ecology/cost;
- improves carbon-related metrics while worsening food/water service.

## 30. Model-vs-baseline qualification

For every more complex REGEN model, first establish a deterministic transparent baseline.

Examples:

- biomass allocation: rule-based conservative baseline before learned ranking;
- soil response: simple explicit regression / interval model before HDC/learned model;
- experiment selection: fixed or random comparator before EIG;
- scenario ranking: Pareto baseline before learned policy.

A more complex model should be promoted only after measured improvement against the frozen baseline under an explicit endpoint set.

## 31. Cross-repository identity

Mycelix owns canonical shared/evidence subject identities. Symthaea adapters should mirror only the minimum DTO/framing required for interoperability and should independently verify any identity commitment they rely on.

JSON or transport encoding is not automatically canonical identity.

A cross-repository REGEN golden-vector tranche should bind at least:

- schema revision;
- domain separator;
- canonical subject identity;
- test vectors;
- rejection of drift.

Neither repository should silently change the shared contract without breaking the vector.

## 32. Recommendation contract

A Symthaea regenerative recommendation should preserve at least:

- study/scenario identity;
- candidate/treatment identity;
- model identity;
- evidence roots;
- assumptions;
- outcome vector;
- uncertainty;
- hard-constraint result;
- comparison baseline;
- explicit authority classification.

The required authority classification for early REGEN work is:

```text
RecommendationOnly
```

## 33. No physical control in REGEN-001

This profile authorizes no:

- irrigation control;
- pump control;
- fertilizer dosing;
- pyrolysis control;
- compost process control;
- vehicle/robot control;
- amendment application;
- crop treatment;
- equipment operation.

A future physical-action bridge requires its own safety and authority theorem and must not rely on model correctness as a security assumption.

## 34. First proving scenario

The first end-to-end proving scenario should be a synthetic community soil loop:

```text
qualified biomass residue
        |
        v
bounded pyrolysis process
      /   \
 biochar   candidate heat
    |           |
    v           v
compost /   qualified heat sink
co-compost
    |
    v
soil treatment + control
    |
    v
repeated observations
    |
    v
prediction-vs-observation comparison
```

It should establish only that the system can:

- preserve material identity;
- conserve modeled mass/nutrients/energy at its declared fidelity;
- preserve unknown coproducts;
- reject contaminated/unqualified feedstock cases;
- preserve control/treatment distinction;
- represent null/adverse outcomes;
- produce recommendation-only analysis.

It should not claim a universal agronomic result.

## 35. Independent-oracle policy

For the first integrated accounting scenarios, implementation-independent small oracles SHOULD be preferred for key conservation and boundary semantics.

Candidate oracle subjects include:

- bulk mass accounting;
- C/N/P/K stock-and-flow accounting;
- simple water balance;
- thermal/energy accounting;
- biomass allocation eligibility;
- deterministic shock campaign.

The oracle should remain smaller than the production model and import no production Symthaea implementation code.

## 36. Immediate tranche sequence

The first Symthaea work should remain narrow:

```text
REGEN-001  this profile
    |
    +--> REGEN-030 terrestrial PIE projection
    |
    +--> REGEN-031 biomass-allocation screen
    |
    +--> REGEN-032 pyrolysis accounting
    |
    +--> REGEN-033 compost/co-compost accounting
    |
    +--> REGEN-034 nutrient accounting
    |
    +--> REGEN-035 soil-response shadow model
    |
    +--> REGEN-037 heat-quality bridge
    |
    +--> REGEN-080 synthetic community soil-loop example
    |
    +--> REGEN-081 independent integrated accounting oracle
```

The exact ordering may branch where dependencies are independent, but REGEN-080 must not precede the conservation/evidence semantics it claims to compose.

## 37. Promotion gates

No REGEN production model should be called qualified merely because it compiles.

A stronger promotion ladder is:

```text
architecture boundary
-> authored implementation
-> exact-head compile/test/lint
-> deterministic synthetic corpus
-> independent oracle parity where applicable
-> adversarial / malformed cases
-> integrated synthetic scenario
-> frozen real-data protocol
-> observational/field evidence
-> replication in distinct context
```

Each rung proves only its declared proposition.

## 38. Deliberate non-claims

REGEN-001 does not establish:

- agronomic efficacy;
- biochar superiority;
- compost superiority;
- carbon removal;
- climate-credit eligibility;
- food safety;
- water safety;
- emissions compliance;
- equipment qualification;
- farm recommendations;
- community self-sufficiency;
- resilience superiority;
- causal generalization;
- physical-action safety;
- execution authority.

It defines only the research/composition boundary for future Symthaea regenerative-systems work.
