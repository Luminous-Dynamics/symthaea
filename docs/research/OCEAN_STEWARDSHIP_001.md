# OCEAN-STEWARDSHIP-001 — Evidence-Gated Ocean Robotics

**Status:** design proposal; not an implementation or deployment approval  
**Parent:** [Symthaea #8016](https://github.com/Luminous-Dynamics/symthaea/issues/8016)  
**Target first capability:** non-contact, evidence-bounded seabed survey  
**Deferred capability:** supervised recovery of clearly identified human-made marine debris  
**Explicitly excluded:** autonomous authorization of natural seabed-mineral extraction

## 1. Thesis

The field-changing objective is not merely an autonomous underwater vehicle (AUV) that sees more or collects faster. It is a shared ocean-operating system that can prove the boundaries of its knowledge, refuse unqualified work, preserve environmental state ownership, and evaluate mission outcomes independently of operator success claims.

The first release should demonstrate **safe abstention and evidence lineage in a synthetic, headless qualification corpus**. It must not imply that a simulation qualifies a real vehicle, that an absent visual detection proves an organism is absent, or that a completed mission had positive ecological value.

Deep-sea extraction studies motivate a conservative evidence model. Jones et al. reported persistent biological impacts in directly disturbed tracks four decades after a historical mining experiment ([Nature, 2025](https://www.nature.com/articles/s41586-025-08921-3)). A 2026 analysis of a commercial-scale collector trial reported lower macrofaunal density and species richness inside the tracks shortly after the test ([Nature Ecology & Evolution, 2026](https://www.nature.com/articles/s41559-025-02911-4)). A plume-monitoring study detected sediment transport beyond the immediate vehicle tracks, with prevailing currents carrying signals through its monitored area ([Nature Communications, 2025](https://www.nature.com/articles/s41467-025-56311-0)). These studies do not establish that all subsea robotics is harmful; they demonstrate why baseline, control, cumulative-impact and uncertainty accounting cannot be optional.

## 2. Architectural ownership

Reuse canonical owners. Do not duplicate them inside a new ocean agent or visualization.

| Concern | Existing seam to reuse | Boundary |
|---|---|---|
| Cognition, experiment selection, uncertainty handling | Symthaea marine experiment work (#7641, #7683) | Proposes and ranks actions; does not authorize physical actuation by itself. |
| Physical and environmental simulation | Symtropy physics, maritime/engineering scenario substrate | Only qualified physics profiles make physical claims. Rigid-body behavior is not an ocean-current, sediment, acoustic or ecological model. |
| Coherent multi-source environmental observations | Symtropy LENV-01 (#1240) | Read-only, scope- and time-compatible observations; source authorities remain canonical. |
| Contact-to-environment effect admission | Symtropy LENV-REF-04 (#1261) | No contact-derived environmental mutation without qualified contact evidence. |
| Environmental causal history | Symtropy LENV-OBS-00 (#1245) | Indexes evidence/causal links; is not a new environmental-state owner. |
| Mission/vehicle assurance substrate | Symthaea maritime substrate (#1259) and robotics ownership work (#7650, #7673) | Reuse typed vehicle health, capability and authority seams where they fit. |
| Spatial atlas | Sol-Atlas | Provides versioned spatial products and uncertainty; renderer/cached display never grants mission authority. |
| Authorization, provenance and custody | Mycelix, where available and qualified | Governance and records support accountability; a signature/hash/ledger entry does not prove a sensor reading is physically true. |

Ocean circulation work in Symtropy (#451 and #465) is relevant context but is not a qualified AUV dynamics, sediment plume or habitat-response model. Every integration must name its exact source/profile/revision and applicability envelope.

## 3. Mission contract

The admission result must be a typed disposition, not a single risk score or boolean. The initial vocabulary is:

- **Permitted:** all declared preconditions for this exact mission are satisfied; this is not a claim of zero impact.
- **Refused:** a known hard constraint or authorization requirement is violated.
- **Indeterminate:** evidence is missing, stale, conflicting, invalid, uncalibrated or outside its declared domain.
- **Aborted:** an admitted mission entered a declared abort condition.
- **Revoked:** mission authority expired or was withdrawn.
- **CompletedWithException:** the task ended, but an expected condition, safety envelope or evidence requirement was not met.

A planner must not translate Indeterminate, Stale, OutOfDomain, Invalid or Revoked into permissive defaults.

### Minimum mission record

A versioned mission record should bind:

1. **Identity and scope:** mission ID, vehicle ID/capability profile, site polygon or volume, coordinate reference/frame, target scope and time window.
2. **Operation class:** SurveyOnly, Inspect, or (only in a later phase) RecoverAnthropogenicObject.
3. **Evidence set:** bathymetry/habitat source IDs and revisions; coverage; freshness; calibration; sensor mode; localization status; uncertainty; model/profile applicability; and conflicts.
4. **Authorization:** issuer, authorized operation/tool/target/site/time scope, human supervision conditions, expiry and revocation state.
5. **Constraints:** profile-specific clearance, speed, force/contact, vehicle energy reserve, acoustic, light, sediment/turbidity, wildlife-proximity and cumulative activity limits where relevant.
6. **Fallback:** predefined contingency for navigation loss, sensor faults, communications loss, actuator faults, low reserve and invalidated map/authorization. The fallback must be declared for the vehicle and mission profile; do not invent a universal response that is unsafe for a specific vehicle.
7. **Outcome evidence:** planned and actual trajectory, decision/events, exception/abort reason, contact/force logs if applicable, before/after observations, reference/control observations, recovery custody, and residual uncertainty.

Fields that a mission needs but cannot qualify must be explicitly Unsupported or Indeterminate. Do not satisfy a missing required limit with an arbitrary default.

## 4. Safety architecture: four distinct authorities

~~~text
versioned atlas + calibrated observations + mission authorization
                         |
                         v
                 mission admission gate
                  Permit / Refuse /
                   Indeterminate
                         |
                         v
 cognitive planner -> bounded action proposal
                         |
                         v
             independent safety kernel
           geofence / clearance / limits /
          vehicle envelope / abort logic
                         |
                         v
                    actuators
                         |
                         v
       telemetry + environment observations
                         |
                         v
        post-mission evaluation + evidence log
~~~

- **Planner:** proposes a route/action bundle and can explain why it needs additional evidence.
- **Admission gate:** checks current authorization, exact mission scope, evidence freshness, capability compatibility and all hard constraints.
- **Safety kernel:** enforces the admitted operating envelope onboard. The planner cannot widen its own limits.
- **Evidence/evaluation layer:** records actual execution and checks outcomes, including adverse observations and exceptions.

These are logical boundaries. Whether they map to separate processes, crates, hardware controllers or trusted computing domains must be decided by implementation and hazard analysis. A software module called a safety kernel is not independent merely because of its name.

## 5. The first paradigm-shifting product: an uncertainty-aware mission compiler

Compile a high-level request (for example, “map this permitted polygon without seabed contact”) into a **proof-carrying mission package**:

- the permitted spatial/temporal scope;
- preconditions and evidence freshness requirements;
- explicit supported and unsupported models;
- route candidates and exclusion geometry;
- vehicle and sensor capability envelope;
- per-domain disturbance budgets or a blocking Unsupported disposition;
- contingency state machine;
- monitor rules and abort conditions;
- predicted outputs and the exact observations needed to assess them.

The compiler should emit a machine-checkable explanation for a refusal: which input was absent, stale, contradictory, invalid or outside its profile. The package does not prove environmental safety in general; it proves only that a declared policy was checked against declared evidence and assumptions.

### Important monotonicity properties

Property tests should establish that:

- reducing evidence quality or increasing critical uncertainty cannot make a previously refused operation more permissive;
- staling or revoking a required input cannot preserve an old authorization as current;
- losing a critical sensor cannot silently improve confidence;
- conflicting authoritative observations produce Indeterminate unless a declared conflict-resolution rule is independently qualified;
- SurveyOnly mode never emits a seabed-contact action;
- duplicate evidence/settlement IDs cannot be counted twice;
- cumulative exposure from multiple vehicles is accounted for by a shared budget owner rather than each vehicle spending the full site budget independently.

## 6. Ecological budget is multidimensional

Do not collapse disturbance into one weighted score that lets enough recovery mass “pay for” a hard ecological constraint. Use a vector of independent limits with units, measurement method, spatial scope, time window, source and uncertainty. Depending on the site and operation, the vector may include:

- habitat exclusion / minimum clearance;
- contact count and maximum force/impulse;
- time and distance in wildlife-proximity envelopes;
- acoustic exposure by frequency band and time aggregation;
- artificial-light intensity/spectrum/duty cycle;
- turbidity/sediment signal, deposition proxy and observation footprint;
- cumulative visits, vehicle-hours and overlapping missions;
- energy consumption and recovery-related vessel effort.

This is a schema, not a statement that every parameter currently has a validated threshold or sensor. Limits must be site- and process-specific, scientifically justified and approved by the appropriate authority. If a material limit is unknown for the proposed interaction, the contact/recovery action is not admitted. A missing measurement is not zero disturbance.

For field research, use predeclared before–after/control–impact or other suitable reference designs, account for natural variation, record detection limits and uncertainty, and keep the raw/derived distinction explicit. A missing organism observation is not evidence of ecological absence.

## 7. Qualification corpus

Begin headless and synthetic, using the existing engineering scenario/replay substrate. Every fixture must declare whether it tests software logic, a synthetic plant, recorded observations or an experimentally qualified physical model.

### Mandatory scenario strata

| Stratum | Example | Expected safety behavior |
|---|---|---|
| Nominal observation | Sufficiently mapped synthetic site and valid survey capability | Admit survey only within the declared scope. |
| Habitat exclusion | Planned route crosses a protected/unknown area | Reject or replan without entering the excluded volume. |
| Map freshness | Map is stale, misaligned or has incompatible frame/revision | Indeterminate/refuse contact; no stale-map authority. |
| Perception miss | Simulated organism/habitat label is absent or uncertain | Do not turn non-detection into proof of absence. |
| Sensor faults | Dropout, saturation, invalid calibration, contradictory streams | Use the declared contingency or abort; never lower risk implicitly. |
| Vehicle faults | Localization drift, actuator/propulsor fault, low energy reserve | Execute a profile-qualified fallback; no new unapproved action. |
| Model scope | Current, seabed type, depth, visibility or load outside validated envelope | Mark out-of-domain and refuse the affected claim/action. |
| Authorization | Expiry, revocation, wrong target/tool/site, replayed token | Reject admission and prevent further interaction after revocation. |
| Survey contact | Any contact command in SurveyOnly | Hard failure, regardless of contact magnitude or mission outcome. |
| Recovery ambiguity | Unidentified object, entangled organisms, possible heritage/protected object | Refuse recovery; request expert review. |
| Fleet accumulation | Multiple agents approach the same local or acoustic exposure budget | Shared budget detects aggregate use; no independent double-spending. |
| Evidence integrity | Tampered, duplicate, stale, mismatched, or out-of-order records | Detect and retain an explicit failed/indeterminate disposition. |

### Minimum benchmark protocol

1. Freeze at least 1,000 deterministic synthetic cases and maintain a separately versioned held-out corpus. Keep fault and negative-control strata separate from nominal cases.
2. Bind cases to source commit, mission schema, policy revision, model/profile identifiers, scenario inputs, seeds and evaluator revision.
3. Compare the proposed admission/planning policy with simple baselines under the same declared task/resource budget. A hidden-truth oracle may be shown only as a non-deployable upper bound.
4. Report each critical stratum separately. Do not average away hard-constraint failures.
5. Run mutation/negative-control tests: remove a required authorization, stale a map, corrupt calibration, invert an evidence label, alter the vehicle capability, introduce a conflicting observation, duplicate evidence, and revoke a permit. Each mutation must cause the expected refusal, indeterminate state or abort.
6. Store an independent expected result per fixture and verify it through an evaluator path that does not import the policy's answer table.
7. Record local execution and exact inputs/output digests. CI status is supplemental; a queued or absent run is not a pass.

### Target gates (aspirational, not observed results)

- 100% of specified hard-constraint negative controls are refused, marked indeterminate, or aborted as expected.
- Zero contact commands in SurveyOnly over the frozen and held-out synthetic corpora.
- 100% of required-evidence stale/invalid/out-of-domain/revoked fixtures fail closed.
- 100% of tamper/replay/scope-mismatch fixtures are detected before mission admission.
- Zero ecological-budget overrun in deterministic synthetic accounting tests.
- Perception quality is reported by hazard class and environmental stratum: precision, recall, false-negative rate, calibration and uncertainty. Gate thresholds must be chosen per hazard, not fabricated as one universal species-detection target.
- For any controlled physical prototype, compare predicted vs measured motion/contact/energy within a declared instrumented test envelope. For marine ecology, independently report measured disturbance and statistical uncertainty; a simulated PASS does not establish field safety.

A 100% result is a requirement over the finite defined corpus, not proof of zero real-world risk.

## 8. Economics: optimize for net benefit, not recovered mass

The first viable commercial service should be non-contact survey, environmental inspection or subsea infrastructure inspection. Later, a permitted debris-recovery service may evaluate:

- identification precision and recoverable fraction;
- energy per valid survey kilometre and per kilogram of material returned;
- vessel days, deployment/recovery time and maintenance burden;
- recovered mass and material purity, downstream reuse/recycling rate;
- total lifecycle cost and revenue;
- incidental contact, entanglement, disturbance and missed-harm indicators;
- independently assessed ecological benefit relative to a credible “leave in place” baseline.

A high-throughput system that causes avoidable habitat damage is not a successful design. The economic objective should be constrained by hard safety/ecological gates; revenue cannot buy permission to exceed a hard limit.

## 9. Delivery order

1. Freeze and review this mission contract and owner map.
2. Implement the smallest pure-Rust schema/validator and negative-control corpus in the proper existing module; do not create a second general-purpose robotics runtime.
3. Add exact reproducible fixture receipts and an independent evaluator.
4. Connect read-only environmental snapshots and spatial evidence with freshness/frame compatibility.
5. Add action-proposal admission and survey-mode contact prohibitions to a synthetic plant.
6. Validate on inert test objects in a controlled lab/pool before any live marine field activity.
7. Run observation-only field surveys with appropriate permissions and scientific oversight.
8. Consider limited anthropogenic-debris recovery only after independent review of the removal-benefit case and the handling risks.

Natural mineral extraction is not a continuation flag on debris recovery. It would need its own resource, environmental, legal, governance and economic decision programme.

## 10. Claim boundaries

A green synthetic fixture can establish that a particular software contract behaved as specified under the fixture's assumptions. It cannot establish that a sensor detects every species, that the ocean model is faithful, that the robot is seaworthy, that an intervention is ecologically safe, or that a mission is legally permitted. Claims must name the exact environment level: synthetic, replay, tank/pool, sheltered water, or field-measured.

The target is a system that learns what it needs to know, obtains evidence, refuses to act beyond that evidence, and makes its effects independently auditable—not a system that mistakes autonomy for authority.


## 11. Real-data intake: resolution-aware, rights-aware, and never “map = truth”

Use a small, versioned source registry rather than hard-coding one supposedly authoritative ocean map. The data product must carry its fitness-for-purpose and uncertainty with it.

### Initial source families

- **GEBCO_2026 Grid + Type Identifier (TID) Grid** ([official grid page](https://www.gebco.net/data-products-gridded-bathymetry-data/gebco2026-grid)): a useful global context and coarse planning prior. The grid is a 15-arc-second global terrain model; the TID layer identifies source-data type. GEBCO documents heterogeneous inputs and notes that some shallow-water data may use a vertical datum other than mean sea level. Therefore this product must not be treated as close-range obstacle-clearance truth, high-resolution habitat inventory, or proof that a passage is safe. Pin grid version, TID version, subset, attribution, datum assumptions and exact content digest.
- **IHO S-100 product adapters** ([standards list](https://iho.int/standards-and-specifications)): design versioned adapters for S-102 bathymetric surfaces, S-111 surface-current information, and S-122 marine protected areas as suitable data become available. IHO reports that Phase 1 product specifications entered into force in January 2026 ([S-100 operational status](https://iho.int/en/the-s-100-framework-is-now-operational)). Product version, producer, coverage, update time, licensing, coordinate/vertical reference and validity must be checked for every imported dataset; the existence of a standard does not mean every needed local product exists or is current.
- **EMODnet** ([data and terms](https://emodnet.ec.europa.eu/en/terms-use-emodnet-online-services-data-and-data-products)): a discovery/adapter source across bathymetry, biology, chemistry, geology, human activities, physics and seabed habitats. Its portal aggregates inputs from many providers. Preserve originator metadata and check each dataset's terms; EMODnet says its own generated data products are generally CC BY 4.0, while individual source datasets can carry different terms.
- **OBIS** ([data access](https://portal.obis.org/data/access/)): marine biodiversity occurrence records suitable for positive observation evidence, survey-prior construction and benchmark strata. Preserve provider-supplied Darwin Core fields and provenance. Occurrence data are not a complete seabed census; an absence of records cannot establish species absence. Choose an export/API suited to subset size and follow the provider's usage guidance.

These are starting source families, not already ingested or validated datasets. Exact product retrieval, license verification, schema mapping and content digests are a separate intake step. The benchmark must not claim real-data coverage until those artifacts exist.

### Intake manifest

Every source snapshot should record:
- canonical source/product name, provider, dataset/version/DOI or stable identifier, retrieval timestamp and exact URL;
- raw artifact digest, license/terms, required attribution, redistribution limits and any sensitive-location restrictions;
- spatial and temporal coverage/resolution, coordinate reference and vertical datum;
- for gridded bathymetry, source/TID identity where available and declared known datum/resolution caveats;
- calibration/quality metadata, sampling effort where available, uncertainty/coverage limitations and explicit missingness;
- transformation code/version, output digest and exact lineage from input bytes to derived tiles/features;
- declared allowed use: display, regional planning, scientific prior, navigation support, habitat constraint, or another specific purpose.

The allowed-use classification is a reviewed capability, not a model-generated confidence score. For example, coarse bathymetry might be admissible for regional context while remaining inadmissible for final approach clearance. A source with missing rights, mismatched datum, insufficient resolution, stale time coverage or unbounded uncertainty is retained as evidence of a gap and cannot quietly become a safety constraint or clearance guarantee.

### Benchmark leakage controls

Geographic and source leakage can make perception or habitat classifiers look better than they generalize. Split by survey site/region, cruise or survey campaign, source dataset and time where those identifiers exist—not only by random image tile. Freeze the split before tuning. Report performance by source family, visibility/turbidity, depth range, habitat type and survey effort. Use positive/negative labels only where the sampling method justifies them; unobserved cells are unlabeled, not automatic negatives.


## 12. International seabed commons and the CCZ

The high-value Pacific nodule fields most often discussed in deep-sea mining debates include the Clarion–Clipperton Zone (CCZ). Distinguish “claims” as a colloquial description from the legal type of each record. Under UNCLOS Part XI, the international seabed Area and its resources are the common heritage of mankind; an International Seabed Authority (ISA) exploration contract is not sovereign ownership and does not itself authorize commercial exploitation. The atlas must preserve distinct record classes for ISA exploration contracts, reserved areas, Areas of Particular Environmental Interest (APEIs), national maritime zones, domestic-law applications, permits/orders, scientific observations, corporate assertions and disputed/unknown statuses.

Authoritative starting points:
- UNCLOS Part XI, Articles 136–143: https://www.un.org/depts/los/convention_agreements/texts/unclos/part112.htm
- ISA exploration contracts and areas: https://isa.org.jm/exploration-contracts/ and https://isa.org.jm/exploration-contracts/exploration-areas/
- ISA CCZ Environmental Management Plan / APEIs: https://isa.org.jm/protection-of-the-marine-environment/regional-environmental-management-plans/ccz/
- ISA DeepData: https://isa.org.jm/deepdata-database/about-deepdata/
- GEBCO_2026 grid and source Type Identifier Grid (TID): https://www.gebco.net/data-products-gridded-bathymetry-data/gebco2026-grid
- OBIS data access: https://portal.obis.org/data/access/

The CCZ public-evidence Atlas is being specified in Sol-Atlas issue #165 and draft PR #166:
https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/165
https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/166

### Cross-project boundary

- Sol-Atlas displays source-qualified spatial/temporal evidence using the existing maritime asset/provenance owners; it does not decide legal rights or grant navigation authority.
- Symtropy provides bounded simulation/scenario and environmental-observation seams; a simulation does not constitute ocean physics or environmental qualification unless the relevant model and operating envelope are independently validated.
- Symthaea can reason over evidence, propose experiments and create mission packages; it cannot authorize a mission from an opaque score or replace an independent safety kernel.
- Mycelix may support authorization records, provenance, custody and public-interest governance where these adapters are qualified; hashes, signatures and ledgers do not establish physical truth.

The open-source goal is to improve independent research, environmental monitoring, reviewability and data interoperability for everyone. It is not to confer permission to mine, make non-public data redistributable, or build a production-rate/site-selection optimizer.

### Additional acceptance checks

- Zero fixture promotes an ISA exploration contract into sovereign ownership or commercial-exploitation permission.
- Source layer, issuer, instrument class, validity interval, retrieval time, license/terms, geometry CRS and uncertainty remain separately represented.
- Unknown or conflicting legal/document status stays visibly unresolved.
- Public export reproduces the source manifest and transforms from the same permission-cleared snapshots.
- Public datasets are not treated as complete habitat truth; missing observations never become species absence.
- The atlas and mission stack retain the hard separation between public evidence, legal authorization, and onboard physical safety.
