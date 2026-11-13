# Climate Mobility and Destination Service-Capacity Protocol v0.1

**Status:** Research draft; synthetic-contract scope only  
**Owner:** Symthaea scenario orchestration, with domain-owned inputs from Symtropy, Sol-Atlas and Mycelix  
**Claim ceiling:** This protocol does not forecast real migration, determine who should move, approve relocation, or qualify any city as safe.

## 1. Purpose

Connect climate-risk scenarios to the capacity of origin and destination communities to sustain essential services and preserve meaningful choices.

The target is not to predict a single number of "climate migrants." It is to evaluate how plausible climate, livelihood, infrastructure and governance futures alter a community's feasible options:

- remain in place with adaptation;
- move temporarily and potentially return;
- migrate voluntarily;
- participate in a community-led planned relocation;
- experience displacement after a shock;
- remain because movement is constrained, unsafe, unaffordable or unwanted.

Voluntary immobility and constrained immobility must remain distinct. Neither a climate hazard nor an observed movement is sufficient to attribute an individual's or household's motive to climate change.

The model is a decision-support and scenario-comparison instrument. It must never become an automated relocation, admission, exclusion, land-allocation or resource-denial authority.

## 2. Why this is needed

Climate-related mobility is shaped by interacting environmental, economic, political, cultural, demographic and institutional conditions. The IPCC finds that climate hazards can drive displacement directly and indirectly through livelihood degradation; it also finds that housing, infrastructure, water, schools and health services in receiving locations shape outcomes. It further distinguishes deliberate immobility from immobility caused by lack of agency.

The World Bank's Groundswell II estimate of up to 216 million internal climate migrants by 2050 across six regions is a scenario result, not a deterministic forecast for a particular city. A 2023 review of 30 climate-migration forecasting models describes the field as immature and warns that data gaps and omitted non-climate drivers make numerical projections speculative. IOM's 2026 World Migration Report chapter emphasizes locally grounded, disaggregated evidence and local/community action.

This protocol therefore starts with transparent scenarios, local capacity constraints and honest unknowns. It does not turn global projections into local population targets.

## 3. Paradigm shift

**From forecasting people as a flow to evaluating the capability of places and institutions to preserve safe choices.**

A conventional mobility forecast asks: "How many people will leave this origin, and where will they go?"

This programme must additionally ask:

1. Which services and livelihoods are becoming less reliable, for whom, and with what evidence?
2. What adaptation could make staying viable without imposing unacceptable ecological or financial costs?
3. What would make temporary movement, voluntary migration or return safer and more feasible?
4. Which potential receiving locations have sufficient *additional* service capacity, not merely nominal capacity?
5. What happens to existing residents if demand rises?
6. Which assumptions are uncertain, contested, missing or outside the model's validity?
7. Who has authority to choose, contest, correct or reject a proposed intervention?

A receiving place is not "available" merely because land is mapped or a population forecast is positive. Water, sanitation, housing, energy, health, transport, livelihoods, ecological limits, tenure, rights and local consent are non-substitutable constraints.

## 4. System boundaries and ownership

| System | Owns | Must not silently own |
|---|---|---|
| **Symthaea** | Scenario composition, causal hypotheses, alternative futures, mobility-pathway comparisons, uncertainty-aware explanation and abstention | Source observations, physical solver truth, community consent or relocation authority |
| **Symtropy** | Specialist physical models and hazard/infrastructure trajectories; solver validity, conservation/budget checks, impact-chain qualification and replay receipts | Human migration motives or rights decisions |
| **Sol-Atlas** | Spatial identities, catchment/city/district geometry, CRS/datum, spatial source registry, exposure overlays, destination constraints and spatial provenance | A model's physical validity or a legal determination of land suitability |
| **Mycelix** | Federated observations, consent/authority records, community proposals, water stewardship decisions, resource commitments, disputes and auditable coordination | Self-authorizing water rights, journals, relocation, or physical operational commands |

Integrations must use stable, versioned contracts. Each owner retains its own validation, security and claim boundaries.

Existing work to compose:
- Symthaea Living City context and service floors: https://github.com/Luminous-Dynamics/symthaea/issues/7686
- Symtropy natural-hazard programme: https://github.com/Luminous-Dynamics/symtropy/issues/2232
- Symtropy Johannesburg hazard integration cell: https://github.com/Luminous-Dynamics/symtropy/issues/2267
- Sol-Atlas Highveld resilience binding: https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/630
- Sol-Atlas water–power stress sweeps: https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/147
- Mycelix scenario evidence contract: https://github.com/Luminous-Dynamics/mycelix/issues/5417
- Mycelix South African source-vintage acquisition: https://github.com/Luminous-Dynamics/mycelix/issues/5382

This issue owns the mobility-to-destination-capacity contract only. It must reuse the above owners rather than create parallel hazard, spatial, water, service-floor or governance engines.

## 5. Scenario contract

A scenario must be immutable by identity and version and must include:

### 5.1 Spatial and temporal frame
- Origin and destination area identifiers, boundary revisions, CRS, horizontal/vertical datum where relevant, resolution and aggregation method.
- Valid time, observation time, publication time, horizon, timestep and time-basis semantics.
- Explicit distinction between local, municipal, catchment, bulk-supply and national-system boundaries.

### 5.2 Environmental forcing
- Hazard family and exact source/model identity.
- Drought, heat, flood, coastal, fire, ecosystem or other forcing only where a suitable specialist model exists.
- Units, baseline, scenario pathway, ensemble member, quantile, temporal window, uncertainty and validity envelope.
- No climate index presented as a complete water balance or hazard model.

### 5.3 Population and mobility evidence
- Aggregated population counts or distributions with source vintage, spatial resolution, uncertainty, coverage and lawful-use decision.
- Observed departures, arrivals, temporary displacement, return or other mobility events with event definition and observation completeness.
- Voluntary survey/community evidence only under explicit consent, purpose limitation and withdrawal rules.
- No individual-level location tracking, inferred climate motive, identity-proxy prediction or hidden demographic imputation.
- Missing reports must remain missing, not be interpreted as zero movement or zero impact.

### 5.4 Socioeconomic and institutional drivers
- Livelihood and employment conditions, service access, housing/tenure, transport connectivity, social protection and relevant policy/operational context.
- Each variable must be sourced and typed as observed, estimated, modelled, synthetic or unknown.
- Governance or conflict variables must not be inferred from protected identity characteristics.

### 5.5 Adaptation and mobility pathways
Each scenario declares one or more alternatives, such as:
- in-place adaptation;
- temporary displacement with return conditions;
- voluntary migration;
- community-led planned relocation;
- post-disaster recovery;
- no feasible movement / constrained immobility.

A pathway is an analytical alternative, not a recommendation or assigned fate. The evaluator reports feasibility, constraints, service outcomes and uncertainty without selecting people or compelling movement.

## 6. Destination service-capacity contract

For every origin and potential destination area, evaluate an explicit service vector. Do not collapse it into one "carrying capacity" score.

Minimum domains:

| Domain | Candidate measures | Required treatment |
|---|---|---|
| Drinking water | Reliable volume, continuity, pressure, quality compliance, source diversity, storage/autonomy, affordability | Separate raw-water availability, bulk treatment, distribution and household access |
| Sanitation | Safe collection, treatment capacity, effluent quality, overflow and sludge handling | Include downstream receiving-water constraints |
| Housing | Habitable and affordable units, tenure security, exposure and access | Land area alone is not housing capacity |
| Energy | Reliable service, critical-load support, pump/treatment energy and backup | Model water-energy dependencies explicitly |
| Health and care | Reachable service capacity, surge capability and access barriers | Preserve lower-tail and underserved-area results |
| Transport | Essential journey accessibility and route disruption | Model access and evacuation as separate questions |
| Livelihoods | Job access, income opportunities and service-worker capacity | Do not assume population growth creates employment automatically |
| Ecology and land | Catchment limits, protected areas, flood exposure, water-source protection and ecosystem flows | Regional screening cannot be promoted to parcel approval |
| Governance and rights | Local decision authority, consent, appeal, tenure and lawful allocation | Human/community decisions remain authoritative |

Each service domain declares:
- the service-floor rule and its source/authority;
- the population or community unit to which it applies;
- demand assumptions and uncertainty;
- physical and institutional dependencies;
- time-to-degradation and recovery;
- evidence needed to classify a result as meets, breaches or indeterminate.

An unknown threshold, missing observation or unqualified model must produce **indeterminate**, not "meets floor." Aggregate city averages cannot hide a known community-level floor breach. The v0.1 synthetic fixture checks aggregate capacity sufficiency/shortfall only; it does not define or evaluate a protected minimum-service floor. Capacity sufficiency is not a floor-pass, safety or destination-suitability determination.

## 7. Water-security requirements

Water is a coupled catchment-to-tap service, not a reservoir-volume scalar.

The water profile must distinguish:
1. catchment inflows, drought and climate variability;
2. surface water, groundwater, environmental flows and inter-basin transfers;
3. reservoir/storage state and operating rules;
4. bulk abstraction, treatment and transfer;
5. municipal distribution, pump energy, pressure, leakage and continuity;
6. drinking-water quality and contamination response;
7. wastewater collection, treatment, reuse and receiving-water quality;
8. household/community access, affordability and coping capacity.

Use established specialist models as references or adapters where appropriate. EPANET is a mature drinking-water distribution hydraulic and water-quality model; WNTR is an EPANET-compatible tool for resilience analysis and disruptive incident/recovery scenarios. Symtropy's natural hydrology and surface-water authority must remain distinct from a pressurised distribution-network solver. Do not rewrite EPANET hydraulics just to obtain a Rust-native label.

Water accounting must use multiple indicators. Non-revenue water (NRW) alone cannot separate physical leakage, apparent/commercial losses and authorised unbilled consumption. Use a complete water balance, infrastructure leakage index (ILI), pressure/continuity, quality, energy, affordability and service-floor measures together.

### Protected service-floor semantics

A service-capacity calculation and a protected minimum-service-floor assessment are different claims. A capacity surplus is not proof of delivered service, legal compliance, safety, affordability or destination suitability.

Every floor assessment must bind:
- the exact pathway and area/service-scope identity it qualifies, with the assessment invalidated if either changes;
- an exact, versioned floor profile and the responsible authority/reference;
- a named aggregate service scope (such as a declared community cell), metric and unit;
- the exact observation bundle and its validity/coverage;
- the observation-valid-from and observation-valid-to timestamps, the assessment timestamp, and a maximum observation-age budget supplied by the versioned policy; missing timing or stale observations remain indeterminate, future/inverted windows are rejected;
- a conservative observed lower/upper interval;
- covered and total units for the declared scope;
- the evidence class, qualification level and limitations.

Use non-compensatory evidence semantics:
- **Meets floor:** full declared-scope coverage and the observed lower bound is at least the authorised threshold.
- **Breached floor:** full declared-scope coverage and the observed upper bound is below the authorised threshold.
- **Observed breach:** the covered subset is definitely below the threshold, but coverage is incomplete; do not generalise to the whole scope.
- **Indeterminate:** evidence, authority, threshold, unit, validity or coverage is incomplete, timing/freshness is missing or stale, the uncertainty interval crosses the threshold, or the evidence/qualification is only synthetic or contract-only. Synthetic fixtures may test the logic but cannot produce a real floor outcome.

Unknown thresholds are never guessed from a global benchmark. Where the applicable legal or community-approved threshold is not confirmed, the result remains indeterminate. The v0.1 fixture uses arbitrary synthetic units and is not a South African water-rights or minimum-service determination.

## 8. Benchmark protocol

All benchmark cases must be versioned, content-addressed, comparable and independently reproducible. Separate synthetic contract mechanics, analytic/reference correctness, published benchmark agreement, held-out local validation and prospective operational evaluation.

### 8.1 Mobility forecast benchmarks

Targets must be well-defined aggregates (for example, origin–destination counts per time interval, displacement events by hazard/time/area, or arrival volumes). Use:
- MAE and RMSE for count/flow errors;
- WAPE only when denominator and zero-flow handling are explicit;
- Brier score and reliability diagrams for event probabilities;
- CRPS or other proper scoring rules for predictive distributions;
- empirical coverage and sharpness for declared prediction intervals;
- precision/recall and false-alarm/miss rates for event detection;
- lead time, detection delay and missingness coverage where event observations support them.

Do not use MAPE blindly on sparse/zero flows. Publish sample counts and uncertainty around every score.

### 8.2 Baselines and ablations

On identical frozen cases, compare against a deliberately broad, low-complexity-first ladder:
1. last-observation persistence and recent-flow/seasonal averages;
2. ARIMA and Bayesian ARIMA where the series length and stationarity/seasonality support them;
3. population-and-economic-trend baseline without climate inputs;
4. gravity and radiation models for spatial allocation, not as assumed temporal forecasters;
5. multiplicative-component/spatial-structure models where the origin–destination structure is stable;
6. gradient-boosted or other machine-learning baselines only after the data volume and leakage controls justify them;
7. climate-only and service-disruption-only ablations;
8. integrated climate, services, livelihoods and policy-context model.

A 2026 evaluation of 21 Australian internal-migration forecasts across five method families found no single method consistently best on bias, accuracy and empirical coverage; simple models performed as well as or better than more complex ones in several comparisons. This is not a climate-migration benchmark, but it is a warning against assuming that additional model complexity guarantees better forecasts. Therefore, the integrated model must beat the strongest task-appropriate simple baseline on preregistered, held-out cases and must not worsen calibration or lower-tail service outcomes.

Use rolling-origin temporal evaluation and region/event holdouts. No random row split where nearby times or places leak the same event into train and test. Any calibration set must be separate from evaluation data.

### 8.3 Water and infrastructure benchmarks

Start with analytic and manufactured cases, then matched external references:
- water/volume balance closure and boundary-flux reconciliation;
- hydraulic head/pressure and flow residuals against EPANET/reference outputs;
- tank/storage trajectory, pump energy and operating constraint checks;
- contaminant transport, water age and quality observables where input data permit;
- leak/pipe/pump/valve failure and recovery trajectories;
- service-floor breach duration, unmet demand and recovery-time bounds;
- distributional service availability, not only network-wide averages.

Use the published benchmark case's exact topology, units, controls, demand patterns, initial/boundary conditions, time grid and expected tolerances. Do not claim a fair comparison if the equations, solver scope, data or spatial resolution differ.

### 8.4 Integrated decision-quality metrics

- Worst-case and lower-tail essential-service outcomes across the declared scenario set.
- Community-level unmet demand and floor-breach duration.
- Number and severity of indeterminate assessments.
- Pareto/regret comparison across cost, service continuity, equity, ecological limits and emissions.
- Robustness under correlated failures; never assume independent hazards or multiply marginal probabilities without evidence.
- Reproducibility, runtime, memory, output size, model-call count and observation freshness.

No single score may compensate for an essential-service floor breach or a rights violation.

### Water digital-twin maturity and operating-value qualification

Do not call a static model, GIS layer or dashboard a digital twin. Use an explicit maturity ladder:

1. **Digital model:** versioned representation; updates are manual or scenario-driven.
2. **Digital shadow:** qualified physical observations update the digital representation, but there is no demonstrated bidirectional state/command loop.
3. **Synchronised digital twin:** the physical and digital states are time-bound and traceably synchronised; bidirectional exchange and feedback are demonstrated under an approved operational authority, with safe fallback and audit.

A June 2026 systematic review of 48 water digital-twin studies found that urban water infrastructure dominates the studied applications; distribution networks have the strongest share of operational real-world deployments, while natural-system applications are more often conceptual or simulation-based. The review identifies data availability, integration, institutional fragmentation, cybersecurity, cost-benefit evidence and social equity as persistent gaps. This is a maturity map, not a head-to-head performance benchmark.

For each maturity claim, record:
- telemetry source and exact valid/observed/received times;
- synchronization latency distribution and stale-data behavior;
- missingness, quality, calibration and out-of-range handling;
- physical-to-digital state reconciliation and residuals;
- whether any digital-to-physical action path exists, who authorizes it and the independent safety interlock;
- cyber boundary, identity, least privilege, audit, recovery and offline behavior;
- interoperability round-trip loss and owner-of-record for each field;
- lifecycle cost and value per verified unit of service delivered.

A simulation-only or data-poor pilot can be valuable; it must be labelled as a model or shadow, not as an operational twin. The first Johannesburg demonstrator should begin read-only and shadow-mode. No actuation, abstraction, valve, pump, rationing or relocation authority is introduced by this programme.

## 9. Proposed qualification gates and targets

These are **proposed research targets**, not existing results or universal standards:

1. **Contract correctness:** 100% of frozen synthetic known-answer cases pass; every declared adversarial case is rejected or remains indeterminate as specified.
2. **No false-safe promotion:** zero cases with missing required evidence, stale inputs, invalid units/CRS, out-of-domain models or known floor breaches are reported as verified-safe.
3. **Reproducibility:** byte-stable canonical outputs for frozen deterministic inputs; all input/model/config/output revisions and digests retained.
4. **Forecast calibration:** on sufficiently large, held-out probabilistic corpora, nominal 90% intervals should achieve empirical coverage within five percentage points of nominal, with uncertainty intervals published. If sample size cannot support this test, report insufficient evidence.
5. **Baseline challenge:** the integrated model must beat the strongest simple baseline on pre-registered held-out tasks with uncertainty intervals, while not worsening declared tail-risk or calibration measures. A point-estimate win is not enough.
6. **Spatial generalisation:** hold out at least one area/event family from model development before claiming transfer beyond the calibration region.
7. **Service-floor preservation:** no candidate is called robustly better when any protected service floor has a possible regression under the declared comparison rule.
8. **Data governance:** zero individual tracking or climate-motive inference; 100% of non-public observations have a documented purpose, authority, access class and retention rule.
9. **External validation:** no operational claim until an independent domain owner reviews data, model validity, uncertainty, failure modes and the decision context.

The first benchmark should be small and interpretable. Scale only after the reference cases, evidence ledger and holdout protocol are correct.

## 10. Candidate South African pilot

Use the existing Johannesburg/Highveld work as the first integration environment, not as a claim of national representativeness.

Recommended boundary hierarchy:
- a bounded Johannesburg sub-catchment or service area;
- the local municipal distribution and wastewater service boundary, where authoritative data are lawfully available;
- the relevant bulk-supply and Integrated Vaal River System dependency;
- candidate receiving-community areas only when local population, housing, service and consent evidence is adequate.

The distinction matters: South African government reporting in February 2026 attributed Gauteng service interruptions to a combination of ageing infrastructure, high demand, leaks, power interruptions affecting pumping and maintenance backlogs. Regional dam storage alone therefore cannot certify local tap-level reliability.

Prioritise public, rights-cleared source candidates and record exact vintages. Current source leads include:
- DWS Blue Drop, Green Drop and No Drop releases and municipal performance records;
- Stats SA sub-national water resource accounts (clearly retaining their experimental status and catchment coverage);
- City of Johannesburg climate action and spatial-risk material, subject to reuse/access verification;
- CHIRPS rainfall and other climate products with exact product/version/coverage metadata;
- WRI Aqueduct 4.0 for regional prioritisation only, not as a locally validated operational model;
- IOM DTM and IDMC disaster-displacement records for mobility context, preserving observation coverage and the distinction between weather-related displacement and climate-attributed migration;
- official population, housing and service datasets with geographic and temporal limitations recorded.

Do not ingest proprietary operational feeds or sensitive utility geometry without permission and security classification. Do not infer zero incidents from absent records.

## 11. First implementation sequence

**P0 — Contract and evidence**
- Add versioned scenario and mobility-pathway schema.
- Add service-capacity vector and protected-floor contract.
- Add evidence-state and abstention rules, reusing owner-native evidence contracts.
- Add synthetic fixtures and negative controls for missing data, unsupported climate attribution, involuntary immobility, capacity double-counting, mismatched geography/time and service-floor masking.

**P1 — Analytic baselines**
- Implement deterministic, auditable persistence/gravity/radiation count baselines only where the required inputs exist.
- Add small synthetic origin–destination and capacity-constrained toy cases with hand-calculated answers.
- Add rolling-origin and spatial-holdout evaluation harness specifications.

**P2 — Owner adapters**
- Sol-Atlas: spatial IDs, boundary and source-vintage binding, destination constraints.
- Symtropy: qualified hazard, water and infrastructure trajectories.
- Mycelix: consented observations and community decisions; no physical command authority.
- Symthaea: scenario orchestration, alternative comparison, uncertainty and explanations.

**P3 — Johannesburg/Highveld retrospective pilot**
- Acquire and license-check sources before ingestion.
- Select a bounded event and time interval only after verifying compatible observations.
- Establish a population/service baseline and data-coverage report.
- Compare specialist model outputs and mobility baselines on held-out periods/areas.
- Convene local water, planning, community and technical reviewers before any prospective use.

**P4 — Prospective evaluation**
- Shadow-mode only initially.
- No operational action or allocation from model output.
- Record forecast, uncertainty, human decision, observed outcome and later revisions for independent evaluation.

## 12. Required negative controls

At minimum:
- Global or regional climate-risk score cannot be promoted to parcel-level safety.
- Weather-related displacement cannot be relabelled as climate-caused migration without a justified attribution method.
- A missing movement report cannot be scored as zero migration.
- A person/community staying cannot automatically be labelled resilient or safe.
- A community choosing to stay cannot be overridden by an optimizer.
- Lack of money, legal status or transport cannot be treated as voluntary immobility.
- Destination land area cannot substitute for water, housing or service capacity.
- Nominal reservoir volume cannot substitute for delivered potable service.
- NRW cannot be used as a proxy for physical leakage without the supporting water balance.
- Scenario count cannot be converted into event probability.
- An aggregate city average cannot mask a protected community service-floor failure.
- A synthetic profile cannot be promoted to observed or calibrated evidence.
- Stale, conflicting, out-of-domain or unlicensed source records cannot improve the confidence class.

## 13. Claim ceiling

A passing synthetic evaluator demonstrates contract and arithmetic behavior on its frozen fixtures only. Agreement with EPANET/WNTR or another reference establishes matched-case consistency, not general real-world accuracy. A retrospective pilot establishes only the stated region, time window, variables, data vintage and holdout scope. Community legitimacy, human rights, legal compliance, engineering safety, public-health safety and operational utility each require their own accountable review and evidence.

**The success condition is not "the model predicts migration." It is that a community can inspect the assumptions, preserve its decision rights, compare credible adaptation choices, see where essential services fail, and obtain independently checkable evidence before a decision is made.**

## 14. Research update — 13 November 2026

Newer evidence strengthens the Johannesburg pilot rationale while reinforcing the need to keep service, mobility and rights claims separate:

- Johannesburg Water's 29 October 2026 statement reports high demand and excessive draw-off limiting reservoir/tower recovery, low pressure and intermittent/prolonged outages in multiple named supply systems. This is an operator statement about system conditions and active interventions; it is not a customer-level measured time series.
- The South African Human Rights Commission's 2 November 2026 statement on its Gauteng water-services inquiry identifies interconnected failures across the water-service value chain, including ageing infrastructure, losses, maintenance, storage/pumping capacity, electricity, wastewater treatment, skills and finance. It also warns against normalising recurring tanker delivery as an inferior permanent service.
- SAnews reported on 3 November 2026 that DWS warned of El Niño-related 2026/27 summer pressure, heat-driven demand, and high Gauteng consumption and loss context. This is a risk/stress context, not a quantified local inflow forecast for a particular sub-catchment.
- A September 2026 systematic review of 244 climate-(im)mobility articles finds persistent disciplinary separation, quantitative emphasis on large-scale movement, underrepresentation of internal/intra-regional mobility and local context, and weaker treatment of immobility and lived agency in predominantly quantitative studies.
- The OECD's March 2026 migration forecasters' toolkit says gravity models can explain some historical flows but typically struggle with trend breaks and forced displacement; it cautions against treating a model family as a universal temporal forecaster.
- A June 2026 systematic review of 48 water digital-twin studies finds the strongest operational maturity in distribution networks, with many natural-water-system applications still conceptual/simulation-led; data integration, cybersecurity, cost-benefit evidence and social equity remain significant gaps. This motivates an explicit model → shadow → synchronised digital-twin maturity ladder rather than premature twin claims.
- The Johannesburg Water customer-notices archive exposes six dated pages (29 October–5 November) with 25 image attachments now visually transcribed into a metadata/evidence register in [Mycelix PR #6110](https://github.com/Luminous-Dynamics/mycelix/pull/6110). The transcribed operator notices include planned reservoir closures, burst-pipe repair notices with named areas, reservoir/tower status and system-recovery snapshots, plus one separate infrastructure-security notice.
- Several image notices report a citywide system-capacity sequence of 37% (1 November baseline), 45% (3 November), 52% (4 November) and 50% (5 November). These values are **operator-reported context**, not independent telemetry. The common calculation method, measurement coverage and sensor/asset aggregation have not been acquired; do not fit a trend, interpret the small differences as measured dynamics, or compare them as a calibrated time series until the metric definition and source data are obtained.
- A source-specific event catalogue can now be built from reported notice times and named locations, but the notices are not proof of customer-level delivery or outage duration. Planned closure windows are distinct from actual restoration times, and missing ETR remains unknown rather than imputed.

Pilot implication: treat Johannesburg's near-term problem first as a measurable delivered-service and infrastructure-reliability problem. Mobility research should consume those qualified service trajectories as one input, while separately incorporating agency, livelihoods, housing, rights and receiving-place capacity. Do not conflate the 2026 pressure episode with a climate-attributed migration event.
- Siriban et al., *A review and evaluation of internal migration forecasting models* (Population Studies, published online 26 May 2026; 21 Australian bilateral-flow forecasts across five method classes): https://doi.org/10.1080/00324728.2026.2645544
- OECD, *Migration Anticipation and Preparedness* (2026), forecasters' toolkit: https://www.oecd.org/en/publications/2026/03/migration-anticipation-and-preparedness_c7c13bc4/full-report/forecasters-toolkit-choosing-the-right-model-for-the-task_af4817f2.html
- Systematic review of 244 climate-(im)mobility studies (18 September 2026): https://link.springer.com/article/10.1007/s11111-026-00535-9
- Qureshi et al., *A systematic literature review of digital twin applications for sustainable water systems in natural and built environment* (17 June 2026; 48 studies): https://link.springer.com/article/10.1007/s43621-026-03825-2
- Johannesburg Water, customer notices archive (six dated pages; 25 image-backed notice records visually transcribed into the metadata register in Mycelix PR #6110; underlying image bytes/digests are not pinned): https://www.johannesburgwater.co.za/media/media-statement/customer-notices/
- Mycelix Johannesburg Water source/evidence register, draft PR #6110: https://github.com/Luminous-Dynamics/mycelix/pull/6110
- South African Human Rights Commission, Gauteng water-services inquiry release (2 November 2026): https://sahrc.org.za/index.php/sahrc-media/news-2/item/4568-media-statement-the-south-african-human-rights-commission-releases-an-inquiry-report-highlighting-neglected-water-and-sanitation-infrastructure-and-heavy-reliance-on-road-based-water-distribution-water-tankers-in-gauteng
- SAnews / Department of Water and Sanitation, El Niño and water-conservation statement (3 November 2026): https://www.sanews.gov.za/south-africa/dws-urges-responsible-water-use-amid-el-ni%C3%B1o-threat
- Johannesburg Water, urgent interventions statement (29 October 2026): https://www.johannesburgwater.co.za/blog/2026/10/29/johannesburg-water-implements-urgent-interventions-to-stabilise-severely-constrained-systems/

## References

- IPCC AR6 WGII, Technical Summary — migration, displacement, immobility and planned relocation: https://www.ipcc.ch/report/ar6/wg2/chapter/technical-summary/
- IPCC AR6 WGII, Summary for Policymakers — urban/settlement adaptation: https://www.ipcc.ch/report/ar6/wg2/chapter/summary-for-policymakers/
- World Bank, Groundswell II (2021): https://www.worldbank.org/en/news/press-release/2021/09/13/climate-change-could-force-216-million-people-to-migrate-within-their-own-countries-by-2050
- IOM, World Migration Report 2026, Chapter 6 — local and community action: https://publications.iom.int/books/world-migration-report-2026-chapter-6
- IOM DTM, Who are Climate Migrants? data update (2025): https://dtm.iom.int/dtm-insights/january-2025-edition/data-update-climate-who-are-climate-migrants
- Zickgraf et al., review of climate-migration forecasting models (2023): https://www.frontiersin.org/journals/climate/articles/10.3389/fclim.2023.1189125/full
- WRI, Aqueduct 4.0 methodology and limitations: https://www.wri.org/research/aqueduct-40-updated-decision-relevant-global-water-risk-indicators
- US EPA, EPANET: https://www.epa.gov/water-research/epanet
- US EPA, WNTR documentation: https://usepa.github.io/WNTR/
- South African Government, Gauteng water challenges (13 February 2026): https://www.gov.za/news/media-statements/gauteng-provincial-government-addresses-water-challenges-13-feb-2026
- South African DWS, 2026 Budget Speech water security statement: https://www.dws.gov.za/Communications/MinisterSpeeches/2026/speech%20-%20Budget%20speech%20May%202026.pdf


