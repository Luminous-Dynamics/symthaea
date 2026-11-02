# CSS-001: water-energy-food shock and essential-service allocation

**Benchmark version:** 0.1.0  
**Run kind:** synthetic benchmark  
**Scenario fixture:** CSS_001_WATER_ENERGY_SHOCK_MANIFEST_V0_1.json  
**Symtropy executable:** https://github.com/Luminous-Dynamics/symtropy/pull/3391  
**Coordination:** https://github.com/Luminous-Dynamics/symthaea/issues/10076

## 1. Research question

Under a temporary water-yield and electricity-supply shock, which inter-region allocation policies maintain essential water, energy and food services, for which populations, with what resource transfers, reserve consequences and uncertainty?

This is not a universal civilization-health score. It is a controlled, disaggregated test of resource allocation and resilience. The primary outputs are per-region/per-resource service coverage, unmet demand in native units, conservation residuals, and transfer receipts.

## 2. What is executable now

Symtropy's standalone Rust kernel models:
- water stocks in m³, energy stocks in kWh and food stocks in kg;
- population-scaled essential demand per tick;
- exogenous production;
- ordered inter-region transfer requests;
- unrestricted or opening-stock-reserve-limited transfers;
- service coverage and unmet demand, separately for each resource;
- beginning + production + imports − exports − served demand − ending conservation.

The CSS-001 runner adds three policy conditions over identical synthetic inputs:
1. **No transfer:** local stocks/production only.
2. **Fixed solidarity:** a constant north-to-south transfer each tick.
3. **Adaptive service deficit:** transfer only the receiver's current-tick essential-service deficit, bounded by donor stock after its own demand and a half-tick reserve buffer.

The adaptive policy computes a transfer from current state and the declared production/demand input. It does not observe future ticks. It is a simple deterministic controller, not a learned agent or general optimizer.

## 3. Synthetic scenario

The manifest contains two abstract regions, four daily ticks, and an exogenous production shock beginning at tick 2. All populations, stocks, demands and production are invented fixture values. The regions are not Johannesburg, South Africa, named catchments or real grids.

The scenario deliberately begins with enough aggregate resources to meet aggregate four-tick demand, but with an uneven spatial distribution. This creates a coordination/allocation problem rather than a global-scarcity problem. A model that reports only total stocks would miss the service failures in the receiving region.

### Analytical expectations (not yet an execution receipt)

The fixture is hand-calculated in the JSON manifest and is pinned by Rust tests in the Symtropy runner:
- no transfer should create four region-resource-tick breaches below the 0.90 service floor, all in the receiving region;
- fixed solidarity and adaptive service-deficit policies should keep all region-resource-tick coverage at or above 0.90;
- fixed solidarity should deliver 240 m³ water, 1,000 kWh energy and 320 kg food;
- adaptive service-deficit should deliver 154 m³ water, 760 kWh energy and 290 kg food.

Do **not** sum these transfer quantities into a single volume or efficiency number: m³, kWh and kg are dimensionally incomparable. Do not report these hand-derived expectations as a measured Rust run until the exact-head CI and captured program output verify them.

The adaptive controller is expected to use less transferred quantity than fixed solidarity in each resource for this particular fixture. This is a synthetic, scenario-conditional result; it does not establish policy dominance on other scenarios, catchments or real infrastructure.

## 4. Required metrics

Report, without collapsing dimensions:
- water/energy/food coverage for every region and tick;
- unmet water (m³), energy (kWh), and food (kg);
- beginning stock, production, imports, exports, essential service and ending stock;
- requested and delivered transfer separately for each resource;
- resource-specific conservation residual and pass/fail;
- count of region-resource-tick outcomes below the declared 0.90 service floor;
- minimum regional coverage and population-weighted coverage, separately by resource;
- donor closing stock and reserve buffer;
- controller decisions and the observations available when each decision was made.

A service-floor pass is a conditional outcome for the manifest's exact inputs. It is not proof of ecological sustainability, affordability, infrastructure feasibility or social legitimacy.

## 5. Paradigm shifts

### 5.1 From aggregate sufficiency to spatially distributed access

A system can have enough total water, energy or food and still fail to supply a particular region. Measure service accessibility and unmet demand at region/resource/time resolution.

### 5.2 From fixed allocation to state-contingent coordination

Compare static transfer rules with adaptive policies that use current stock, production and demand, while preserving explicit donor buffers and fail-closed conservation.

### 5.3 From optimizing one objective to constrained essential-service guarantees

First ask whether all regions meet declared essential-service floors. Then examine the trade-off surface across service coverage, resource transfers, reserve stocks and other measured consequences. Do not hide distributional harms in a weighted average.

### 5.4 From scenario narrative to reproducible experiment

Freeze scenario inputs, policy code, version, seed schedule, units, update ordering and boundaries. Give each policy the same initial state and exogenous forcing. Retain negative controls and ablations.

### 5.5 From assumed sustainability to explicit ecological constraints

The present ledger does not model river environmental flows, groundwater recharge, soil water, biodiversity, ecological thresholds, storage evaporation, conveyance losses or pumping energy. The opening-stock reserve is an operational buffer, not an environmental-flow standard. Add ecological constraints only with sourced thresholds, units, spatial/temporal scope and uncertainty.

### 5.6 From an agent's claim to rights-aware evidence

A measured observation, a historical report, a model inference and a scenario assumption are different epistemic objects. Mycelix should preserve origin, custody, licence/consent, correction and dispute metadata. A signature or ledger entry is not proof that a claim is true.

## 6. Relationship to current tools and SOTA

CSS-001 is deliberately not positioned as a replacement for established water or integrated-assessment models.

- **WEAP** is an established water-resources planning tool. Its integrated approach places demand-side patterns, efficiencies, reuse and allocation alongside supply-side streamflow, groundwater, reservoirs, transfers and ecosystem considerations. CSS-001 should compose with this class of model after its stock-flow kernel and data interfaces are qualified, not claim to replace it. Source: https://www.sei.org/tools/weap/
- **ISIMIP4** launched in August 2026 with a common simulation protocol and forcing-data framework. Its ISIMIP4a arm supports historical model evaluation/attribution, while ISIMIP4b supports climate-impact projections under scenario forcings. CSS-001 should adopt matched forcing, version control, counterfactuals and cross-sector scenario consistency where those data apply. Sources: https://www.isimip.org/protocol/ and https://protocol4.isimip.org/
- **IAMC Scenario Evaluation Criteria (2026 update)** provide machine-readable checks for scenario consistency, empirical plausibility, technical feasibility and social/ecological desirability. Use these as an external scenario-quality layer instead of inventing a competing universal score. Source: https://scenario-evaluation-criteria.iamconsortium.org/stable/
- **Global water-model research** identifies data availability/quality, poorly quantified human influence and region-specific process heterogeneity as important uncertainty sources. CSS-001 therefore requires explicit data gaps and region-specific limitations before scaling beyond the synthetic fixture. Source: https://doi.org/10.1002/wat2.70025

A credible improvement over these systems would not be a larger simulator. It would be a verified cross-project evidence loop: source-qualified observations → calibrated resource models → independently checked scenario runs → transparent policy comparisons → contestable, rights-aware claims. This is a proposed direction, not an achieved capability.

## 7. Four-project handoff

| Project | Narrow responsibility for CSS-001 | Explicit non-responsibility |
|---|---|---|
| Symtropy | Conserved stock-flow state, ordered transfers, policy execution, service outcomes | It does not establish the truth or rights status of input data |
| Symthaea | Policy-learning contenders, counterfactual reasoning, benchmark scoring, uncertainty and transfer evaluation | It does not override physical constraints or make governance decisions |
| Mycelix | Claim/source provenance, custodian rights, licence/consent, disputes, correction and revocation | It does not turn signed claims into facts |
| Sol-Atlas | Render evidence-aware temporal/spatial records and scenario comparisons | Visualization does not independently validate a model or source |

These are composition boundaries, not claims that all four runtime integrations are complete. The corresponding source work remains in draft PRs/issues.

## 8. Real-data pilot gate: Vaal/Johannesburg candidate

Do not replace synthetic values with real numbers until every variable has an identified source snapshot, unit, spatial boundary, temporal window, update frequency, missingness and reuse condition.

Candidate sources to assess—not yet ingested:
- **Department of Water and Sanitation Hydrological Services:** audited hydrological information, near-real-time stage/flow/rainfall, Vaal/Orange daily flow/dam information and weekly state of dams. Confirm the exact reservoir/catchment-to-service-area relationships and the licence/terms of each series. https://www.dws.gov.za/hydrology/
- **Eskom Data Portal:** hourly actual/forecast demand and available capacity; the portal offers data requests for up to five years. Access may require submitting a request; preserve the returned extract and terms. https://www.eskom.co.za/dataportal/
- **Statistics South Africa Census 2022:** municipality-level population context and official census downloads. Use a versioned table and avoid treating municipal population as the population actually served by a water/electricity asset. https://www.statssa.gov.za/publications/Report-03-01-76/ and https://www.statssa.gov.za/?p=16716
- **Johannesburg Water Annual Report 2023/24** and the **City of Johannesburg Annual Integrated Report 2025:** service-delivery and operational indicators; these are not substitutes for continuous hydraulic or electricity time series. https://johannesburgwater.co.za/wp-content/uploads/2025/04/160512-JHB-WATER-ANNUAL-REPORT-2024-15-04-2025.pdf and https://joburg.org.za/work_/Documents/Annual-Integrated-Report-2025/City-of-Johannesburg-Annual-Intergrated-Report-2025.pdf

For each dataset, store exact file digest, retrieval time, publisher/version, transformation code, spatial/time aggregation, missingness, rights/consent and limitations. If public downloadable data are absent, record withheld_by_custodian or unavailable_from_source; do not silently fill.

## 9. Qualification ladder and proposed targets

These are proposed engineering targets, not community standards or achieved scores.

| Gate | Acceptance target |
|---|---|
| T0 Input integrity | 100% of scenario fields have units, scope and explicit epistemic class |
| T1 Accounting | Every region/resource ledger closes within documented numeric tolerance on every tick |
| T2 Determinism | Same source/config/seed produces byte-identical normalized outputs |
| T3 Policy fairness | Identical initial states, forcing, action budget and observation boundary across policies |
| T4 Essential services | Report per-region/resource floor breaches; no aggregate average can mask a region failure |
| T5 Uncertainty | Sensitivity ensembles vary production, demand, stock, transfer capacity and service-floor assumptions independently |
| T6 Robustness | Policy remains acceptable on held-out shock timing, magnitude, topology and regional demand patterns |
| T7 Data qualification | Every real input has exact source digest, transformation receipt, rights and missingness; no synthetic/observed blending |
| T8 Independent review | A separate verifier replays exact source/config/model digest and compares outputs |

Before any superiority claim, compare against no-transfer, fixed transfer, proportional deficit allocation, reserve-aware allocation and relevant WEAP/sector-model baselines under matched inputs and budgets. Use paired scenario sets and report Pareto frontiers over service floors, resource transfers, regional equity and ecological constraints; never claim a single aggregate welfare score is sufficient.

## 10. Limitations and next engineering steps

1. Compile/test the CSS-001 Rust runner and verify its analytical expectations on exact head.
2. Export normalized outputs plus a model-run receipt bound to source/config/code/seed digests.
3. Add multiple topologies, transfer capacities, losses, pumping energy and storage constraints.
4. Add sourced environmental flow and groundwater recharge constraints before making ecological-resilience claims.
5. Acquire rights-cleared Vaal/Johannesburg source snapshots and build a separate historical replay/hindcast. Keep it distinct from CSS-001 synthetic runs.
6. Connect Mycelix and Sol-Atlas receipts; only then begin Symthaea policy-learning comparisons over identical scenarios.

**No results are reported in this protocol until the exact-head Rust tests, output capture, and independent receipt checks complete.**
