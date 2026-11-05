# SAWEM BESS Underwriting Profile v1

**Status:** research fixture / not authoritative.  
**Parent:** Symthaea [SAWEM-BESS-UNDERWRITE-001 #10512](https://github.com/Luminous-Dynamics/symthaea/issues/10512).  
**Scope:** standalone battery storage and PV+BESS market-readiness / project-underwriting interfaces in South Africa.

## 1. Why this profile exists

A generic PV+BESS financial model is not enough to underwrite participation in the South African Wholesale Electricity Market (SAWEM). The market path includes registration and admission, technical Trading Unit declarations, day-ahead schedules, network constraints, intra-day re-declarations, reserve commitments, real-time instructions, metered outcomes, imbalance/balancing payments, settlement statements and credit-cover requirements.

The profile separates:

1. **Rule publication** — which Market Code / rule revision is being interpreted.
2. **Rule effectiveness** — whether the relevant instrument is approved, effective, partially effective, suspended or unknown.
3. **Market operational status** — whether the market is in pre-launch testing, limited live operation or commercial operation.
4. **Participant admission** — whether this specific participant has been admitted for testing or commercial participation.
5. **Asset capability** — what the asset can do physically, what it declares, what is registered, what the System Operator certifies and what meters actually record.

These are not interchangeable. A published rule is not evidence that a particular rule is effective. An invitation to pre-launch testing is not a commercial participation licence. A vendor capability is not a registered Trading Unit value, and a registered value is not a measured operating result.

## 2. Current source baseline and rule-status boundary

The initial profile identifies **SAWEM Market Code Rev 2.3 (publication period 2026-02)**. The official NTCSA wholesale-market page is the publisher entry point:

- https://www.ntcsa.co.za/wholesale-market-code/

The first profile crosswalks these Code areas:

| Code area | Underwriting semantics retained |
|---|---|
| §7.4 storage Trading Unit standing data | Maximum continuous charging rating, minimum stable charging point, maximum/minimum stored energy, cycle efficiency and mode-transition times |
| §9.4 energy-constrained Trading Units | Daily energy limit, next-six-days limit, rolling seven-day total and preferred-run flag |
| §9.5 day-ahead scheduling | Energy co-optimization with regulating, instantaneous and ten-minute reserves; keep unconstrained and network-constrained schedules distinct |
| Chapter 10 intra-day market | Six-hour auction / re-declaration cadence and revised constrained schedule |
| §§13.2–13.4 imbalance and balancing payments | Instructed-versus-metered energy, Trading Period basis and potentially cumulative payment components |
| Chapter 15 settlement and credit cover | Settlement statements, invoice/due-date chronology, reruns, actual and undefined-potential exposure, business-day recalculation and credit-cover notices |

This is a profile-level mapping for research. It is not a certified legal interpretation of the Code or Rules 1–4. Before a real decision uses any field, freeze the exact official Code and relevant Rules as raw bytes, record their content digests, check amendments and effective instruments, and obtain South African market / legal review.

The 30 October 2026 pre-launch testing report is represented separately as **secondary reporting**, not as proof of commercial market operation or participant admission:

- https://www.engineeringnews.co.za/article/sawem-trading-platform-opens-to-85-external-entities-for-pre-launch-testing-2026-10-30

Government progress reporting on 30 October 2026 says work on the wider policy and regulatory framework continued:

- https://csp.treasury.gov.za/news-events/operation-vulindlela-phase-2-q2-26-27-progress-report?type=news
- https://www.sanews.gov.za/south-africa/operation-vulindlela-implementation-continues-make-strides

Accordingly, the synthetic fixture has:
- publication_state = published;
- effective_status = unknown;
- market_operations_state = pre_launch_testing_reported;
- participant_admission_state = unknown;
- commercial_trading_state = unknown.

The raw source digest is intentionally null and the source freeze state is not_frozen. This fixture cannot move to source_frozen_for_review or qualified_for_policy_evaluation until the exact sources, release digests, effective-date evidence and local interpretation have been independently reviewed.

## 2A. Industry reference baseline and the competitive test

The objective is not to reproduce every market optimizer. The objective is to qualify the complete chain from physical capability and market rights through dispatch, settlement, liquidity and the investment decision. Published vendor documentation already describes substantial capability in the optimisation layer; feature presence alone is not a defensible differentiation claim.

| Reference system | Published strengths to include in a fair benchmark | Luminous responsibility |
|---|---|---|
| PLEXOS 12 / Energy Exemplar | Market simulation and optimisation; storage SOC targets, build-duration and aggregation optimisation; degradation/maintenance logic; PTDF/LODF congestion analytics; AC power-flow interoperability; ELCC/EFC analysis. | Verify the exact study inputs, project rights, rule versions, contracts, model domains, financial transformations and evidence lineage. |
| PLEXOS Intelligence | Separate workflow layer: study-aware analytical agents, SQL/output querying, AI Studio and cloud workflow automation. | Benchmark workflow provenance, reproducibility, correction burden, source grounding and permission boundaries separately from the PLEXOS solver. |
| Aurora Chronos | Granular battery dispatch forecasts calibrated against historical data, imperfect foresight and engineering/regulatory/geographic/operational constraints. | Independently validate forecast vintages, actual-vs-backcast calibration, contract eligibility, source provenance and realized outcomes. |
| HOMER Front 1.6 | Imperfect-foresight price inputs, forecast/realized differences, multiple time-of-delivery contracts and multi-year participation/availability assumptions, alongside storage dispatch. | Avoid duplicating those features as supposed novelty; benchmark evidence quality, SAWEM settlement rules and underwriting conclusions. |
| NREL SAM | Detailed PV/storage performance and front-of-meter financial modelling. | Qualify technology assumptions, degradation, replacement, cash-flow lineage and model applicability. |
| NREL REopt | Techno-economic sizing/dispatch and resilience/cost optimisation for sites, campuses, communities and microgrids. | Extend the underwriting evidence graph across contracts, participant rights, grid capacity, collateral and financing. |
| PyPSA / GridPath | Transparent, inspectable energy-system and network optimisation formulations. | Use as open reference formulations and independent feasibility comparators where the model boundary matches. |

Primary vendor references:
- PLEXOS August 2026 product update: https://www.energyexemplar.com/product-news/whats-new-august-2026
- Aurora Chronos description: https://auroraer.com/company/about
- HOMER Front release history: https://homerenergy.com/homer-front/releases
- SAM: https://sam.nrel.gov/
- REopt: https://www.nrel.gov/reopt/

PLEXOS Intelligence must be benchmarked as an analyst-workflow layer, not confused with the mathematical optimisation engine. Likewise, a vendor's description of global optimality is scoped to its documented supported action space, model formulation and configuration; it is not a proof of global optimality for the whole real-world market or investment decision.

### Paradigm shifts worth owning

1. **Executable investment thesis.** Every material thesis is a typed dependency graph of falsifiable claims, proof obligations, evidence, contradictions and invalidation conditions. Strong economics cannot compensate for absent critical grid rights or participant admission.
2. **Evidence-time correctness.** Preserve both valid time and known-at time. A document published later cannot enter a historical investment decision. Replays must reconstruct the exact evidence frontier and model versions.
3. **Continuous underwriting.** Contract amendments, rule changes, provider downgrades, battery warranty changes, grid reassessments and model-domain exits selectively invalidate only their dependent claims.
4. **Active diligence.** Prioritise the next evidence request by expected decision improvement, cost, delay, irreversibility and the value of waiting. Do not request another solar survey when unresolved grid allocation dominates decision risk.
5. **Model disagreement as evidence.** Maintain a disagreement ledger across PLEXOS, HOMER, Chronos, SAM, REopt and open formulations. Attribute differences to data vintage, boundaries, algorithms, degradation, constraints and validation domain; do not average away disagreement.
6. **Counterfactual engineering.** Search a constrained Pareto frontier over storage duration, PV/BESS sizing, POC, export limit, dispatch, contracts and financing. Report feasibility, downside, evidence gaps and model-domain limits—not one opaque score.
7. **Proof-carrying investment decisions.** Bind exact project/evidence revisions, content digests, model/evaluator versions, scenarios, calculation outputs, independent checks, unresolved assumptions and invalidation triggers into a reproducible certificate.
8. **Liquidity is a first-class physical constraint.** Treat cash held as collateral, invoice acceleration, settlement reruns and cure deadlines as a chronological funding problem, not a final spreadsheet line item.

### Matched-information evaluation protocol

Run the evaluation in two market contexts and do not mix them:

- **Internal / pre-launch simulation:** reproduce the market state, rules, access cohort and synthetic or authorized test data actually available to the internal environment.
- **External commercial participation:** evaluate only after production access, effective rule instruments, participant admission and authorized data are independently evidenced. An invitation into pre-launch testing is not admission to commercial trade.

For each frozen project, all participants receive the same evidence frontier, decision timestamp, forecast vintage, topology, technology boundary, price and contract assumptions, temporal resolution, degradation settings and permitted action space. Freeze a separate sealed holdout by project and time. Compare incumbents and Luminous with blind independent reviewers; later backtest every forecast and material decision against realized measurements and settlements where rights-cleared data exists.

Required benchmark dimensions:
- Mathematical correctness: 100% of preregistered known-answer cases; exact units, date conventions, cash-flow roots and accounting invariants.
- Evidence integrity: zero false acceptance on the finite hostile corpus for wrong project/POC, wrong issuer/authority, superseded evidence, future-known evidence, duplicate provenance and inaccessible artifacts.
- Physical qualification: zero accepted out-of-domain results; interval energy, SOC, reserve response, availability, curtailment, export and degradation reconciliations.
- Forecast quality: rolling-origin evaluation, project/time holdout, pinball loss and weighted interval score, empirical P50/P90/P95 exceedance calibration with confidence intervals, and calibration by regime/horizon. Evaluate realised dispatch value as well as forecast error.
- Settlement/liquidity: zero false "funded" states; exact reconciliation of source-required cover vs posted cash; peak unrestricted-cash draw, maximum shortfall, days under shortfall, notice/deadline misses, cash lock-up duration and financing cost.
- Decision quality: preregister a hypothesis of at least 20% relative improvement in critical-risk recall at matched false-alarm rate and 25% lower diligence time without sacrificing critical-risk recall. These are targets, not results.
- Human review: reviewer correction burden, time to reproduce a conclusion, unsupported-claim rate, material-risk discoveries and inter-reviewer agreement.
- Value of information: whether the next evidence request changed the decision enough to justify its cost and delay.

Do not report a single "beats SOTA" score. Publish the case-level result, uncertainty interval, exclusions, missing data, model/version identity, and each system's supported domain. Agreement among vendor tools is not independent ground truth. Use metered outcomes, official operator records, independent engineering review and adjudicated historical cases as separate validation sources.

### Treasury liquidity waterfall: from static headroom to event-time solvency

The separate `liquidity_stress.rs` check is a single-snapshot calculation. The new `liquidity_waterfall.rs` module extends that model over an explicit dated event sequence.

It evaluates three paths:
- **Committed:** source-backed commitments and expected obligations/receipts that are treated as committed in the scenario.
- **Forecast:** committed events plus forecast events.
- **Stress:** committed and forecast events plus adverse stress adjustments.

For each path it reports minimum unrestricted cash, minimum cash before the Credit Cover deadline, cash at the deadline, first operating-cash-floor breach, maximum cover shortfall observed after source-backed reassessments, and the potential funding / external funding gap at the deadline. Events use civil-date ordinals, local minute and a deterministic sequence number; the full horizon is bound to a source-digested calendar.

This is a scenario projection, not an executed treasury ledger. A projected Collateral Reserve Account deposit or early invoice payment is a cash use in the waterfall, but does **not** increase Posted Credit Cover. The current MO Required Credit Cover and Posted Credit Cover remain source-bound inputs; future posted-cover changes require a separate receipt-backed event in the credit-cover ledger. A forecast receipt is never silently treated as cash already received.

The event engine is designed to expose an important failure hidden by single-snapshot arithmetic: a project can have enough cash above its operating floor at the Credit Cover deadline but breach the floor earlier in the week, or it may survive the earlier outflows but fail to fund by the required deadline. Both temporal outcomes must be visible.

Benchmark case: `CREDIT-COVER-LIQUIDITY-WATERFALL-001`. The initial tests use synthetic values and do not establish source authenticity, transfer cut-off certainty, cash availability, or a real project's ability to meet a Market Operator deadline.

### Forecast-vintage replay and historical decision integrity

The new `asof_replay.rs` module addresses a different problem from forecast accuracy: **could this forecast or realized value actually have been known when the investment decision was made?**

Every forecast record carries:
- target valid-time interval;
- forecast issue timestamp;
- source available-at timestamp;
- model identity and model digest;
- source identity and source-content digest;
- the forecast value.

Every realized price carries the target interval, observation timestamp and source availability timestamp. At a frozen decision timestamp, the replay selects the latest forecast issue that was available at that time. A realized observation can supersede a forecast only after the target interval has ended and the observation has become available. Future-known records are excluded and counted; missing target coverage stays missing. Multiple engines require an explicit model scope; ambiguous latest vintages fail closed.

This does not claim to improve Chronos or HOMER forecasting accuracy. It supplies a reconstructible information set for fair model comparison and historical underwriting replay. The initial case `FORECAST-VINTAGE-ASOF-001` contains synthetic timestamps and price values only; real calibration, rolling-origin backtests, forecast coverage and realised value-capture comparisons remain `not_run`.

### South African market phase as of 5 November 2026

Engineering News reported on 30 October that invitations had been issued to 85 external entities for pre-launch simulation/testing, while broader launch was scheduled for April 2027. On 5 November, the Energy News Network interviewed the Acting NERSA CEO, who described finalisation of the market rules as a priority ahead of external market opening in April 2027. Both are secondary reporting; neither proves that any specific participant is admitted or commercially trading. The profile therefore records `pre_launch_testing_reported`, `participant_admission_state=unknown` and `commercial_trading_state=unknown`, with a null raw-source digest and `source_freeze_state=not_frozen`.

Sources:
- 30 October 2026 pre-launch invitation report: https://www.engineeringnews.co.za/article/sawem-trading-platform-opens-to-85-external-entities-for-pre-launch-testing-2026-10-30
- 5 November 2026 NERSA leadership interview: https://energy-news-network.com/industry-news/exclusive-nomfundo-maseti-takes-nersa-into-south-africas-competitive-power-market/

## 3. Machine-readable contract

- [SAWEM BESS Underwriting Profile v1 JSON Schema](sawem-bess-underwriting-profile-v1.schema.json)
- [Synthetic standalone-BESS profile](sawem-bess-underwriting-profile-v1.synthetic.json)
- [Comparator run-manifest schema](sawem-bess-comparator-manifest-v1.schema.json)
- [Synthetic comparator roster](sawem-bess-comparator-manifest-v1.synthetic.json)
- [Matched-engine benchmark protocol](COMPARATOR_PROTOCOL_V1.md)
- [Reserve economics and collateral liquidity crosswalk](RESERVE_ECONOMICS_AND_COLLATERAL_V1.md)
- [Machine-readable 30-case benchmark corpus](sawem-bess-benchmark-corpus-v1.json)
- [Benchmark corpus JSON Schema](sawem-bess-benchmark-corpus-v1.schema.json)

The JSON Schema uses Draft 2020-12. The synthetic fixture contains fictional project identifiers, ratings, energy declarations, dispatch, metered response, balancing components and credit-cover amounts. It is only a schema and interface example.

### Storage Trading Unit

Retain separate values for:
- maximum continuous charging MW;
- minimum stable charging MW;
- declared maximum discharging MW as a physical-feasibility extension (not an assertion that this is an additional mandatory Code registration field);
- maximum and minimum stored energy MWh and the energy-basis convention;
- charge-cycle efficiency and the boundary at which it is measured;
- all six directed mode-transition times between standstill, charging and discharging;
- registration / declaration / certification / measurement status.

Do not collapse nameplate, vendor_target, registered_value, operator_certified and independently_measured into one capability field.

### Energy constraints

Preserve daily, next-six-days and seven-day limits independently. The declaration vector is seven Trading Days and uses an explicit MWh basis. The sum of the six-day horizon and the seven-day total must not be silently interchanged.

### Dispatch and reserves

Keep the unconstrained day-ahead schedule separate from the network-constrained schedule and later intra-day revisions. Record reserve products separately. The same MW/MWh must not be committed to mutually incompatible energy and reserve obligations. A synthetic reserve commitment is not evidence of System Operator certification or an actual award.

### Physical and metered outcomes

Symtropy owns qualified physical trajectories, SOC, usable energy, efficiency, thermal limits, degradation, auxiliary load, mode transitions and response-envelope evidence. Actual operator instruction and authenticated meter data remain separate evidence classes.

A simulated dispatch trajectory cannot satisfy a meter-settlement or reserve-performance evidence gate.

### Settlement and liquidity

Do not reduce SAWEM exposure to a single expected revenue number. Keep:
- each balancing payment category and status;
- indicative vs initial settlement statement vs invoice vs settled vs rerun;
- actual exposure separate from undefined potential exposure;
- required credit cover and cash-call timing;
- collateral liquidity and settlement-delay exposure.

Forecast prices, synthetic balancing components and modeled credit cover do not establish actual settlement, invoice liability or a credit-cover notice.

## 4. Ownership across the four systems

- **Mycelix:** source artifacts, authority profiles, participant admission, contracts, evidence lifecycle, access and provenance. Do not create a second evidence or contract ledger.
- **Sol-Atlas:** site, POC, network zone, grid constraints, spatial coverage, locational charging opportunity and discharge deliverability.
- **Symtropy:** interval-level BESS/PV physical capability and independently qualified operating envelope.
- **Symthaea:** market-rule composition, schedule/settlement scenarios, dated cash flows, downside, liquidity, value of information and underwriting certificate.

The market profile should not become a fifth simulator. It is a typed rule/obligation layer consumed by the existing systems.

## 5. Dependency-free Rust reference kernel

The first executable open reference is in [tools/sawem-market-reference](../../../../tools/sawem-market-reference/). It is a standalone Cargo workspace with no third-party dependencies.

It currently implements and tests:
- charge/discharge energy balance using an explicitly documented symmetric square-root round-trip-efficiency assumption;
- minimum stable charge and continuous charge/discharge power limits;
- explicit transition downtime and mode-switch timing;
- SOC minimum/maximum feasibility;
- bidirectional reserve headroom from the net operating point: upward reserve may reduce existing charging, while downward reserve may reduce existing discharging and/or increase charging;
- minimum-stable-charge deadband validation for reserve target setpoints;
- SAWEM Rev 2.3 product-specific response requirements: regulating up/down starts within 10 s and fully activates within 10 min; instantaneous up responds within 10 s and sustains for at least 10 min; ten-minute reserve responds within 10 min;
- separate asset response/sustain evidence from market-rule requirements; missing sustain duration for products where the Code definition is silent remains indeterminate;
- compare a reserve's evidenced full-activation time against the registered directed mode-transition time; a faster asserted response is rejected as contradictory, and a mode transition longer than the product's market deadline is infeasible;
- per-product reserve energy headroom/footroom, evaluated against the activation sustain duration supplied by the applicable operational contract/profile;
- duplicate reserve product/direction detection and aggregation rather than silent double-counting;
- network import/export caps;
- seven-day declaration, next-six-days aggregate and seven-day energy caps;
- checked integer arithmetic for credit cover and multiple signed balancing-payment components;
- product-specific reserve opportunity-price calculation for Regulating Up/Down, Instantaneous Up and Ten-Minute Up using a validated generating incremental price curve, distinct elbow-selection semantics and checked signed arithmetic.

The reserve mapping is grounded in NTCSA Market Code Rev 2.3 §§2.1(107), 2.1(167), 2.1(199), 9.1(4), 9.5(2)(b) and 7.4: https://www.ntcsa.co.za/wp-content/uploads/2026/02/Market-Code-v2.3.pdf. Regulating and Ten-Minute reserve definitions do not themselves specify a minimum sustain duration. The reference therefore requires an explicit contract/operator sustain profile for energy qualification and reports indeterminate when absent; it does not assume that all products need 10 minutes. Section 7.4 records mode-change times in all directions; the reduced BESS abstraction now stores all six directed transitions among standstill, charging and discharging. The full Code's four-state SCO/pumping/generating vocabulary must be mapped separately before representing pumped-storage units.\n\nThe kernel is a deterministic feasibility auditor, not a market optimizer. Its simplified symmetric efficiency assumption must be replaced by boundary-specific measured charge/discharge/auxiliary-loss models before engineering performance claims. It does not model temperature, electrochemical dynamics, degradation, warranty, power flow, market price formation, unit commitment, stochastic policies, SAWEM legal interpretation, authenticated dispatch/meter data or settlement integration.

The first CI run executes formatting and dependency-locked unit tests in isolation from the repository root workspace. The open reference is intended to become the hand-auditable feasibility oracle for the finite benchmark corpus, not a replacement for PLEXOS, Chronos or EnCompass.

The reference package now also contains separately scoped financial/settlement evaluators:

- `credit_cover.rs`: source-referenced unpaid-invoice / uninvoiced-Statement exposure components, supplied-UPE arithmetic, receipt-backed early-invoice / collateral remediation, duplicate-source/receipt controls and source-calendar coverage. It labels its arithmetic as a **synthetic cover requirement**, not the MO's full §15.10.3 ASE/UEF result.
- `liquidity_options.rs`: exact rational simple-interest comparison of cash held in the Collateral Reserve Account versus early payment of an issued outstanding invoice. It models only the two paths stated in §15.10.2(3), requires source-referenced eligibility/rates/invoice evidence, and does not admit generic guarantees or letters of credit as presumed acceptable Credit Cover under Rev 2.3.
- `liquidity_stress.rs`: source-bound point-in-time treasury headroom calculation using the MO Required Credit Cover, Posted Credit Cover, unrestricted cash, mandatory non-cover outflows and an externally approved operating cash floor. It reports potentially fundable cash separately from Posted Credit Cover and never promotes one into the other without a posting receipt.
- `lib.rs` also includes bounded SAWEM reserve-capacity payment arithmetic and constrained-schedule adjustment arithmetic. They consume separately qualified capacities, product-bound Market Operator reserve-category prices, and signed scheduled-energy increments; they do not authenticate those inputs or claim settlement.

The credit-cover deadline layer separately calculates the daily Business Day notice, second-Working-Day collateral deadline, two-Business-Day Credit Call replenishment, ten-Business-Day qualifying-bank replacement after provider cessation, and one-Business-Day advance notice before a Posted Credit Cover change. Business Day and Working Day remain separate source-backed calendars.

These modules are exact-arithmetic / synthetic or prequalified-input references. They are not live market interfaces, bankability determinations, legal opinions, or independent authentication of the records they consume.

## 6. Required next conformance work

The next implementation slice should validate:
1. market-code publication status and effective status are separate;
2. unknown effective date does not imply live eligibility;
3. participant admission is project/participant-specific;
4. Trading Unit charge/discharge and energy fields have valid units and boundaries;
5. daily / six-day / seven-day energy limits reconcile;
6. unconstrained and constrained schedules remain separately traceable;
7. energy and reserve commitments do not double-count available power / energy;\n8. reserve up/down capability includes charge-reduction/discharge-reduction headroom and respects minimum stable charging;\n9. reserve product response deadlines and sustain durations are distinct, source-backed fields;\n10. missing product sustain profile remains indeterminate;
11. intra-day re-declaration respects the selected Code / Rule version and gate timing;
12. instructed-versus-metered differences use the correct Trading Period and sign convention;
13. balancing payment components can be cumulative;
14. settlement reruns and due-date chronology are preserved;
15. credit cover includes actual and undefined potential exposure;
16. synthetic, vendor-declared, registered, operator-certified and measured quantities never silently promote one another.

## 7. Nonclaims

This profile does not establish:
- that a particular SAWEM rule or product is in force;
- that any external entity or project is admitted to trade;
- grid allocation, dispatch entitlement or System Operator certification;
- actual measured reserve response or settlement;
- a revenue guarantee, bankability, market alpha or commercial investment suitability.

No bid submission, market operation, equipment actuation, payment, collateral posting or settlement execution is in scope.
