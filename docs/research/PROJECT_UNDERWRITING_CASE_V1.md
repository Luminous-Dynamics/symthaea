# Project Underwriting Case v1

**Status:** Public synthetic research contract, 4 November 2026.  
**Parent:** [SYM-INV-UNDERWRITE-001](https://github.com/Luminous-Dynamics/symthaea/issues/10439).  
**Claim ceiling:** Case-input conformance and fail-closed gating only. It is not a bankability standard, investment recommendation, engineering certification, or evidence about any real project.

## 1. Why this contract exists

The ecosystem already contains financial primitives, Mycelix project/investment records, evidence and accounting work, Symtropy energy-model qualification, and Sol-Atlas economic portfolio analysis. The missing seam is the *individual project*:

- What exact project, site, design and revision are being assessed?
- What was known at the decision time?
- Which claims are supported by which source families?
- Are the physical models being used inside their qualified domains?
- Is the grid/offtake status evidenced by the party with authority to establish it?
- Do cash flows, currency transformations and day-count conventions reconcile?
- Are downside variables jointly modelled, or has independence merely been assumed?
- What remains unknown, disputed, stale, superseded or outside the model boundary?

The case format freezes these inputs before any agent generates an assessment.

## 2. System ownership and authority

| System | Owns | Does not gain |
|---|---|---|
| Mycelix | Source-owned project, ownership, contract, financing, payment, consent and evidence records | Engineering certification or investment decision authority |
| Symthaea | Claim graph, financial calculations, analytical hypotheses, uncertainty, scenario analysis, value of information and report generation | Broker credentials, custody, discretionary capital-allocation authority |
| Symtropy | Physical/operational model definitions, qualified domain, simulation and measurement/verification outputs | Authority to assert that a model output is a measured physical fact |
| Sol-Atlas | Geospatial observations and infrastructure dependency projections | Authority to convert a regional map into a site-specific connection right |

A derived claim must retain its source refs and transformation/model identity. A system's output cannot silently inherit another system's authority.

## 3. Evidence vocabulary

### 3.1 Evidence classes

- \`self_asserted\`: a party's own declaration. Useful as a lead; not independent proof of the declared proposition.
- \`issuer_signed\`: evidence issued by the party whose authority is relevant to the proposition (for example, an executed offtake or grid-connection agreement). Signature authenticity and authority still require profile-specific verification.
- \`authenticated_meter\`: device-authenticated observation. Authentication does not alone prove meter calibration, correct placement or complete measurement.
- \`independently_reconciled_measurement\`: measured evidence reconciled to a declared boundary and independent method. This does not certify every downstream inference.
- \`validated_model_output\`: output from an identified model version inside an explicitly qualified domain, with calibration/validation lineage. It is still a model output, not a measurement.
- \`unvalidated_model_output\`: exploratory result, not eligible to satisfy a qualified-model gate.
- \`analyst_assumption\`: explicit assumption used for sensitivity or scenario analysis, never silently promoted to observation.
- \`contradictory_document\`: material evidence that conflicts with another source or claim.

### 3.2 Record state

\`current\`, \`superseded\`, \`disputed\`, and \`retracted\` are distinct. A retracted or superseded item remains historical evidence but cannot satisfy a current gate. Conflicting items must remain visible; do not majority-vote the number of documents, especially when multiple documents share one upstream provenance family.

### 3.3 Two clocks

Every source carries:

- \`valid_time\`: when the source proposition applies or the measurement period occurred;
- \`known_at\`: when the information became available to the decision process.

Historical observations may have a valid period that ends before the decision date; that does not invalidate them as historical observations. But a contract, tariff or approval that purports to be in force at the decision date must cover that date. No record with \`known_at > decision_time\` may enter the decision-time evidence frontier.

The case format records timestamps in UTC RFC 3339. A later implementation must reject malformed timestamps and explicitly document any provider-timezone conversion.

## 4. Claim graph

Each claim has:

- a stable ID and precise proposition;
- materiality (\`critical\`, \`high\`, \`medium\`, \`low\`);
- an explicit assessment (\`unresolved\`, \`supported\`, \`contradicted\`, \`not_applicable\`);
- references to evidence records;
- a claim-specific gate where the proposition is necessary to valuation or risk assessment.

Do not infer claim support from fluent explanation text. A claim is supported only under a named rule that checks evidence class, record state, source authority, temporal frontier, subject identity, units and scope. Evidence counts are not evidence strength.

## 5. Physical-to-financial boundary

The physical model record binds:

- model ID, version and family;
- qualification status and explicit applicable boundary;
- maximum capacity and validity interval;
- output values and units;
- calibration/validation evidence refs;
- the exact source/evidence frontier used.

For v1, model qualification is one of:

- \`not_qualified\`;
- \`limited\`;
- \`qualified\`.

A project outside model capacity, site, technology, dispatch, climate, operating or time domain cannot inherit the model's qualified status. A valid model result may produce a cash-flow input only through an explicit, versioned transformation. Do not convert predicted annual MWh into revenue without the applicable price, curtailment, loss, availability, dispatch, settlement and contract assumptions.

### Important industry comparison

The System Advisor Model (SAM) already couples technology performance outputs to project cash flows and provides distinct financial model contexts, including front-of-meter PPA/merchant plants and behind-the-meter cases. Its battery functionality includes degradation and replacement modelling for supported battery configurations. We should use SAM as a serious independent baseline and interoperability target—not reimplement a weaker version and claim novelty.

The current SAM reference documentation:
- https://sam.nlr.gov/financial-models.html
- https://sam.nrel.gov/battery-storage/

The Electricity Annual Technology Baseline (ATB) is a useful *prior/reference family* for technology-specific CAPEX, O&M, capacity factor and scenario assumptions. The 2025 update was released on 2 September 2026 and includes utility-scale PV-plus-battery and utility-scale storage profiles:
- https://www.nlr.gov/news/detail/program/2026/electricity-annual-technology-baseline-2025-update-now-available
- https://atb.nlr.gov/electricity/2025/technologies

Global cost benchmarks are not site-specific bids. IRENA's July 2026 report, *Renewable Power Generation Costs in 2025*, reports a global weighted-average utility-scale solar PV LCOE of USD 44/MWh for 2025. This is a sector reference—not a South African project forecast, a PPA tariff, a financing rate or a substitute for a site/resource/EPC assessment:
- https://www.irena.org/Publications/2026/Jul/Renewable-Power-Generation-Costs-in-2025

## 6. Measurement and verification boundaries

For PV monitoring, use IEC 61724-1:2021 as the monitoring vocabulary/reference for system performance, monitoring classes, sensors and analysis. It does not authenticate a particular project's data collection merely because the project claims alignment:
- https://webstore.iec.ch/en/publication/65561

For measurement and verification, evaluate the selected plan against the Efficiency Valuation Organization's IPMVP Core Concepts 2022 and an explicitly declared measurement boundary, baseline and adjustment method:
- https://www.evo-world.org/en/library/download-protocol-documents-mainmenu-en
- https://www.evo-world.org/en/products-services-mainmenu-en/protocols/ipmvp

Do not conflate:
\`modelled output != metered output != independently reconciled savings != contractual performance acceptance\`.

## 7. South African grid and market evidence

A grid-capacity map or provincial estimate is a discovery aid, not a project right. A decision-time grid claim must bind the specific point of connection, requested and allocated MW, operator/authority, queue/allocation stage, connection agreement, validity, curtailment basis and the exact source revision.

The NTCSA/Eskom GCCA 2025 material includes reports, results and GIS datasets. Eskom's Distribution Grid Library lists updated connection guidance, including an August 2026 Grid Connection Guideline. Grid/wheeling participation also has specific connection, voltage, tariff and registration requirements; do not generalize from a national generation headline:
- https://www.ntcsa.co.za/gcca/
- https://www.eskom.co.za/distribution/grid-library/
- https://www.eskom.co.za/distribution/tariffs-and-charges/wheeling/

Use local legal, grid, market and contract review for any live project. This research contract is not a regulatory interpretation.

## 8. Financial calculation contract

- Cash-flow amounts are decimal strings in the interchange corpus; don't round through JavaScript \`Number\` or binary floating point while parsing or serializing source records.
- Each valuation schedule has one declared cash-flow currency. If reporting and schedule currencies differ, an explicit FX observation, rate source, timestamp, unit, direction and conversion/rounding policy are required.
- Dated valuation v1 is \`actual_365_fixed\`, matching the first XNPV/XIRR implementation slice. Other day-count conventions require separate named implementations and known-answer tests.
- The date schedule must be ordered, with duplicate-date aggregation rules explicit. Debt service, interest, principal, tax, replacement and decommissioning must not be silently omitted.
- The first v1 corpus contains deliberately synthetic cash flows. It does not assert that the project is economically attractive.
- No DSCR is admissible without a defined debt schedule, cash-flow available for debt service definition, currency, period alignment and covenant definition.
- No IRR is a decision by itself. Multiple roots, no roots, unusual sign patterns and bounded-search limitations must be surfaced.

## 9. Joint uncertainty and scenario boundaries

Each scenario identifies its dependence treatment:

- \`joint_modelled\`;
- \`common_factor_modelled\`;
- \`independence_assumed\`;
- \`unknown\`.

If dependence is unknown, the system may report marginal sensitivities but must not describe a joint tail probability or CVaR as calibrated. Empirical weighted CVaR, a stress case and worst-case distributionally robust risk are different objects and must be labelled separately.

The scenario corpus should include shared weather/resource, curtailment, power-price, FX, interest-rate, construction-delay, supply-chain and replacement-cost factors where relevant.

## 10. Public v1 corpus

Files:

- \`public-corpus.schema.json\`: JSON Schema Draft 2020-12 structure for solver-visible inputs.
- \`public-cases.json\`: seven transparent synthetic cases.
- \`reference-expected.json\`: transparent expected dispositions for development.

The cases exercise:
1. a synthetic baseline with supported evidence;
2. a developer-only grid assertion;
3. a physical model applied outside its qualified capacity domain;
4. missing FX conversion evidence;
5. a tariff record known only after the decision time;
6. unknown scenario dependence;
7. conflicting current grid evidence.

The expected result is a *readiness/risk disposition*, not an investment verdict. In particular, \`eligible_for_synthetic_valuation\` means only that the declared synthetic inputs pass this limited contract's gates.

The reference file is in the public repository, so it is not a hidden oracle. Certification must use a separately controlled, post-freeze holdout set that the candidate system cannot read during development. Do not call a public fixture set "blind."

## 11. Qualification gates and benchmarks

### Hard correctness gates

- 100% known-answer pass on frozen date-aware NPV/XIRR vectors for the specified convention.
- 100% rejection of non-finite inputs, invalid timestamps, currency/unit mismatch, missing FX, unsupported day-count convention and arithmetic overflow.
- 100% rejection of future-known evidence from historical decision-time replays.
- 100% rejection of critical claims that lack the required evidence class, source authority or model scope.
- 100% detection of explicit contradictory current evidence in the synthetic negative controls.
- 100% preservation of unknown/missing/conflicting status; no silent default imputation.
- Zero financial outputs emitted when the selected required calculation gate is blocked.

### Evidence and agent quality

Evaluate FinQA, ConvFinQA, FinanceBench and FinBen for generic finance reasoning, but report them separately and don't treat a QA score as project-underwriting capability. FinanceBenchmark's October 2026 methodology tracks 42 benchmark families across five financial domains, with source attribution and distinct verification/document/numerical/agent task categories:
- https://financebenchmark.ai/benchmarks
- https://financebenchmark.ai/methodology

For the project-underwriting corpus measure:
- claim-level evidence precision/recall;
- unsupported material-claim rate;
- contradiction recall;
- source-family deduplication;
- temporal leakage rate;
- unit/currency transformation accuracy;
- correct abstention and indeterminate classification;
- probability calibration (Brier score/log loss) where probabilities are justified;
- prediction interval coverage *and width*.

### Physical and financial model comparison

Use matched boundaries and exact versions when comparing Symtropy against SAM or another engineering baseline. Evaluate by output channel and horizon, not a single average:
- energy yield and availability;
- curtailment and dispatch;
- battery degradation and replacement;
- capex/opex and construction schedule;
- annual and dated cash-flow error;
- NPV/XIRR error;
- downside and forecast interval calibration.

Compare against independent measured operating records where the evidence and rights permit it. A simulation match to another simulation is not physical validation.

### Decision-quality evaluation

Compare the system with:
1. conventional project-finance DCF and spreadsheet workflow;
2. SAM + the matching ATB assumptions where the physical/financial boundary matches;
3. independent engineering and project-finance underwriters;
4. existing AI financial research agents.

Use identical information frontiers and comparable time budgets. Blind the reviewer to method identity where practical. Track critical risk discovery, diligence omissions, evidence gaps, analyst time/cost, calibration, decision regret and eventual operating outcomes.

Do not claim alpha or investment superiority from backtests or synthetic cases. Require a prospective shadow study with pre-registered outcomes and realistic transaction, financing, tax, liquidity and operating costs before performance claims.

### Point-in-time and reflexivity

The July 2026 NBER work by Koijen and Levy is an important asset-pricing comparator because it evaluates contemporaneous earnings-announcement explanations using only information available at the event time and highlights both look-ahead leakage and market reflexivity:
- https://www.nber.org/papers/w35431

Reuse the point-in-time discipline, but do not transfer its equity-return metric to infrastructure project underwriting.

## 12. Paradigm shifts to test

1. **Falsifiable thesis graph:** every material thesis contains necessary conditions, competing explanations, contradictory evidence and explicit failure boundaries.
2. **Continuous qualification envelope:** qualification is domain- and time-bounded, not a permanent green badge. New sites, operating regimes, designs or model versions may require requalification.
3. **Active verification:** choose the next survey, meter check, contract confirmation or model run by expected value of information and cost, rather than maximizing data volume.
4. **Physical-to-financial digital thread:** each revenue and cost line traces back to a physical output, contract/market observation, transformation and date.
5. **Uncertainty composition:** propagate correlated parameter, observation, model, scenario and source-family uncertainty without pretending that all terms are independent.
6. **Counterfactual redesign:** when a project fails, identify whether a change in storage duration, interconnection point, dispatch strategy, contract, phasing or design would change the failure condition. This generates engineering alternatives, not authority to invest.
7. **Decision regret, not persuasive prose:** measure whether the system discovers material issues earlier and improves decisions under matched information—not whether reviewers prefer its narrative.
8. **No universal score:** maintain separate financial, physical, resilience, distributional and ecological coordinates with explicit objective profiles.

## 13. Nonclaims and safeguards

- No model, Φ/consciousness metric, reputation score or Harmony value is an investment-quality or creditworthiness signal.
- No system output grants lending, brokerage, payment, custody, capital allocation or equipment-actuation authority.
- No source document, signature or DHT record alone proves physical performance, legal title, regulatory approval or settlement finality.
- No LCOE comparison alone establishes bankability or project value.
- No synthetic corpus pass establishes empirical accuracy.
- No AI explanation is provenance; provenance must be machine-inspectable.
- This is not financial, legal, tax, engineering or investment advice and does not certify any real asset.

## 14. Next implementation steps

1. Implement a deterministic Rust evaluator that reads the public corpus and expected dispositions, checks structural references and emits a machine-readable receipt.
2. Add adversarial mutations: known-at after decision time, expired agreement, wrong site/point of connection, copied source family, changed model digest, out-of-domain capacity, currency substitution, duplicated cash flow, and contradictory tariff.
3. Add date-aware calculations from the XNPV/XIRR PR, retaining exact decimal strings until an explicit analytical conversion boundary.
4. Bind Symtropy M&V, joint-scenario and model-calibration receipts by exact digest and declared model boundary.
5. Bind Sol-Atlas spatial evidence by exact feature/source revision and coverage.
6. Add a controlled holdout protocol separate from public repository fixtures.

The next goal is not a polished dashboard. It is a deterministic evaluator that can say **why a project is not yet underwritable**, with exact machine-readable evidence, rather than producing a plausible-looking valuation from incomplete inputs.
