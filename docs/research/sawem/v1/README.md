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

## 3. Machine-readable contract

- [SAWEM BESS Underwriting Profile v1 JSON Schema](sawem-bess-underwriting-profile-v1.schema.json)
- [Synthetic standalone-BESS profile](sawem-bess-underwriting-profile-v1.synthetic.json)

The JSON Schema uses Draft 2020-12. The synthetic fixture contains fictional project identifiers, ratings, energy declarations, dispatch, metered response, balancing components and credit-cover amounts. It is only a schema and interface example.

### Storage Trading Unit

Retain separate values for:
- maximum continuous charging MW;
- minimum stable charging MW;
- maximum and minimum stored energy MWh and the energy-basis convention;
- charge-cycle efficiency and the boundary at which it is measured;
- mode-transition times;
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

## 5. Required next conformance work

The next implementation slice should validate:
1. market-code publication status and effective status are separate;
2. unknown effective date does not imply live eligibility;
3. participant admission is project/participant-specific;
4. Trading Unit charge/discharge and energy fields have valid units and boundaries;
5. daily / six-day / seven-day energy limits reconcile;
6. unconstrained and constrained schedules remain separately traceable;
7. energy and reserve commitments do not double-count available power / energy;
8. intra-day re-declaration respects the selected Code / Rule version and gate timing;
9. instructed-versus-metered differences use the correct Trading Period and sign convention;
10. balancing payment components can be cumulative;
11. settlement reruns and due-date chronology are preserved;
12. credit cover includes actual and undefined potential exposure;
13. synthetic, vendor-declared, registered, operator-certified and measured quantities never silently promote one another.

## 6. Nonclaims

This profile does not establish:
- that a particular SAWEM rule or product is in force;
- that any external entity or project is admitted to trade;
- grid allocation, dispatch entitlement or System Operator certification;
- actual measured reserve response or settlement;
- a revenue guarantee, bankability, market alpha or commercial investment suitability.

No bid submission, market operation, equipment actuation, payment, collateral posting or settlement execution is in scope.
