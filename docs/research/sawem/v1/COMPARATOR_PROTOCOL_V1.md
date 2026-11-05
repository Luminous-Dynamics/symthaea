# SAWEM BESS Benchmark and Comparator Protocol v1

**Status:** research protocol / synthetic comparator roster / no comparative execution claimed.  
**Parent:** Symthaea [#10512 SAWEM-BESS-UNDERWRITE-001](https://github.com/Luminous-Dynamics/symthaea/issues/10512).  
**Profile:** [SAWEM BESS profile v1](README.md).  
**Purpose:** produce defensible, repeatable comparisons between battery-project models, market simulators, physical-system models and evidence-qualified underwriting workflows.

## 1. The strategic boundary: do not rebuild the wrong thing

The industry already has strong, specialized tools. Published documentation describes:

- **PLEXOS 12:** market and system modelling; its August 2026 R04 update lists storage SOC targets, battery build-duration/aggregation optimisation, degradation and maintenance logic, congestion analytics, and improved AC power-flow interoperability. Sources: [PLEXOS](https://www.energyexemplar.com/plexos), [August 2026 updates](https://www.energyexemplar.com/product-news/whats-new-august-2026).
- **Aurora Chronos:** battery dispatch and valuation, degradation cases, balancing-market analysis, site-specific modelling and revenue stacking. Aurora's current global-presence menu lists South Africa in EMEA, but that does not prove a specific South African price dataset or feature entitlement is included in a given licence. Source: [Chronos](https://auroraer.com/software/chronos).
- **EnCompass:** zonal capacity expansion, production-cost modelling and embedded nodal power flow; its product materials describe single-pass co-optimisation of energy, capacity and ancillary services, storage/hybrid representations, congestion and deliverability analysis. Source: [EnCompass](https://www.yesenergy.com/products/encompass).
- **HOMER Front 1.6:** released 19 October 2026; official release notes describe imperfect price foresight, multiple simultaneous ToD contracts, multi-year participation assumptions, real-time volume controls and other storage/hybrid modelling improvements. Sources: [Front 1.6](https://homerenergy.com/homer-v16), [release notes](https://homerenergy.com/homer-front/releases), [changelog](https://support.homerenergy.com/docs/changelog).
- **SAM:** detailed PV-battery and generic system-battery models; front-of-meter PPA and merchant financial models; dispatch and degradation. It does not expose a dedicated standalone-battery model, though NREL documents a generic-battery configuration for that use. Source: [SAM battery storage](https://sam.nlr.gov/index.php?Itemid=360&id=94&option=com_content&view=article).
- **PyPSA:** economic dispatch, unit commitment, storage, linear network-constrained OPF, security-constrained LOPF and rolling-horizon optimisation. Sources: [PyPSA](https://github.com/PyPSA/PyPSA), [optimisation API](https://docs.pypsa.org/latest/api/networks/optimize/).
- **GridPath:** current documentation (v2026.14.0) describes production-cost, capacity-expansion, asset-valuation and reliability modelling; storage energy tracking; upward/downward reserve headroom and footroom; and network constraints. Its documentation also distinguishes open-source capabilities from features that may require additional access. Pin the source release, local build and feature entitlements before execution. Sources: [GridPath introduction](https://gridpath.readthedocs.io/en/stable/introduction.html), [functionality](https://gridpath.readthedocs.io/en/latest/functionality.html), [Sylvan overview](https://www.sylvan.energy/gridpath).

These platforms are not interchangeable, do not necessarily use South African market rules, and may require licensed datasets or specialist configuration. A published feature list is not evidence that a specific product/version supports every detail in this protocol.

**Differentiation hypothesis to test:** Symthaea + Symtropy + Sol-Atlas + Mycelix should add an evidence-qualified decision layer around these specialized engines:
- Mycelix: source, authority, evidence lifecycle, contract, access and receipt provenance;
- Sol-Atlas: POC, topology, grid constraints, locational deliverability and source coverage;
- Symtropy: physical battery/PV envelope, warranty, degradation and response evidence;
- Symthaea: market-rule composition, scenarios, non-anticipative policy, cash-flow/credit exposure, uncertainty and underwriting decision.

Do not claim superior market clearing, dispatch optimisation, price forecasting or battery degradation until a matched benchmark demonstrates it.

## 2. Benchmark tracks

Keep five distinct tracks and report their scores separately.

### Track A — physical feasibility and battery fidelity

Question: does a policy remain feasible under the same physical battery and grid constraints?

Metrics:
- hard constraint violations (target: **zero**);
- SOC and usable-energy conservation error;
- charge/discharge efficiency and auxiliary load;
- ramp, minimum stable charge and mode-transition violations;
- reserve headroom/footroom from the scheduled net injection point, including both reducing charge and increasing discharge (up) and reducing discharge and increasing charge (down);
- product-specific response and sustain checks for regulating, instantaneous and ten-minute reserve;\n- all six directed mode transitions and cross-checks between physical transition time, product deadline and asset response evidence;
- temperature, state-of-health, throughput, cycle life and warranty-envelope violations;
- network export/import and congestion-limit violations;
- model-to-meter energy deviation, unfulfilled scheduled energy and overdelivery, with schedule/meter/settlement coverage reported separately;
- SOC-dependent charge/discharge capability-envelope error across the calibrated SOC range; never extrapolate beyond field-measured or manufacturer-guaranteed breakpoints;
- multi-dimensional SOC × temperature × SOH cell coverage, per-cell sample-count adequacy and lower-quantile calibration; no operational use outside evidenced cells;
- load-dependent inverter/transformer efficiency and BMS curtailment residuals, once synchronized field measurements exist;
- source-bound meter record and settlement line-item reconciliation, with incomplete coverage withheld rather than scored as zero.

Do not count vendor targets or simulated response as independently measured capability.

### Track B — market-rule and settlement conformance

Question: does the engine or integration implement the correct rule version and settlement semantics?

Finite corpus target: **100% pass** for every frozen Code/Rule example; no tolerated false-safe.

Cover:
1. published vs approved vs effective vs superseded status;
2. participant and Trading Unit eligibility;
3. storage registration data, including charge capability, energy basis, efficiency and transition times;
4. daily, next-six-day and seven-day energy constraints;
5. unconstrained vs network-constrained day-ahead schedule;
6. reserve products and co-optimization without double-counting;
7. six-hour intra-day re-declarations only under the applicable rule version;
8. dispatch instruction vs authenticated meter outcome;
9. imbalance direction and cumulative balancing-payment components;
10. settlement statement revisions, queries, invoices, due dates and reruns;
11. actual vs undefined-potential credit-cover exposure and recalculation;
12. rule, contract, participant or source revocation/supersession during a project life.

A passing synthetic fixture is only conformance to its declared corpus. It does not prove real operator integration or legal interpretation.

### Track C — dispatch, forecast and decision quality

Compare:
- deterministic forecast dispatch;
- scenario-tree non-anticipative dispatch;
- rolling-horizon dispatch;
- rule-based operating baseline;
- human-underwritten baseline;
- perfect-information oracle, reported separately as a mathematical bound only.

Metrics:
- net revenue after market charges, balancing, degradation and replacement;
- delivered energy and reserve shortfall;
- realized imbalance;
- CVaR 95% / worst-case downside;
- forecast MAE/WAPE, q10/q50/q90 pinball loss, empirical quantile coverage, CRPS where a full predictive distribution exists, and weighted interval score;
- calibration error and interval coverage, scored only on forecasts available by the historical decision timestamp and realized observations available by evaluation time;
- leakage rate: future-known forecast/realized records excluded / records matching the target set; target: **zero leakage**;
- schedule fulfilment, unfulfilled MWh and booked balancing cost / gross trading revenue on the same field trace;
- rolling-horizon regret against the best **implementable non-anticipative** policy;
- non-anticipativity violations: **zero** across all shared information nodes; allow divergence only after the information frontier branches;
- runtime, memory and solver gap;
- strategy robustness under regime shift and distribution shift.

Do not compare an implementable forecast policy against a perfect-foresight oracle and call the difference model error. Do not give a model realized future data before its decision frontier.

### Track D — bankability, liquidity and lifecycle risk

Metrics:
- net present value, IRR and debt-service metrics under common finance assumptions;
- peak credit cover and collateral calls;
- peak working capital before settlement receipts;
- settlement cash-conversion time and late/revised statements;
- counterparty default / downgrade scenario;
- warranty exclusions and replacement/augmentation liability;
- FX, inflation, tax, fees and financing sensitivity;
- risk-adjusted NPV and downside distribution.
- realized-net / theoretical-gross retention, shown by cohort and forecast horizon;
- evidence-attributed bridge coverage (% of gross-to-net difference backed by unique source references);
- unexplained residual in currency and as a percentage of gross revenue; never force the residual to zero;
- category-level forecast regret, availability loss, execution deviation, balancing, fees, degradation/warranty, financing/collateral and contract effects, with overlap adjudication;
- attribution completeness and source duplication rate (target: 100% required-category coverage, 0 duplicated adjustment sources).

A market-revenue forecast is not an executed offtake contract. A credit-cover estimate is not a collateral notice. An expected insurance/warranty recovery is not certain cash.

### Track E — evidence and reviewer productivity

Measure whether the underwriting conclusion can be reproduced and challenged.

Metrics:
- source claim coverage (% material claims linked to content-addressed source);
- authority and licence verification coverage;
- decision-time validity and known_at correctness, including distinct forecast issue, publication/availability and target valid timestamps;
- model build identity and digest binding for every forecast comparator;
- meter/settlement source coverage and duplicate-record rejection;
- missing/contradictory evidence detection;
- stale/revoked evidence invalidation latency;
- false-safe rate and material-risk recall;
- independent reproduction success;
- reviewer minutes per completed case;
- percentage of findings with exact source/line/page and model-input lineage.

Target **zero false-safe** for the finite hostile corpus; report false-blocks and indeterminate decisions separately rather than silently lowering the gate.

## 3. Paired-comparison protocol

A head-to-head claim is allowed only if all material dimensions are matched or a documented transform is applied:

| Dimension | Required matching evidence |
|---|---|
| Market/rule version | Same effective-dated Code/Rules, products, settlement and eligibility assumptions |
| Information frontier | Same decision timestamp; no inputs published after that time |
| Forecast vintage | Same forecast issuance time and source vintage, not just same realized series |
| Scenario | Same scenario seed, probabilities, stress draws and actual-outcome window |
| Network | Same node/POC, topology, ratings, outage state, losses and constraint enforcement |
| Battery | Same charge/discharge powers, usable energy, SOC limits, efficiency, auxiliary load and modes |
| Warranty | Same warranty/augmentation envelope and cost assumptions |
| Time resolution | Same market/technical intervals or explicit conservative aggregation with a documented transform |
| Objective | Same revenue/constraint treatment, terminal SOC and risk preference |
| Finance | Same inflation, FX, tax, debt, fees, COD and settlement timing |
| Settlement | Same actual/indicative/final status, balancing formulas and credit-cover assumptions |
| Eligibility | Same rule/participant/product eligibility; unknown is not eligible |

Each dimension must be marked matched, mismatched, not_comparable or not_assessed. A mismatched dimension must either exclude the case from a head-to-head score or be reported as its own comparison. Never fill missing model capability with an invented default.

The machine-readable [comparator manifest schema](sawem-bess-comparator-manifest-v1.schema.json) and [synthetic roster](sawem-bess-comparator-manifest-v1.synthetic.json) make model version, source vintage, information frontier, topology, constraints, licence, adapter and output evidence explicit.

## 4. Baseline tiers and comparator roster

| Tier | Comparator | What to compare | Claim boundary |
|---|---|---|---|
| Market/system optimisation | PLEXOS 12 | Production-cost/dispatch, network constraints, storage and reserves | Compare only a matched SAWEM-like case; do not assume SAWEM rule support |
| Battery valuation | Aurora Chronos | Dispatch, degradation, balancing performance, revenue stacking | Require matched geography, market data and licensed configuration |
| Market/system planning | EnCompass | Nodal/zonal model, power flow, storage/ancillary co-optimisation | Compare network/economic outputs, not evidence-ledger functions |
| Project/hybrid economics | HOMER Front 1.6 | Imperfect foresight, simultaneous contracts, multi-year BESS assumptions | Its new feature set makes it a close project-economics comparator |
| Battery technical / financial | SAM | PV/BESS performance, degradation, merchant/PPA cash flows | Preserve its documented scope and standalone-battery modelling workaround |
| Transparent open reference | PyPSA | Dispatch, unit commitment, storage, network and rolling horizon | Exact environment and solver pinned |
| Additional open planning reference | GridPath | Production-cost, planning and storage case where supported | Add only after version, licence and feature mapping are frozen |
| Independent human reference | Two independent SA market/finance reviewers | Risk detection, eligibility, cash-flow assumptions and decision quality | Blind review with adjudication protocol |
| Luminous composition | Four-system workflow | Evidence lineage, policy/contract/physical integration and decision certificate | No superiority claim before matched execution |
| Revenue-realization calibration | Modo Energy GB model-vs-realized calibration | Compare only on same GB cohort, horizon and gross/net definitions; 82% central with 77%/87% low/high in Oct 2026 methodology | Calibration-practice comparator, not a SAWEM performance target |

For commercial software, record exact version/build, solver, licence/feature entitlements, adapter, machine and runtime. For open-source tools, record exact commit, dependency lock, solver build and execution environment.

## 5. Oracle and information-leakage rules

Three policies must never be conflated:

- **Perfect-information oracle:** sees realized future prices, outage and availability outcomes. Non-implementable; report only as a lower/upper bound depending on objective.
- **Forecast-informed policy:** only sees data available by the declared information frontier.
- **Rolling-horizon / stochastic non-anticipative policy:** makes each decision from the then-available information; future scenarios may shape policy, but the implemented action must not use future realized outcomes.

Every forecast and external dataset must retain source_vintage_utc, issue/receipt time, content digest, and decision_time_utc. A backtest using latest revised data instead of what was available then is not a valid historical decision replay.

## 6. Scenario corpus

Start with a small, exact, hand-auditable corpus before scaling to thousands of stochastic runs.

- **BASE-001:** normal standalone BESS; no binding network outage; day-ahead + constrained + intraday schedule.
- **PVHYBRID-001:** PV+BESS with limited export and source-of-charge tracking.
- **SOC-FLOOR-001:** reserve request conflicts with minimum SOC.
- **SOC-CEILING-001:** otherwise-curtailed charging conflicts with upper SOC.
- **POWER-HEADROOM-001:** reserve and energy commitment exceed discharge MW.
- **MODE-TRANSITION-001:** charging/discharging reversal violates transition time.
- **DAILY-LIMIT-001:** daily MWh limit binds.
- **ROLLING-WEEK-001:** seven-day MWh total binds across day boundaries.
- **NETWORK-CHARGE-001:** charging is restricted while local surplus is available.
- **NETWORK-DISCHARGE-001:** discharge recreates the network constraint.
- **INTRADAY-GATE-001:** re-declaration arrives after gate closure.
- **INSTRUCTION-METER-001:** instructed energy differs from authenticated metered output.
- **CUMULATIVE-BALANCING-001:** simultaneous applicable balancing-payment components.
- **SETTLEMENT-RERUN-001:** M+1 revision changes a prior cash-flow assumption.
- **CREDIT-COVER-001:** actual and undefined-potential exposure require collateral before invoice settlement.
- **WARRANTY-001:** dispatch maximizes short-term revenue but breaches warranty envelope.
- **PRICE-SPIKE-001:** forecast misses a rare price/negative-price period.
- **FORECAST-SHIFT-001:** forecast regime shifts after model selection.
- **AVAILABILITY-001:** forced outage removes a previously declared service.
- **RULE-SUPERSESSION-001:** new rule or effective date invalidates prior eligibility assumptions.
- **RESERVE-UP-CHARGE-REDUCTION-001:** upward reserve is provided by reducing charging before increasing discharge.
- **RESERVE-DOWN-DISCHARGE-REDUCTION-001:** downward reserve is provided by reducing discharge before increasing charging.
- **RESERVE-DEADBAND-001:** reserve target falls into the minimum-stable-charge deadband.
- **RESERVE-PRODUCT-DURATION-001:** regulating, instantaneous and ten-minute products use separate response/sustain requirements.
- **RESERVE-UNKNOWN-SUSTAIN-001:** missing operator/contract sustain profile remains indeterminate when the Code definition is silent.
- **RESERVE-TRANSITION-001:** claimed full activation conflicts with the registered mode-transition path.
- **RESERVE-PAYMENT-QUALIFICATION-001:** contracted MW, effective available MW and capacity qualifying for payment differ.
- **RESERVE-LOST-OPPORTUNITY-001:** product-specific reserve price uses the correct piecewise price-curve elbow selection.
- **CONSTRAINED-SCHEDULE-COMPENSATION-001:** unconstrained-to-constrained schedule adjustment changes the constrained purchase/sales payment.
- **CREDIT-COVER-LIQUIDITY-TIMELINE-001:** daily credit-cover recalculation triggers a working-day-specific posting deadline and liquidity need.
- **CREDIT-COVER-LIQUIDITY-STRESS-001 / CREDIT-COVER-LIQUIDITY-WATERFALL-001:** distinguish point-in-time treasury headroom from dated committed/forecast/stress cash paths; preserve peak cover shortfall and first operating-floor breach.
- **FORECAST-VINTAGE-ASOF-001:** replay only forecast vintages actually available at a frozen decision timestamp; reject future publication and ambiguous/model-unscoped vintages.
- **FORECAST-CALIBRATION-EXACT-001:** score one exact model build against later-known realized prices using exact median errors, quantile pinball loss, coverage and interval scores.
- **NON-ANTICIPATIVITY-POLICY-AUDIT-001:** enforce identical actions across scenario paths sharing a source-digested decision-time information set; diverge only after the information tree branches.
- **DISPATCH-EXECUTION-FIDELITY-001:** compare signed scheduled export/import energy with meter records and booked balancing settlement lines over a complete horizon.
- **BMS-SOC-CAPABILITY-ENVELOPE-001:** assess charge/discharge power against separate conservative SOC-dependent capability curves; reject extrapolation and synthetic operational claims.
- **BMS-MULTIDIM-CAPABILITY-SURFACE-001:** evaluate direction-specific power in a complete SOC × temperature × SOH cell grid; reject missing cells, duplicate evidence, out-of-range states and unsupported sample counts.
- **BMS-CAPABILITY-CELL-CALIBRATION-001:** fit a deterministic lower empirical order statistic per cell from BMS available-power telemetry or explicit capability tests; reject ordinary delivered power and future-known samples.
- **REVENUE-REALIZATION-BRIDGE-001:** reconcile theoretical gross revenue to realized net cash using source-bound adjustments by forecast regret, availability, execution, balancing, market fees, degradation/warranty, financing/collateral and contract/curtailment; withhold a complete residual when a required category is missing.

For each case, freeze expected feasible/infeasible state, violated constraint IDs, settlement direction, exposure and evidence disposition. Independent reviewers adjudicate the golden expected outcome.

## 7. Proposed target thresholds

These are **research targets**, not measured current performance or industry-certified thresholds.

| Dimension | Initial acceptance target | Mature target |
|---|---:|---:|
| Finite rule/settlement conformance | 100% | 100% |
| Hard physical/network violations | 0 | 0 |
| Double-counted energy/reserve MWh | 0 | 0 |
| False acceptance of unknown/ineligible participant | 0 | 0 |
| Forecast calibration | Publish by horizon and regime | Stable under measured distribution shift |
| Matched policy regret | Report baseline and variance | 5% or less vs best implementable reference in agreed corpus |
| Material-risk recall | At least 95% on independently adjudicated corpus | 99%+ |
| False-safe rate | 0 on finite hostile corpus | 0 on release gate; confidence intervals on field sample |
| Forecast look-ahead leakage | 0 | 0 |
| Shared-information action divergence | 0 violations | 0 violations on all scenario-tree nodes |
| Execution evidence completeness | Report schedule, meter and settlement coverage independently | 100% for a scored complete field replay |
| Unfulfilled energy delivery | Reproduce published M5BAT 238 kWh / 61-hour result on the same input trace | ≤ 120 kWh on the same 61-hour trace |
| Balancing cost / trading revenue | Reproduce published 1.8% result on the same tariff and settlement scope | ≤ 1.0% on the same trace |
| System RTE model error | Reproduce published 2.2 percentage-point deviation | ≤ 1.0 pp |
| Inverter efficiency error | Reproduce published 0.55 percentage-point deviation | ≤ 0.30 pp |
| SOC capability extrapolation | 0 | 0; all decisions stay inside measured/certified coverage |
| Multi-dimensional capability cell coverage | 100% of states scored must map to an evidenced cell | 100%; no unmeasured state fallback |
| BMS per-cell sample sufficiency | 100% of field cells meet declared minimum sample count | 100%, with separate calibration/validation campaigns |
| Revenue bridge required-category coverage | 100% of declared applicable categories or incomplete status | 100% on source-complete historical case |
| Revenue adjustment source duplication | 0 | 0 |
| Revenue bridge unexplained residual | Always report in currency and % gross | Explain ≥ 95% of absolute gross-to-net difference; residual remains visible |
| Independent receipt reproduction | 100% | 100% |
| Source/authority coverage | 100% of material claims | 100%, with stale-source invalidation |
| Reviewer time | Establish baseline | 50% reduction without reduced risk recall |

A 5% regret target is only meaningful if the policy reference is implementable, the objective is common, and the scenario distribution is independently frozen. Report absolute cash impact alongside percentage regret.

## 8. Execution phases

1. **P0 — Freeze the contract:** schema, scenario corpus, input/forecast manifest, objective and exact model versions.
2. **P1 — Build an open reference:** implement the synthetic storage/market case using PyPSA or a small independent reference model, with no vendor adapter needed.
3. **P2 — Verify equations:** hand-check SOC, energy limits, mode switching, reserve headroom, network/export feasibility, dispatch instructions and balancing/settlement formulas.
4. **P3 — Vendor runs:** execute only when licensed products and the identical input/feature configuration are available.
5. **P4 — Human blind review:** two independent reviewers, hidden model identity, adjudicated false-safe/false-blocks and review time.
6. **P5 — Historical shadow replay:** use lawfully obtained operator/participant data with frozen source vintages and no lookahead.
7. **P6 — Prospective shadow mode:** compare expected outcomes with actual schedules, meters, settlement and credit-cover records before any operational use.
8. **P7 — Continuous re-underwriting:** invalidate and recompute decisions when rules, contracts, equipment, network topology, source authority, warranty or counterparties change.

The first open reference and finite cases should precede expensive vendor licences. Runners are not a prerequisite for writing the model-independent test oracle.

## 9. Decision gates

A result can be:
- pass — the declared finite test and evidence requirements pass;
- fail — a hard rule, physical or evidence invariant fails;
- indeterminate — inputs or authority are unavailable;
- not_comparable — engine scopes or inputs cannot be matched;
- not_run — no execution occurred.

Never turn indeterminate, not_comparable or not_run into a numerical zero or a pass. Never rank engines by a single weighted score before publishing hard constraints, false-safe rate, downside, runtime and comparability dimensions separately.

## 10. Current status

The synthetic vendor roster remains **not executed**. Product capability claims are literature/documentation signals, not independent feature verification. Product versions, vendor access, solver builds and common adapters are not yet fully pinned. The benchmark corpus currently contains 44 cases; the exact-head source/test mapping and conformance receipt should be refreshed after each module addition. The public M5BAT field dataset is identified (RWTH DOI 10.18154/RWTH-2025-06555), but its ZIP has not yet been staged, so the field replay and all M5BAT performance targets remain `not_run`. The Modo revenue-realization comparison is methodology-only; no same-cohort backtest has been executed. The current SAWEM rule profile keeps effective status, participant admission and commercial trading unknown.

The next useful result is a complete, independently hand-verified open reference case—not an ungrounded claim that one platform beats the market.
