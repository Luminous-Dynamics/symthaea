# Luminous Aqua — Task-Selective Delivery and Boost Qualification Protocol v0.1

**Status:** software contract and unoccupied benchmark preregistration; no Aqua boost hardware or task-performance result exists  
**Workstream:** [AQUA-TASK-SELECTIVE-DELIVERY-001 #12798](https://github.com/Luminous-Dynamics/symthaea/issues/12798)  
**Compiler interface:** #12749  
**Spray-field owner:** #12692  
**Pressure / method owner:** #12784 / #12817  
**Power + DWHR:** #12816  
**Lifecycle:** #12727

## 1. Paradigm shift — allocate useful flow to task phases, not the whole shower

Aqua should investigate a sequence of task phases:
- preparation / warm-up;
- low-flow base delivery;
- optional, explicitly user-authorized boost for a defined action;
- dwell / pause / water break;
- rinse;
- return-to-ready, cleaning and purge.

The research question is whether a task-specific delivery sequence reduces fresh water and purchased energy per **completed task** while preserving independently assessed task outcome, spray-field quality, temperature stability, user agency and serviceability.

A low-flow base plus boost is not a new feature by itself. Commercial hold-to-boost precedent exists. Aqua's potential contribution is the integrated, evidence-qualified service plan and matched task-level performance.

## 2. Commercial baseline — Hansgrohe WaterBoost and a documentation ambiguity

Hansgrohe's South African SKU **28036000** (Activera Select S 95 2jet EcoSmart+ with booster) lists at 3 bar:
- Rain regular mode: 2.9 L/min;
- ImpactRain regular mode: 2.9 L/min;
- ImpactRain + Booster: 6.6 L/min;
- Rain + Booster: 9.4 L/min;
- flow regulator: 4 L/min (possible tolerance −20%);
- maximum flow rate: 4 L/min;
- minimum flow pressure: 0.8 bar;
- Water Boost feature and QuickClean+ nozzle-cleaning feature.

The public sheet therefore contains an unresolved scope conflict: boosted-mode values of 6.6 / 9.4 L/min appear alongside a 4 L/min regulator / maximum-flow declaration. Do not assume the boost bypasses the regulator, that the regulator applies only in base mode, or that any value is a typo. Until the manufacturer clarifies the exact flow path or an independent lab reproduces the exact SKU and mode, the boosted values are **manufacturer-declared but not eligible as a verified comparator result**.

Sources:
- Hansgrohe South African SKU 28036000 page: https://pro.hansgrohe.co.za/articledetail-activera-select-s-hand-shower-95-2jet-ecosmart-with-booster-28036000
- Official South African datasheet: https://pro.hansgrohe.co.za/download-datasheet?lang=en_ZA&productCode=28036000
- Global manufacturer's page: https://www.hansgrohe.com/articledetail-activera-select-s-hand-shower-95-2jet-ecosmart-with-booster-28036000

Aqua must reproduce **the exact article, exact spray mode, actual regulator configuration and boost activation state**. A family-level or marketing-page claim is not sufficient.

## 3. Formal method baseline — UWL boost function test

The Unified Water Label (UWL) Taps & Showers Technical Criteria, January 2023, defines a boost function as a device that temporarily increases flow and automatically cancels when the flow control is closed.

Clause 12.1 test sequence:
1. Install the product as used and apply UWL P2.
2. Open regular flow without boost and record (Q_{reg}).
3. Activate boost and record (Q_{boost}).
4. Close the flow control.
5. Re-open at maximum possible flow **without actively using boost** and record (Q_{max}).

Clause 12.1.2 requires:
- automatic return to regular flow after flow-control shutoff;
- (Q_{max} = Q_{reg});
- (Q_{boost} < 2 Q_{reg});
- the regular flow (Q_{reg}) is the value listed on the scheme label.

For UWL Type 1 P2, dynamic pressure is applied to **each inlet** at 0.30 MPa +0.02/−0 MPa (300–320 kPa). This is not one upstream gauge reading and is not interchangeable with terminal-fitting-inlet, faceplate-centre or EN point-of-discharge measurements.

Source: [UWL Taps & Showers Technical Criteria, January 2023, sections 3.12, 4.2 and 12.1](https://uwla.eu/wp-content/uploads/2023/05/TAPS-Showers-Technical-2023.pdf).

### Aqua software treatment

The Rust `standard_methods::UwlBoostFunctionObservation` record keeps:
- exact specimen, SKU, mode and hardware configuration;
- separate hot/cold Type 1 P2 inlet pressure channels;
- regular, boosted and post-close/reopen flow observations;
- flow instrument/calibration identity and declared measurement uncertainty;
- boost activation and close/reopen sequence receipts;
- observation that boost has reset after flow-control closure.

Its conservative checks require the boost increase to be resolved above regular-flow uncertainty, the boosted upper bound to remain strictly below twice the regular lower bound, and post-close flow to agree with regular flow within their combined declared measurement uncertainty. This is a software evidence-shape and numeric-bound validator; it is not a UWL conformity test or independent test report. Where a laboratory's method uncertainty or scheme interpretation requires a different decision rule, that must be resolved before formal use.

## 4. ETL user-behaviour features are not synonymous with boost

The UK Energy Technology List (ETL) is a **non-domestic** programme. It defines user-behaviour features (UBFs) as built-in features that affect or react to user behaviour to reduce total water volume through reduced shower duration; its examples include push valves, duration/resource displays, timed shutoff and water or temperature breaks.

A hold-to-boost button is not automatically an ETL UBF. It may increase flow for a task and be useful under the UWL boost test, but ETL UBF eligibility depends on the function actually reducing total shower water/duration under the scheme's method.

ETL's current electric-shower pathways are also separate:
- standard maximum 4.5 L/min;
- up to 9 L/min when power is modulated to minimize electricity while maintaining stable output temperature with ETL-compliant wastewater heat recovery;
- up to 10.6 L/min when qualifying UBFs are also included.

These are non-domestic UK programme criteria, not South African requirements or Aqua results.

Sources:
- [UK ETL Showers — definitions and criteria](https://etl.energysecurity.gov.uk/products/showers/showers)
- [ETL 2025/26 criteria update, 15 September 2026](https://etl.energysecurity.gov.uk/news-events/news/202526-annual-research-programme-updated-criteria-now-live)

## 5. Exact product capability contract

The Rust `task_phase` module validates an advisory plan against a capability catalogue. Each capability is bound to:
- exact market variant / SKU;
- hardware configuration digest;
- delivery-condition digest covering pressure, inlet water, temperature, room, mounting and task context;
- method digest;
- test qualification receipt;
- role (base, boost or rinse);
- qualified flow at its exact pressure point;
- local activation requirements;
- maximum continuous boost duration where the local product qualification defines one.

Manufacturer-declared, ambiguous, stale or revoked modes can be stored for competitor/lab research, but cannot be selected in a qualified candidate service plan. A boost mode requires an exact, independently qualified capability, explicit task-specific user consent, direct user hold with automatic return on flow-control closure, and a locally qualified maximum continuous duration. This module does not actuate or time a physical valve.

## 6. Candidate phase vocabulary and ordering

| Phase | Mode required? | Contract |
|---|---|---|
| PREPARE | No | Warm-up / task preparation; all resources still counted by owner ledgers |
| BASE_DELIVERY | Exact base mode | At least one required |
| OPTIONAL_BOOST | Exact boost mode | Only after base/rinse; explicit task-specific consent; direct local hold + auto-return; duration within exact local qualification |
| PAUSE | No | Water break / dwell; it is not assumed to be zero-resource until measured |
| RINSE | Exact rinse mode | Only with exact qualified rinse capability |
| RETURN_TO_READY | No | End of task phase sequence; cleaning / purge and readiness are separate accounted operations where applicable |

Plans must start with PREPARE and end with RETURN_TO_READY. Internal PREPARE/RETURN phases are rejected. Every phase has a positive bounded duration; arithmetic overflow is rejected. The plan summary reports time allocation only—no water volume is inferred from nominal flow.

## 7. Matched benchmark arms

Use one exact task profile, exact hardware installation, pressure/temperature context, water condition and independent outcome method across arms:

| Arm | Delivery strategy | Research question |
|---|---|---|
| B0 | Constant low-flow base | Minimum-flow service baseline |
| B1 | Constant high-flow | Strong performance / water-use control |
| B2 | Manual hold-to-boost | Commercially familiar task-selective control |
| B3 | Deterministic phase schedule | Value of explicit task sequencing without AI |
| B4 | Symthaea bounded recommendation | Incremental value of intent interpretation and preference consent |
| B5 | Time/resource display or pause / water-break feature | Duration and interruption effect; evaluate as a separate UBF candidate |

Do not compare different tasks, duration, temperature, installation, pressure, water batch or accounting boundaries. Do not describe an interface feature as saving water unless the measured complete task demonstrates it.

## 8. Task protocol — shampoo wash / rinse as the first surrogate

Begin with an unoccupied, repeatable surrogate task:
- fixed surrogate hair/fibre material and surface area;
- frozen, non-hazardous model soil / rinse tracer and application mass;
- fixed pre-wet, application, dwell and rinse sequence;
- fixed inlet-water temperature profile, pressure method and shower mode;
- spatially resolved residual tracer / soil measure after rinse;
- standardized warm-up and return-to-ready boundary.

The surrogate method must first demonstrate measurement repeatability and a meaningful relationship to the task endpoint. It is not equivalent to human hair-washing experience. Human-factors validation and any human exposure study require separate approval.

A failed or incomplete rinse remains a failed attempt; its water and energy are retained in the accounting. It cannot be removed to make a candidate appear more efficient.

## 9. Required outcomes and accounting

Primary functional unit: one completed, independently assessed task with non-inferior task outcome.

Report separately:
- litres of fresh water per completed task;
- purchased kWh per completed task;
- base-flow seconds, boost seconds, pause seconds and rinse seconds;
- number of boosts, user-held duration and premature/repeated activations;
- spatial rinse / residue endpoint and its uncertainty;
- standardized spray-force / coverage outcomes where applicable;
- bather-plane temperature stability;
- noise and pressure transients;
- failed/aborted tasks and retries;
- warm-up, transition, cleaning, purge, standby allocation and return-to-ready burden;
- user corrections and manual intervention (human study only);
- maintenance and post-clean performance.

Symtropy owns canonical hydraulic/thermal and water/energy resource closure. Mycelix owns instrument identity, calibration, uncertainty and provenance. Aqua must not calculate a parallel water/energy ledger from nominal flow x phase duration or turn surrogate residue into a comfort/hygiene score.

## 10. Research targets — hypotheses, not results

- **UWL boost-method comparator:** (Q_{max}=Q_{reg}) after flow-control close/reopen and (Q_{boost}<2Q_{reg}), under the controlled UWL 12.1 procedure.
- **Task-level water stretch:** at least 20% lower fresh water per completed task than matched constant-high-flow control, with independently assessed task outcome non-inferior and no unacceptable increase in task time, intervention or failure.
- **No unqualified boosts:** 100% of candidate boost phases bind to exact SKU/configuration, current qualification receipt, user consent and a local direct-hold / auto-return capability.
- **No silent boost persistence:** no boost after flow-control closure, user release, or qualified maximum-duration limit.
- **Energy:** any claimed reduction must be measured at whole-system purchased-energy boundary, including heater, pump, control, warm-up and standby; do not infer it from flow alone.
- **Robustness:** repeat task and boost tests at pressure/method points permitted by the exact article and apparatus; UWL P2 boost qualification does not prove low-pressure performance at 50–100 kPa.

The 20% target is a research hypothesis, not a standard limit or a result. Determine the task non-inferiority margin, pilot variance and confirmation sample size before unblinding.

## 11. Claim ceiling and physical boundary

This protocol authorizes software-contract, competitor, synthetic and unoccupied test planning work only. It does not authorize powered/pressurized fixture control, prototype procurement, human exposure, occupied testing, construction, certification or comfort/hygiene claims. Local thermostatic and mechanical protections, water-quality gates and emergency isolation remain independent of Symthaea.
