# EV mission-conditioned battery capability evaluator v0.1

## Purpose and ownership

This is the first executable slice of [EV-MISSION-CAPABILITY-001 #11589](https://github.com/Luminous-Dynamics/symthaea/issues/11589).

The evaluator composes qualified upstream models; it does **not** create another equivalent-circuit model, battery digital twin, SOH predictor, thermal solver or BMS. Its job is to decide whether one exact time-varying battery-side duty profile fits the exact pack capability envelope over its full time horizon.

Existing owners remain authoritative for their layers:
- #11249/#11252: observations, SOH labels, calibration and transfer evaluation;
- #11271: topology-aware pack descriptor and bottleneck population;
- #11384/#11438: fixed-pulse cell-limited envelope and model-derived beta-power;
- #11485: dynamic 2-RC electrothermal model, time-grid bounds and active identification;
- #11404: dynamic sequential diagnostic policy;
- #11495: leakage-safe multi-output early-window targets;
- Symtropy #2112/#2114: electric vehicle mission profiles, platform boundaries and physical verification.

This crate consumes per-interval capability slices from those layers. It does not assert that those upstream bounds are valid outside their own stated domains.

## Paradigm shift: peak rating is not mission capability

The system must stop treating one SOH or peak-kW value as a complete statement of useful battery capability.

A battery may pass an instantaneous peak-power check and still fail:
- sustained energy/reserve over a long duty segment;
- repeated pulses after thermal accumulation;
- a later weak-cell voltage/SOC/temperature constraint;
- regenerative charge acceptance at high SOC or low temperature;
- a required end-of-mission reserve.

The output is mission-conditioned:

`vehicle + exact pack/configuration + duty profile + operating conditions
  -> time-indexed evidence-qualified capability decision`

The question becomes: does this exact battery support this exact requested trajectory over the entire mission, under conservative bounds and the evidence/model support available?

## v0.1 input contract

### Mission profile

A mission binds vehicle, pack, configuration digest, topology digest, profile identity/digest, exact start/end and declared cell population. It contains an ordered sequence of contiguous intervals. Each interval has:
- exact start/end time in seconds;
- requested **battery-terminal** power in kW;
- symmetric power uncertainty in kW;
- separately metered external/trolley/charger energy.

Sign convention:
- positive battery power = pack discharge;
- negative battery power = regenerative/charging power.

External energy is reported separately and **never credited** against the battery's own energy reserve. This prevents trolley, charger or site-supplied energy from masking battery demand or being counted twice.

The mission also declares:
- lower-bound energy initially available above the required end reserve;
- lower-bound regenerative energy efficiency;
- minimum and maximum cell SOC;
- maximum cell temperature;
- minimum and maximum cell voltage.

### Time-indexed capability slice

Each exact mission interval must have a corresponding upstream slice. The slice binds:
- same vehicle/pack/configuration/topology;
- exact same interval boundaries (no silent interpolation);
- model version, evidence reference and SHA-256 digest;
- source class;
- SOC and temperature support domain;
- conservative lower bound for pack discharge power;
- conservative lower bound for pack charge acceptance;
- complete cell population with SOC lower/upper, temperature lower/upper and voltage lower/upper bounds.

The capability slices may originate from synthetic known truth for software tests or an upstream qualified physical model. A model input classified as QUALIFIED_PHYSICAL_MODEL is **not independently revalidated by this evaluator**; the output says so explicitly.

## v0.1 decision logic

### Exact time and configuration coverage

- Mission intervals must be ordered and contiguous.
- There must be exactly one capability slice for every step.
- Every slice's time boundaries must exactly match its step.
- Asset, pack, configuration, topology and model digests must match.
- Cell identity population must match the declared pack. Missing units make the result NOT_COMPUTABLE; undeclared or duplicate cells create CONFLICTING_EVIDENCE.
- No extrapolation, interpolation, stale carry-forward or cross-configuration pooling is permitted.

### Conservative discharge and charge acceptance

For requested battery power (P) and symmetric uncertainty (delta_P), define:
- worst-case discharge request: (max(P+delta_P,0));
- worst-case charge request: (max(delta_P-P,0));
- guaranteed regenerative power: (max(-P-delta_P,0)).

The first two must fit the conservative lower discharge and charge-acceptance bounds supplied by the upstream pack model. Guaranteed regenerative energy is credited only when the entire uncertainty interval remains in the charging direction, capped by the upstream conservative charge-acceptance bound and multiplied by the declared lower-bound regenerative efficiency.

This also handles uncertainty intervals that straddle zero: the evaluator checks both possible power directions rather than assuming the nominal sign is certain.

### Cumulative energy and reserve

For a step of duration (Delta t):
- subtract worst-case discharge energy (P_{dis,worst}Delta t/3600);
- credit only guaranteed regenerative energy, capped by the conservative charge-acceptance bound:

`E_regen_credit = min(P_regen_guaranteed, P_charge_acceptance_lower) × eta_regen_lower × Delta_t / 3600`

The energy ledger therefore never credits regenerative energy above the upstream qualified acceptance bound. An over-acceptance request remains INFEASIBLE even though the accounting ledger only credits the conservative accepted amount.

The lower-bound energy remaining above reserve is updated after every fully evaluated step. It must never cross below zero. A mission that passes every instantaneous power limit but crosses its cumulative energy reserve is INFEASIBLE.

The report includes `battery_energy_accounting_complete`. If any interval is missing capability evidence, has incomplete cell coverage, conflicts with the mission identity, or is outside the model domain, that flag is false and the mission-wide minimum-reserve value is null. Accumulated discharge/regeneration totals are partial in that case and must not be interpreted as a full-mission result. A fully evaluated but infeasible mission retains its modeled negative reserve result.

### Cell-level constraints and domain

For every declared cell and every time slice, compare its supplied conservative trajectory bounds against:
- mission minimum/maximum SOC;
- mission maximum temperature (which may be below 0 °C for cold-soak missions);
- mission minimum/maximum voltage;
- the model's declared SOC and temperature support domain.

A cell outside the upstream model domain returns UNSUPPORTED_DOMAIN. A cell that violates a mission constraint returns INFEASIBLE. These are different: the former means the model cannot justify the claim; the latter means the supplied qualified bound violates the requested mission limit.

A limiting cell may change through the mission. The evaluator reports the hottest cell separately from the limiting cell. A pack-level power/charge bottleneck is attributed to a cell only when the upstream model supplies a matching limiting-cell reference. A pure energy-reserve shortfall does not invent a cell attribution. Pack averages do not hide cell constraints.

## Result dispositions

- `FEASIBLE_WITHIN_DECLARED_ENVELOPE`: all intervals, evidence bindings, power/charge limits, cell constraints and energy reserve are covered.
- `INFEASIBLE`: a qualified supplied bound violates a requested power, energy, SOC, thermal or voltage constraint.
- `NOT_COMPUTABLE`: one or more exact mission intervals or declared cells lack capability evidence.
- `UNSUPPORTED_DOMAIN`: the supplied cell state falls outside the upstream model's declared applicability domain.
- `CONFLICTING_EVIDENCE`: asset, configuration, topology, time or cell identities do not agree.
- `REJECTED_INPUT`: malformed mission profile, units/time ordering, duplicate identities or invalid bounds.

A disposition is a read-only model/evidence evaluation, not a permit to operate the vehicle.

## Frozen synthetic corpus

Twelve cases exercise:
1. Mixed discharge, regen and cruise mission that fits.
2. Peak power fits but sustained energy crosses reserve.
3. Repeated pulse mission whose later temperature bound crosses the mission limit.
4. Regen request that exceeds charge acceptance despite adequate discharge power.
5. Missing cell coverage -> NOT_COMPUTABLE.
6. Temperature outside model support -> UNSUPPORTED_DOMAIN.
7. Wrong configuration digest -> CONFLICTING_EVIDENCE.
8. External/trolley energy recorded separately and not credited to the pack reserve.
9. Non-contiguous mission steps -> REJECTED_INPUT.
10. Exact end-reserve boundary -> FEASIBLE.
11. Power uncertainty interval straddles zero -> check both possible discharge and charge acceptance, but credit zero guaranteed regen.
12. Cold-weather mission with negative Celsius limits and full temperature interval inside a negative-Celsius model domain -> FEASIBLE.

All values, assets, identities, model references, trajectories and evidence digests are synthetic. The fixtures validate software arithmetic and fail-closed contract behavior only. They are not measured vehicle missions, real battery limits, or physically validated thermal trajectories.

## Benchmark tournament

Compare:
- instantaneous peak-power-only screen;
- total usable-energy-only screen;
- static derated power/energy screen;
- time-indexed pack power without cumulative energy accounting;
- time-indexed pack power with reserve accounting;
- full time-indexed cell/SOC/temperature/voltage capability slices from #11485;
- worst-case robust evaluator with power/model uncertainty.

Required metrics:
- false-feasible rate (primary);
- false-infeasible rate and useful-service rejection;
- power/charge acceptance conservative-bound coverage;
- energy reserve shortfall detection;
- thermal/voltage/SOC limit-crossing detection and lead time;
- limiting-cell attribution and bottleneck migration accuracy;
- mission completion and useful work per energy/downtime;
- abstention and unsupported-domain rates;
- runtime and memory against interval count and cell count.

### Proposed research targets (not achieved)

Software:
- zero false-feasible results in frozen negative controls;
- 100% profile/asset/configuration/topology and declared-cell binding;
- zero unapproved interpolation or extrapolation;
- exact deterministic replay and permutation invariance for non-temporal arrays.

Future physical benchmark:
- preregister an upper confidence bound for false-feasible decisions;
- use independent pack-disjoint mission holdouts and matched reference protocols;
- report 90%/95% coverage of conservative power/energy/thermal envelopes by chemistry, temperature, SOC, age and duty domain;
- measure useful-work delivered, missed service, reserve violations and downtime at matched mission boundaries.

For a binary false-feasible endpoint, use a one-sided exact binomial confidence bound and count independent pack–mission pairs, not time intervals from one pack as independent trials. With zero false-feasible outcomes, 59 independent cases give a one-sided 95% upper bound of 4.95%; 299 independent cases are required for an upper bound below 1%. These are sample-size implications under the independent Bernoulli assumption, not evidence that the current fixtures or any physical model meet those rates. Clustered missions from the same pack/vehicle require cluster-aware analysis and a larger effective sample.

A small synthetic fixture pass does not validate physical model accuracy or safety.

## Research and standards context

SAE J1634's current 2026 revision is **J1634_202604**, revised 13 April 2026. It defines light-duty BEV whole-vehicle energy/range test procedures using UDDS/HFEDS and supplementary US06/SC03/cold-FTP cycles. These profiles are useful whole-vehicle baselines; a battery-only capability report must not be presented as a J1634 range result: https://doi.org/10.4271/J1634_202604

Le (28 September 2026), *Beyond standard battery testing: a physics-informed framework for validating next-generation battery packs*, proposes a progression from protocol-oriented to stress-, mechanism-, physics-informed and predictive validation. Its practical direction is application-specific duty cycles, coupled physical loads, distributed sensing and mechanism-relevant metrics. This supports extending standardized testing rather than claiming to replace it:
https://doi.org/10.3389/fbael.2026.1966110

ISO 12405-4:2018 remains published and under review; it defines pack/system performance, reliability and electrical-functionality test procedures for high-power and high-energy applications:
https://www.iso.org/standard/71407.html

SAE J3356_202609 and its SOC/SOH reports were issued on 28 September 2026. They provide diagnostic and reporting context (fault identification/classification/system reaction, SOC calculation/influence/warnings and SOH calculation/influence/reporting), but they are Information Reports, not certification:
https://doi.org/10.4271/J3356_202609
https://doi.org/10.4271/J3356/1_202609
https://doi.org/10.4271/J3356/2_202609

The novel engineering target is not a new battery simulation or generic digital twin. It is a conservative, evidence-bound, independently replayable decision about one particular battery and one particular time-varying mission.

## Safety and claim ceiling

The evaluator cannot charge, discharge, command a charger, change BMS limits, operate contactors, derate torque, authorize service or approve a vehicle mission. It does not establish functional safety, battery safety, certification or standard conformance. It evaluates mission profiles only against the supplied upstream capability slices and their declared evidence/domain.