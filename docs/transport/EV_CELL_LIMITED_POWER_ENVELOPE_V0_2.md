# EV Cell-Limited Power Envelope v0.2

**Status:** Synthetic evidence-qualified electrical/thermal envelope and β_power evaluator. Not a BMS, validated battery model, safety approval or vehicle qualification.
**Owner:** [EV-BETA-POWER-002 #11438](https://github.com/Luminous-Dynamics/symthaea/issues/11438).
**Predecessor:** [EV-POWER-ENVELOPE-001 PR #11384](https://github.com/Luminous-Dynamics/symthaea/pull/11384).
**Consumer:** [Pack inconsistency metrics v0.2 PR #11324](https://github.com/Luminous-Dynamics/symthaea/pull/11324).

## Why this work exists

The 2026 Nature Energy study defines six cell-inconsistency metrics. Its power-utilization ratio is:

[
\beta_{power}(t)=\frac{P_{pack}}{\sum_{i=1}^{N}P_{cell,i}}
]

where the numerator is maximum available pack power under the declared pack conditions and the denominator is the sum of each constituent cell's individual maximum available power. Source: Zhou et al., *Nature Energy* (2026), https://www.nature.com/articles/s41560-026-02131-5.

v0.1 calculated only the common-current pack envelope. v0.2 adds every cell's independently admissible maximum-power point under the same pulse, OCV/SOC/temperature reference, model, parameter bounds, voltage/current/thermal constraints and evidence protocol.

## v0.1 pack model retained

For each cell under a constant-current discharge pulse of duration t:

- Conservative cell voltage: V_i = OCV_i,lower − I R_i,upper.
- Voltage current ceiling: I_V,i = max(0, (OCV_i,lower − V_min,i) / R_i,upper).
- Rated current ceiling: I_C,i = I_rating,i,lower.
- Thermal current ceiling: I_T,i = sqrt((T_max,i,lower − T_base,i,upper) / G_i), if thermal margin is positive, otherwise zero.
- G_i = R_i,upper R_th,i,upper [1 − exp(−t/(R_th,i,upper C_th,i,lower))].
- T_base,i,upper is the maximum of the cell's upper-bound initial temperature and pulse ambient upper bound.

The pack's common admissible current is the minimum of all cell voltage/current/thermal ceilings. The pack's maximum terminal power is the maximum of P_pack(I)=I[ΣOCV_i,lower − IΣR_i,upper] over that common current interval.

## v0.2 cell denominator and β_power

For each cell independently:

[
I_{cell,i}^{*}=\min\left(I_{admissible,i},\frac{OCV_{i,lower}}{2R_{i,upper}}\right)
]

[
P_{cell,i,max}=I_{cell,i}^{*}(OCV_{i,lower}-I_{cell,i}^{*}R_{i,upper})
]

For the pack:

[
P_{pack,max}=\max_{0\le I\le I_{common}} I\left(\sum_i OCV_{i,lower}-I\sum_iR_{i,upper}\right)
]

Then:

[
\beta_{power,model}=\frac{P_{pack,max}}{\sum_iP_{cell,i,max}}
]

The output is labelled a model-derived ratio under the declared first-order model and adverse parameter selection. It is not a statistical lower confidence bound, not a universal BMS limit and not a physically validated fleet metric. Because each constituent cell's individual feasible current interval contains the common pack current, the pack numerator must not exceed the sum of independent cell maxima; values above 1 (outside numerical tolerance) force rejection, never clamping.

When the pack has no admissible discharge or the individual-cell denominator is zero, β_power is NOT_COMPUTABLE. No fabricated zero or infinite ratio is emitted.

## Adverse-direction uncertainty and lineage

OCV uses its lower bound; resistance and thermal resistance use upper bounds; thermal capacity uses its lower bound; cell and ambient temperature use upper bounds; minimum allowed voltage and temperature use lower bounds; current rating uses its lower bound. OCV reference state is bound to the cell's current SOC and temperature within declared uncertainty.

Receipts bind vehicle/pack configuration, hashed cell inventory, cell identity/configuration, topology qualification and authority reference, model/equation/parameters, protocol, calibration/domain, pulse, and every cell measurement and hard-limit digest with evidence class.

## Frozen synthetic corpus

Eleven synthetic cases: four computed, one explicit no-discharge, six rejected controls. Independent reference arithmetic gives:

| Case | Limiting condition | Pack maximum power | Sum of individual cell maxima | β_power |
|---|---|---:|---:|---:|
| Equal series reference | Cell voltage | 464.2857 W | 464.2857 W | 1.0000 |
| Weak-cell voltage limit | Cell 003 voltage | 185.5567 W | 363.7174 W | 0.5102 |
| Thermal limit | Cell 001 temperature | 497.3047 W | 497.3047 W | 1.0000 |
| Current-rating limit | Cell 002 current rating | 285.3000 W | 404.6238 W | 0.7051 |
| Exhausted thermal margin | No admissible discharge | Not emitted | 0 W | Not computable |

All fixtures are synthetic arithmetic/gate tests, not physical cell or pack results. Identical cells yield β_power=1 in this model; the intentionally weak cell reduces usable pack power to about 51.0% of the sum of its cells' individual maxima.

## Deliberate limits

The lumped first-order thermal model is a research approximation. It does not capture electrochemical polarization, nonlinear OCV/impedance, spatial gradients, cooling-plate dynamics, heat generation beyond I²R, ageing-dependent thermal parameters, charge capability, regenerative braking, parallel-string imbalance or switched topologies. The model is limited to a complete, qualified series-single-parallel string under one fixed discharge pulse.

It does not compute β_energy, authorize BMS action or establish safety/certification. Physical validation requires independent cell pulse/HPPC tests and matched module/pack pulse tests using the same SOC, temperature and duration, followed by fleet shadow.

## Run

`cargo fmt --manifest-path tools/ev-cell-limited-power-envelope/Cargo.toml -- --check`

`cargo test --manifest-path tools/ev-cell-limited-power-envelope/Cargo.toml`

`cargo run --manifest-path tools/ev-cell-limited-power-envelope/Cargo.toml -- --schema schemas/transport/ev_cell_limited_power_envelope_v0_1.schema.json --corpus fixtures/ev_battery/cell_limited_power_envelope_scenarios_v0_1.json`
