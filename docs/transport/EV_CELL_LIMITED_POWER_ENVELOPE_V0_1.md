# EV Cell-Limited Power Envelope v0.1

**Status:** Synthetic evidence-qualified electrical/thermal envelope prototype. Not a BMS, validated battery model, safety approval or vehicle qualification.
**Owner:** [EV-POWER-ENVELOPE-001 #11384](https://github.com/Luminous-Dynamics/symthaea/issues/11384).
**Related:** [Pack inconsistency metrics v0.2 PR #11324](https://github.com/Luminous-Dynamics/symthaea/pull/11324).

## Why this work exists

The pack inconsistency programme correctly leaves β_power NOT_COMPUTABLE when it has only resistance dispersion. This prototype supplies a narrowly scoped power calculation prerequisite: for a complete, explicitly qualified series-single-parallel string, determine the largest conservative common discharge current allowed by every cell's voltage, current-rating and pulse-temperature limits.

It does not yet calculate the Nature Energy paper's β_power ratio. That requires implementing the exact numerator/denominator aggregation and matching its stated operating point and topology assumptions to the model outputs.

## v0.1 model

For each cell under a constant-current discharge pulse of duration t:

- Conservative cell voltage: V_i = OCV_i,lower − I R_i,upper.
- Voltage current ceiling: I_V,i = max(0, (OCV_i,lower − V_min,i) / R_i,upper).
- Rated current ceiling: I_C,i = I_rating,i,lower.
- Thermal current ceiling: I_T,i = sqrt((T_max,i,lower − T_base,i,upper) / G_i), if thermal margin is positive, otherwise zero.
- G_i = R_i,upper R_th,i,upper [1 − exp(−t/(R_th,i,upper C_th,i,lower))].
- T_base,i,upper is the maximum of the cell's upper-bound initial temperature and the pulse ambient upper bound.

The pack's maximum admissible common current is min_i min(I_V,i, I_C,i, I_T,i). The limiting cell and binding constraint are preserved. Within that feasible current interval, the conservative terminal power envelope uses the maximum of P(I)=I[ΣOCV_i,lower − IΣR_i,upper], with its unconstrained optimum clipped to the admissible interval. Output power is DC at the declared pack terminals for the specified pulse horizon.

## Adverse-direction uncertainty

OCV uses its lower bound; resistance and thermal resistance use upper bounds; thermal capacity uses its lower bound; cell temperature and ambient temperature use upper bounds; minimum allowed terminal voltage and temperature use lower bounds; current rating uses its lower bound. A zero or negative thermal margin produces NO_ADMISSIBLE_DISCHARGE, not a positive power estimate.

All cell identities, the hashed pack cell-inventory manifest, asset/configuration digests, topology qualification and authority reference, model/equation/parameter digests, protocol, calibration and domain evidence are bound into the output receipt with evidence class. Voltage/current limits and ambient conditions also carry evidence digests and evidence classes.

## Frozen corpus

Nine synthetic cases include equal-cell reference, weak-cell voltage limit, thermal limit, current-rating limit, an exhausted thermal-margin no-discharge result, and four fail-closed controls (SOC out of domain, temperature out of domain, unsupported topology and invalid calibration). These are arithmetic and gate tests only.

## Deliberate limits

The lumped first-order thermal model is a research approximation. It does not capture electrochemical polarization, nonlinear OCV/impedance, spatial gradients, cooling-plate dynamics, heat generation beyond I²R, ageing-dependent thermal parameters, charge capability, regenerative braking, parallel-string imbalance or switchable topologies. It assumes the same current through every cell in one series string. The model must not be used for control or safety decisions.

## Validation ladder

Synthetic known-truth and mutation tests → independently measured cell pulse/HPPC data → module/pack pulse tests at matched SOC, temperature and duration → fleet shadow comparisons. At each stage report absolute power error, conservative-bound coverage, limiting-cell identification, voltage/temperature margin, worst chemistry/SOC/temperature domain, compute and evidence coverage. No stage can be skipped by passing software tests.

## Run

`cargo fmt --manifest-path tools/ev-cell-limited-power-envelope/Cargo.toml -- --check`

`cargo test --manifest-path tools/ev-cell-limited-power-envelope/Cargo.toml`

`cargo run --manifest-path tools/ev-cell-limited-power-envelope/Cargo.toml -- --schema schemas/transport/ev_cell_limited_power_envelope_v0_1.schema.json --corpus fixtures/ev_battery/cell_limited_power_envelope_scenarios_v0_1.json`
