# Battery formation × charge-policy co-design v0.1

Status: synthetic factorial campaign contract and read-only reference audit. No battery performance, safety, cycle-life, cost or manufacturing improvement is claimed.

## Objective

Test whether the in-service charge policy that maximizes lifetime service depends on the formation protocol that created the cell's initial interphase, lithium inventory and electrode-utilization state.

The experimental unit is the physical cell. Cells are blocked by lot and/or line; treatment identity is the exact design revision × formation profile revision × charge-policy revision.

## Why this is not two independent optimizations

A 2024 Joule study evaluated 186 Li-ion cells across 62 formation protocols and reported a 50% average cycle-life increase for its high-first-cycle-current formation group. The authors linked this to lithium loss and a shifted electrode-specific utilization window. It is a study-specific result, not a universal effect: https://doi.org/10.1016/j.joule.2024.07.024

A 2026 *Advanced Science* study developed and validated a model-informed protocol for a specific commercial cylindrical cell. It reported 20–80% SOC charging in 15 minutes (versus 18 minutes for the manufacturer's specification and 25 minutes for a commercial EV profile) with less than 5% capacity loss over 500 cycles: https://doi.org/10.1002/advs.75306

These are separate cell/protocol tasks. Neither establishes that the best formation and lifetime charge policies are independent, nor that combining the reported settings would work. The research hypothesis is an interaction:

`joint effect - formation-only effect - charge-only effect + reference effect`

## Factorial design

For each exact cell design, freeze a complete formation × charge-policy matrix:

| Arm | Formation | In-service charge policy |
|---|---|---|
| Reference | Reference | Reference |
| Formation-only | Candidate | Reference |
| Charge-only | Reference | Candidate |
| Joint | Candidate | Candidate |

Extend the matrix when more than two factor levels are scientifically justified. Randomize physical cells within lot/line blocks. Record the executed waveforms and environment, not just the intended protocol names.

The v0.1 software contract requires at least two distinct formation profiles, two distinct charge policies, every Cartesian pairing per design revision, and exactly one declared reference arm. It also declares a blocking axis (lot or line) and rejects any block that does not contain every arm, so the interaction is not inferred from unpaired missing treatment groups.

## Compose existing owners

- Symthaea #11671: mission-conditioned battery inverse design and experiment selection.
- Symthaea #11686: unseen-design cycle-life transfer and early-window prediction.
- Symthaea #11485: dynamic electrothermal cell model and active identification.
- Symthaea #11271 and #11589: pack bottleneck/capability and mission feasibility.
- Symthaea #11404: diagnostic experiment planner.
- MAT-013 #4344, MAT-PSPP #6224, MAT-INTERFACE #6227 and MAT-UQ #6225: physical lineage, process/structure, interface state, uncertainty and metrology.
- EV-FIELD-EVIDENCE #11421: rights by operation.
- Symtropy #4833: manufacturing lot/line robustness; #4848: design-to-mission evidence closure.

This programme consumes those owners' outputs. It does not add another electrochemical model, BMS, manufacturing ontology, uncertainty engine, data-rights registry, cell genealogy service or experiment scheduler.

## Benchmark protocol

Compare, on the same design space, duty-cycle distribution, physical groups and experiment budget:

1. reference formation + reference charge;
2. formation-only optimization with reference charge;
3. charge-only optimization with reference formation;
4. factorial joint formation/charge co-design;
5. physics-guided or mechanistic hybrid model;
6. DOE / response-surface baseline;
7. Bayesian or multi-objective active-learning policy;
8. random and diversity/uncertainty sampling controls.

Use exact outer holdouts by physical cell, lot, production line/equipment, formation/charge protocol pair and design family where the source supports them. The split axis must be declared; a group must never straddle train/validation/test/prospective partitions on that axis. All scaling, feature selection, calibration, hyperparameter tuning and adaptation stay inside the training partition.

For early-window predictions, every feature has its own availability cycle and admitted evidence reference, and that cycle must be no later than the cell's declared cutoff. Post-cutoff capacity, impedance, endpoint, failure and lifetime information is forbidden. Start with cutoffs 1, 5, 10, 20 and 50 only where the source contains the necessary observations.

## Required physical outcomes

- Formation duration, energy, resource use and throughput.
- Executed charge time over declared SOC intervals (20–80% where applicable).
- Capacity retention, energy/power, DCIR/impedance growth and coulombic efficiency.
- Lifetime delivered energy and charge throughput under an exact EOL rule.
- Temperature and pressure trajectories; plating-sensitive diagnostics; gas/swelling; failure modes under the responsible qualified test protocol.
- Between-cell, between-lot and between-line spread; worst-group and lower-tail outcomes.
- Formation × charging interaction effect with uncertainty and independent-lot replication.
- Test time, material, energy and compute consumed to select a physically supported policy.

Right-censored life is not an observed EOL event. Competing failure, invalid test, lost specimen and incomplete endpoint remain separate outcomes.

## Baselines and comparison targets

First reproduce the two published studies on their own exact cell designs, datasets, splits and definitions, where rights and data access permit. Do not pool their cells or compare unlike cycle-life denominators.

A joint-policy advantage is demonstrated only when the factorial policy beats the best single-axis policy on a preregistered physical confirmation set with independent lot/line groups. Report the charge-time/lifetime-throughput/formation-cost Pareto frontier; do not reduce it to an opaque scalar.

Set physical percentage-improvement thresholds only after measuring baseline repeatability, metrology uncertainty and statistical power. Software targets are stricter: zero factor-pair omissions, zero holdout leakage, zero post-cutoff feature use, zero censored-to-observed promotion, zero unadmitted physical claims, and deterministic replay.

## Data rights

The 186-cell formation study's associated dataset is identified in the Symtropy manufacturing benchmark as CC BY-NC 4.0. Keep it commercial-use-blocked unless additional permission is obtained. The 2026 fast-charge paper is open access, but article access does not imply availability or rights for every underlying raw-data or derived-model operation. Use Symthaea #11421 to qualify each operation (local processing, evaluation, training, derived-data release, model-weight release, commercial use and redistribution) before ingestion.

## Safety boundary

All optimization is offline and read-only. No software in this campaign schedules tests, purchases materials, synthesizes cells, runs formation, charges/discharges cells, operates a chamber/charger, changes BMS settings or authorizes hazardous work. Physical tests remain under the responsible lab's qualified protocol, interlocks and safety authority.

## Claim ceiling

A passing software audit proves only factorial coverage, identity, split, cutoff, status and evidence semantics. Retrospective reproduction proves only the exact source task. Neither proves a new battery lifetime gain, absence of plating, safety, pack qualification or commercial readiness. Prospective physical confirmation, independent replication and appropriate product qualification remain necessary.
