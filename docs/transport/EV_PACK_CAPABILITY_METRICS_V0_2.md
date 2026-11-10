# EV Pack Inconsistency Metrics v0.2

**Status:** Synthetic evidence-qualified metrics evaluator. Not a battery model, BMS or automotive qualification artifact.
**Owner:** Symthaea [EV-PACK-INCONSISTENCY-METRICS-001 #11324](https://github.com/Luminous-Dynamics/symthaea/issues/11324).
**Predecessor:** [EV pack capability profile v0.1 PR #11274](https://github.com/Luminous-Dynamics/symthaea/pull/11274).

## Research anchor

Zhou et al., *Quantifying the impact of cell-to-cell inconsistency on electric vehicle battery degradation and utilization*, Nature Energy, published 25 August 2026: https://www.nature.com/articles/s41560-026-02131-5

The study analyzes more than three years of operational data from 116 NMC passenger cars and 17 LFP buses. It defines six different quantities. Its fleet findings (including β_life of about 0.823 / 0.772 and β_energy around 0.807 / 0.73) are population- and method-specific context, not universal constants or target labels for our synthetic fixtures.

## Metric contract

| Metric | v0.2 behavior | Exact prerequisites |
|---|---|---|
| σ_SOH | Computed | Comparable per-unit SOH; always labels full population vs observed subset |
| β_SOH | Computed when qualified | Complete cell population and explicitly qualified series-single-parallel topology |
| β_life | Computed when qualified | Same topology/population plus observed EOL and EFC evidence for every declared cell under a declared EOL rule |
| β_SOC | Computed when qualified | Same topology/population, homogeneous nominal cell-capacity basis, and a synchronized, complete end-of-charge cell SOC snapshot |
| β_power | NOT_COMPUTABLE | Qualified cell and pack electrothermal power model is not part of this profile version |
| β_energy | NOT_COMPUTABLE | Requires lifetime-average SOC utilization over the same lifetime boundary as β_life; a single SOC snapshot is insufficient |

### Definitions

For each unit, SOH_i = capacity_i / nominal_capacity_i.

σ_SOH is the population standard deviation of the observed SOH values. If population coverage is incomplete, its status says `observed_subset_only`; it is not extrapolated to missing units.

For a complete, cell-resolution, qualified series-single-parallel pack with a homogeneous nominal cell-capacity basis:

- β_SOH = min(SOH_i) / mean(SOH_i).
- β_life = N × min(EFC_i at EOL) / Σ EFC_i at EOL.
- β_SOC = 1 − Σ[(SOC_max − SOC_i) × SOH_i] / Σ SOH_i for one synchronized end-of-charge snapshot.

The topology and nominal-capacity restrictions are intentional: these formulas encode the paper's series-connected, single-parallel operating assumptions. β_SOH and β_SOC additionally require a homogeneous nominal cell-capacity basis so a weakest-to-mean ratio or SOH-weighted SOC gap is not misrepresented across unequal rated cell sizes. v0.2 must not generalize β_SOH, β_life or β_SOC to arbitrary module, parallel-group or reconfigurable pack arrangements.

EFC and EOL evidence are supplied per physical cell, not inferred from pack age or mileage. The paper uses a declared SOH EOL threshold of 0.85 for its studied fleet because the vehicles had not yet reached the commonly used 0.80 threshold; v0.2 does not hard-code either value. The threshold and rule are part of the evidence record. A right-censored or unknown EOL record makes β_life NOT_COMPUTABLE; records are never silently removed.

β_power remains NOT_COMPUTABLE. Resistance dispersion is not a power estimate. The paper's power metric needs cell and pack power over an explicit prediction horizon, current/voltage limits, ECM parameters and a valid model domain.

β_energy remains NOT_COMPUTABLE. The paper defines it as β_life multiplied by the lifetime-average β_SOC. v0.2 has one snapshot only, not a qualified time-weighted SOC history across the complete service life.

## Evidence and output rules

Every computed metric carries status, value, unit, population scope and a list of evidence references. Each reference binds its role, exact digest and evidence class, so a digest cannot be detached from its classification (measurement, derived estimate, prediction, simulation, synthetic fixture or vendor claim). A topology-dependent metric also binds the topology qualification record and its responsible authority reference. A digest establishes byte identity, not truth or authority; the authority reference identifies the accountable qualification source but does not itself constitute a regulatory certificate. Non-computable metrics carry a stable reason code. No single weighted battery-health score is emitted. Hard topology, coverage and EOL gates are not compensable.

The SOC snapshot binds the exact pack configuration, end-of-charge event, reference condition, evidence digest, evidence class for each cell, and a maximum sample-skew window. The synchronization tolerance is declared in the fixture/protocol; it is not silently widened to accommodate late or stale samples. A snapshot missing any declared cell or exceeding the synchronization window cannot produce β_SOC.

The lifetime contract binds the pack configuration, EOL rule and digest, service-history digest, EFC method and digest, per-cell EFC, EOL event, evidence digest and evidence class. A censored/unknown unit is retained and prevents β_life.

Per-unit capacity/resistance uncertainty is preserved but not propagated into population confidence intervals in v0.2.

## Frozen synthetic corpus

Eight cases cover a fully qualified simple-series pack, partial population, unknown topology, parallel-series topology, a censored EOL record, asynchronous SOC rejection, duplicate EOL identity rejection, and module-level observations. Expected metric results include the four supported computed/not-computable states above. All source rows are explicitly synthetic.

## Run

`cargo fmt --manifest-path tools/ev-pack-capability-metrics-audit/Cargo.toml -- --check`

`cargo test --manifest-path tools/ev-pack-capability-metrics-audit/Cargo.toml`

`cargo run --manifest-path tools/ev-pack-capability-metrics-audit/Cargo.toml -- --schema schemas/transport/ev_pack_capability_metrics_v0_2.schema.json --corpus fixtures/ev_battery/pack_capability_metrics_v0_2.json`

## Claim ceiling

This evaluator is a deterministic metric calculation and evidence-gate prototype. It is not a topology-aware electrothermal battery model, a BMS, diagnostic service, safety approval, certified test, field validation, pack capability qualification or proof that balancing/repair improves a battery.
