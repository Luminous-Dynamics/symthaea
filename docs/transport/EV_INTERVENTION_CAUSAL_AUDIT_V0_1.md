# EV Intervention Causal Audit v0.1

**Status:** Synthetic known-truth software prototype. It does not establish a physical battery benefit or identify causal effects in field data.
**Owner:** Symthaea [EV-INTERVENTION-EFFECT-001 #11289](https://github.com/Luminous-Dynamics/symthaea/issues/11289).

## Purpose and paradigm shift

The evaluation target is not whether a service event was recommended or whether a work order was closed. It is whether an independently verified intervention changed a declared battery outcome relative to a defensible counterfactual, under comparable reference conditions, with the uncertainty and missing follow-up kept visible.

v0.1 implements one narrow estimator: matched-pair difference-in-differences (DiD). It is a software and arithmetic qualification step, not a claim that DiD assumptions hold in any real fleet.

For matched pair i:

`DiD_i = d[(Y_{T,post}-Y_{T,pre})-(Y_{C,post}-Y_{C,pre})]`

where d=+1 for higher-is-better metrics and d=-1 for lower-is-better metrics. Before estimating the post-period effect, v0.1 compares one pre-intervention trend interval for each matched set and abstains when the treated/comparator trend gap exceeds the scenario's declared tolerance. This is a negative-control screen, not proof of parallel trends. If the screen passes, the scenario effect is the unweighted mean of the matched-pair effects. The naive treated-only pre/post change is reported separately to expose regression-to-the-mean and common-trend traps.

## Hard evidence gates

- The scenario must identify a synthetic fixture, the outcome and its unit, outcome direction, and the prespecified matched-DiD estimand.
- Each matched set binds exactly one treated unit and one comparator to the same matching-profile digest.
- Vehicle, pack, before/after configuration, observation, protocol and reference-condition identities are preserved.
- Pre/post protocol and condition digests must match; matched units must use the same reference protocol and conditions.
- A treated unit must say `verified_complete` and include procedure, service event, authority and completion-evidence identities. `work_order_only`, unknown or unperformed states are rejected.
- A comparator must declare `no_intervention`; contaminated comparators are rejected.
- A unit with incomplete follow-up remains in the report as censored. v0.1 abstains on the entire scenario rather than silently dropping that unit or treating missing outcomes as zero.
- Any missing or inconsistent evidence blocks an estimate.

## Frozen synthetic corpus

Eight cases exercise:
- a positive difference-in-differences effect (+7 kWh);
- a regression-to-the-mean negative control with a +4.5 kWh naive treated-only change but zero DiD effect;
- a harmful intervention (-5 kWh);
- a censored follow-up that must force ABSTAIN while retaining the censored unit identity;
- a pre-trend mismatch that must force ABSTAIN before a treatment effect is reported;
- a lower-is-better resistance outcome, verifying that beneficial reductions have a positive normalized effect;
- a work-order-only control that must be rejected;
- protocol drift that must be rejected.

The fixture's expected effects are known-truth synthetic labels, not empirical performance. The audit checks accepted, abstained and rejected dispositions, effect arithmetic, and a deterministic report digest.

## v0.1 boundaries

This prototype does not prove parallel trends: its single pre-period comparison is only a screen. It does not estimate confidence intervals, adjust for covariates, perform propensity weighting, fit interrupted time series, account for staggered treatment, model recurrent failures, estimate survival effects, or diagnose unmeasured confounding. It does not establish parallel trends, no interference, exchangeability, treatment consistency or absence of concurrent changes. The matching-profile digest is an evidence binding, not proof that the matching design is statistically adequate.

No physical benefit, battery-life extension, cost saving, safety improvement, service recommendation, warranty conclusion, BMS authority or automotive qualification follows from this synthetic audit. A recommendation, work order, verified completion, and independently measured outcome remain distinct states.

## Benchmark progression

1. **Software invariants:** exact outcome arithmetic; row-order invariance; 100% retention and disclosure of censored units; 0 synthetic-to-physical claim promotion; 0 estimates from unverified service or mismatched protocols.
2. **Synthetic causal stress tests:** regression to the mean, common trend, non-parallel trends, time-varying confounding, selection into treatment, non-adherence, measurement drift, configuration change, informative censoring, retirement and adverse-event competing risks.
3. **Estimator baselines:** naive pre/post, matched-pair change, DiD, event-study/pre-trend checks, interrupted time series, synthetic controls and propensity-weighted comparisons. Only compare methods when their assumptions and target estimands are aligned.
4. **Statistical qualification:** bias, RMSE, false-benefit and false-harm rates, interval coverage, interval width, sensitivity to unmeasured confounding, censoring sensitivity, subgroup heterogeneity and robustness to alternate valid specifications.
5. **Physical studies:** controlled repeatability tests, matched packs, preregistered follow-up, independent reference instrumentation, fleet-level held-out replication. Numeric field targets require baseline variance, minimum practically important effect, a power calculation and an independently approved protocol.

## Industry and standards alignment

IEC 62660-1:2018 specifies propulsion-cell performance and life testing. The third-edition IEC 62660-1 work is still a draft/CDV and must not be treated as a published replacement. ISO 12405-4:2018 is the pack/system performance-testing reference; its public ISO status page says it is under review and to be revised. The software should record applicable standard edition, protocol, deviations and raw test evidence, but does not certify compliance.

For EU Battery Passports, use the Commission's current category-specific guidance and Regulation (EU) 2023/1542 as authoritative inputs. The Commission lists 18 February 2027 for the relevant passport obligation and says the registry became operational on 20 July 2026. Its FAQ says simple repair does not require a new passport; repurposing and remanufacturing do, with the new passport linked to the previous one. Do not hard-code all 71 guidance data points as universally mandatory: applicability differs by category and legal timing.

## Run

`cargo fmt --manifest-path tools/ev-intervention-causal-audit/Cargo.toml -- --check`

`cargo test --manifest-path tools/ev-intervention-causal-audit/Cargo.toml`

`cargo run --manifest-path tools/ev-intervention-causal-audit/Cargo.toml -- --schema schemas/transport/ev_intervention_causal_scenario_v0_1.schema.json --corpus fixtures/ev_battery/intervention_causal_scenarios_v0_1.json`
