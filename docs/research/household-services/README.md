# Verified Household Services — research index

This programme develops household equipment and utility services that are measurable at their actual delivery boundaries, independently checkable, locally operable and practically repairable.

The governing principle is:

> A product cycle, controller status, tank inventory or repair log is not proof that the household received the intended service.

## Programme owners

| Area | Owner | Purpose |
|---|---|---|
| Portfolio and product boundary | [HOME-SVC-001 #10322](https://github.com/Luminous-Dynamics/symthaea/issues/10322) | Shared product/service thesis and cross-project boundaries |
| Household utility coordination | [HOME-UTILITY-COORD-001 #10342](https://github.com/Luminous-Dynamics/symthaea/issues/10342) | Coordinate water, energy and thermal service without centralizing appliance safety authority |
| Service continuity and recovery | [HOME-SERVICE-CONTINUITY-001 #10445](https://github.com/Luminous-Dynamics/symthaea/issues/10445) | Measure declared service floors during interdependent utility disruptions |
| Water resilience | [WATERLOOP-APPL-001 #10323](https://github.com/Luminous-Dynamics/symthaea/issues/10323) | Household storage, delivery, leak resilience and bounded non-potable use |
| Repairability and lifetime service | [HOME-REPAIR-QUAL-001 #10358](https://github.com/Luminous-Dynamics/symthaea/issues/10358) | Parts, firmware, maintenance, repair and post-repair requalification |
| Repair recurrence and durability | [HOME-REPAIR-RECURRENCE-001 #10431](https://github.com/Luminous-Dynamics/symthaea/issues/10431) | Measure recurrence, repeated repair burden and censored follow-up |
| Metrology and evidence strata | [HOME-SVC-METROLOGY-001 #10341](https://github.com/Luminous-Dynamics/symthaea/issues/10341) | Repeatability, reproducibility, production variation and household validity |

## Current research specifications under review
- [WaterLoop machine-readable competitor ledger v0.1](waterloop-competitive-comparator-ledger-v0.1.json) — source-level claims, exact model/class scope, converted listing units, unresolved discrepancies and functional-unit boundaries; includes JoJo, Moen, Hydraloop, AQUALOOP and Phyn.
- [WaterLoop competitive evidence v0.1](waterloop-competitive-evidence-v0.1.md) — separates outage backup, greywater source substitution and leak containment; reconciles manufacturer capacity with certification listing scope, source-stream classes, local jurisdiction and offline connectivity.

- [#10342 — household water/energy/thermal coordination](https://github.com/Luminous-Dynamics/symthaea/issues/10342): [household-resource-arbitration-contract-v0.1.md](household-resource-arbitration-contract-v0.1.md) defines non-compensatory safety/quality/consent gates, local appliance authority, uncertainty-bounded resource planning and 14 adversarial HRA scenarios. Design-only; no live control.

- [PR #10386 — household services charter, physical validation protocols and competitive frontier](https://github.com/Luminous-Dynamics/symthaea/pull/10386). Those research files are on the PR branch and should not be assumed present on main until merged.
- [PR #10432 — repair recurrence and post-repair durability contract](https://github.com/Luminous-Dynamics/symthaea/pull/10432). Defines repair episode identity, independent restoration, use exposure, competing events, recurrent burden and censoring.
- [PR #10446 — household service continuity and recovery benchmark](https://github.com/Luminous-Dynamics/symthaea/pull/10446). Defines service envelopes, matched outage comparators, recovery trajectories and service-deficit metrics.

The PR links are used intentionally: a research specification on an unmerged branch must not appear as a relative path that looks available on main.

## Current software evidence chain

The following pull requests form the household synthetic-evidence and receipt sequence. They are separate from physical appliance qualification:

1. [#10386 — research charter and competitive frontier](https://github.com/Luminous-Dynamics/symthaea/pull/10386)
2. [#10388 — offline synthetic fixture validator](https://github.com/Luminous-Dynamics/symthaea/pull/10388)
3. [#10389 — deterministic water and repair evaluator](https://github.com/Luminous-Dynamics/symthaea/pull/10389)
4. [#10407 — typed receipts bound to exact source bytes](https://github.com/Luminous-Dynamics/symthaea/pull/10407)
5. [#10476 — restore the exact known-good Cargo dependency/lockfile pair](https://github.com/Luminous-Dynamics/symthaea/pull/10476), stacked on #10407
6. [#10480 — interval-valued qualified service-deficit evaluator](https://github.com/Luminous-Dynamics/symthaea/pull/10480), stacked on #10476 and tracked by [#10474](https://github.com/Luminous-Dynamics/symthaea/issues/10474)

The #10476 exact-head focused evaluator workflow passed formatting, locked tests, strict Clippy and all 20 synthetic case evaluations. The #10480 exact-head focused workflow passed on commit `2399a0d2c00e90d3e259dc9835c85c0776a8ce9e`: 42 tests passed (31 existing + 11 new), formatting, strict Clippy and all 20 synthetic case evaluations; batch artifact uploaded at [run #41902426526](https://github.com/Luminous-Dynamics/symthaea/actions/runs/41902426526). Repository-wide CI, PR Governance and Workflow Syntax were skipped, not passed. The earlier pre-rebase PR #10475 is closed as superseded.

The latest #10480 implementation also separates electrical kWh from useful thermal kWh, preventing cross-carrier comparisons. It is tracked by [HOME-SVC-ENERGY-CARRIER-001 #10486](https://github.com/Luminous-Dynamics/symthaea/issues/10486). Its exact-head focused workflow passed on commit `f3a1e47bf541d96d3a768979e48725fe2d3e7f55`: 44 tests passed (31 existing + 13 new), formatting, strict Clippy and all 20 synthetic scenarios; artifact uploaded at [run #41904012267](https://github.com/Luminous-Dynamics/symthaea/actions/runs/41904012267). Repository-wide CI, PR Governance and Workflow Syntax were skipped, not passed.

WaterLoop's separate competitive work is tracked by [#10483](https://github.com/Luminous-Dynamics/symthaea/issues/10483) and draft [PR #10485](https://github.com/Luminous-Dynamics/symthaea/pull/10485), which separates outage backup, greywater substitution and leak containment and adds source-stream, local-installation and offline-operation qualification.

The current follow-on research documentation is under review:
- [#10432 — repair recurrence and post-repair durability](https://github.com/Luminous-Dynamics/symthaea/pull/10432)
- [#10446 — household service continuity and recovery](https://github.com/Luminous-Dynamics/symthaea/pull/10446)

## Evidence boundaries

- Synthetic fixture validation establishes software-contract conformance only.
- Content digests establish byte identity, not source authenticity, calibration validity or physical truth.
- Independent assay evidence is distinct from appliance-process telemetry and model estimates.
- E1 repeatability, E2 reproducibility, E3 production variation and E4 household validity remain separate.
- A completed repair is not durable restoration without post-repair qualification and observed follow-up.
- Local appliance safety, manual fallback and user authority remain independent of any household coordinator.

No item on this index implies a physical test result, product certification, legal compliance, household demand proof, market superiority or deployment authority.
