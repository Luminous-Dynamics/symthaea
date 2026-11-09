# EV Pack Capability Profile v0.1

**Status:** Research contract and synthetic evaluator. Not a battery model, BMS or automotive qualification artifact.  
**Owner:** Symthaea [EV-PACK-CAPABILITY-001 #11271](https://github.com/Luminous-Dynamics/symthaea/issues/11271).  
**Inputs:** EV battery observation contract [PR #11249](https://github.com/Luminous-Dynamics/symthaea/pull/11249) and SOH transfer benchmark [#11252](https://github.com/Luminous-Dynamics/symthaea/issues/11252).  
**Integration:** Symtropy battery architecture/circularity [#2155](https://github.com/Luminous-Dynamics/symtropy/issues/2155) and EV reference configuration [#4346](https://github.com/Luminous-Dynamics/symtropy/issues/4346).

## Why this profile exists

A scalar pack SOH can conceal a weak tail of cells or modules. The 2026 Nature Energy fleet study *Quantifying the impact of cell-to-cell inconsistency on electric vehicle battery degradation and utilization* reports that cell inconsistency materially affects usable pack health, lifetime, power capability and lifetime energy utilization in its studied passenger-car and bus fleets.

Source: https://www.nature.com/articles/s41560-026-02131-5

This profile describes a **population of comparable electrochemical units** and its observed health distribution. It is a diagnostic descriptor, not a complete electrical pack model.

## v0.1 scope

The input is one battery-pack configuration, one declared comparison resolution, one common reference condition for capacity, one common reference condition for resistance, and a set of comparable measured or inferred units.

Allowed resolutions are `cell`, `parallel_group` and `module`. A profile must not mix these resolutions. Declared and observed unit counts are distinct. Missing units remain missing; they are never imputed as healthy, average or nominal.

Capacity and resistance are separate measurement records. Each binds its value, absolute uncertainty, evidence class, method identifier, evidence digest and exact common-condition reference.

Capacity conditions include temperature, C-rate, voltage cutoffs and a protocol digest. Resistance conditions include temperature, SOC, pulse current, pulse duration and a protocol digest. Measurements made at materially different conditions cannot be compared merely because their JSON shapes match.

## v0.1 report metrics

For each comparable unit:

`SOH_i = capacity_i / nominal_capacity_i`

The evaluator reports descriptive statistics over the **observed subset**: mean, median, minimum, nearest-rank p05/p95 and population standard deviation of unit SOH; mean, median, population standard deviation and coefficient of variation of resistance; coverage fraction; weakest observed unit identifier; and evidence-class counts.

Quantiles use nearest-rank: sort ascending, take index `ceil(p × N) - 1`, clamped to the valid index. This makes the quantile method reproducible across implementations.

Per-unit uncertainty remains in the input. v0.1 does not yet propagate it into confidence intervals for population metrics, and the report marks that limitation.

## What v0.1 deliberately does not calculate

It does **not** infer pack usable kWh, charge/discharge power, driving range, thermal margin, remaining useful life or a pack-limiting cell from distribution statistics.

Those claims require actual series/parallel topology; cell/group electrical limits; BMS balancing and reconfiguration policy; relevant thermal boundary conditions; a validated electrothermal model; and evidence that the operating point lies inside that model's validation domain. The evaluator therefore returns `NOT_COMPUTABLE_TOPOLOGY_MODEL_NOT_QUALIFIED` for pack energy/power capability.

This is intentional fail-closed behavior, not a gap to bypass with a pack-average approximation.

## Evidence classes and claim ceiling

`reference_test_measurement`, `derived_estimate`, `model_prediction`, `simulation_output`, `synthetic_fixture` and `vendor_claim` remain distinguishable. A SHA-256 digest establishes byte identity/integrity, not sensor truth, calibration quality, measurement correctness or physical validation.

A synthetic fixture is always synthetic. It cannot substantiate battery-health, fault-detection, safety, performance or lifetime claims.

## Benchmark plan

1. Contract/schema validity and negative-control rejection.
2. Complete versus partial coverage; cell versus module versus parallel-group populations.
3. Exact permutation invariance of descriptive metrics.
4. Matched-condition versus deliberately mismatched-condition inputs.
5. Leave-one-cell/unit-out and leave-one-pack/vehicle-out transfer, composed with #11252; no adjacent windows or units from one pack cross a pack-held-out split.
6. Baselines: mean-only pack summary, weakest-observed-unit descriptor, SOH/resistance distributions, transparent ECM/thermal model, and validated topology-aware model when available.
7. Calibration and selective-risk evaluation from #11252; numerical accuracy gates only after label quality, independence, source rights and statistical power are assessed.

Do not claim SOTA from synthetic fixtures or pooled RMSE. Report per-pack and worst-domain behavior, coverage, abstention, uncertainty calibration, false-alert burden and resource cost.

## Safety and authority

This is a read-only advisory/evaluation profile. It must not actuate contactors, inverter torque, charging current, thermal systems, brakes or steering; replace BMS protections; authorize charging/driving; or declare a battery safe. Certified BMS protections, hardware fuses, thermal cutouts and qualified engineering processes remain authoritative.

## Versioning

The schema ID is `https://luminousdynamics.io/schemas/transport/ev-pack-capability-profile/0.1`. Breaking changes require a new schema version and frozen fixture corpus. Reports bind exact schema and corpus digests.
