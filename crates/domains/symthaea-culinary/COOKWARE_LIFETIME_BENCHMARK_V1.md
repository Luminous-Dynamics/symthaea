# Cookware Lifetime Benchmark V1

**Schema:** `CookwareLifetimeManifestV1`  
**Implementation:** `src/cookware_lifetime.rs`  
**Composes:** `CookwareBenchmarkManifestV1` from `src/cookware.rs`  
**Status:** proposed validation contract; no physical cookware trial has been run.

## Objective

The original cookware benchmark compares matched observations under declared conditions. The lifetime contract adds the **trajectory of each physical specimen**: new, exposed to a defined duty cycle, reassessed, maintained/repaired where applicable, failed or retired.

A pan does not become a better product merely because its first heat map looks good. The programme must ask how performance, safety qualification, repairability, costs and practitioner experience change over real use.

## Data model

Each lifecycle study embeds the existing v1 benchmark rather than defining a second measurement/evidence system. Each benchmark run must be bound to exactly one lifecycle `StageAssessment` event, and that assessment must refer to the same candidate and physical cookware-unit identity.

A unit’s event sequence is contiguous and ordered. The first record is a new, zero-cycle baseline. Cumulative counters distinguish:

- thermal cycles;
- cooking-load cycles;
- cleaning cycles;
- abrasion cycles;
- declared misuse/abuse cycles.

Only an explicit completed-use batch may advance counters. Counter decreases are invalid. Repeated measurements on the same unit are not new independent units.

The lifecycle record distinguishes use, maintenance, repair, failure, and retirement. A repair requires named parts, positive repair duration, a recorded cost, and a declared currency code. A post-maintenance or post-repair assessment requires a preceding matching intervention event; a failed repair outcome must remain representable rather than being silently recast as success. A retired specimen cannot have later events.

## Experimental protocol

Freeze the intended use profile, geometry, heat source and power trajectory, ingredient/load profile, ambient conditions, measurement methods, calibration, wear/duty-cycle definition, checkpoint schedule, randomization, primary outcomes, and analysis before data collection.

At minimum, compare multiple suitable construction families with multiple independently selected physical units per candidate. The exact count must follow the research question and sample-size rationale; schema minimums are not evidence of statistical power.

At baseline and each preregistered checkpoint, reuse the corresponding task-specific v1 measurement vector. Depending on the intended use, that may include thermal spread/recovery, energy per edible output, cooking outcome, food release/sticking, cleanup burden, warp, surface/coating degradation, corrosion, handle/joint condition, ergonomics and practitioner experience.

Record every specimen, including failures and withdrawals. Do not remove unsuccessful units from results without recording the reason. Publish per-specimen trajectories and uncertainty, not only a pooled mean.

## Repair and economics

For every maintenance/repair event, record the parts affected, availability constraints, tools/procedure, hands-on time, downtime, costs, before/after condition, and post-intervention measured performance. A repair record is not proof that the product has been restored to safe or qualified use.

Report **cost per successful edible serving** only with an explicit boundary that covers relevant acquisition, use-phase energy, cleaning resources, maintenance, parts, repair labor, downtime/rework, cookware-attributable food loss, and end-of-life recovery/disposal where supportable. Keep each line item and source available, and run sensitivity analysis for uncertainty such as useful life and energy mix.

Any environmental or life-cycle claim must disclose its functional unit, system boundary, allocation rules, data provenance and uncertainty. A product's mass, “recyclable” label, or modeled longevity alone cannot establish sustainability.

## Claim ceilings

- A validation PASS establishes only internal structural consistency.
- A cited reference does not authenticate the record that it names.
- A repaired or worn article does not automatically inherit a food-contact qualification.
- Accelerated cycles do not equal calendar years without a separately justified and evidenced equivalence model.
- A small exploratory pilot can develop the method, but cannot establish population-wide service life or superiority.
- A lower cost or better thermal result cannot offset an unresolved safety gate or a meaningful loss of practitioner agency.
- No aggregate score may hide failed units, repair blockers, uncertainty, subgroup outcomes or adverse findings.

## Engineering acceptance goals

1. Every lifecycle event identifies a physical unit, candidate, sequence, event evidence, method and counter state.
2. Every stage assessment resolves to exactly one existing benchmark run with matching unit/candidate identity; no run is left unlinked or linked twice.
3. Every physical unit has a baseline and required in-service checkpoints, or the validation report identifies the deficiency.
4. Counter regression, counter changes outside use batches, missing intervention prerequisites, missing repair data, unsupported calendar-life extrapolation, post-retirement activity and duplicate event IDs are rejected.
5. Failed or retired units remain in the evidence and are never counted as successful repairs by implication.
6. Results publish the number of independent physical units separately from repeated time points, with raw metric vectors and uncertainty.
7. Safety, food-contact qualification, cooking outcome quality and practitioner-defined agency constraints are hard gates—not compensatory terms in a single score.

These are design targets. They are not achieved product results, a cookware standard, or a claim of superiority over commercial products.

## Reuse and research context

- Symthaea owns the evidence contract, measurement protocol and uncertainty-aware comparisons.
- Symtropy should reuse existing lifecycle, repairability and process-planning systems for product/process decisions instead of growing a parallel universal lifecycle engine.
- Mycelix may support supplier/lot provenance, repair/contributor attribution and permission governance, but provenance does not substitute for independent measurement.
- Consumer Reports provides useful product-comparison dimensions (heating, distribution, food release, durability, handle behavior, cleaning and cooking performance).
- ASTM F1786-97(2021) is a methodological analogue for energy and cooking performance of commercial braising pans, not a direct skillet certification.
- NSF/ANSI 51 addresses materials used in commercial food equipment; applicability to any particular cookware article or claim must be established before citing it as qualification.
- EU Regulation 2024/1781 provides a general ecodesign framework emphasizing topics such as durability, repairability, maintenance/refurbishment, resource use and end-of-life. This document does not claim a cookware-specific delegated requirement already applies.

## Related implementation

- Parent benchmark: [Symthaea PR #8190](https://github.com/Luminous-Dynamics/symthaea/pull/8190).
- Mission: [COOKWARE-CRAFT-001 / issue #8182](https://github.com/Luminous-Dynamics/symthaea/issues/8182).
- Lifecycle follow-on: [COOKWARE-LIFETIME-001 / issue #8191](https://github.com/Luminous-Dynamics/symthaea/issues/8191).
- Human-craft charter: [HUMAN-CRAFT-001 / issue #8183](https://github.com/Luminous-Dynamics/symthaea/issues/8183).

This is intended as a reproducible engineering contract for later testing, not an alternative to physical testing, competent metrology, applicable food-contact requirements, or research review.