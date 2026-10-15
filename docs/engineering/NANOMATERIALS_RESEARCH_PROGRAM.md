# Evidence-Bounded Nanomaterials Discovery Program

**Status:** Research architecture and benchmark preregistration. This document does not establish a discovered material, validated atomistic model, synthesis recipe, product qualification, or authority to run physical experiments.

**Tracking:** [NANO-001 #7453](https://github.com/Luminous-Dynamics/symthaea/issues/7453)  
**Parent research boundary:** [MAT-HYPOTHESIS-001 #5254](https://github.com/Luminous-Dynamics/symthaea/issues/5254)  
**Date:** 2026-10-15

## 1. Mission

Build a reusable, evidence-bounded workflow that helps researchers decide **which nanomaterials hypotheses, structures, and calculations deserve the next unit of scientific effort**.

The goal is not to claim that Symthaea is an atomistic simulator or that reasoning alone can discover useful matter. The goal is to test whether Symthaea adds measurable value to an established scientific toolchain by improving retrieval, hypothesis formulation, candidate diversity, experiment selection, or label efficiency under fixed budgets.

Start with a narrow computational test involving atomic structures and catalyst-surface data. Keep nanostructured construction materials, batteries, membranes, sensors, self-assembly, and molecular devices as follow-on applications that must pass their own domain-specific evidence gates.

Core claim ladder:

```
analogy or generated idea
!= literature novelty
!= chemically valid candidate
!= stable structure
!= accurate predicted property
!= experimentally synthesized structure
!= measured performance
!= safe, scalable, economically useful material
```

Likewise:

```
simulation reproduced
!= simulation validated against reality
solver agreement
!= all solvers correct
ranked candidate
!= discovery
promising measurement
!= independently replicated result
```

## 2. Existing ownership: compose, do not duplicate

This is an application profile over existing canonical owners, **not permission to add a parallel material, quantity, observation, evidence, provenance, split, or experiment type**.

| Concern | Canonical owner to reuse | NANO contribution |
|---|---|---|
| Material/formulation identity | MAT-007 and its current qualified successor | Reference the exact subject and revision |
| Conditioned material properties | MAT-008; SE-SEM-001; FIELD / SE-OBS | Name target properties, method, conditions, units, and uncertainty |
| Campaigns, objectives, Pareto trade-offs | MAT-009 and qualified MAT-OPPORTUNITY / MAT-PORTFOLIO successors | Freeze campaign objective and budget |
| Failed, null, contradictory, out-of-domain results | MAT-010 | Preserve the complete candidate/failure denominator |
| Physical sample, precursor, batch, and process lineage | MAT-013; Mycelix manufacturing/circularity owners | Reference lineage once a physical specimen exists |
| Human-authorized experiment proposals | MAT-014 | Keep candidate selection proposal-only |
| Material mechanism hypotheses | MAT-HYPOTHESIS #5254 | State falsifiable competing explanations |
| Evidence dependencies and independence | ENG-EVID-INDEP #4918 | Define holdout groups and common-mode checks |
| Evidence-bound claims and supersession | ENG-EVID-CLAIM #5771; ETK/SE owners | Separate proposal, prediction, measurement, and replication |
| Research split semantics | Existing research-split owner and its current qualified successor | Provide domain split policy and frozen manifest only |
| External-source identity, access and governance | Mycelix provenance/source/review owners | Link immutable source, rights and review records |
| Deterministic scenarios | Symtropy engineering-simulation boundary, including issue #1467 | Use only where its actual model class is appropriate |

MAT owner IDs above are references to the existing campaign map and may have historical implementations. Before any code is proposed, resolve current qualified successors from the owners; do not infer qualification from an issue number or old PR.

The hard default remains **zero new common production types**. If a future implementation appears to require a new type, first produce an adversarial fixture demonstrating that the exact current owner cannot represent the semantics losslessly, then obtain the applicable owner review. A domain document or adapter is not, by itself, grounds to create a new ontology.

## 3. System roles

### Symthaea — hypothesis and selection policy

Symthaea may contribute:
- literature- and evidence-aware candidate retrieval;
- HDC representations for grouping, similarity, and analogy, with similarity interpreted as a representation statistic rather than a physical law;
- candidate diversity and coverage analysis;
- proposal generation and competing, falsifiable explanations;
- candidate-selection policies that can be compared against simple baselines;
- uncertainty/abstention and applicability warnings when these are empirically calibrated;
- analysis of negative, failed, contradicted, or out-of-domain outcomes.

It must not silently turn an HDC distance, Phi metric, confidence score, fluent explanation, or consensus into a physical property, novelty finding, safety assessment, or evidence grade. Temporal models may analyze actual time-series or trajectory data only when the target and evaluation are specified; they do not replace molecular dynamics.

### Scientific engines — atomistic calculations and reference labels

Use established, version-pinned external tools and datasets for calculations that need chemistry-specific physics. Depending on the exact question, the eventual toolchain may involve density-functional theory, established machine-learned interatomic potentials, molecular dynamics, or a materials data service. The selected method must document its approximation, parameterization, supported chemistry, and applicability domain.

Possible tooling to evaluate—not dependencies chosen by this proposal:
- [Open Catalyst Project / OC20 / OC22](https://opencatalystproject.org/) for an open catalyst-surface benchmark and reference labels.
- [LAMMPS](https://docs.lammps.org/stable/) for supported molecular/materials dynamics with declared interatomic potentials.
- [Atomic Simulation Environment (ASE)](https://docs.ase-lib.org/) as a research environment for setting up atomistic calculations and invoking calculators.
- DFT engines such as Quantum ESPRESSO or CP2K if the target study, license, and compute budget justify them.

These are alternatives with different scientific models and licensing. NANO-001 must select the minimum suitable stack only after assessing the specific benchmark, version pinning, provenance, hardware needs, tool license, and redistribution model. No solver is treated as ground truth merely because it is expensive or widely used.

The production Symthaea integration should remain Rust-first. Scientific executables may be invoked as separately versioned external processes with frozen inputs and machine-readable outputs; this document introduces no production Python runtime or new solver implementation. Research tooling used to reproduce a paper may be language-specific, but it must remain explicit, isolated, versioned, and outside production claims.

### Symtropy — scenario projection, not atomistic physics

Symtropy may support interactive exploration, experiment-workflow rehearsal, or a physical-system/mesoscale scenario when an appropriately validated model already exists and the actual physics fits the engine. It may also visualize atomistic structures supplied by a scientific tool, but rendering a structure is not a chemistry calculation.

Symtropy's current rigid-body engine is **not** an interatomic potential, DFT engine, molecular-dynamics engine, fracture solver, or atomistic materials model. Do not encode bond chemistry or chemical energies as arbitrary rigid-body forces. Do not use consciousness-coupled or research-coupled physics as the scientific reference profile. Any later coupled scenario must bind the external model identity, validity domain, assumptions and synthetic-evidence label.

### Mycelix — provenance, review and research coordination

Mycelix should link source-rights decisions, dataset releases, review/replication records and eventual physical-sample lineage through the existing canonical owners. Preserve which result came from which code, dataset, method, model, sample and independent reviewer. Replication and common-mode dependencies must remain visible.

Mycelix governance can decide whether a community funds or prioritizes a study; a vote does not establish that a model is accurate, a material is safe, or a result is physically true. Do not create a second materials registry, measurement ledger or scientific truth oracle.

## 4. First proving domain: catalyst-surface data efficiency

### Research question

On a version-pinned, rights-cleared slice of OC20/OC22—or an explicitly justified alternative admitted under the same rules—does a Symthaea-assisted acquisition/selection policy improve the amount of useful scientific information obtained from a fixed label or compute budget compared with simpler policies?

This first test is about **candidate/data selection**, not about claiming that Symthaea has invented a new catalyst or trained a state-of-the-art atomistic potential.

OC20/OC22 are relevant because the Open Catalyst Project provides open datasets and benchmark tasks built around catalyst structures and DFT-derived calculations. The published dataset license and required attribution must be checked for the exact release and artifact used. The data repository's software license does not automatically settle the license of every external model weight, calculator, or downstream artifact.

### Task selection gate

Before modeling:
1. Inspect the exact available release, license, task definitions, labels, official splits, known baselines, data schema, download size, and required attribution.
2. Choose one declared task and one primary target. Keep the official task/split result distinct from any additional domain-shift experiment.
3. Document why the target is scientifically useful for the specified surface/reaction class. Do not assume that a lower adsorption energy is universally better; catalytic usefulness is reaction-dependent and may require a defined energy window or multiple descriptors.
4. Define the unit of selection: a structure, a structure/trajectory, or another canonical observation unit. Explicitly state whether initial and relaxed structures share identity.
5. Freeze the dataset release, license evidence, source URL, content hashes, normalization, exclusions, unit conversions, seed, label budget, compute budget, and split manifest before inspecting held-out results.

If no suitable rights-cleared, reproducible dataset slice or compatible baseline can be established, record `data_access_unresolved` or `baseline_unavailable` and stop the accuracy claim. Do not silently switch datasets, targets, or splits after seeing results.

### Acquisition-policy comparison

Where the chosen task supports it, compare at minimum:
1. **Random selection**, with frozen seeds and repeated trials.
2. **Declared simple baseline**, such as a documented descriptor/nearest-neighbor strategy.
3. **Established reference baseline**, such as a compatible model or acquisition policy from the selected benchmark.
4. **Symthaea-assisted selection**, with the HDC/features, policy, scoring, candidate diversity rule, abstention policy and resource ceiling specified before held-out evaluation.

Keep the downstream predictor/model family and label budget comparable across acquisition strategies. When precomputed labels are used to simulate label acquisition, labels from the evaluation partition must remain hidden from the selector and every fitted transform until the final evaluation. Simulated acquisition over an existing corpus demonstrates label-efficiency under that corpus's assumptions; it does not prove wet-lab efficiency or that new structures can be synthesized.

### Leakage and negative controls

The independent split auditor should inspect, as applicable:
- exact and near-duplicate atomic configurations;
- structure relaxations and trajectory frames derived from a common initial structure;
- shared slabs, surface families, compositions, adsorbates, or reaction families;
- shared parent calculation or dataset lineage;
- normalization/statistics fit using held-out records;
- tuning, feature engineering or early stopping informed by test labels;
- re-used or near-identical material structures spanning train and test.

Retain the dataset's standard official splits and report their results separately. Add chemically meaningful group holdouts for generalization only if group definitions can be justified and frozen; do not relabel a custom holdout as the official benchmark.

Required negative controls:
- random ranking/selection;
- shuffled-target or shuffled-label control;
- simple, intentionally weak baseline;
- a diagnostic split known to be leakage-sensitive, labelled as a diagnostic rather than a generalization result.

The shuffle control must not produce credible predictive or acquisition value. If it does, treat the benchmark as invalid until the leak or implementation error is explained. Distinct IDs alone are not evidence of independent samples.

## 5. Metrics and evidence contract

Freeze primary and secondary metrics before the final holdout is unsealed.

### Core metrics

- **Label efficiency:** downstream error or selection quality versus number of acquired labels/calculations.
- **Energy/force prediction error:** only if the selected official task and baseline support those targets; report units and metric definitions.
- **Top-k utility/regret or recall:** only against a predeclared task-specific utility function and held-out candidate pool.
- **Uncertainty calibration and coverage:** required before making calibrated-uncertainty claims; report selective risk as well as coverage.
- **Candidate diversity:** composition/surface/adsorbate coverage, duplicate rate, and collapse onto a narrow family.
- **Validity and abstention:** malformed/chemically invalid inputs, unsupported elements or states, out-of-domain cases, and explicit abstentions.
- **Cost:** wall time, CPU/GPU model, peak memory, number of external calculations, and compute budget.
- **Statistical variation:** repeated random baselines where appropriate, confidence intervals, and the fixed analysis plan.

Do not select the winner based on one scalar alone. Better mean error does not compensate for failed mandatory controls, unreliable uncertainty, leakage, invalid structures, or severe domain-shift failure. Improvement claims require a predefined practically meaningful margin and uncertainty analysis; if none is supported, report a null or inconclusive result.

### Independent verification

The checker must be separate from the Symthaea selector and should consume the frozen manifest plus produced artifacts. It should independently recompute split membership, sample counts, unit handling, metrics, and control expectations. It must verify:
- exact dataset/code/model/configuration identities;
- no overlap forbidden by the split contract;
- labels are not available to the selector before acquisition;
- pre-processing and scaling were fitted only on allowed data;
- all selected and rejected candidates remain accounted for;
- reported metrics can be regenerated from saved predictions/labels;
- manifest and outputs have stable content digests.

Rerunning the same selector is not independent verification. Passing this checker demonstrates conformance to the frozen benchmark contract; it does not prove all scientific assumptions correct.

## 6. Compute and integration discipline

Avoid full-scale training until the benchmark's expected memory, disk, runtime and hardware are documented. Start with a CPU-readable metadata/data-integrity slice, then increase the sample/budget only after the baselines and split auditor work. Use a declared resource limit and record GPU model/VRAM, runtime and any fallback behavior. An OOM, partial download, missing external solver, or unsupported model must become an explicit failed/incomplete run, not a silently smaller or different experiment.

No change to Symthaea's core numeric semantics or cognitive architecture is justified merely to run this benchmark. Prefer an isolated experiment adapter and canonical references. Do not add a new model serving layer, benchmark ontology, custom database, or solver fork unless a demonstrated limitation and owning project require it.

## 7. Candidate safety and lifecycle

Physical size, shape, surface chemistry, coating, aggregation/agglomeration, dissolution, oxidation, exposure medium, dose, persistence, by-products and end-of-life can change nanomaterial behavior. The safety and lifecycle questions are part of candidate assessment, not an afterthought.

For every future physical candidate, the study plan must establish, with qualified domain experts, appropriate characterization and risk review covering relevant:
- composition, size distribution, morphology, surface chemistry and stability;
- behavior in the intended medium and during ageing/transport;
- worker exposure and safe handling;
- toxicity/ecotoxicity and environmental fate where relevant;
- leaching, dissolution, by-products and disposal/recovery;
- manufacturing repeatability, scale-up constraints, cost, energy and life-cycle impacts.

Use material- and application-specific methods. NIST's nanotechnology measurement protocols are a starting point for reproducible characterization; a general protocol inventory is not a substitute for a material-specific hazard assessment. Do not infer safety from a model, a bulk-material analogue, low solubility alone, or missing toxicity data. Unknown critical risks remain explicit blockers.

This plan is initially computational. It does not authorize synthesis, biological work, environmental release, medical use, or autonomous lab actuation. Any physical work requires qualified laboratory ownership, suitable containment/risk review, a reviewed experimental protocol, traceable sample lineage, matched positive/negative controls, calibrated measurements, and human authorization.

## 8. Progressive program

### N0 — admission and baseline contract
- Confirm dataset/data/model license and citation requirements for exact artifacts.
- Freeze the target, task, split manifest and resource budget.
- Build a data-quality and leakage report before fitting a selector.
- Record source-controlled dataset identities and incomplete fields.

**Exit:** data, rights, splits, baselines and target are reproducible. Otherwise stop without an accuracy claim.

### N1 — simple-baseline reproduction
- Run the chosen reference baseline on the frozen slice.
- Independently recompute its primary metric and units.
- Establish variance and compute cost before evaluating Symthaea.

**Exit:** baseline reproduces within a predeclared tolerance. If not, resolve the discrepancy first.

### N2 — Symthaea ablation
- Compare random, descriptor/similarity, reference, and Symthaea-assisted policies.
- Freeze features and policy before the held-out run.
- Run the full negative-control suite and independent checker.

**Exit:** a measured result—including null results—with no unresolved leakage/control failure.

### N3 — candidate review
- Emit candidate references, not autonomous promotion or synthesis instructions.
- Assess domain applicability, chemical validity, uncertainty, diversity and utility with a domain scientist.
- Choose any next expensive simulation through a documented human review.

**Exit:** independent reference calculations or an expert review supports further investigation.

### N4 — cross-domain extension
Choose only one next application after N2 is qualified:
- low-carbon cementitious systems and nano-additives, composed with BIO-CEM/CEM and material owners;
- battery/electrode interfaces;
- membranes, adsorption and resource recovery;
- durable coatings and corrosion barriers;
- nanophotonic or sensor materials.

Every new application needs its own targets, methods, baseline, split/holdout rationale, safety/lifecycle profile and validation plan. A result on catalytic surfaces does not transfer automatically to any of these domains.

### N5 — laboratory collaboration
Propose a small, preregistered and human-authorized study only when computational value is repeatable and an appropriate lab can perform independent characterization. Predefine controls, replicates, analysis, sample custody, failed outcomes, material-specific safety review and criteria for advancement. Require independent replication before a broader performance claim.

## 9. Success criteria and claim ceiling

NANO-001 is a success if it produces a reproducible and independently checked answer to the narrow question—not only if Symthaea wins. A well-demonstrated null result is useful evidence against wasting compute on an ineffective method.

At the end of each stage, report:
- exact commit, dataset release/content hashes, model and solver versions;
- commands/configuration and machine/resource identity;
- split/lineage manifest and independent-check result;
- baseline and Symthaea outcomes with units, confidence intervals and failed/null cases;
- whether the primary hypothesis was supported, rejected or unresolved;
- the next permitted action and the unresolved limitations.

Never describe a queued CI job as passed. Never claim compilation or tests that were not executed. A source audit, deterministic replay, benchmark pass, external-solver result, physical measurement, and independent replication are different evidence classes.

## 10. References

1. ACS Nano (2025), *Artificial Intelligence for Materials Discovery, Development, and Optimization*: https://pubs.acs.org/doi/10.1021/acsnano.5c04200
2. Open Catalyst Project, datasets and project background: https://opencatalystproject.org/
3. Open Catalyst Dataset repository, dataset workflow and license notes: https://github.com/Open-Catalyst-Project/Open-Catalyst-Dataset
4. NIST, *Nanotechnology Measurement Protocols* (page updated 13 July 2026): https://www.nist.gov/mml/nanotechnology-measurement-protocols
5. Ren et al., *Autonomous experiments using active learning and AI*, Nature Reviews Materials (2023): https://www.nature.com/articles/s41578-023-00588-4
6. LAMMPS stable documentation (30 September 2026): https://docs.lammps.org/stable/
7. Atomic Simulation Environment documentation: https://docs.ase-lib.org/
