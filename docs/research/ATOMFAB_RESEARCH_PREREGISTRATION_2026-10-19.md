# AtomFab: Evidence-First Atomic-Scale Manufacturing Research

**Status:** Research proposal and benchmark preregistration; no physical-manufacturing claim is made.  
**Date:** 2026-10-19  
**Implementation boundary:** Rust orchestration and evidence; established atomistic solvers remain external, versioned tools.

## Executive decision

Start AtomFab as a falsifiable research track inside Symthaea. Its first deliverable is not an atom printer or a new interatomic potential. It is a reproducible benchmark kernel and research protocol that can answer a narrower, valuable question:

> Can Symthaea's planning, evidence management, and experiment-selection logic improve the reliability or sample efficiency of established atomistic workflows, without hiding uncertainty or claiming physical validation it does not have?

The first computational benchmark should use selected, license-compatible subsets of MS25, an existing 2025 materials-science benchmark spanning MgO surfaces, liquid water, zeolites, a Pt surface reaction, high-entropy alloys, and disordered Zr oxides. MS25 explicitly tests derived observables as well as energy, force, and stress errors, and reports transferability failures between different zeolite frameworks. These are directly relevant to avoiding a common failure: a model looks accurate on average but produces the wrong physical conclusion.

Sources:
- Maxson et al., *MS25: Materials Science-Focused Benchmark Data Set for Machine Learning Interatomic Potentials*, Journal of Chemical Information and Modeling (2025), [DOI: 10.1021/acs.jcim.5c01262](https://doi.org/10.1021/acs.jcim.5c01262).
- Public data landing page and license metadata: [Argonne ACDC/MS25](https://acdc.alcf.anl.gov/mdf/detail/74d6b3d9-c33c-47c4-9c9a-c14ca773d3c8-1.0/).

## Why use Symthaea and Symtropy together?

Use both, but keep the scientific, atomistic, and apparatus-physics responsibilities distinct.

- **Symthaea / AtomFab — scientific decision layer:** specify objectives and constraints; select candidates and the next calculations or experiments; orchestrate specialist solvers; manage calibrated uncertainty, abstention, evidence provenance, and prediction-to-measurement updates.
- **Specialist atomistic solvers — molecular/electronic physics:** calculate electronic structure, interatomic forces, reaction pathways, and material properties within validated domains. Symtropy's rigid-body engine must not be treated as a substitute for these methods.
- **Symtropy — apparatus/process digital twin:** represent the workcell, tool/stage geometry, reachable regions, collisions, actuator limits, process timing, drift and sensor-fault scenarios, and replayable task-level dynamics. Its visualization is not, by itself, validation.
- **Metrology — physical evidence:** actual observations retain their instrument, calibration, sample identity, acquisition settings, uncertainty, and raw-data digest. Simulated observations must never be labelled measured.
- **Versioned contract — integration seam:** begin with explicit engine-neutral proposal/evaluation/measurement/decision records and shared fixtures, not direct coupling of internal data structures or a new tight dependency between the repositories.

This separation is consistent with the current code: Symtropy provides N-dimensional rigid-body / collision mechanics and deterministic single-thread replay on a given platform, and already has a Symtropy–Symthaea embodiment adapter. Its roadmap describes the robotics bridge as a skeleton, so AtomFab should reuse the existing integration pattern only where it fits, not assume lab-grade robotic control already exists.

The central design rule is **one accountable owner per kind of truth**. Symtropy can decide whether a tool path collides with the simulated stage; it cannot establish whether a chemical reaction is favorable. A DFT or interatomic-potential solver can estimate atomistic behavior; it cannot establish that the tool can safely reach the sample. A real measurement can calibrate either model, but its provenance must remain independent.

The companion Symtropy proposal defines contract fields, failure semantics, workcell scenarios, and a joint benchmark suite: [Symtropy PR #2140](https://github.com/Luminous-Dynamics/symtropy/pull/2140).

For digital-twin credibility, use verification, validation, and uncertainty quantification (VVUQ) from the outset. NIST's 2026 manufacturing digital-twin work highlights interoperability, VVUQ, cybersecurity, and traceable data as ongoing obstacles—not solved properties of a digital twin. [NISTIR 8620](https://doi.org/10.6028/NIST.IR.8620).

## State-of-the-art and industry comparison

The competitive strategy and exact benchmark lanes are maintained in [AtomFab SOTA and Industry Comparison](ATOMFAB_SOTA_AND_INDUSTRY_COMPARISON_2026-10-19.md). The machine-readable lane registry is [benchmark-matrix-v1.json](../../crates/domains/symthaea-atomfab/benchmarks/benchmark-matrix-v1.json), parsed and checked by the new Rust benchmark-registry module.

Do not position AtomFab as a replacement for OMat24-scale pretrained interatomic potentials or DFT. OMat24 reports more than 110 million DFT calculations and leading Matbench Discovery performance; UniFFBench (2026) demonstrates a persistent gap between computational MLIP scores and experimental-property accuracy. The defensible competitive hypothesis is that a decision/evidence layer can improve *validated decision quality per total cost* when using a strong specialist model, and that Symtropy can reduce false-feasible process plans. Both hypotheses require matched baselines and explicit ablations. References: [OMat24](https://doi.org/10.1038/s43588-026-00996-w), [UniFFBench](https://doi.org/10.1038/s43588-026-01019-4), [Matbench Discovery](https://docs.materialsproject.org/services/ml-and-ai-applications/matbench-discovery).

## Paradigm shifts

### 1. Compile a desired property into a feasible process, not just an atomic structure

Start from measured requirements (e.g. conductivity, catalytic activity, thermal stability, cost, toxicity constraints, or durability). Generate candidate structures, then process plans and verification procedures. The output must include known process constraints, assumptions, unresolved unknowns, and a rejection reason when no feasible route is supported. A good-looking structure that cannot be made is not a successful design.

### 2. Exploit physics before insisting on arbitrary atom-by-atom placement

General-purpose placement of every atom is a far-future goal. Many useful structures may be made more practically through controlled growth, epitaxy, area-selective deposition, surface chemistry, templating, or self-assembly. AtomFab should select the coarsest fabrication method that can meet the specification, and reserve direct manipulation for tasks where explicit placement adds measurable value. Compare methods on final function, defects, throughput, resource use, and repeatability—not on atomic resolution alone.

### 3. Optimize observables and decision quality, not only model loss

Track energy/force/stress errors, but also validate task-specific quantities such as reaction barriers, surface energies, lattice constants, volume, defect energetics, and stability. MS25's results support this shift: similar conventional error metrics can coexist with differences in derived properties and cross-framework transferability. The system must never infer manufacturing readiness from a scalar MAE.

### 4. Make uncertainty an action constraint

Maintain calibrated uncertainty and detect distribution shift. When the candidate structure or reaction pathway lies outside validated support, AtomFab should abstain, request a higher-fidelity calculation, or ask for an experiment. The desired capability is not merely to predict more; it is to know when a prediction is not decision-grade.

### 5. Treat experiment selection as an economic optimization problem

Count expensive reference calculations and physical experiments. Compare active learning against random sampling, diversity-only selection, uncertainty-only selection, and a simple cost-aware hybrid. Optimize the expected reduction in uncertainty or decision regret per unit time, compute, money, and instrument occupancy. Report both acceleration factor and final quality. A method that uses fewer calls but chooses worse candidates is not a win.

Reference: Adesiji, Wang, Kuo, and Brown, *Benchmarking self-driving labs*, Digital Discovery (published 17 December 2025), [DOI: 10.1039/D5DD00337G](https://doi.org/10.1039/D5DD00337G). The review formalizes acceleration and enhancement metrics while noting their sensitivity to search-space properties and benchmark design.

### 6. Close the loop from model to metrology

Design the system around a loop of hypothesis → simulation → candidate process → measurement → comparison → model update. Instrument readings, calibration state, sample identity, environmental conditions, drift, and failure observations are first-class data. No hardware is directly actuated by this initial milestone; the first implementation is retrospective and simulation-only.

### 7. Make every conclusion carry its evidence

Keep proposed, predicted, simulated, measured, and independently replicated claims separate. Each result should retain dataset version, split definition, input digest, solver/model version, configuration, uncertainty, output digest, and reproduction instructions. Formal verification can establish properties of the recorded protocol or software; it does not prove that a simulation matches reality.

### 8. Test Symthaea's distinct contribution instead of assuming it

Use specialist equivariant machine-learning interatomic potentials and trusted reference solvers as the physics baseline. Hyperdimensional computing may be useful for organizing experiment history, retrieving analogous failures, or binding context across a workflow, but it should not replace geometrically appropriate atomistic representations without comparative evidence. The ablation question is explicit: does the Symthaea planning/evidence layer improve the outcome beyond the same solver plus a conventional acquisition strategy?

### 9. Co-design manufacturability instead of optimizing a material in isolation

Make desired properties, synthesis/growth window, available precursors, equipment reachability, yield, energy consumption, material criticality, toxicity, repairability, and end-of-life recovery part of the design problem. Search for a **Pareto frontier** across function, process feasibility, resource demand, and uncertainty instead of treating manufacturability as a final-stage filter. A candidate outside the known process window should be rejected or explicitly marked speculative even when its predicted material properties look excellent.

### 10. Use multi-fidelity evidence routing rather than one universal simulator

Route each question to the lowest-cost adequate model, then escalate when expected decision value justifies it:

1. **Static/constraint checks:** composition, units, conservation and known process rules.
2. **Cheap candidate screen:** a pinned surrogate or interatomic potential inside its validated domain.
3. **High-fidelity atomistic evaluation:** a trusted reference method for finalists, uncertain reactions, ranking disputes or out-of-domain structures.
4. **Symtropy workcell simulation:** apparatus-level geometry, trajectory feasibility, timing, faults, collision margins and replay—not molecular chemistry.
5. **Metrology:** physical evidence for prospective claims, with calibration and uncertainty.

Record discrepancies between levels instead of averaging them into a single confidence score. The system should spend expensive evaluations where they could change the decision, and abstain if all available evidence remains inadequate.

### 11. Optimize falsification, not just candidate generation

Reserve part of every campaign for attempts to break the current best hypothesis: unfamiliar material families, perturbed structures, hidden process constraints, sensor drift, bad calibration, and deliberately contradictory evidence. Report false-feasible rate, decision regret, and failures detected, not only candidates found. A workflow that finds impressive examples but misses counterexamples is not ready to guide manufacturing.

## Current research implications (literature checked 2026-10-19)

The most recent work makes the benchmark design more specific:

- **Test equilibrium and far-from-equilibrium configurations separately.** Tahmasbi et al. (published 28 July 2026) benchmark universal potentials with equation-of-state tests and minima-hopping structural searches. They report that search efficiency and structural fidelity can diverge, and that a smoother learned potential-energy surface does not guarantee correct energetic ordering. AtomFab therefore needs structural recovery and energy-ordering metrics in addition to energy/force errors. [DOI: 10.1038/s41524-026-02251-2](https://doi.org/10.1038/s41524-026-02251-2).
- **Treat atom manipulation as a narrow physical task, not general atomic printing.** An integrated AI framework published in December 2025 demonstrated autonomous STM atom manipulation for silver on a specific Si(111)-(7 × 7) surface at room temperature, including drift correction, tip conditioning, atom detection, site localization, and manipulation-condition evaluation. This is a meaningful precedent for a bounded controller, not evidence of arbitrary-material or three-dimensional manufacturing. [DOI: 10.1021/acs.nanolett.5c04982](https://doi.org/10.1021/acs.nanolett.5c04982).
- **Benchmark the structure found as well as the energy assigned to it.** A 2026 supported-nanoparticle benchmark explicitly studies the decoupling of energy accuracy from structural exploration. It reinforces the need for candidate recovery, correct energetic hierarchy, and task-level fidelity tests. The machine-readable registry now has a distinct `supported-nanoparticle-search` lane with separate accuracy and search-discovery metrics, equal-budget policy comparisons, and explicit deduplication/held-out-family rules. The Rust kernel also implements a normalized finite-prefix top-weighted ranking-overlap metric with duplicate-ID, ranking-length, and persistence checks. Its rank-position/counting algorithm has expected linear work in candidate count rather than repeated prefix-set intersections. Before campaigns, freeze canonical structure identity, deduplication tolerance, and ID mapping across policy outputs; overlap scores are not interpretable if the same structure can receive different IDs.

For uncertainty-based abstention, the kernel computes risk-versus-coverage points sorted by uncertainty and a stepwise area under the risk-coverage curve. Equal-uncertainty samples enter as a group, with a canonical secondary sort to make floating-point accumulation permutation-deterministic. The evaluation plan requires comparison to random and shuffled uncertainty scores at matched coverage; report accepted coverage, risk, false-feasible, and missed-feasible counts together.

For nanoparticle search, report candidate discovery overlap separately from ranking quality. The matched-ranking helper computes candidate-set Jaccard on the original outputs and top-weighted overlap only after filtering both rankings to shared canonical structure IDs. No intersection means matched-ranking quality is undefined; do not substitute zero, silently invent an ID mapping, or mix the lack of discovered structures into the ranking score. [DOI: 10.1038/s41524-026-02043-8](https://doi.org/10.1038/s41524-026-02043-8).

These papers refine the roadmap but do not replace the initial MS25 reproduction. They are research evidence for choosing tests, not measurements of AtomFab.

## Initial benchmark design

### Dataset and leakage controls

1. Begin with small, well-understood subsets of MS25; do not commit multi-gigabyte datasets into the repository.
2. Record the dataset DOI/landing page, exact selected files, license, checksums, source split, unit conventions, structure identifiers, and filtering steps before running comparisons.
3. Keep correlated configurations from the same parent structure or material family in the same split. Add at least one grouped out-of-distribution (OOD) holdout by material family or framework. A random frame split alone is insufficient to establish transferability.
4. Do not compare results produced with different reference methods, unit conventions, convergence settings, or effective information budgets as if they were equivalent.
5. Publish unsuccessful runs and missing-data decisions. No cherry-picking the best seeds.

### Baselines

- **Reference:** published split and reference values, respecting their method and tolerance.
- **Simple model:** a documented, reproducible baseline available for the chosen subset.
- **Specialist model:** one established interatomic-potential implementation, pinned by version and configuration; compare other models only when they can be run under matched conditions.
- **Acquisition baselines:** random sampling, diversity-only selection, uncertainty-only selection, and a predeclared hybrid that accounts for query cost.
- **Ablations:** same physics solver without Symthaea selection; selection without uncertainty; selection without history retrieval; and a deliberately misleading random-frame split as a negative control for leakage detection.

The first phase compares orchestration and active learning; it does not train a new universal potential or claim to outperform the state of the art.

## Benchmark metrics and proposed gates

These are targets for a future preregistered experiment, not achieved results or universal physical tolerances.

| Metric | Definition and reporting | Proposed gate |
|---|---|---|
| Property prediction | MAE, RMSE, maximum absolute error, units, sample count, and per-family results | Do not accept aggregate-only reporting; task tolerances must be set before viewing test outcomes |
| Derived observables | Errors for the relevant barrier, surface, lattice, volume, or stability observable | Correct physical ranking and decision thresholds must be validated independently of energy/force MAE |
| Structure discovery and ranking | Fraction of reference structures recovered; pairwise agreement between predicted and reference energetic orderings | Report recovery and ordering separately; neither metric substitutes for geometric validity or full physical validation |
| Manufacturability co-design | Pareto frontier/hypervolume across material function, process feasibility, yield, cost, resource demand, and uncertainty | Freeze objectives and normalization before comparing; hard constraints cannot be traded away by a weighted average |
| Decision regret | Gap between the campaign's selected candidate and the best known validated candidate at equal query/compute budget | Report complete learning curves; identify performance regressions hidden by averages |
| Uncertainty calibration | Empirical coverage of fixed prediction intervals and mean interval width | For nominal 90% intervals, target 85–95% coverage on in-distribution validation; report grouped OOD coverage separately |
| Active-learning acceleration | AF = reference queries / strategy queries to reach the same preregistered target | Stretch target: median AF ≥ 2 across at least 20 fixed-seed campaigns, with a 95% bootstrap confidence interval whose lower bound exceeds 1 |
| Equal-budget enhancement | Best validated objective after the same number and cost of queries | Report full learning curves and uncertainty; do not select a single favorable budget after looking at results |
| OOD behavior | Error, calibration, abstention rate, and false-confidence rate on grouped holdouts | Explicit synthetic OOD negative controls must trigger abstention; no OOD case may be silently labeled verified |
| Reproducibility | Replay exact split, seed, config, and versions | Same deterministic inputs reproduce identical ranking where deterministic; numerical outputs remain within declared tolerances |
| Process feasibility | Satisfied/violated constraints with cited evidence | No physical actuation unless every hard constraint is satisfied and a human-approved operating envelope exists |
| False-feasible rate | Incorrect feasibility accepts divided by all predicted-feasible cases, with false-accept count and missed-feasible count | Zero false-feasible accepts on the frozen adversarial suite; if no candidates are accepted, report rate as undefined, not zero |
| Workcell twin validity | Replay fidelity, geometry/contact residuals, calibration error, and per-sensor uncertainty | No general validity claim from synthetic data alone; validate each physical channel against a stated reference |
| Hard-bound rejection | Count of accepted proposals that violate a hard constraint under adversarial test fixtures | Zero false-feasible proposals on the frozen constraint suite; every rejection has a machine-readable reason |
| Fault response | Detection precision/recall, safe-stop latency, recovery rate, and critical misses under drift/dropout/saturation/collision faults | Preregister per-fault limits; a missed critical fault cannot be hidden by an aggregate score |
| Cross-repository contract | Shared valid/invalid fixtures parsed identically by the Symthaea and Symtropy adapters | 100% agreement on the frozen fixture corpus; unknown critical values and missing units fail closed |
| Fabrication outcome (later phase) | Yield, defect density, thickness/placement error, throughput, rework, energy/material use | Set targets per instrument and material before prospective experiments; never substitute simulated yield for measured yield |

Acceleration factor follows the self-driving-lab literature and only has meaning when both methods reach the **same target on the same benchmark**. Fix the target and budget before the campaign. Use at least 20 seeds for the stretch gate, retain all runs, and report confidence intervals and full learning curves. If the benchmark is too small or expensive for 20 campaigns, explicitly label the result exploratory rather than relaxing the protocol after seeing outcomes.

For atom manipulation specifically, later hardware benchmarks should use lattice-relative placement error, success rate per attempted move, unintended neighbor displacement, recovery rate, time per accepted placement, and defect-free assembled area. Report temperature, surface, tip/instrument, environmental conditions, and the fraction of attempts excluded. No single length tolerance or throughput goal is declared before choosing the physical system.

## Milestones that do not depend on CI runners

### M0 — Benchmark kernel and interop contract (this change)

- Implement pure Rust metrics for scalar prediction errors, prediction-interval coverage, pairwise energy ordering, normalized finite-prefix top-weighted ranking overlap at preregistered persistence p=0.94, uncertainty-selective risk-coverage/AURC, reference-structure recovery, false-feasible decisions, and active-learning acceleration, with fail-closed input validation.
- Add contract-v1 JSON Schema, serde-backed envelope/payload types, and semantic validation for source-kind, provenance, uncertainty, digests, measurement calibration, abort guards, and hard-constraint outcomes.
- Add a shared valid process-proposal JSON fixture and a negative no-abort-guard fixture.
- Add a machine-readable benchmark registry covering ten scientific, experimental, active-learning, multi-fidelity routing, workcell and prospective-metrology lanes; validate it for duplicate IDs, required policy baselines, and nontrivial performance gates.
- Add versioned campaign records and a campaign JSON Schema so completed, failed, and incomplete runs retain evidence and cannot silently change the model, dataset, target, or budget between paired policy comparisons.
- Keep solver adapters, dataset loading, and instrument drivers out of the metric kernel.
- Add deterministic unit tests with hand-calculated examples, semantic negative controls, benchmark-registry integrity checks, and matched-campaign compatibility checks.
- Provide `cargo run -p symthaea-atomfab --bin atomfab-validate-registry` as a local, fail-closed check of the embedded benchmark matrix without needing external datasets or model downloads.
- Companion workcell/interop contract: [Symtropy PR #2140](https://github.com/Luminous-Dynamics/symtropy/pull/2140).
- Record this research protocol and literature pointers.

### M1 — Dataset manifest and reference reproduction

- Select one tractable MS25 subset and record its exact source files, license, source method, and hashes.
- Pin one strong available pretrained interatomic potential and one simpler baseline; record model/license, checkpoint digest, effective defaults, hardware, and convergence settings.
- Reproduce published reference metrics with checksums and pinned versions; report a matched error and task-level observable table.
- Add grouped material-family splits and a leakage negative control.
- Do not publish an AtomFab performance claim until this passes.

### M2 — Retrospective active learning

- Replay the same query budget across the predeclared acquisition baselines.
- Measure AF, equal-budget enhancement, uncertainty coverage, OOD behavior, and compute/query cost.
- Use an ablation to determine whether Symthaea contributes beyond conventional active learning.

### M2A — Multi-fidelity evidence-action routing

- Compare fixed cheapest-first escalation, random eligible action, uncertainty-only escalation, calibrated cost-aware value of information, the full Symthaea router, and a history/retrieval ablation.
- The new Rust evidence router selects only among actions in one request, evidence domain, decision contract, cost/benefit unit, and budget. Atomistic calculations, workcell simulation and metrology are separate comparison lanes.
- Calibration of the expected decision-loss reduction must be measured on disjoint validation campaigns before prospective routing; the router is not allowed to fabricate its own value estimates.
- Run shuffled-value and high-value/out-of-domain negative controls; retain blocked actions, failures, abstentions, and actual cost in the report.
- Measure validated objective and unique discoveries per total cost, target regret, time-to-target, expected-value calibration, fidelity mix, abstention and domain-rejection rates. Apply the Pareto frontier rather than a single scalar score.
- Do not count a recommended/selected action as a successful experiment. The selected action is a plan; only its recorded outcome can supply benchmark evidence.

### M3 — Symtropy process digital twin

- Implement the proposal/evaluation contract described in [Symtropy PR #2140](https://github.com/Luminous-Dynamics/symtropy/pull/2140), beginning with shared fixtures and typed process constraints.
- Build a simulation-only workcell scenario for stage/tool reachability, collision envelopes, sequence timing, drift, sensor dropout, and actuator saturation.
- Require a machine-readable rejection trace and fail-closed hard bounds.
- Connect atomistic solver adapters and process simulation through versioned interfaces; do not use the rigid-body engine as an atomistic solver.
- No remote or unattended instrument control.

### M4 — Prospective experiment (only with appropriate laboratory access)

- Pre-register material, equipment, calibrations, acceptance criteria, exclusion policy, and operating envelope.
- Have a qualified operator supervise all instrument actions.
- Compare prospective observations to frozen predictions and archive both successes and failures.

## Decision rules

- **Advance** only if the method beats a matched baseline on a predeclared criterion and preserves uncertainty reporting.
- **Revise** if average error improves but derived observables or OOD reliability regress.
- **Stop or narrow** if Symthaea adds no measurable value beyond the specialist solver and baseline acquisition logic.
- **Never claim** arbitrary atom-by-atom 3D printing, practical throughput, or demonstrated material superiority from this software scaffold alone.

## Primary references

1. Barros-Luque et al. (2026). “The Open Materials 2024 (OMat24) inorganic materials dataset and models.” *Nature Computational Science*. [https://doi.org/10.1038/s43588-026-00996-w](https://doi.org/10.1038/s43588-026-00996-w).
2. Riebesell et al. (2025). “A framework to evaluate machine learning crystal stability predictions.” *Nature Machine Intelligence*. [https://doi.org/10.1038/s42256-025-01055-1](https://doi.org/10.1038/s42256-025-01055-1). Official guide: [Matbench Discovery](https://docs.materialsproject.org/services/ml-and-ai-applications/matbench-discovery).
3. Mannan et al. (2026). “UniFFBench: evaluating universal machine learning force fields against experimental measurements.” *Nature Computational Science*. [https://doi.org/10.1038/s43588-026-01019-4](https://doi.org/10.1038/s43588-026-01019-4).
4. Maxson et al. (2025). “MS25: Materials Science-Focused Benchmark Data Set for Machine Learning Interatomic Potentials.” *Journal of Chemical Information and Modeling* 65(15), 8097–8112. [https://doi.org/10.1021/acs.jcim.5c01262](https://doi.org/10.1021/acs.jcim.5c01262). Dataset landing page: [Argonne ACDC](https://acdc.alcf.anl.gov/mdf/detail/74d6b3d9-c33c-47c4-9c9a-c14ca773d3c8-1.0/).
5. Tahmasbi, Knüpfer, Kühne, and Mirhosseini (2026). “Benchmarking universal machine learning interatomic potentials on elemental systems.” *npj Computational Materials* 12, 251. [https://doi.org/10.1038/s41524-026-02251-2](https://doi.org/10.1038/s41524-026-02251-2).
6. “Integrated AI Framework for Room-Temperature Atom Manipulation in Scanning Probe Microscopy.” *Nano Letters* (2025). [https://doi.org/10.1021/acs.nanolett.5c04982](https://doi.org/10.1021/acs.nanolett.5c04982).
7. Adesiji, Wang, Kuo, and Brown (2026). “Benchmarking self-driving labs.” *Digital Discovery* 5, 14–27. [https://doi.org/10.1039/D5DD00337G](https://doi.org/10.1039/D5DD00337G).
8. Open Catalyst Project. [OC20/OC22 datasets, baselines and leaderboards](https://opencatalystproject.org/).
9. Shao, Hightower, and Schindel (2022). “Credibility Consideration for Digital Twins in Manufacturing.” *Manufacturing Letters*. [https://doi.org/10.1016/j.mfglet.2022.11.009](https://doi.org/10.1016/j.mfglet.2022.11.009).
10. Frechette, Shao, and Pease (2026). *Digital Twins Workshops Summary Report*, NISTIR 8620. [https://doi.org/10.6028/NIST.IR.8620](https://doi.org/10.6028/NIST.IR.8620).


## Multi-fidelity routing artifact (2026-10-20)

The executable contract and decision policy are documented in [ATOMFAB_MULTIFIDELITY_EVIDENCE_ROUTING_2026-10-20.md](ATOMFAB_MULTIFIDELITY_EVIDENCE_ROUTING_2026-10-20.md). Schemas and an explicitly synthetic fixture are included under `crates/domains/symthaea-atomfab`. The router produces recommendations only; it does not execute solvers, process twins, laboratory instruments, or physical actions.


## Paired acceleration statistics v1 (2026-10-20)

The Rust module `campaign_statistics.rs` implements a deterministic percentile bootstrap for the **median paired acceleration factor**:

`AF(seed) = baseline_total_cost(seed) / candidate_total_cost(seed)`

The input is a pair of traceable `CampaignRecord` values per preregistered seed. Each baseline/candidate pair must validate against the same lane, model, data, evaluator, shared environment, target definition, budget contract, objective and cost unit. Baseline and candidate identities and each policy's configuration must remain stable across seeds. Costs must be finite and positive; seed IDs and run IDs must be unique.

The interval is deliberately fail-closed: the function emits no acceleration interval unless at least 20 pairs are provided and **every paired run is completed and reaches the same frozen target**. Do not filter to successful seeds after looking at results; target failures remain in the campaign table, and the interval is withheld until the preregistered cohort/targeting rule is satisfied. Use the failed/incomplete campaigns to report target reachability separately. It uses a pinned bootstrap PRNG seed, at least 1,000 resamples, a declared confidence level, and linearly interpolated percentile bounds. The generated report has a v1 JSON Schema and semantic validation; independent implementations should recompute the summary from the raw campaign records rather than trusting the summary alone.

This reports **cost-to-target**, not general wall-clock speed, model throughput, experimental yield, or manufacturing rate. Even a lower confidence bound greater than 1 does not establish the overall policy win on its own: report the paired median/interval, failures, and target-reaching coverage, then separately require no worse reference-validated objective quality at equal total cost. If results fail the gate, publish the non-win rather than changing the target, sample set, or statistic.

Artifacts: `src/campaign_statistics.rs`, `schemas/paired-acceleration-summary-v1.schema.json`, and the `active-learning-policy` lane in `benchmarks/benchmark-matrix-v1.json`. This code and its unit tests are not external MLIP comparisons; no speedup has been measured.


### Frozen seed roster and statistical claim boundary

The current proposed extension pins seeds **101–120** in `policy_comparison.seed_manifest`. Its canonical SHA-256 is `c8dd76e3828125d3984e95bef0cd50603cee2bef7933f1fec178866dd973ba59`, computed over UTF-8 bytes of `atomfab-seed-manifest-v1\n` followed by each unique seed in ascending decimal form, one seed per line. The Rust `SeedManifestV1` checks this digest and rejects duplicate/unsorted rosters.

The paired acceleration function receives the manifest explicitly, requires exactly one baseline/candidate pair for every seed in that roster, and writes both the roster and digest into its report. A missing seed, extra seed, duplicated seed, or mismatched roster blocks the interval. A campaign with a failed/non-target run is retained as a failure and blocks this strict target-reaching interval; it must not be silently removed to make the policy look faster. The seed roster and hash are proposal inputs and must remain fixed before outcomes are observed.

The illustrative seed-manifest fixture is a test artifact, not evidence of any completed campaign. The accepted report still requires successful execution on an exact code head, traceable completed campaign records, and independent recomputation of the statistics from those records.
