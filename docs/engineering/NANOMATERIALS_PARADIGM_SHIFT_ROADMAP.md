# NANO-001: Paradigm-Shift Roadmap for Symthaea-Assisted Materials Discovery

**Status:** Research proposal and preregistered hypotheses; not a validated atomistic model, material discovery, synthesis protocol, or authorization for physical experiments.  
**Date:** 2026-10-15  
**Parent:** [NANO-001 #7453](https://github.com/Luminous-Dynamics/symthaea/issues/7453)  
**Companion:** [Evidence-Bounded Nanomaterials Discovery Program](./NANOMATERIALS_RESEARCH_PROGRAM.md)  
**Data gate:** [NANO-001A #7455](https://github.com/Luminous-Dynamics/symthaea/issues/7455)  
**Model/rights gate:** [NANO-001C #7465](https://github.com/Luminous-Dynamics/symthaea/issues/7465)

## Executive thesis

The highest-leverage shift is not “make a larger model predict a material.” It is to build a **closed-loop, evidence-budgeted discovery policy** that chooses which calculation, measurement, or uncertainty-reducing test should happen next—and can prove that this policy beats simple alternatives under equal budgets.

That changes the optimization target from prediction accuracy alone to **decision quality per unit of evidence, compute, money, time, and risk**. A model may be inaccurate but useful for triage; a high-accuracy model may be useless if it is miscalibrated, duplicates known candidates, ignores synthesis constraints, or recommends expensive tests that do not change a decision.

No claim below assumes Symthaea already has atomistic physics. Existing components are candidate policy, representation, memory, temporal reasoning, and audit contributions. Each must be measured against a simpler baseline before being promoted.

## Paradigm shifts worth testing

### P1 — From “predict the property” to “buy the most useful evidence”

For a fixed budget, select the next candidate/calculation by expected decision improvement, not just predicted performance or model uncertainty.

A possible utility is:

- expected reduction in decision regret
- minus declared compute cost
- minus declared time cost
- minus unresolved-risk penalty

This is a decision-policy sketch, not a calibrated formula. Terms and weights must be frozen with domain experts before evaluation. Compare seeded random acquisition, descriptor/diversity selection, uncertainty-only acquisition, cost-aware expected improvement or information gain, and Symthaea-assisted acquisition.

**Success criterion:** better held-out decision utility or label-efficiency curve at the same total budget, with uncertainty intervals and all failures retained. A score improvement without decision benefit is not a win.

### P2 — From one “best material” to a diverse, constraint-aware frontier

Search for a *portfolio* of candidates spanning useful trade-offs: performance, cost, abundance or criticality of elements, durability, processability, environmental burden, and known safety gaps. Do not collapse these into one opaque score.

Return a Pareto set and show which candidate dominates another under which assumptions. Penalize duplicate structures and near-identical surface/adsorbate families. A candidate with missing hazard or manufacturability evidence is “unknown,” not “safe” or “feasible.”

**Success criterion:** improved non-dominated coverage and lower redundancy than descriptor-only selection at equal budget, without changing objective weights after seeing outcomes.

### P3 — From benchmark rows to causal- and lineage-aware scientific units

A row is not necessarily a candidate; a configuration is not necessarily an independent system; a relaxed endpoint is not interchangeable with an initial input. Track material/system, slab/bulk lineage, adsorbate, trajectory, calculation method, and split membership separately.

Group evaluation by meaningful scientific families where source keys support it—for example related bulk parent, composition, surface family, adsorbate, or trajectory. Test whether performance collapses when a whole family is held out. Preserve ambiguity instead of guessing a join.

**Success criterion:** reproducible source-to-pair crosswalk and split audit before any accuracy claim. The current OC22 admission/crosswalk blocker remains first; this roadmap does not bypass it.

### P4 — From “confidence” to calibrated abstention and explicit unknowns

A selector should sometimes say “insufficient support—run a reference calculation or request expert review.” Separate predictive uncertainty, distance from the training/applicability domain, data/provenance uncertainty, method/DFT mismatch, and safety/manufacturability unknowns.

Do not treat HDC similarity, Phi, ensemble spread, or fluent reasoning as calibrated probability by default. Calibrate on held-out groups, report selective risk versus coverage, and include out-of-domain and malformed-input tests.

**Success criterion:** abstain more often on genuinely difficult/out-of-domain cases and achieve a predeclared error-risk target at useful coverage. If calibration fails, retain abstention as a heuristic label, not a probability.

### P5 — From positive examples only to learning from failed and contradictory evidence

Record failed relaxations, non-converged calculations, unstable candidates, negative experiments, inconsistent measurements, and superseded hypotheses. Separate “failed under this method/condition” from “universally impossible.” Track failure mode and conditions so the selector can avoid repeating known unproductive work without erasing negative evidence.

**Success criterion:** lower repeated-failure rate or higher information gained per budget on a frozen replay, compared with a baseline that ignores failure history. No silent deletion of failures from the denominator.

### P6 — From one fidelity to a verified ladder of evidence

Use cheap filters for triage, not as substitutes for higher-fidelity evidence. A candidate can progress through source/data checks, low-cost descriptors or screening, qualified surrogate, converged reference calculation, human-reviewed experimental proposal, and independent characterization/replication. Every transition states what new evidence it adds and what uncertainty remains.

This ladder depends on the material question. Do not run a DFT/ML potential outside its supported chemistry, spin, periodicity, or domain just because it is available.

**Success criterion:** demonstrate that a multi-fidelity policy achieves a predeclared decision target with less total cost than an equal-budget single-fidelity policy, while preserving mandatory reference checks and uncertainty bounds.

### P7 — From “novel structure” to “could it be made and survive?”

Computational novelty is not synthesizability. Add manufacturability and lifecycle evidence as separate gates: precursor availability, plausible processing window, phase/defect sensitivity, scale-up, aging, dissolution/leaching, worker exposure, ecotoxicity where relevant, recovery/disposal, and measurement feasibility. Missing evidence stays missing.

**Success criterion:** for a retrospective, expert-labeled candidate set, improve identification of candidates worth a synthesis-planning review without increasing false “safe” or “synthesizable” assertions. This is an expert-review aid, not autonomous synthesis.

### P8 — From static model to continual, auditable learning

After each approved calculation or measurement, update the evidence graph and acquisition state using immutable run identities. Never overwrite old labels silently: preserve source revision, method, conditions, units, raw-result digest, transformations, and supersession. Detect drift when instruments, solvers, or protocols change.

**Success criterion:** an independent checker can regenerate the candidate ranking from a frozen evidence snapshot and explain each ranking change through new evidence or a versioned policy change.

## What more of Symthaea can contribute

These are role hypotheses, not claims that every capability is production-ready or qualified for scientific use. First inventory actual modules and tests at the pinned commit; reuse existing owners and avoid a parallel materials ontology.

| Symthaea capability family | Potential contribution | Required guardrail / ablation |
|---|---|---|
| Hyperdimensional computing (HDC) | Fast grouping, duplicate/near-duplicate triage, coarse candidate coverage and retrieval over descriptors/metadata | Not a geometry-aware atomistic potential. Test against exact hashes, conventional descriptors, and random grouping. Never infer energies from vector similarity. |
| Temporal/CfC/LTC-style models, where supported by actual interfaces | Analyze genuine relaxation trajectories, time-series measurements, aging or process histories; identify transitions and changes | Never fabricate a time axis from unordered rows. Compare with simple trajectory statistics; validate on held-out trajectory families. |
| Free-energy / predictive-processing or active-inference mechanisms, if exposed and testable | Formulate candidate actions and compare expected evidence value against cost; model explicit observation/update loops | A cognitive free-energy quantity is not thermodynamic free energy. Name the objective precisely and test it against standard acquisition functions. |
| Global-workspace/attention and working-memory mechanisms, where present | Route scarce review attention to conflicting claims, high-impact unknowns, or candidates that could change the Pareto frontier | Attention or salience is not scientific confidence. Compare triage yield with a simple deterministic priority rule. |
| Competing hypotheses / counterfactual reasoning, if supported by concrete interfaces | Keep multiple explanations alive and propose discriminating calculations rather than one confident narrative | Each hypothesis must yield a falsifiable prediction and a test that distinguishes alternatives. Score with blind outcomes, not explanation quality. |
| Retrieval and semantic/literature tooling | Map prior art, known failure conditions, methods and claims to candidate provenance; identify likely rediscovery | Literature coverage is never proof of novelty. Keep retrieval snapshots and human review; evaluate recall on a labeled set. |
| Metacognitive monitoring, uncertainty and abstention components | Detect unsupported claims, missing evidence, out-of-domain requests, and mismatch between claim strength and evidence | Audit calibration separately. Phi, coherence, consensus, or self-reported confidence cannot stand in for empirical calibration. |
| Independent verifier / typed evidence contracts | Check artifact identity, source lineage, units, splits, metrics, controls and claim boundaries outside the selector | Verifier must not reuse selector decisions as proof. Keep fail-closed controls and negative tests. |
| Symthaea quantum-chemistry interface | Potentially support a separate small-molecule or method-specific comparison if its method and scope match | Current source audit has not qualified it for periodic OC22 oxide slabs. Do not use it as OC22 ground truth without a separate validation study. |
| Mycelix provenance/review/evidence graph | Preserve source rights, claims, independent review, revisions, physical sample lineage and supersession | Mycelix records provenance/authority; consensus does not certify that a scientific claim is true. |
| Symtropy deterministic scenarios / visualization | Visualize candidate families and rehearse workflow/cost scenarios using validated upstream outputs | Visualization or rigid-body simulation is not atomistic chemistry. Keep synthetic scenario output distinct from physical evidence. |

### Important architectural rule

**Do not start by wiring every cognitive subsystem into one giant nanomaterials agent.** That creates confounding, makes failures hard to attribute, and risks turning internal scores into pseudo-physics. Start with a thin, deterministic selector interface and add one Symthaea component per ablation. Keep the scientific predictor/calculator fixed across acquisition arms.

A future minimal interface should accept frozen candidate IDs, allowed descriptors/metadata, acquisition budget, declared constraints, prior acquired-label IDs, and explicit unknowns. It should output ranked *requests for evidence*, rationale fields tied to inputs, diversity group, abstention flag, cost basis, and policy version.

Forbidden inputs include hidden validation/test labels, target energies before acquisition, post-hoc outcome fields, and unapproved network data. An independent audit must reconstruct the eligible pool and selected actions from a frozen manifest and assert that no forbidden label was read.

This is an interface contract for a future experiment, not a claim that an implementation already exists.

## A practical first experiment: “Which next calculation is worth paying for?”

Run only after dataset admission, pair crosswalk, split/lineage audit, and model-eligibility gates pass.

1. Freeze a small reproducible candidate pool and acquisition budget; use group-aware train/validation separation.
2. Choose a downstream predictor and keep architecture, training recipe, compute, and seed protocol matched across arms.
3. Compare random, descriptor/diversity, uncertainty-only, standard cost-aware acquisition, HDC-assisted diversity, and one carefully specified Symthaea policy.
4. Use a label oracle that exposes labels only after selection. Log every query, rejection, cost, duplicate, failure and abstention.
5. Include label-shuffle, hidden-label-canary, duplicated-family, out-of-domain, malformed-input, and random-selector controls.
6. Repeat stochastic arms across preregistered seeds; report label-efficiency curves, final held-out error, top-k regret/utility where meaningful, diversity, abstention risk/coverage, wall time, peak memory and confidence intervals.
7. Have an independent checker regenerate metrics from frozen outputs. Keep the final holdout sealed until policy and thresholds are frozen.

Do not run a huge model first. A null result is useful: this experiment tests whether Symthaea's selection and evidence-management add value, not whether prose can disguise missing atomistic physics.

## Recommended implementation sequence

1. **Finish admission tooling and actual data access.** The Rust archive verifier and JSON pair-manifest contract are infrastructure, not data validation. The official OC22 archive remains unverified locally and verifier CI must pass on the exact head.
2. **Add a source-inventory report** once bytes are available: hashes, schema, record counts, unique SIDs, split counts, duplicates, non-finite fields, units, and lineage.
3. **Build the smallest deterministic benchmark harness** around the official loader or an independently verified equivalent. Keep selector separate from scientific calculation.
4. **Establish a simple reference baseline** and cost/feasibility profile for target hardware before enabling cognitive components.
5. **Ablate one Symthaea contribution at a time**; preserve negative controls and all failed outcomes.
6. **Only then expand** to adsorption energies, batteries, membranes, coatings, nano-additives, or experimental planning. Each domain needs its own source, target, baseline, rights, safety and validation gates.

## Claims prohibited until evidence exists

- “Symthaea discovered a new nanomaterial.”
- “HDC/Phi/active inference predicts atomistic energy.”
- “The candidate is stable, synthesizable, safe, scalable, or commercially useful.”
- “The model is novel relative to literature” without defensible prior-art review.
- “The selector beats the state of the art” without an exact, independent, task-matched benchmark.
- “Autonomous experimentation is authorized” without qualified lab ownership, reviewed protocol, containment, and explicit human authorization.

## Research references

- FairChem, [OC22 task/data contract](https://github.com/facebookresearch/fairchem/blob/main/docs/catalysts/datasets/oc22.md): official IS2RE records pair initial structures with relaxed structures/energies; preserve task distinction and source lineage.
- Schwalbe-Koda et al., [Model-free estimation of completeness, uncertainties, and outliers in atomistic machine learning using information theory](https://www.nature.com/articles/s41467-025-59232-0), *Nature Communications* (2025): information-theoretic analysis can support dataset diversity, data efficiency, model-free uncertainty and outlier analysis.
- RSC, [Uncertainty in the era of machine learning for atomistic modeling](https://pubs.rsc.org/en/content/articlelanding/2025/dd/d5dd00102a) (2025): uncertainty must be characterized and managed for reliable atomistic-model use.
- [FAIR Chemistry custom datasets](https://facebookresearch.github.io/fairchem/ase-dataset-creation/): supported routes for custom structures/calculations.
- [FairChem pretrained models and task matrix](https://github.com/facebookresearch/fairchem): model compatibility, checkpoint exposure, and each model's license must be assessed separately.

**Evidence status:** literature-informed architecture proposal. No raw OC22 data rows, atomistic predictions, model training, synthesis, or experimental result are claimed by this document.
