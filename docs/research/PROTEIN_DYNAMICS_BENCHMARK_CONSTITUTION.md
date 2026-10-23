# Protein Dynamics Benchmark Constitution (Draft v0.1)

**Status:** research protocol proposal only. No model has been qualified by this document, no benchmark has been run, and no biological discovery is claimed.

**Parent program:** [BIO-MILLENNIUM-000, issue #6271](https://github.com/Luminous-Dynamics/symthaea/issues/6271). This document narrows the program's existing requirement that static structure, conformational ensembles, dynamics, and experimental observations remain distinct. It is not a second scientific-reasoning stack.

## 1. Decision to test

Test whether Symthaea can use a bounded adaptive controller—potentially combining HDC retrieval, state/uncertainty tracking, and active selection of the next expensive evaluation—to reach a declared protein-ensemble evaluation target with less compute than simpler methods, without reducing coverage of experimentally supported states.

This is a falsifiable systems hypothesis, not an assumption that HDC or CfC is inherently better at molecular modeling.

The initial target is **condition-specific conformational ensemble reasoning**, not a claim to solve all protein folding. A static structure is not an ensemble; an ensemble is not a kinetic model; a kinetic model is not automatically experimentally correct.

### Claims this protocol does not authorize

- “Symthaea solves protein folding” or “beats AlphaFold” as a general claim.
- Protein-scale molecular-dynamics capability based on the existing small-molecule quantum-chemistry MD module.
- Biological truth from agreement among models, simulated trajectories, or a formal proof about a model.
- A physical discovery, binding result, or therapeutic claim from retrospective computational scores.
- An unqualified aggregate score that combines different experimental observables or evidence classes.

## 2. Initial source audit

The code and benchmark boundaries below are the initial facts that motivate a dedicated protocol. Recheck them against the exact commit used for any implementation; this audit is not an enduring assertion about future repository HEAD.

1. [symthaea-proteins](../../crates/domains/symthaea-proteins/src/lib.rs) describes **protein secondary-structure prediction**. Its labels are three-state helix/sheet/coil; benchmark.rs reports Q3 and per-class secondary-structure metrics. That is useful as a bounded baseline, but it does not evaluate tertiary 3D structure, conformational populations, or transition kinetics.
2. [cb513.rs](../../crates/domains/symthaea-proteins/src/cb513.rs) describes a hardcoded, small subset (approximately 40 entries of at most 80 residues), not an evaluation over the full CB513 corpus. Results from this subset must not be reported as full-CB513 performance or as a modern 3D-fold benchmark.
3. [molecular_dynamics.rs](../../crates/domains/symthaea-quantum-chemistry/src/molecular_dynamics.rs) explicitly marks its initialization as non-physical thermal sampling: it uses a deterministic hash and uniform velocity components rather than Maxwell–Boltzmann sampling, does not remove center-of-mass momentum, obtains forces by finite differences, and documents a deliberately loose short-trajectory energy-drift bound. This implementation may support bounded small-system experiments; it is not a protein-scale MD baseline or a validated source of protein conformational ground truth.
4. [biophysics.rs](../../crates/core/symthaea-core/src/physics/biophysics.rs) labels its HDC biophysics encodings as exploratory. An encoding or analogy is not, by itself, a trained structural predictor or a physically calibrated energy model.

These limits define a clean development opportunity: compose with specialist structure/ensemble/simulation tools and measure whether Symthaea's orchestration reduces wasted computation.

## 3. Scientific basis and dataset roles

The protocol deliberately separates experimental ensemble data from simulated trajectories.

### Primary retrospective source candidate: Protein Ensemble Database (PED)

[PED](https://proteinensemble.org/) is an open-access, curated repository of conformational ensembles, with experiment and method metadata and links to primary publications. It includes experimental entries based on methods such as NMR, SAXS, and single-molecule FRET, as well as computationally predicted entries. The primary test set must use **experimental records and the observable(s) actually reported for each record**; predicted entries are quarantined from the primary evidence set unless used explicitly as training data with exposure recorded.

For each candidate PED experiment, retain the accession, version/retrieval date, construct and residue range, sequence and modification identity, oligomer/ligand state, experimental conditions, evidence method, primary publication, available observables, data/license terms, and downloaded asset hashes. Distinct conditions or constructs are not interchangeable observations simply because their sequences overlap.

### Secondary computational source candidate: mdCATH

[mdCATH](https://www.nature.com/articles/s41597-024-04140-z) is a large molecular-dynamics dataset intended to support data-driven computational biophysics. It can support controlled experiments on trajectory forecasting, adaptive sampling, conformational-state discovery, and simulation-budget allocation. However, a trajectory generated by a force field is **simulation output**, not independent experimental validation. Record the force field, preparation, solvent, temperature, integrator, and simulation provenance, and do not pool simulation-derived labels into experimental success counts.

### Static structure / secondary structure sources

The existing CB513 subset remains a separate, low-cost software regression task for Q3 and per-class secondary-structure metrics. If a tertiary-structure task is later added, freeze an appropriate structure-prediction benchmark and its training/exposure controls independently. Do not reinterpret Q3 as TM-score, lDDT, fold recovery, ensemble fidelity, or kinetics.

### Literature motivation

- Jumper et al., “Accurate structure prediction of biomolecular interactions with AlphaFold 3,” Nature (2024), [doi:10.1038/s41586-024-07487-w](https://doi.org/10.1038/s41586-024-07487-w). The paper describes limitations including that repeated prediction seeds do not produce an approximation to the solution conformational ensemble.
- “Protein dynamics prediction by integrating biophysics and artificial intelligence,” Current Opinion in Structural Biology (2026), [doi:10.1016/j.sbi.2026.103223](https://doi.org/10.1016/j.sbi.2026.103223). It highlights combining experimental data and biophysical constraints.
- “The landscape of machine learning approaches for modeling protein conformational ensembles,” Current Opinion in Structural Biology (2026), [doi:10.1016/j.sbi.2026.103253](https://doi.org/10.1016/j.sbi.2026.103253). It identifies lack of standardized benchmarks as a barrier to measuring progress.

These publications motivate the task and metric boundaries; they do not provide qualification evidence for Symthaea.

## 4. Preregistered hypotheses

Freeze the target list, split manifest, metrics, resource cap, baseline implementations, software/model versions, and stopping rules before looking at held-out results.

- **H1 — compute efficiency:** at a matched resource budget, the Symthaea-controlled search produces better declared ensemble/observable fidelity than a simple fixed-budget baseline; alternatively, it reaches a preregistered fidelity threshold using fewer expensive evaluations.
- **H2 — adaptive value:** adaptive selection of the next simulation, conformer, or analysis call improves the quality-versus-cost curve over uniform or fixed schedules.
- **H3 — representation value:** HDC retrieval/relational features add measurable value beyond sequence-similarity retrieval, geometry-aware nearest-neighbor baselines, and conventional clustering.
- **H4 — calibration value:** controller uncertainty ranks failures or high-value follow-up evaluations better than simple baseline uncertainty proxies.

Each hypothesis may be rejected. A result is not a pass merely because a candidate is generated, a workflow is queued, a simulation completes, or a score improves on a training-like example.

## 5. Evaluation tracks

Do not combine these tracks into a single undifferentiated “protein score”.

### Track A — experimental ensemble compatibility

For each eligible experimental record, use only its declared, condition-matched observables. Depending on the entry these may include radius of gyration, end-to-end distance, SAXS profiles, FRET distance distributions, NMR chemical shifts, residual dipolar couplings, or paramagnetic-relaxation-enhancement constraints.

Primary scoring must measure agreement with the **available observable distributions or measurements**, with a metric chosen before evaluation and documented for that assay class. Report missing observables as not applicable, not as zero or success. Where observation uncertainty is available, carry it into the score. Do not assume one deposited coordinate ensemble is the unique native truth for a disordered protein.

### Track B — simulated trajectory tasks

Use held-out mdCATH trajectories or another explicitly versioned computational dataset for state discovery, next-window forecasting, rare-state retrieval, or adaptive sampling. Choose one task and its forecast/sampling horizon before the run.

Evaluate state coverage and observables relevant to that task. Transition probabilities or kinetics are eligible only when the sampling, time resolution, and reference data justify them. Report force-field and simulation ancestry. Success here establishes performance against the declared simulation target, not biological validity.

### Track C — static structure prediction

This is a separate optional track. Use appropriate structure metrics and a time/family-held-out protocol. A predicted static structure cannot satisfy an ensemble or dynamics criterion. Do not claim this track until a real 3D prediction pipeline, data license/exposure audit, and independent evaluation implementation exist.

## 6. Required controls and baselines

At minimum compare:

- **B0 — simple baseline:** a preregistered trivial/fixed strategy, including a fixed random or uniform-budget selection policy where applicable.
- **B1 — conventional retrieval/clustering:** sequence and/or structural-neighbor retrieval and a standard clustering/sampling strategy, chosen for the task.
- **B2 — specialist baseline:** an appropriate existing ensemble predictor or MD/sampling workflow with its exact version and configuration recorded.
- **B3 — Symthaea controller without HDC:** same compute budget and upstream tools, with HDC retrieval/analogy disabled.
- **B4 — full Symthaea controller:** all predeclared components enabled.
- **B5 — optional active-selection ablation:** the same Symthaea pipeline with adaptive choice disabled, to isolate whether adaptivity adds value.

Do not select whichever baseline makes the result look best after seeing the test answers. Freeze eligible implementations and versions first. Give all compared methods the same test inputs and information access; track extra candidate generations, upstream calls, GPU-hours, CPU-hours, wall-clock time, and simulated nanoseconds where applicable. Preserve shared model, training corpus, template, database, and force-field ancestry; correlated models do not become independent witnesses by voting.

## 7. Splits, leakage, and exposure controls

A residue-level or random-row split is not sufficient for a protein generalization claim. Split by grouped biological subject and experimental lineage.

- Keep all conformers, trajectory frames, replicas, chains/complex partners, alternate records, and repeated measurements from the same target/construct/experiment in one partition.
- Use sequence-family and structure-family/scaffold-aware grouping where applicable. Record the clustering method and identity/coverage thresholds instead of relying on a generic label such as “nonredundant”.
- Keep close homologs, engineered variants, and shared templates out of evaluation partitions when they would reveal the target answer. State exceptions explicitly.
- For prospective-style evaluations, freeze a publication/data cutoff and document model/database/template exposure. If exposure cannot be ruled out, report the result as retrospective, not prospective.
- Use separate split manifests for PED experimental records and mdCATH simulation trajectories; identical sequence does not imply identical condition or interchangeable target state.
- Run a duplicate and near-duplicate audit over sequence, construct, source experiment, structure/trajectory, and publication identities before opening held-out labels.

Every split file must be versioned and content-addressed. Any post-result change to target selection, exclusion rules, score implementation, model version, or threshold creates a new benchmark lineage.

## 8. Compute-efficiency primary endpoint

The primary endpoint is a **quality-versus-resource curve**, not raw accuracy alone. Before execution, declare:

1. the target metric(s) and acceptable/failed validity range;
2. the fixed total resource budget or a common set of budget checkpoints;
3. which operations count as expensive evaluations;
4. resource counters and how they are collected;
5. how failures, timeouts, missing outputs, and invalid structures are scored;
6. the aggregation method across targets and uncertainty interval.

Report each target and the full census, not just favorable examples. Include wall-clock time, CPU/GPU resource use, number of specialist calls, number of candidates evaluated, and simulated nanoseconds where applicable. Use area under the quality-vs-cost curve only after the x-axis, y-axis, normalization, interpolation, and missing-run policy are frozen. Never compare unlike resource units by silently adding them into one invented scalar.

For Track A, keep each assay's observable-level score visible. Any cross-assay summary must be predeclared, justified, and accompanied by the per-assay result vector. For Track B, simulation-based performance stays in a separate table from Track A.

## 9. Immutable run record

Each run should record at least:

- protocol ID/version and exact repository commit;
- exact target, construct, condition, dataset accession/version, source publication, and content hashes;
- split manifest hash, exclusion/duplicate audit, and exposure cutoff;
- model/provider identity and exact version/revision where available, upstream data/template ancestry, inputs, settings, and raw output hashes;
- random seeds, execution environment, hardware, wall time, CPU/GPU use, specialist-call count, and simulation length;
- frozen metric implementation/version, per-target metrics, uncertainty/calibration outputs, warnings, failed runs, and terminal disposition;
- any deviations from preregistration, recorded without rewriting the original plan.

A hash proves content identity, not data correctness, independent execution, model calibration, or biological truth.

## 10. Stop/go gates

Do not proceed to a headline performance claim until all relevant gates pass:

1. **Inventory gate:** target data, licensing, condition metadata, reference observables, and model/data exposure are documented.
2. **Leakage gate:** grouped split and near-duplicate checks pass under an independent checker.
3. **Reproducibility gate:** one target can be replayed from the frozen manifest and yields the same declared metric output within the declared numerical tolerance.
4. **Baseline gate:** all baseline implementations and resource accounting are frozen before opening the test partition.
5. **Coverage gate:** every selected target has a terminal disposition, including failures and exclusions with reasons.
6. **Ablation gate:** comparisons isolate HDC and active-selection contribution, under equivalent upstream access and budgets.
7. **Claim gate:** wording matches the actual track and evidence class. Simulation success cannot be described as experimental validation; retrospective results cannot be described as prospective discovery.

CI is one possible executor of these checks, not a substitute for a local/source audit, and queued CI is not a pass.

## 11. Smallest useful next execution sequence

1. Re-resolve the code audit against the exact commit and record the current status of the existing secondary-structure tests; make no broader folding claim from them.
2. Produce a read-only PED candidate inventory limited to experimental records with downloadable ensemble assets and usable, condition-matched observables. Record exclusions rather than silently discarding them.
3. Produce an independent mdCATH inventory for one narrowly defined simulated-trajectory task; keep it separate from experimental evidence.
4. Select one tractable pilot target set and one endpoint. Freeze targets, resource caps, baselines, metric code, and family/experiment holdouts before running any model.
5. Run the fixed/random and conventional baselines first. Add Symthaea-controlled selection and ablations only once resource accounting and result capture work end to end.
6. Publish the complete result vector—including null/negative results—and decide whether HDC/adaptivity should be expanded based on measured contribution.

No new canonical crate/type is justified by this document. Prefer a thin benchmark runner or adapter over a new protein intelligence stack. Keep any wet-lab work outside this initial computational scope and under appropriate external authorization and oversight.
