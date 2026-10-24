# Evidence-Bounded Nanomaterials Discovery Program

**Status:** Research architecture and benchmark preregistration. This document does not establish a discovered material, validated atomistic model, synthesis recipe, product qualification, or authority to run physical experiments.

**Tracking:** [NANO-001 #7453](https://github.com/Luminous-Dynamics/symthaea/issues/7453)  
**Parent research boundary:** [MAT-HYPOTHESIS-001 #5254](https://github.com/Luminous-Dynamics/symthaea/issues/5254)  
**Date:** 2026-10-15  
**Initial data-admission companion:** [NANOMATERIALS_OC22_ADMISSION_CHECKLIST.md](./NANOMATERIALS_OC22_ADMISSION_CHECKLIST.md)
**Cross-domain strategy:** [NANOTECHNOLOGY_FRONTIER_PORTFOLIO.md](./NANOTECHNOLOGY_FRONTIER_PORTFOLIO.md) — staged opportunities beyond the initial OC22 benchmark.
**Benchmark standards:** [NANOTECHNOLOGY_BENCHMARK_CHARTER.md](./NANOTECHNOLOGY_BENCHMARK_CHARTER.md) defines common evidence gates and task-specific benchmarks across the nano portfolio.  
**New Rust selection prototype:** [tools/nano-policy](../../tree/research/nano-001-atomistic-discovery/tools/nano-policy) provides random, cheapest-first, max-min diversity, and cost-aware diversity policies for controlled evaluation.
**Domain atlas:** [NANOTECHNOLOGY_DOMAIN_ATLAS.md](./NANOTECHNOLOGY_DOMAIN_ATLAS.md) — expands the application taxonomy without assuming scientific transfer across fields.
**NIST Fe–Co–Ni measured thin-film benchmark:** [NANOTECHNOLOGY_NIST_FECO_NI_BENCHMARK_PLAN.md](./NANOTECHNOLOGY_NIST_FECO_NI_BENCHMARK_PLAN.md), tracked by [NANO-007](https://github.com/Luminous-Dynamics/symthaea/issues/7644). It is an offline replay of 921 historic measured samples and is blocked on explicit data-use terms, local integrity/schema validation, and baseline reproduction.
**Independent policy benchmark:** [NANOTECHNOLOGY_CLOSED_LOOP_BENCHMARK_PLAN.md](./NANOTECHNOLOGY_CLOSED_LOOP_BENCHMARK_PLAN.md), tracked by [NANO-003](https://github.com/Luminous-Dynamics/symthaea/issues/7519). This tests discovery-policy value in MADE's convex-hull task and does not replace OC22 validation.  
**Checkpoint eligibility gate:** [NANO-001C #7465](https://github.com/Luminous-Dynamics/symthaea/issues/7465)

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

### Current Symthaea materials modules: suitability gate

The source-level review for [NANO-001B #7460](https://github.com/Luminous-Dynamics/symthaea/issues/7460) found that the existing materials modules are useful in their present domain but are not yet an atomistic OC22 representation/predictor:

| Existing module | What current source implements | Permitted role in NANO-001 | Explicit non-claim |
|---|---|---|---|
| `symthaea-materials::MaterialHdcEncoder` | Maps eight normalized bulk engineering property values into a 16,384D HDC vector | Bulk-material metadata similarity where those properties exist and are measured/qualified | Not an atomistic structure encoder; it does not encode atom coordinates, periodic cells, adsorbate geometry, or local atomic environments |
| `MaterialAgingModel` | Evolves an HDC state and sets `remaining_strength` from state similarity | No use as a physical strength or energy oracle until calibrated against independent outcomes | HDC state similarity is not measured retained strength and does not predict OC22 DFT energy |
| `symthaea-psych-bench::MaterialsDesign` | Checks property ordering and similarity across a small set of bulk-material presets | May remain a bulk-property/similarity smoke test with its declared scope | Does not establish atomistic accuracy, catalyst ranking, novelty, or synthesis success |
| `symthaea-quantum-chemistry` | Source documents Gaussian-basis molecular integrals, Hartree–Fock, and a limited LDA molecular DFT path | Candidate for separate method-specific capability review | No source evidence found in the audited public interface that qualifies it for periodic OC22 oxide-slab calculations or reproduces OC22 calculation settings |

References audited at source snapshot `b9af191f380d635f0459b24284c1d1fd6a5516aa`:
- [Material encoder](https://github.com/Luminous-Dynamics/symthaea/blob/b9af191f380d635f0459b24284c1d1fd6a5516aa/crates/domains/symthaea-materials/src/encoder.rs)
- [Bulk property schema](https://github.com/Luminous-Dynamics/symthaea/blob/b9af191f380d635f0459b24284c1d1fd6a5516aa/crates/domains/symthaea-materials/src/properties.rs)
- [Aging output](https://github.com/Luminous-Dynamics/symthaea/blob/b9af191f380d635f0459b24284c1d1fd6a5516aa/crates/domains/symthaea-materials/src/aging.rs)
- [Bulk material benchmark](https://github.com/Luminous-Dynamics/symthaea/blob/b9af191f380d635f0459b24284c1d1fd6a5516aa/crates/domains/symthaea-psych-bench/src/benchmarks/science/materials_design.rs)
- [Quantum chemistry module overview](https://github.com/Luminous-Dynamics/symthaea/blob/b9af191f380d635f0459b24284c1d1fd6a5516aa/crates/domains/symthaea-quantum-chemistry/src/lib.rs)

Therefore, NANO-001 must not simply feed OC22 rows to the current bulk-material encoder and call that atomistic selection. The first useful integration should preserve atomistic structure/label identity through a pinned upstream data reader, reproduce a compatible reference baseline, and then test an atomistic representation whose units, periodicity, element handling, and symmetries are independently checked. HDC may contribute metadata retrieval, grouping, or candidate-diversity logic if those semantics are supported; any HDC atomistic encoder is a new research hypothesis and must earn its role against appropriate baselines.

The OC22 source documents a mapping from system IDs to metadata including `bulk_id`, `bulk_symbols`, `miller_index`, `traj_id`, `slab_sid`, `ads_symbols`, and `nads`. When fields are present, use them to audit related-system grouping and possible train/test leakage. Missing mapping fields must stay missing—not be inferred from unrelated examples.

### Reference-checkpoint eligibility, training exposure, and license

The baseline choice is a separate qualification gate, tracked in [NANO-001C #7465](https://github.com/Luminous-Dynamics/symthaea/issues/7465). Do not choose a reference model merely because it is recent or reports high benchmark accuracy.

Current upstream sources show:
- FairChem's pretrained-model page recommends UMA for many new atomistic applications.
- The current UMA task matrix lists `oc22` as available only in UMA 1.2, at PBE+U/VASP with spin polarization, for oxide catalysis/supports and energy conversion/storage.
- The current FairChem repository/model release notes say UMA 1.2 expanded training coverage to include OC22 and OC25. Thus, UMA 1.2 is task-matched for OC22-style oxide calculations in principle, but it has OC22 training exposure. Exact checkpoint and split exposure must still be established before interpreting it as an independent benchmark comparator.

References:
- [FairChem pretrained models and legacy OC22 S2EF checkpoints](https://github.com/facebookresearch/fairchem/blob/3801dac0cc0458a2f8121259a2ce8b23d4dcc5a1/docs/catalysts/models.md)
- [FairChem UMA task/domain matrix](https://github.com/facebookresearch/fairchem/blob/c22fb1bedb2b4ca453376749f42999be1589856d/docs/core/uma.md)
- [FairChem UMA changelog: OC22 added in UMA 1.2](https://github.com/facebookresearch/fairchem/blob/c22fb1bedb2b4ca453376749f42999be1589856d/docs/core/uma_changelog.md)
- [FairChem current README and supported tasks](https://github.com/facebookresearch/fairchem)
- [UMA checkpoint model card](https://huggingface.co/facebook/UMA)
- [UMA license text and acceptable-use policy](https://huggingface.co/facebook/UMA/blob/9f578aa/LICENSE)

Treat this as a training-lineage question requiring resolution for the exact checkpoint/release and included data splits, not as proof of either held-out independence or contamination of a specific test partition. For every candidate checkpoint, bind its exact hash, config, software revision, selected task/head, DFT level, model paper, training datasets/splits, license revision, and inference recipe. Specifically document exposure to any target, source trajectory, parent slab/bulk, train/validation/test records, and derived labels in the evaluation corpus.

Task match and training exposure are separate gates. The current FairChem pretrained-model page lists legacy OC22 S2EF-Total checkpoints but not a direct OC22 IS2RE checkpoint in that page's released-checkpoint table. S2EF predicts energy/forces for a supplied structure; that is not the same target as relaxed total energy from the initial structure. An OC22 S2EF potential can be used in a relaxation-based comparator only if the entire optimizer/constraints/convergence/failure/units/cost pipeline is frozen and evaluated end-to-end. Report that result as a relaxation-based baseline, not as a direct single-pass IS2RE predictor. UMA 1.2 supports an `oc22` task and is a plausible task/DFT-matched energy/force potential, but an UMA relaxation pipeline still needs a frozen protocol and checkpoint audit.

**Do not use an UMA 1.2 checkpoint with OC22 training exposure as the primary selector/predictor for an OC22 label-efficiency claim.** If the model has already trained on candidate structures or labels that the experiment is supposed to hide, hiding those labels only from the new training loop does not create a genuinely label-blind baseline. UMA may be an external, non-independent task-matched comparator for a strictly separate official held-out evaluation only after its exact training split and benchmark partition independence are established. Label-efficiency experiments should train the same predictor family from scratch for each acquisition arm, or use a pretraining corpus whose non-overlap with the benchmark pool is demonstrable.

#### Current candidate-model disposition (source-document audit)

| Candidate | Appropriate role to evaluate | Current blocker / caveat |
|---|---|---|
| UMA 1.2 with `task_name="oc22"` (e.g. UMA-S 1.2.1) | Task-matched energy/force potential; possible relaxation-based comparator | UMA 1.2 includes OC22 training exposure; exact checkpoint, included split/record lineage, license, inference recipe, and memory/runtime on the target GPU must be frozen. Not a label-blind acquisition oracle if the query pool has already been seen. |
| Legacy `GemNet-OC-S2EFS-OC22` | Energy/force potential for a fully specified relaxation pipeline | The current legacy model table reports validation energy MAE 0.563 eV and force MAE 0.030 eV/Å on its S2EF task; these numbers are not an IS2RE result. Its OC22 training exposure also means exact official held-out partition separation must be checked. |
| Direct OC22 IS2RE-Total checkpoint | Preferred direct baseline if a verifiable, appropriately licensed checkpoint is located | The current published legacy model table lists OC22 S2EF-Total checkpoints but does not list a direct OC22 IS2RE checkpoint. Do not infer one exists merely because the dataset supports IS2RE. |
| Same predictor family trained from scratch per acquisition arm | Preferred primary comparison for label-efficiency | Must identify a compatible implementation/task config, fit a bounded model/training recipe to the acquired-label budget, and measure feasibility on the intended hardware. Do not claim this is complete until an exact run reproduces. |

This disposition uses the source documentation only. No checkpoint was downloaded, loaded, or benchmarked in this workstream. The UMA 1.2 `oc22` task is based on PBE+U/VASP with spin polarization; model task selection does not make the underlying total energy interchangeable with energies from another DFT level or task.

A pretrained model may be an external comparator, a frozen feature extractor, a model fine-tuned on a declared acquired-label partition, or an active-learning oracle; these roles are not interchangeable. If training data exposure to a benchmark's evaluation records or acquisition pool is unknown, mark it `exposure_unknown` and do not use it as the primary proof of independent accuracy or label-efficiency improvement. A model trained on hidden labels cannot be described as an independent predictor of those records.

The visible UMA license revision `9f578aa` identifies the FAIR Chemistry License v1, last updated 13 October 2026, and incorporates the FAIR Chemistry Acceptable Use Policy. The policy lists use related to the operation of critical infrastructure, transportation technologies, and heavy machinery among prohibited uses. That clause may materially affect future applications in rail, construction, and industrial operations. The intended use needs explicit review; do not deploy UMA-derived outputs for an operational use potentially covered by that restriction without written rights-holder clarification or an alternative license. The publisher also reserves the ability to modify the license as described in its text, so archive the exact license/version with any future run. This note is a technical licensing gate, not legal advice.

Keep code, dataset, checkpoint, and output/use terms separate. OC22's CC BY 4.0 dataset terms and the MIT license for the data-reader software do not grant rights under the UMA weights license. If no suitably documented, permitted, compatible baseline is available, continue only with data-admission and integrity work or a transparent simple baseline until model eligibility is resolved.

### Scientific engines — atomistic calculations and reference labels

Use established, version-pinned external tools and datasets for calculations that need chemistry-specific physics. Depending on the exact question, the eventual toolchain may involve density-functional theory, established machine-learned interatomic potentials, molecular dynamics, or a materials data service. The selected method must document its approximation, parameterization, supported chemistry, and applicability domain.

Possible tooling to evaluate—not dependencies chosen by this proposal:
- [Open Catalyst Project / OC22](https://opencatalystproject.org/) is the frozen first data-admission target for relaxed total-energy label-efficiency; [OC20](https://github.com/facebookresearch/fairchem/blob/main/docs/catalysts/datasets/oc20.md) is a later, separate adsorption-energy benchmark. The [OC22 dataset paper](https://doi.org/10.1021/acscatal.2c05426) reports 62,331 DFT relaxations and approximately 9.85 million single-point calculations; these are calculations/relaxations, not 9.85 million independent candidates.
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

## 4. First proving domain: OC22 oxide-electrocatalyst energy benchmark

### Why this dataset first

Select the published Open Catalyst 2022 (OC22) **Initial Structure to Relaxed Total Energy (IS2RE-Total)** task for the first admission and label-efficiency qualification. It is a tractable way to exercise the data/provenance/split pipeline before paying the much larger cost of broader atomistic benchmarks.

The FairChem OC22 dataset documentation snapshot currently identified at commit `57c71322d5a5be0a0fcdcf05592e1699b120dc79` reports:
- approximately 62,000 oxide systems;
- IS2RE-Total archive containing train, validation, and test sets with in-distribution and out-of-distribution splits;
- compressed archive of approximately 109 MB and expanded size of approximately 424 MB;
- upstream MD5 for the IS2RE-Total archive: `b35dc24e99ef3aeaee6c5c949903de94`;
- metadata mapping artifact upstream MD5: `13dc06c6510346d8a7f614d5b26c8ffa`;
- dataset license stated as CC BY 4.0. The FairChem code/tooling license is separate (MIT in the dataset repository).

Sources and exact artifact paths are frozen for the admission checklist in [NANO-001A #7455](https://github.com/Luminous-Dynamics/symthaea/issues/7455):
- [OC22 dataset docs at pinned source commit](https://github.com/facebookresearch/fairchem/blob/57c71322d5a5be0a0fcdcf05592e1699b120dc79/docs/catalysts/datasets/oc22.md)
- [IS2RE-Total archive](https://dl.fbaipublicfiles.com/opencatalystproject/data/oc22/is2res_total_train_val_test_lmdbs.tar.gz)
- [OC22 metadata mapping](https://dl.fbaipublicfiles.com/opencatalystproject/data/oc22/oc22_metadata.pkl)
- [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
- [OC22 dataset repository](https://github.com/Open-Catalyst-Project/Open-Catalyst-Dataset)

**Admission is not yet complete.** The source page and advertised checksums have been researched, but the data files have not been downloaded or independently checked in this workstream. The source document, access date, exact byte count, SHA-256, split counts, and rights/attribution record must be frozen during admission. Upstream MD5 is only a compatibility/integrity check—not a collision-resistant artifact identity.

### Critical claim boundary

OC22 IS2RE-Total supplies DFT-derived total energies for initial structures and their relaxed outcomes. **Total energy is not catalyst activity, adsorption energy, reaction free energy, selectivity, durability, or synthesizeability.** Energies across different compositions/system sizes are not a valid generic catalyst ranking. This first benchmark qualifies data integrity and simulated label-efficient *total-energy prediction* only. A later OC20 adsorption-energy task, and ultimately application-specific validation, are required before making adsorption-energy or catalyst-performance claims.

### Research question

After admission, test whether a Symthaea-assisted acquisition policy improves **label efficiency for predicting relaxed OC22 total energies** under a fixed label budget, compared with random selection and a declared descriptor-based acquisition baseline.

This is retrospective label masking: the dataset's DFT labels already exist, but the selector/predictor must behave as though only the acquired training labels are available. It can test selection strategy under the dataset's assumptions. It does not demonstrate real DFT-cost savings, synthesis efficiency, or wet-lab performance.

### Scope decision

- Use the published **IS2RE-Total** task, not the 20 GB S2EF-Total archive or optional 34 GB trajectory archive, for the first pipeline qualification.
- Preserve the official train/validation/test and ID/OOD partitions. Never reinterpret the held-out test data as an acquisition pool.
- Keep official-split results distinct from any later chemistry-family or material-family holdout.
- Use a pinned, isolated FairChem reader/data-conversion tool only if needed to read the upstream LMDB; pin its source revision, dependencies, license, and command. It is not a production Symthaea dependency.
- Export only a rights-reviewed compact manifest and machine-readable records needed by the Rust-first benchmark/verifier; do not commit raw OC22 data or re-host the upstream archive.
- Defer OC20 because its documented full IS2RE archive is substantially larger (about 8.1 GB compressed / 97 GB expanded) and its adsorption-energy target is a different scientific task. Treat OC20 as a later domain-specific benchmark, not a substitute silently swapped into this experiment.

### Admission checklist

The exact per-system data contract is specified in [NANOMATERIALS_OC22_PAIR_MANIFEST.schema.json](./NANOMATERIALS_OC22_PAIR_MANIFEST.schema.json). It separates artifact identity, provider-reported versus observed counts, split reconciliation, initial-input/relaxed-target pairs, and independent-review status. It is a documentation-level schema, not a data admission result or new production runtime type.

Execute [NANO-001A #7455](https://github.com/Luminous-Dynamics/symthaea/issues/7455) before implementing model comparison:
1. Retrieve the pinned source documentation and record its commit and access date.
2. Download the archive and mapping to temporary research storage; verify upstream MD5, compute SHA-256 and exact byte size, and preserve those results in the manifest.
3. Inspect archive layout and task/split metadata with the upstream reader; independently count examples and labels per official split.
4. Verify finite values, stated units, structure/label alignment, duplicate IDs, related trajectories/materials, missing/invalid structures, and official ID/OOD semantics.
5. Freeze source rights, attribution/citation, dataset artifact identity, software/runtime identity, transformations, exclusion list, and output manifest schema.
6. Stop on unresolved rights, checksum mismatch, unknown units, ambiguous split membership, hidden-label exposure, or missing structural lineage. Do not produce a benchmark accuracy claim if any gate fails.

### Hosted conversion candidate discovered during admission research (2026-10-15)

A second possible access route is the ColabFit Exchange collection of OC22 in Parquet form:
- [OC22 collection](https://huggingface.co/colabfit/collections)
- [OC22 IS2RE train](https://huggingface.co/datasets/colabfit/OC22-IS2RE-Train)
- [OC22 IS2RE validation in-domain](https://huggingface.co/datasets/colabfit/OC22-IS2RE-Validation-in-domain)
- [OC22 IS2RE validation out-of-domain](https://huggingface.co/datasets/colabfit/OC22-IS2RE-Validation-out-of-domain)

The visible cards identify CC BY 4.0, energy/force properties, a Parquet-based ColabFit schema, and attribute the original data to FairChem. The train dataset's surfaced root revision is [`fb892188938913025086056f6e29461fb49bcaf6`](https://huggingface.co/datasets/colabfit/OC22-IS2RE-Train/commit/fb892188938913025086056f6e29461fb49bcaf6); its large Parquet shard upload is recorded at [`787bf4654488ae585cc98f19a74c3401706fa832`](https://huggingface.co/datasets/colabfit/OC22-IS2RE-Train/commit/787bf4654488ae585cc98f19a74c3401706fa832). At the surfaced train snapshot, the dataset card reports 7,861,269 unique molecular configurations, while the auto-converted viewer estimates 7,236,212 rows (showing only the first 718,431 rows) and 30.6 GB across its Parquet representation. The out-of-domain validation card reports 521,827 unique molecular configurations, the viewer estimates 521,308 rows, and the page reports 2.44 GB for auto-converted Parquet; the repository's source-data folder view reports 2.04 GB. The 2.04 GB versus 2.44 GB figures refer to different representations (source ColabFit files versus auto-converted Parquet), not a verified byte-count discrepancy for one identical file. The card/viewer configuration-count differences still require reconciliation. These hosted datasets are much larger than the official FairChem combined IS2RE-Total/IS2RS archive (~109 MB compressed / ~424 MB expanded, ~62K oxide systems). Visible records carry both IS2RE and S2EF membership labels. This makes a configuration-/trajectory-level representation plausible and raises a material target-contract question: a row containing one configuration and its energy/forces is not automatically the official IS2RE pair of an initial structure with its relaxed-total-energy target. This is a hypothesis from source metadata and scale, not yet a verified row-level conclusion.

Treat ColabFit as a separate potential source/conversion path, not a substitute admitted by its dataset name or license badge. Before using it for this benchmark:
1. Pin exact dataset revision and file hashes, record uploader/processing provenance, and inspect the actual Parquet files.
2. Reconcile rows and identifiers back to OC22 `sid`, source LMDB membership, task labels, units, and official train / validation-ID / validation-OOD membership.
3. Establish whether an official test partition is provided through this path. The currently surfaced collection lists train and the two validation datasets, not a test dataset; do not infer held-out test availability.
4. Detect duplicated structures and trajectory-family overlap; count unique systems separately from configuration rows.
5. Cross-check a deterministic sample against the pinned FairChem reader/source when those original records become accessible.
6. If an exact mapping cannot be established, keep this route as an exploratory dataset with its own contract and do not combine its metrics with official OC22 split results.

No ColabFit rows or labels have been downloaded or independently verified in this workstream. This finding changes the retrieval options, not the admission state.

### ColabFit schema and lineage evidence (source audit, 2026-10-15)

A follow-up review of the visible ColabFit Hugging Face Dataset Viewer makes this alternate route more concrete, but also creates explicit lineage work:

- The viewer schema exposes `configuration_metadata`, `configuration_metadata_id`, `configuration_labels`, `configuration_dataset_ids`, `configuration_id`, `configuration_hash`, and `structure_hash`.
- The train dataset's small aggregate `ds.parquet` metadata file is shown by Hugging Face as 18.5 kB with SHA-256 `843156208bc225e1c7c77690c36975b45c6f0f85cf63600b7ca1dff850b9b8ef` and Xet hash `5bcd29fdf37dc0809be1f8dbc50e6ce49cf9cbec51dc2ffe0784bbf22cb0aa2c` at the visible `fb892188938913025086056f6e29461fb49bcaf6` revision. This is **provider-reported metadata**, not a locally downloaded or independently recomputed hash; `ds.parquet` is not the multi-gigabyte configuration dataset.
- The OOD validation `ds.parquet` file was updated at immutable dataset revision [`bac2ddf46c6b7399a8a796026fe711094c8d7027`](https://huggingface.co/datasets/colabfit/OC22-IS2RE-Validation-out-of-domain/commit/bac2ddf46c6b7399a8a796026fe711094c8d7027); its Git-LFS pointer reports SHA-256 `3621b4c1261e81148afb280a66a89fd47327838e9e275254605eb4eb8c934faf` and 18,677 bytes. That is a metadata-file digest, not the digest of the OOD configuration data.
- Example out-of-domain validation rows expose configuration metadata fields including `lmdb-system-id`, `traj-id`, and `num-adsorbates`. The examples use the original OC22 system identifier/trajectory lineage, which gives us a candidate join key to verify against the upstream OC22 mapping.
- Visible OOD records include configuration labels such as `is2re_val_ood_182` and `s2ef_val_ood_182`, and dataset-ID lists spanning both IS2RE and S2EF datasets. This confirms that at least some configurations are intentionally represented in more than one task-specific collection. If multiple collections are ever combined, cross-task membership and duplicate structures must be reconciled instead of counted as independent candidates.
- The ColabFit OC22 family page lists six datasets and 18.0M total configurations across the family; its IS2RE-train member reports 7,861,269 configurations and its S2EF-train member 8,356,688. The family total is a configuration-level aggregate across distinct dataset memberships, not evidence for 18 million independent OC22 systems. The shared IS2RE/S2EF labels on visible rows show why membership-level deduplication and source-SID counts need separate fields.
- The ColabFit Exchange catalog and Hugging Face train card both report 7,861,269 configurations for IS2RE train (DOI `10.60732/722bcab6)); the Hugging Face auto-converted viewer, which is explicitly limited to its first 5 GB, shows 718,431 rows and estimates 7,236,212 total rows. Treat the viewer number as a preview-derived estimate, not a source-level unique-configuration count. ColabFit's own record page reports three distinct download representations: original files about 9.4 GB, Parquet about 30.6 GB, XYZ about 22.7 GB. The OOD validation catalog reports 520,744 configurations (DOI `10.60732/15fa94f2`), its Hugging Face card reports 521,827, and its auto-converted viewer estimates 521,308 rows. The ColabFit OOD record page lists original files at about 2.04 GB and Parquet at about 2.44 GB. These are different representations and/or catalog/viewer count semantics, not a single byte-count discrepancy. None of the viewer estimates or provider page counts replaces independently computed counts from frozen files.

Sources inspected:
- [ColabFit OC22 collection](https://huggingface.co/collections/colabfit/oc22-open-catalyst-2022-colabfit)
- [OC22 IS2RE train card and viewer](https://huggingface.co/datasets/colabfit/OC22-IS2RE-Train)
- [OC22 IS2RE OOD validation schema and examples](https://huggingface.co/datasets/colabfit/OC22-IS2RE-Validation-out-of-domain)
- [Canonical FairChem OC22 mapping and task definition](https://github.com/facebookresearch/fairchem/blob/main/docs/catalysts/datasets/oc22.md)

Preregister these mapping outcomes before any ColabFit-based metric:
1. Exact column type/schema and revision for every file/shard; exact byte sizes and SHA-256.
2. Whether `lmdb-system-id` agrees with official `sid` for a deterministic comparison sample and then for every record.
3. Number of configurations, unique `lmdb-system-id` values, unique `traj-id` values, and duplicated structure/configuration hashes—reported separately.
4. `configuration_labels` and `configuration_dataset_ids` intersections between train, validation-ID, validation-OOD and the parallel S2EF collections.
5. Target property identity/units and evidence that each IS2RE row stores the initial-structure input and relaxed-total-energy target expected by the official task.
6. The absence/presence of official test membership. The surfaced collection lists six train/validation datasets (three IS2RE and three S2EF) and no IS2RE test dataset. Do not invent or substitute a test partition.

No raw Parquet records have been downloaded, no sample crosswalk has been run, and the count/size differences remain unresolved. This is additional source-level evidence, not data admission.

### ColabFit configuration rows versus official IS2RE sample pairs

The ColabFit schema documentation defines each `co/*.parquet` row as a single configuration: one atomic structure plus its associated calculated properties. Its columns include structure fields (`positions`, `atomic_numbers`, `cell`, `pbc`) and individual properties such as `energy` (eV) and `atomic_forces` (eV/Å). ColabFit describes these configurations as data points in an interatomic-potential dataset. See the [configuration schema](https://materials.colabfit.org/docs/configuration_schema) and [Parquet usage guide](https://materials.colabfit.org/docs/how_to_use_parquet).

By contrast, the [official OC22 task definition](https://github.com/facebookresearch/fairchem/blob/main/docs/catalysts/datasets/oc22.md) says the IS2RE-Total/IS2RS LMDB records contain the initial input structure and output relaxed structure/energy. These are different data contracts. A row in ColabFit with an energy and forces describes the calculated structure represented by that row; it is not, by itself, proof of the relaxed-energy target paired to an initial input for IS2RE. A label such as `is2re_train_*` or `is2re_val_ood_*` only marks dataset membership and does not establish which trajectory configuration is the initial structure or which is the relaxed endpoint.

For any ColabFit conversion to be considered, construct and independently verify an explicit pair manifest:
1. For each official `lmdb-system-id` / OC22 `sid`, identify the initial-structure configuration and the exact relaxed-total-energy target using source lineage and trajectory endpoint evidence.
2. Do not choose a first/last row from file order, `configuration_last_modified`, hash ordering, or label text; none of those fields alone has been shown to encode trajectory order.
3. Confirm initial and target structure/energy units, finite-value checks, atom ordering, composition, cell and periodic-boundary alignment, plus any cell/atom-count changes permitted by the official task.
4. Require one unambiguous pair per eligible official system, report missing/ambiguous/multiple endpoints as explicit exclusions, and reconcile pair counts against the official split manifest.
5. Run a deterministic sample crosswalk against the pinned FairChem loader when the source archive becomes accessible, followed by a complete independent comparison. If the endpoint mapping cannot be proved, ColabFit remains a separate single-configuration dataset and cannot substitute for the official IS2RE benchmark.

This schema-level distinction is now a first-class blocker, not a guess about dataset size. No pair manifest or data rows have been downloaded or verified yet.

### Selector comparison

Keep downstream predictor family, structure representation, training protocol, available label budget, compute budget, and evaluation sets fixed while comparing:
1. deterministic random acquisition over repeated frozen seeds;
2. a simple declared descriptor/diversity selector;
3. Symthaea-assisted acquisition with its HDC/feature construction, scoring policy, diversity/abstention behavior, and resource ceiling frozen before the held-out run.

Do not assume HDC similarity is a physical energy or chemically meaningful geometry by default. The structure encoder and its invariances must be defined and tested; if that encoder is not ready, the first deliverable is data admission plus simple-baseline reproduction, not a manufactured Symthaea result. A reference atomistic potential or pretrained model may be used only as an explicitly versioned downstream predictor shared across acquisition arms.

Required controls: label shuffling, random acquisition, candidate-count reconciliation, train/validation/test isolation, structure/trajectory grouping and an independent metric checker. If a shuffled-label run appears predictive or the selector can see hidden labels, invalidate the result.

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
2. Open Catalyst Project, project background and data portals: https://opencatalystproject.org/
3. FairChem OC22 dataset documentation at source commit `57c71322d5a5be0a0fcdcf05592e1699b120dc79`: https://github.com/facebookresearch/fairchem/blob/57c71322d5a5be0a0fcdcf05592e1699b120dc79/docs/catalysts/datasets/oc22.md
4. Open Catalyst Dataset repository, dataset workflow and separate tooling license: https://github.com/Open-Catalyst-Project/Open-Catalyst-Dataset
5. OC22 data license as reported by upstream: Creative Commons Attribution 4.0: https://creativecommons.org/licenses/by/4.0/
6. Tran et al. (2023), *The Open Catalyst 2022 (OC22) Dataset and Challenges for Oxide Electrocatalysts*, ACS Catalysis 13(5), 3066–3084: https://doi.org/10.1021/acscatal.2c05426
6. FairChem OC20 documentation for the later adsorption-energy benchmark: https://github.com/facebookresearch/fairchem/blob/main/docs/catalysts/datasets/oc20.md
7. NIST, *Nanotechnology Measurement Protocols* (updated 13 July 2026): https://www.nist.gov/mml/nanotechnology-measurement-protocols
8. Ren et al., *Autonomous experiments using active learning and AI*, Nature Reviews Materials (2023): https://www.nature.com/articles/s41578-023-00588-4
9. LAMMPS stable documentation (30 September 2026): https://docs.lammps.org/stable/
10. Atomic Simulation Environment documentation: https://docs.ase-lib.org/
