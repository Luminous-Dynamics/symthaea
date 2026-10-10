# Protein Dynamics Dataset Inventory — initial draft 2026-10-10; source audit refreshed 2026-10-11

**Status:** initial literature/source inventory; not a frozen test manifest. No datasets have been downloaded, checksummed, or benchmarked by this inventory. The exact revision of this document and any future asset manifest must be recorded before experiments.

**Related protocol:** [PROTEIN_DYNAMICS_BENCHMARK_CONSTITUTION.md](PROTEIN_DYNAMICS_BENCHMARK_CONSTITUTION.md)  
**Umbrella:** [BIO-MILLENNIUM-000, issue #6271](https://github.com/Luminous-Dynamics/symthaea/issues/6271)

## Decision summary

Use **one existing experimental-ensemble benchmark framework plus two evidence-separated data lanes**:

1. **Benchmark reuse:** audit PeptoneBench before implementing custom experimental-observable scoring. It already evaluates predicted ensembles against chemical-shift and SAXS observables, and reports pre-/post-refinement results. This is a starting point, not automatic qualification: pin its revision and dependencies, record its preprint/provenance, reproduce a reference case, and keep refinement results separate from primary held-out scoring.
2. **Algorithm/controller pilot:** a small, deterministic, family-aware subset from mdCATH for replayable state/trajectory tasks. This only tests performance against simulation-generated reference trajectories; it is not experimental validation.
3. **Experimental-evidence lane:** a curated set of experimentally measured observables from PED/BMRB/SASBDB, selected only after source, condition, provenance, and observable-leakage audit.

Do not download all mdCATH or treat every deposited PED ensemble as independent physical ground truth. Neither is justified at this stage.

## 1. PeptoneBench — reuse before rebuilding experimental scoring

**Primary code reference:** https://github.com/PeptoneLtd/peptonebench  
**Associated manuscript:** Invernizzi et al., “Advancing Protein Ensemble Predictions Across the Order–Disorder Continuum,” bioRxiv preprint posted 2025-10-18, DOI https://doi.org/10.1101/2025.10.18.680935.

The public repository documents a working evaluation pipeline for protein ensemble predictions against experimental data spanning structured and disordered proteins. It includes dataset definitions for three PeptoneDB collections, scripts to construct chemical-shift and SAXS datasets from BMRB and SASBDB, forward-model processing, and reports with refinement and without refinement. The repository states that its code is Apache-2.0 licensed; individual input datasets and derived assets retain their own source-specific licenses and terms.

### Process decision

Do **not** write a parallel experimental-observable scoring stack before an ownership/compatibility audit of PeptoneBench. First:

1. Pin a precise source commit/tag, Python/runtime dependencies, container/build inputs, dataset manifest versions, and source checksums. The current `main` branch is mutable and is not an adequate version identifier.
2. Read the published protocol and implementation for how targets, experimental observables, forward models, refinement weights, exclusions, and aggregation are handled.
3. Reproduce one small published/reference target from raw inputs to output CSV and verify all assets and outputs by hash.
4. Treat **unrefined/pre-reweighting score** as the primary prediction-quality result only when the prediction and evaluation inputs are frozen before target observables are exposed.
5. Report **post-reweighting/refinement** results separately as an allowed-observable-conditioned diagnostic. When the same experimental observable is used to reweight an ensemble and then scored, it is not a held-out validation result.
6. Audit target overlaps between the benchmark and any model training/fine-tuning data; an openly available benchmark is not automatically a blind or prospective test set.
7. Record the benchmark's preprint status and the authors' stated commercial affiliations in the research record. This is context for independent reproduction, not by itself a reason to reject the benchmark.

### Important limitation

PeptoneBench provides an experimental-observable evaluation path; it does not automatically test whether Symthaea saves compute. To measure that claim, all compared systems must receive equivalent sequence/context and resource budgets, with predictor calls, candidate counts, refinement calls, CPU/GPU time, and wall time recorded. Symthaea's contribution must be isolated through no-HDC and no-adaptive-selection ablations.

**Current disposition:** strong candidate to reuse after reproduction and split/exposure audit; not yet integrated, run, or qualified by this repository.

## 2. mdCATH — candidate for the first controller pilot

**Primary source:** Mirarchi, Giorgino & De Fabritiis, “mdCATH: A Large-Scale MD Dataset for Data-Driven Computational Biophysics,” Scientific Data 11, 1299 (2024), https://doi.org/10.1038/s41597-024-04140-z.

The paper describes:

- 5,398 CATH domains;
- five replicas at each of five temperatures from 320 K to 450 K;
- 134,950 trajectories at the production cut-off described in the paper;
- coordinates and forces sampled every 1 ns, with a mean trajectory length of roughly 464 ns;
- over 62 ms of accumulated simulation time and a total dataset exceeding 3 TB;
- CHARMM22* force-field simulations, TIP3P water, prepared systems, and CC BY 4.0 licensing for the dataset.

The full dataset is unnecessarily large for a process-validation pilot. Start with individual domain assets and exact source manifests, not a broad data mirror. The paper notes that 1 ns sampling limits resolution of fast motions such as solvent-exposed side-chain rotations. Trajectories also reflect force-field and preparation assumptions; they are not experimental truth.

### Candidate initial task

**Task B0 (replay / instrumentation):** given an already sampled, held-out trajectory, test whether a bounded controller selects frames or trajectory windows that cover a predefined set of simulation-derived states/observables more efficiently than fixed-uniform and deterministic-random selection.

This first stage validates the controller, result capture, budget accounting, and ablations. Because it replays precomputed trajectories, it does **not** prove that real MD wall time or GPU-hours were saved. Label all costs in this stage as sampled-frame/window or evaluation-call proxies.

**Task B1 (online follow-up):** only after B0 is reproducible, test an adaptive sampling policy against a fixed schedule using an explicit simulator, force field, integrator, target set, restart policy, hardware, and equal compute budgets. Measure actual wall time, compute use, and simulation length. Keep results separate from experimental evidence.

### Selection rules before any outcomes are inspected

- Freeze a versioned list of available domains, their CATH class/topology/homology-superfamily labels, lengths, trajectory availability, temperatures, replica counts, and file checksums.
- Stratify by declared metadata (such as CATH class and length band), then use a versioned deterministic selection rule to choose the pilot subset. Do not choose targets because their trajectories look especially interesting or because a model performs well on them.
- Keep all temperature conditions and replicas of a domain within the same target partition. For broader generalization claims, enforce separation at CATH homology-superfamily or a stricter declared group level; do not split individual frames randomly across train and test.
- Select the task and target temperature before reading trajectory outcomes. The 320 K tracks are closer to physiological temperature but may contain few large transitions; the 413 K and 450 K tracks can expose unfolding behavior but must be explicitly described as elevated-temperature simulations. Do not silently combine those interpretations.
- Store the exact HDF5/XTC/PDB/PSF source asset identity, license, checksum, and conversion version. A changed or converted asset is a new derived artifact with its own lineage.

### What this lane can prove

It can test deterministic replay, trajectory-state coverage, adaptive-selection logic, cost/proxy accounting, and the effect of HDC or adaptivity ablations against a disclosed simulation source.

It cannot alone prove native-state accuracy, experimental ensemble fidelity, physical folding kinetics, binding efficacy, or that the existing in-repository quantum-chemistry MD module can simulate proteins.

## 3. Protein Ensemble Database (PED) — candidate for experimental compatibility

**Source:** https://proteinensemble.org/ and API documentation at https://proteinensemble.org/api. The PED site describes an open-access, curated repository with experimental and predicted ensembles, experimental/method metadata, primary-publication links, downloadable ensemble assets, and validation information. Its entries include NMR, SAXS, and smFRET-associated work among other methods.

The following are **candidate records for metadata review only**, not selected benchmark targets:

| PED entry | Public listing description | Initial reason to inspect | Required check before eligibility |
|---|---|---|---|
| PED00001 | pSic1 (residues 1–90), with specified phosphorylation sites; 3 ensembles / 32 conformers shown in the public listing | Curated condition- and PTM-specific ensemble; may expose the value of retaining modification context | Confirm source experiment, exact observables, condition metadata, assets/license, and whether candidate construction already used the same observables that would be scored |
| PED00003 | beta-synuclein (residues 1–134); 575 conformers shown in the public listing | More conformers may support distribution-level rather than single-structure comparisons | Same provenance and observable audit; ensure conformer count is not mistaken for independent experimental replicates |
| PED00004 | p27 Kip1 (residues 25–93); 130 conformers shown in the public listing | Additional candidate for checking assay heterogeneity and ensemble comparability | Same provenance, construct, and observable audit |
| PED00017 | Tau-f isoform; 5 ensembles / 995 conformers shown in the public listing | Multiple deposited ensembles may permit a comparison of methods or conditions | Determine which ensembles/conditions are independent and which measurements were used to generate or validate each ensemble |

Counts and descriptions above reflect public PED browse listings inspected on 2026-10-10 and can change. Requery and preserve the retrieved metadata/response identity at inventory time. They are discovery hints—not proof that the records meet the eventual split, license, or observable eligibility rules.

### Experimental-lane safeguards

- Primary test evidence must be an explicitly identified experimental measurement or observable. A deposited computational prediction remains computational evidence even if it appears in the same database.
- Where an ensemble was generated by fitting to a particular experimental observable, scoring it against the same observable is circular unless the protocol explicitly marks this as an in-sample reconstruction test. Prefer withheld observables or an independent source for evaluation.
- Keep data from the same publication, experiment, construct, condition, and ensemble-generation pipeline grouped. Distinguish independent experiments from separate files, conformers, model restarts, or replica calculations.
- A target without enough documented, condition-matched experimental observables is excluded from the primary evaluation with a recorded reason; it must not be scored as a failure or a success.
- Check the applicable data and asset license for each record before redistributing any derived corpus.

PED is the best candidate lane found so far for experimental-observable compatibility, but individual records remain **unqualified** until this audit is complete.



## Current source snapshot (retrieved 2026-10-11)

The current PED public site reports **9,528 proteins, 28,954 entries, 29,120 ensembles, and 3,226,106 conformers**, with statistics updated 2026-10-09. These are live catalog statistics, not a frozen target manifest. The public browse page confirms the following exact candidate records:

- **PED00001 — pSic1 residues 1–90 with six specified phosphorylation sites:** currently listed as 3 ensembles and 32 conformers, with NMR observables including RDCs, chemical shifts, relaxation/T2, PRE, and SAXS tags.
- **PED00003 — beta-synuclein residues 1–134:** 1 ensemble and 575 conformers shown in the public browse listing.
- **PED00004 — p27 Kip1 residues 25–93:** 1 ensemble and 130 conformers shown in the public browse listing.
- **PED00017 — tau-f isoform:** 5 ensembles and 995 conformers shown; tags include NMR/NOESY/RDC/chemical-shift methods.
- **PED00197 — full-length human SMAD2 wildtype, monomer:** NMR and SAXS are listed, with primary experimental data cross-references BMRB 50738 and SASBDB SASDK29. Its methods description reports fitting/selecting an ensemble against SAXS; therefore, the same SAXS observable must not be treated as held-out validation for an ensemble built or selected against it.

References: [PED current catalog](https://proteinensemble.org/), [browse records](https://proteinensemble.org/browse), [PED help and identifier mapping](https://proteinensemble.org/help), [PED00197 details](https://proteinensemble.org/entries/PED00197/).

This fixes the status of these identifiers as live-listed **candidate records**, not qualified held-out targets. Their actual assets, current metadata payloads, license terms, source data, split group, and observable independence still require per-record verification. No record has been downloaded or admitted to a benchmark.

## 4. Suggested data-custody manifest

Before model evaluation, create a machine-readable manifest with one record per source experiment or simulated trajectory group. Preserve, where available:

- source database and entry ID; access/retrieval date and version;
- publication DOI and dataset provenance;
- sequence, construct, residue range, PTMs/mutations, ligand/cofactor/oligomer context, conditions, and units;
- evidence class: experiment, simulation, or prediction;
- available experimental observables or trajectory labels, with provenance;
- source and derived-file hashes, format, conversion code/version, and license;
- sequence-family/structure-family/experiment group IDs and partition assignment;
- known model/training/template exposure and unresolved exposure risks;
- include/exclude decision, reason, reviewer, and decision timestamp.

Do not place held-out observable values in the model-facing manifest. Keep an evaluator-only answer manifest under separate access and exact identity control. Cryptographic hashes bind file identity but do not prove data truth, independent measurement, or correct interpretation.

## 5. Immediate execution queue (independent of CI)

1. Implement a **read-only metadata inventory** first; do not download the full multi-terabyte dataset.
2. Confirm per-record permissions and retrievable assets. Record unavailable assets and exclusions rather than skipping them silently.
3. Freeze a deterministic mdCATH pilot-selection rule and target count before inspecting the selected trajectories.
4. Run a fixed/random selection baseline and verify that the manifest, output hashes, budget counters, and terminal census are replayable.
5. Add the Symthaea/HDC controller only after the baseline instrumentation works. Use ablations to separate representation benefits from the benefits of having a larger budget or more candidate calls.
6. Treat PED experimental-compatibility evaluation as a separate gate after assay/provenance audit; do not let the easier MD replay lane stand in for physical validation.

**Current result:** source inventory only. No target set has been frozen, no asset has been downloaded or verified locally, no controller has been run, and no benchmark result is claimed.
