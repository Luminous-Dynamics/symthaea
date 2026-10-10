# Protein Dynamics Dataset Inventory — 2026-10-10

**Status:** initial literature/source inventory; not a frozen test manifest. No datasets have been downloaded, checksummed, or benchmarked by this inventory. The exact revision of this document and any future asset manifest must be recorded before experiments.

**Related protocol:** [PROTEIN_DYNAMICS_BENCHMARK_CONSTITUTION.md](PROTEIN_DYNAMICS_BENCHMARK_CONSTITUTION.md)  
**Umbrella:** [BIO-MILLENNIUM-000, issue #6271](https://github.com/Luminous-Dynamics/symthaea/issues/6271)

## Decision summary

Use two deliberately separate source lanes:

1. **Algorithm/controller pilot:** a small, deterministic, family-aware subset from mdCATH, because one published data-generation pipeline supplies repeated trajectories and condition labels suitable for replayable state/trajectory tasks. This only tests performance against simulation-generated reference trajectories.
2. **Experimental-evidence lane:** a small, curated set of experimental entries from the Protein Ensemble Database (PED), selected only after primary-observable and provenance audit. This is required before discussing agreement with experimental conformational evidence.

Do not begin by downloading all mdCATH or treating every PED ensemble as independent physical ground truth. Neither is justified at this stage.

## 1. mdCATH — candidate for the first controller pilot

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

## 2. Protein Ensemble Database (PED) — candidate for experimental compatibility

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

## 3. Suggested data-custody manifest

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

## 4. Immediate execution queue (independent of CI)

1. Implement a **read-only metadata inventory** first; do not download the full multi-terabyte dataset.
2. Confirm per-record permissions and retrievable assets. Record unavailable assets and exclusions rather than skipping them silently.
3. Freeze a deterministic mdCATH pilot-selection rule and target count before inspecting the selected trajectories.
4. Run a fixed/random selection baseline and verify that the manifest, output hashes, budget counters, and terminal census are replayable.
5. Add the Symthaea/HDC controller only after the baseline instrumentation works. Use ablations to separate representation benefits from the benefits of having a larger budget or more candidate calls.
6. Treat PED experimental-compatibility evaluation as a separate gate after assay/provenance audit; do not let the easier MD replay lane stand in for physical validation.

**Current result:** source inventory only. No target set has been frozen, no asset has been downloaded or verified locally, no controller has been run, and no benchmark result is claimed.
