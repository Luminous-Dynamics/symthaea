# Protein Dynamics Source Inventory — Refresh 2026-10-11

**Status:** source and benchmark discovery only. This is not a frozen target manifest and no model evaluation has been performed.

## Main change from the previous inventory

Before implementing a custom experimental-observable scoring pipeline, evaluate whether **PeptoneBench** can serve as the external evaluation harness. The project documents datasets spanning ordered and disordered proteins, ensemble generation inputs, forward models for NMR chemical shifts and SAXS, and scoring/reweighting outputs. It provides benchmark results with and without refinement. Source: https://github.com/PeptoneLtd/peptonebench and its cited preprint https://doi.org/10.1101/2025.10.18.680935.

This is a benchmark-reuse candidate, not an endorsement or a qualification pass. The source has now been pinned to upstream commit [`f161b5220a52ebd1a715c91a68beef125101608e`](https://github.com/PeptoneLtd/peptonebench/commit/f161b5220a52ebd1a715c91a68beef125101608e); see the [pinned source audit](PEPTONEBENCH_PINNED_SOURCE_AUDIT_2026-10-11.md). The audit found important execution-contract issues: post-reweighting RMSE uses the observable being fit; ESS defaults to absolute 10; the dataset helper performs no checksum verification; and caught per-directory exceptions/NaN metric rows require fail-closed verification. No reference case has been run. The repository identifies PeptoneBench as Apache-2.0, requires Python >=3.11, and declares scientific Python / MDTraj dependencies. The preprint reports competing interests involving multiple authors employed by or holding shares in Peptone Ltd. Record this provenance and seek an independent evaluator or independently reproduced subset before making headline claims.

## Recommended execution order

1. **PeptoneBench harness audit first (source pin completed; runtime audit pending).** Pin: `f161b5220a52ebd1a715c91a68beef125101608e`. Retain source/license/blob identities; freeze a resolved dependency environment and forward-model image digests. Read the metric implementation, dataset manifests, and refinement path. Determine which targets/observables were used for model development, training, refinement, or reporting. Do not allow the evaluator to tune on its held-out scoring data. See `PEPTONEBENCH_PINNED_SOURCE_AUDIT_2026-10-11.md`.
2. **Reproduce one reference case.** Use the project’s documented reference artifacts, pin all versions, and check whether the published score and output hashes can be recreated. A successful install or completed command is not a benchmark pass.
3. **Define the Symthaea adapter contract.** Symthaea should nominate candidates or allocate computation; a pinned specialist pipeline produces the structural ensemble; a separately versioned evaluator scores it. Record model calls and costs. Symthaea must not see hidden evaluator values or tune after the held-out result is opened.
4. **Separate generation from reweighting.** Report raw ensemble metrics and any reweighted/refined metrics separately. Reweighting to the same experimental observable later used for evaluation is not independent validation.
5. **Ablate the claimed advantage.** Compare fixed/random allocation, conventional allocation, Symthaea without HDC, and Symthaea with HDC/adaptivity. Hold upstream model access and the declared compute budget equal. Report the complete target census, not only successful examples.
6. **Keep mdCATH as a different lane.** Its trajectories are force-field simulation outputs. A replay/selection pilot tests instrumentation and algorithmic behavior against simulation-derived states, not experimental accuracy or real-time MD savings. The full corpus is multi-terabyte; use a deterministic, stratified metadata-based subset and do not download the full corpus for the first pilot.

## Execution record

The machine-readable [run-record schema](PROTEIN_DYNAMICS_RUN_RECORD_SCHEMA_v1.json) is now checked by [the executable validator](../../scripts/research/validate_protein_dynamics_run.py). It compares the target census with the exact frozen target-manifest hash, rejects duplicate/unregistered target IDs and non-finite metrics, checks observable-role labels, requires failure codes, reconciles target conformer counts with resource totals, and verifies local output-file hashes. Its [16 adversarial tests](../../scripts/research/test_validate_protein_dynamics_run.py) cover malformed metadata and integrity failures; see the [usage guide](PROTEIN_DYNAMICS_RUN_RECORD_VALIDATOR.md).

The test suite uses the production schema when run inside the repository. The available local sandbox lacked a repository checkout; the 16 test cases were exercised against a minimal-schema fixture for semantic validation, not against the production schema. No dataset asset, target set, actual run record, or benchmark result has been created.

## Source facts and limits

### PeptoneBench

The repository README describes the following flow: obtain experimental data; generate ensembles (at least 100 configurations per sequence as described); calculate forward-model observables; then run the benchmark to report scores and refinement weights. It contains separate chemical-shift and SAXS datasets and an integrative dataset. Treat its results with/without refinement as distinct result classes.

Potential limitation to inspect: reweighting/refinement can improve observable agreement by construction, so raw predictions and post-refinement results must never be collapsed into a single performance number. Dataset and publication overlap with any proposed Symthaea training or tuning sources must be audited.

### mdCATH

The original dataset paper describes 5,398 CATH domains, five replicas at five temperatures (320–450 K), 1 ns frame sampling, and more than 62 ms aggregate simulation time. Individual domain files are available through the dataset repository; the full corpus is approximately 3.61 TB according to the current repository listing. Source: https://doi.org/10.1038/s41597-024-04140-z and https://huggingface.co/datasets/compsciencelab/mdCATH/tree/main/data.

A 1 ns frame interval cannot resolve fast side-chain/solvent motion. Elevated-temperature simulations must not be described as physiological trajectories without qualification. A replay benchmark counts selected frames/windows, not saved GPU-hours. Actual compute savings require a prospective, equal-budget simulation experiment.

### Protein Ensemble Database (PED)

The current PED home page identifies it as an open-access repository of experimentally derived and computationally predicted ensembles and states that the record type is marked. The public statistics panel was updated 2026-10-09 and showed 9,528 proteins, 28,954 entries, 29,120 ensembles, and 3,226,106 conformers. These are dynamic catalog statistics, not benchmark denominators. Source: https://proteinensemble.org/.

Use PED for discovery and provenance, but do not build a parallel custom score implementation before reviewing PeptoneBench. Keep experimental and predicted entries separate; confirm per-record licensing, source observables, conditions, and whether the candidate ensemble was fit/reweighted to the same observables being scored.

## Initial target-selection contract

No target IDs, split assignments, or inclusion decisions are frozen by this document. Before evaluation:

- Export the metadata list without opening target outcomes; retain the query and retrieval timestamp.
- Assign each candidate an evidence class (experiment, simulation, or prediction), publication/experiment group, construct/condition identity, sequence family, model/data exposure notes, and asset/license status.
- Select by a deterministic, preregistered rule over declared metadata, not by performance or visually interesting trajectories.
- Keep all replicas, frames, conditions, repeated measurements, related constructs, and the same experimental lineage in one partition.
- Keep a separate evaluator-only manifest for reference observables and hashes. The model-facing manifest must not expose hidden scoring values.
- Record exclusions and missing assets with reasons; do not count missing data as success or failure.
- Version the manifest and hash all downloaded/derived assets. A hash binds identity, not scientific truth.

## Current disposition

**Useful work completed:** external source discovery, exact upstream revision pin, static scoring/failure/reproducibility audit, and a reproduction gate. The next highest-value action is metadata-only inspection of Zenodo record 17306061 for `Predictions.tar.gz` and `PeptoneDBs.tar.gz` using the checksum-gated fetch utility, then a byte-capped, checksum-verified acquisition of only what the scorer-only reference case needs. Prefer replay from precomputed predictions before rebuilding the separate, under-pinned forward-model container. No live Zenodo metadata has been retrieved yet. No dataset assets have been downloaded or checksummed, no target set is frozen, no model has been run, and no performance claim is made.