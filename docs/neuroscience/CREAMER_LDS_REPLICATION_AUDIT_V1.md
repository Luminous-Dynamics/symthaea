# Creamer LDS reproduction audit V1

**Status:** source-level audit; no biological data, model fitting, or percentage reproduction has been performed.

**Tracking:** [SCI-CONNECTOMICS-032 / #9578](https://github.com/Luminous-Dynamics/symthaea/issues/9578)

**Purpose:** prevent an invalid “reproduction” of the C. elegans connectome-constrained LDS baseline by confusing a smoke-test configuration with the full synaptic-mask experiment, or by collapsing conflicting manuscript abstract values.

## 1. Current audit decision

The comparator is valuable and worth reproducing. The currently accessible sources do not establish an unambiguous mapping between the reported 82% and 92% values. The bioRxiv v4 abstract (posted 18 May 2026) reports 82%; the indexed PubMed/PMC record still displays 92%. Preserve these as separate source-specific claims until archived manuscript PDFs, supplements, code/data revisions and metric definitions explain the discrepancy.

Do not report “92% for v3 / 82% for v4” as a proven version mapping without the archived source artifacts. Do not select one value as the canonical SOTA baseline merely because it is more favorable.

## 2. Exact public code revision inspected

Repository: https://github.com/Nondairy-Creamer/Creamer_LDS_2026

Latest located commit during this source audit: `bba43302d50a4947804d98b01779856e648237cc`.

The README describes:
- an LDS whose dynamical weight matrix W is masked using the anatomical connectome;
- stimulation input weights H, process covariance Q and emission model parameters;
- Randi et al. 2023 activity data downloaded from OSF: https://osf.io/qxhjd/;
- anatomical chemical and gap-junction connectomes combined from four NemaNode files;
- an experimental fit described as using 80 animals and 154 neurons;
- a rough local fit time of 40 hours on a desktop, or 10 hours on 10 CPUs in the README's estimated setup.

The inspected source files, at that exact repository commit, and their Git blob SHA-1 values are recorded in `CREAMER_LDS_REPLICATION_MANIFEST.example.json`. These SHA-1 values identify Git blobs; they are not SHA-256 hashes of downloaded external data.

## 3. Material configuration hazard

The repository's no-argument `main.py` path selects `submission_params/exp_test.yml`.

That file specifies:
- `dynamics_weights: not_synaptic`;
- 10 datasets;
- 10 EM steps;
- an explicit held-out recording list;
- worm-level holdout.

In `ssm_classes.py`, the `synaptic` mask is constructed as positive chemical or electrical adjacency plus the identity/self terms. The `not_synaptic` mask is the complement of chemical and electrical adjacency plus the identity/self terms. The source helper `solve_masked` fits the coefficients where the mask is true and forces coefficients to zero where it is false.

Therefore, the no-argument `exp_test.yml` path is a reduced test configuration with non-synaptic pair weights permitted. It is **not** a reproduction of the stated connectome-constrained synaptic model.

The separate `submission_params/slurm_example.yml` specifies:
- 80 datasets;
- 200 EM steps;
- the `synaptic` dynamics-weight mask;
- one dynamics lag;
- 45 stimulation-input lags;
- a 120-sample start index;
- 40 tasks per node, 60 GB and a 10-hour Slurm example.

This is a candidate full-run configuration, not yet proven to be the exact configuration used to produce the paper's figures. It must be compared with the archived paper and supplements before being promoted as the reproduction configuration.

## 4. Source-level split and eligibility behavior

The inspected `loading_utilities.load_data()` implementation:
1. Recursively locates `funcon_preprocessed_data.pkl` files and sorts their paths.
2. Removes any explicitly named held-out recordings from the initial training pool and moves them to test.
3. For `hold_out='worm'`, computes the remaining test count from the available recordings and `num_data_sets`.
4. Selects held-out indices starting at `int(hold_out_start * len(remaining))`, with wraparound; the default is 0.9.
5. Aligns cells by identity labels across recordings.
6. Computes neuron-presence frequency from the training recordings only and retains cells meeting `neuron_freq`.
7. Applies the training-derived cell eligibility mask to test recordings.
8. Defaults to blacklisting AWCL/AWCR.

For 110 eligible recording folders and an 80-recording training target, the implied target cohort is 80 train / 30 test, subject to the exact sorted source inventory and explicit holdouts. The README's 80/154 summary is not a substitute for a generated split manifest containing the exact recording IDs and cell list.

## 5. What a valid reproduction must pin

The machine-readable contract is in:
- `CREAMER_LDS_REPLICATION_MANIFEST.schema.json`
- `CREAMER_LDS_REPLICATION_MANIFEST.example.json`

Required before a reproduction can be declared:
- exact manuscript PDF/version and supplement bytes;
- exact code commit and all run configuration files;
- raw OSF recording files and every preprocessed recording artifact;
- each of the four anatomical CSV files and SHA-256 digests;
- exact combined chemical/gap matrices, cell order and mask support;
- sorted recording inventory, exact train/test recording IDs and the training-derived eligible neuron list;
- preprocessing, NaN, stimulus, filtering, normalization and sampling settings;
- model initialization, EM steps, random seeds, solver environment and dependency versions;
- STAM calculation, window, target/response definition, held-out pair denominator and diagonal/eligibility mask;
- test-vs-test reproducibility ceiling and model-vs-test comparison;
- exact score calculation and confidence interval method.

The example manifest deliberately records external data as not materialized and all associated SHA-256 values as null. This is correct: no one should invent hashes or claim source bytes were verified when they were not downloaded.

## 6. Minimum reproduction arms

1. Synaptic-mask connectome-constrained LDS.
2. Fully connected LDS.
3. Shuffled-connectome constrained LDS.
4. Direct-only or one-hop ablation if the paper's implementation supports it.
5. Test-vs-test reproducibility ceiling.
6. Explicit negative control: attempt to reproduce using the default `exp_test.yml`; the preflight must reject it as the paper's connectome-constrained experiment.

The primary reported outputs should include separate STAM and correlation metrics, exact eligible pair denominators, missing pairs, confidence intervals and test-vs-test ceiling values. The 82% and 92% claims can only be compared once their exact outcome definition, pair set and ceiling calculation are proven to match.

## 7. Claim ceiling

This audit establishes source-level facts about the public code and documents the version/configuration ambiguity. It does not reproduce either score, validate the biological model, or demonstrate a Symthaea improvement. Until all inputs and outputs are pinned and run, the machine-readable manifest must remain `INCOMPLETE_REQUIRES_ARTIFACTS`.
