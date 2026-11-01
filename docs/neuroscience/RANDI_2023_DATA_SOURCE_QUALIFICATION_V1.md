# Randi et al. 2023 data-source qualification and leakage rules

**Status:** source dossier; raw artifacts not yet materialized or hashed.  
**Tracking:** #9627; benchmark contract PR #9588; implementation of metrics PR #9624.

## 1. Paper-level experimental data

Randi et al. report systematic optogenetic activation and whole-brain calcium imaging in *C. elegans*. The paper states:

- 113 wild-type-background animals;
- recordings up to 40 minutes per animal, with neurons stimulated one by one at approximately 30-second intervals;
- 23,433 measured neuron pairs (reported as 66% of all possible pairs in the head);
- 186 of 188 head neurons represented;
- a conservative functional-connection criterion under which 1,310 pairs (6%) passed q < 0.05;
- a separate equivalence analysis for functionally non-connected pairs, leaving a third insufficient-evidence state;
- calcium response measurements summarized over a 30-second post-stimulus window.

Data availability in the Nature version of record points to OSF DOI: https://doi.org/10.17605/OSF.IO/E2SYT. The raw OSF project was not materialized in this pass; the OSF web page returned access error through the current browser route. No file digest, split membership or trial-level receipt is claimed.

## 2. Derived WormNeuroAtlas / OpenWorm view

The OpenWorm C. elegans Connectome Toolbox page labels its resource as the Randi 2023 dataset and states that functional values are extracted from WormNeuroAtlas `get_signal_propagation_map()` and `get_signal_propagation_q()` with `max_q = 0.05`: https://openworm.org/ConnectomeToolbox/Randi2023_data/.

The page currently describes its displayed functional view as a 181 × 181 matrix with 1,150 non-zero entries. It is therefore a filtered derived summary, not the 23,433 individual pair measurement records and not an animal/session-level trial table.

## 3. Count differences are unresolved provenance questions

The paper-level figures (186 neurons, 1,310 passing pairs) and the derived view figures (181 nodes, 1,150 non-zero entries) differ. Do not silently force them to match or declare either source wrong. Candidate explanations include package/release differences, node mapping or filtering, cell/region subset, self-connection treatment, or different threshold/equivalence semantics. These remain hypotheses until the exact package version, source data and code path are pinned.

Required reconciliation evidence:

1. exact WormNeuroAtlas package version / Git commit used by the derived view;
2. precise input files and hash digests;
3. cell-ID mapping and exclusions from the 188-neuron paper universe to the displayed 181-node universe;
4. whether the 1,150 entries are directed, include self responses, and represent nonzero means after q filtering;
5. whether `q < 0.05` is the only filter or whether additional tests, missing-data rules or view options are active;
6. how the paper's equivalence-tested non-connected pairs and insufficient-evidence pairs map into the derived matrix representation.

## 4. Evaluation leakage policy

| Artifact granularity | Permitted use | Prohibited claim |
|---|---|---|
| Raw per-trial/per-session observations with verified animal/session IDs | Held-out animal/session intervention prediction, after split creation before preprocessing | None until IDs, controls, outcomes and source digests are verified |
| One aggregate per independently identified animal | Animal-held-out evaluation may be possible if the aggregation has not pooled across animals and split IDs are preserved | Session-held-out prediction if sessions were pooled |
| Cross-animal aggregate matrix (including the current OpenWorm-derived view unless source code proves otherwise) | Exploratory summary, graph visualization, or explicitly leakage-audited training prior | Animal/session-held-out test outcome or blind prospective result |
| Anatomical connectome reconstruction | Structural prior, with its own source and uncertainty | Measured functional response or causal outcome |

An aggregate matrix assembled across the full animal cohort can contain information from animals later assigned to a test fold. It must not be used as held-out test ground truth. A public source being downloadable does not make its aggregation scope transparent.

## 5. Required artifact inventory

Keep separate immutable records for:

- raw optogenetic stimulus and trial schedule;
- per-animal/session neural response traces and registration IDs;
- pairwise response kernels / summaries;
- empirical nulls, q-values and equivalence-test results;
- aggregate functional matrix and exact extraction code;
- anatomical connectome versions (White, Witvliet or other) and cell-ID mapping;
- transcriptomic, neuropeptide/receptor and other annotations used for interpretation;
- analysis scripts, package version, preprocessing parameters and source-data tables.

Each record needs source URI / OSF file ID, release or revision, file name, media type, digest, provenance relation, unit of observation, aggregation scope, rights/access notes and intended evaluation role.

## 6. Benchmark manifest enforcement

The intervention run schema now requires three dataset fields:

- `data_level`: raw trial/session, animal aggregate, cross-animal aggregate, derived matrix, metadata-only, or unknown;
- `aggregation_scope`: trial, animal-session, animal, multiple animals, unknown, or not applicable;
- `evaluation_eligibility`: held-out animal/session outcome, held-out animal outcome, leakage-audited training input, exploratory summary only, or not yet verified.

Cross-animal aggregate and derived-summary artifacts are constrained to `exploratory_summary_only`. The example Randi run remains `unknown` / `not_yet_verified` until actual OSF and package artifacts are materialized.

## References

1. Randi et al. (2023), *Nature*: https://doi.org/10.1038/s41586-023-06683-4.
2. Randi et al. source data, OSF DOI: https://doi.org/10.17605/OSF.IO/E2SYT.
3. OpenWorm derived Randi 2023 view: https://openworm.org/ConnectomeToolbox/Randi2023_data/.