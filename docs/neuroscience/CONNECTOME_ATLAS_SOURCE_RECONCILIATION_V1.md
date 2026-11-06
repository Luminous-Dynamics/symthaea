# C. elegans connectome atlas source reconciliation V1

**Tracking:** #10754 — SCI-CONNECTOMICS-095
**Composes:** #10752/#10753 cohort transport; #9587 perturbation modelling; #10603/#10675 query-critical worlds and intervention design.
**Status:** metadata-only source inventory contract. No atlas download, biological reanalysis, or q/q_eq reproduction is claimed.

## 1. Purpose

Prevent an apparently complete scientific benchmark from being built on a partial or mismatched data release. The source archive, paper export, analysis code, event exclusions, animal identities, stimulated traces, null controls and derived q/q_eq tables are separate evidence objects. Each must be inventoried and joined explicitly.

## 2. Canonical sources

| Source | Pinned identity | Current evidence |
|---|---|---|
| Paper | Randi et al., Nature 623, 406–414 (2023), DOI `10.1038/s41586-023-06683-4` | Published methods and Source Data; raw-control completeness must be assessed independently. |
| DANDI | `DANDI:001075`, version `0.240930.1859`, DOI `10.48324/dandi.001075/0.240930.1859` | Official metadata currently declares 223 assets, 4.1 TB, 113 subjects, NWB, Open Access and CC-BY-4.0. The paginated asset manifest must be materialized and hashed before file-level claims. |
| OSF | DOI `10.17605/OSF.IO/E2SYT` | Paper's exported WT/unc-31 data are referenced. Exact storage inventory, file digests and relation to DANDI remain to be materialized. |
| Analysis | `https://github.com/leiferlab/pumpprobe` plus `wormdatamodel`, `wormbrain` and conversion tooling | Public repository is identified; exact commit SHAs and dependency revisions must be pinned before reproduction. |

## 3. Evidence layers

Never collapse these layers:

1. **Archive metadata:** the repository claims a release exists and exposes counts/size/license.
2. **Asset inventory:** exact version-pinned file paths, identifiers, sizes and content checksums.
3. **Materialized source files:** locally or immutably available bytes with independently verified checksums.
4. **Animal/session reconciliation:** each paper, DANDI and OSF record joined by explicit identifiers, dates, sessions and neuron-label lineage.
5. **Analysis inputs:** stimulation events, exclusions, autoresponses, ROI identities, stimulated traces and unstimulated controls.
6. **Derived labels:** paper-source q/q_eq tables versus independently recomputed q/q_eq from their actual trial-level null controls.

An official archive count is not evidence that every paper analysis input is present. A paper Source Data table is not equivalent to the raw control recordings that generated the table.

## 4. q and q_eq reproduction

The paper's functional-connection q values compare stimulated downstream response distributions with empirical no-stimulation controls using response amplitude and temporal second-derivative statistics, combine tests with Fisher's method and control FDR using Storey–Tibshirani q values.

The functional-non-connection q_eq path uses a two-one-sided-test equivalence procedure against the empirical control distribution, within `epsilon = 1.2 sigma`, followed by FDR correction. The equivalence margin is part of the scientific definition, not a cosmetic threshold.

Record separate evidence for stimulated responses, no-stimulation null controls, source exclusions, neuron identity mapping and exact analysis code. If any required null-control or lineage input is unavailable, q/q_eq must remain a paper-provided reference or unavailable result; do not describe it as fully reproduced.

## 5. Independent audit leads are not accepted facts

A recent third-party reproduction audit on DANDI:001075 reports several discrepancies worth testing: 110 of 113 subjects have traces/stimulation tables, three contain imaging files only; some DANDI-derived pair/stimulation denominators differ from the paper; 950 export events are flagged as excluded; and q/q_eq null controls were not located in the public DANDI/OSF materials inspected by that auditor. The page also reports only partial reproduction of some per-pair results.

These are **unverified leads**, not paper-authoritative statements. The source-reconciliation tool must retain their URL and exact claim while refusing to promote them to verified facts until checked against primary archives and paper exports.

## 6. Reproduction tiers

| Tier | Required evidence | Allowed claim |
|---|---|---|
| `FULL_RAW_REPRODUCED` | Stimulated and unstimulated raw trial/control data materialized and hashed; exact event/exclusion and neuron-identity lineage; pinned analysis code; independently reproduced q and q_eq methods and values. | Full raw reproduction for the pinned cohort and method. |
| `PAPER_SOURCE_DATA_REPRODUCED` | Paper's derived Source Data tables materialized/hashed and independently checked, but raw null controls or full trial-level inputs not independently available. | Reproduction of published source tables only. |
| `PARTIAL_RESPONSE_REANALYSIS` | Some raw response traces available, but one or more exclusion, identity, null-control or source joins incomplete. | Partial response reanalysis with explicit coverage and differences. |
| `METADATA_ONLY` | Archive metadata and/or remote manifest metadata only; no full source bytes verified. | Metadata inventory only. |
| `NOT_REPRODUCIBLE` | Material source or method dependencies cannot be resolved sufficiently for the requested endpoint. | State the missing dependency; no reproduction claim. |

The verifier computes an upper bound on the tier supported by the submitted evidence. A caller-supplied higher tier must be rejected.

## 7. Inventory protocol

1. Freeze DANDI DOI/version, metadata response, paginated asset manifest and all file checksums. Keep exact API URLs and retrieval time.
2. Freeze OSF project/component inventory, file versions and checksums. Do not assume the main project download includes its components.
3. Pin all analysis repositories and dependencies by commit SHA; record their licenses and source relations.
4. Build explicit animal/session/source joins. Keep unmatched, ambiguous, imaging-only, trace-only, and excluded records separate.
5. Pilot one complete WT and one unc-31 recording plus their OSF exports before attempting the 4.1 TB archive in bulk.
6. Reproduce published per-pair observation denominators and response summaries before q/q_eq or model fitting.
7. Publish a discrepancy matrix that records source A, source B, reconciliation rule, outcome, evidence digest and unresolved status.

## 8. Link to the transport benchmark

#10753 uses four disjoint partitions: training, tuning/validation, calibration and lockbox. Those splits must be created only after animal/session reconciliation. The new-animal/same-pair track is distinct from new-pair/known-animal and new-animal/new-pair tracks. q/q_eq labels whose raw null-control inputs are not available remain paper-provided labels, not independently adjudicated ground truth.

## 9. Claim ceiling

A passing metadata-only audit proves only that a pinned archive metadata record matches the declared release identity. It does not prove all 223 assets have been fetched, that paper exports match DANDI, that controls are public, or that q/q_eq results were reproduced. Biological calibration, cohort transport, and SOTA comparisons require an achieved reproduction tier appropriate to the endpoint.