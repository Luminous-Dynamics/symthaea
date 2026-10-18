# CropFoundry dataset discovery inventory v0

**Snapshot date:** 2026-10-18  
**Status:** discovery shortlist only; no external dataset files have been downloaded, parsed, normalized, or scored for this PR.  
**Purpose:** make the first data-ingestion decision auditable, rights-aware, and matched to a real benchmark question.

**Machine-readable registry:** [data-source-manifest-v0.json](./data-source-manifest-v0.json). The Rust validator at `cargo run -p xtask -- cropfoundry-validate` checks that these candidates remain non-trainable unless artifact-specific rights, access authorization, an exact pinned digest, intended use, and schema review are recorded.

## Decision

Do not start by bulk-ingesting every crop database. First build a source manifest and deeply inspect a small, complementary set:

1. One **structured tomato trial dataset** for agronomic/phenotypic schema mapping and a feasibility check.
2. One **multimodal greenhouse tomato dataset** if its exact dataset license permits the intended use.
3. The **existing Mycelix lettuce/DWC profile and cultivar-admission work** as an integration/contract lane, not as measured yield evidence.
4. An **ARC partnership/access request** for one locally relevant underutilized leafy vegetable rather than assuming its trial records are openly downloadable.
5. A **germplasm discovery** lane through Genesys or the relevant national/institutional collections, subject to access terms and material-transfer conditions.

A Portuguese tomato trial is not evidence that the same cultivar or model transfers to Johannesburg. It is initially useful for testing metadata mapping, split design, and pipeline correctness. South African transfer needs local data or new validation.

## Candidate source register

| ID | Source and known contents | Potential role | Access / license status as checked | Decision |
|---|---|---|---|---|
| DS-001 | [LiveSeeding tomato breeding-line trials (Portugal, 2023), Zenodo 21789988](https://zenodo.org/records/21789988). Record describes agronomic, phenological, fruit morphology, yield-related, sensory and quality evaluations; XLSX is listed. | First structured-trial candidate; test schema extraction, trait harmonization, missingness, and trial grouping. | The rendered Zenodo record was marked Open, but its visible License field was blank. Inspect the exact record metadata and file terms before reuse or derived redistribution. | **P0 discovery; quarantine until rights confirmed.** |
| DS-002 | [LiveSeeding tomato shuttle-breeding trials (Portugal, 2024–2025), Zenodo 22008471](https://zenodo.org/records/22008471). Record describes agronomic, morphological, yield and sensory characteristics; XLSX is listed. | Complementary season/management dataset; possible independent study if definitions and experimental populations align. | The rendered record was marked Open, but its visible License field was blank. Exact rights and related-study dependencies need verification. | **P0 discovery; quarantine until rights confirmed.** |
| DS-003 | [Horti-M3-Tomato, DOI 10.5281/zenodo.17217565](https://doi.org/10.5281/zenodo.17217565). Paper describes three seasons (2023–2025), RGB images, sensor logs, soil/environment data, agronomic records and yield. | Multimodal growth-stage/short-horizon prediction and sensor-to-outcome alignment, if permitted. | Paper reports a public Zenodo data package; the article metadata surfaced a CC BY-NC-ND 4.0 notice. Verify the **dataset's own license and its relationship to article/supplement licenses** before ingestion, redistribution, derivative creation, or training use. | **P0 discovery; legal/terms review required.** |
| DS-004 | [TomatoMAP: multi-angle/multi-pose tomato phenotyping](https://www.nature.com/articles/s41597-026-06926-9), *Scientific Data* (2026). The article describes a fine-grained tomato imaging/phenotyping dataset. | Image-only phenotyping tasks such as plant/fruit localization, segmentation, pose and annotation-quality evaluation—not yield prediction unless linked observations support it. | Article is described as CC BY 4.0; verify data availability, supplementary files, and any third-party image exceptions at the point of use. | **P1 after artifact/license verification.** |
| DS-005 | [ARC Indigenous/Traditional African Leafy Vegetables program](https://arc.agric.za/arc-vopi/Pages/Crop%20Science/Indigenous-Crops.aspx). Research scope includes *Amaranthus*, *Cleome gynandra*, *Corchorus*, *Solanum retroflexum*, *Vigna* and others; its overview discusses water/fertilizer needs, planting/harvest methods, nutrition and field/pot trials. | Local species/cultivation knowledge; identify candidate partners, variables and evidence gaps for an underutilized crop. | This is a program/research page, not a published downloadable trial dataset. Do not scrape researcher/contact data or imply records are open; pursue the official contact/partner path and agreements. | **P0 collaboration/access discovery.** |
| DS-006 | [Genesys PGR](https://www.genesys-pgr.org/) and its [API guidance](https://docs.genesys-pgr.org/docs/api/). The July 2026 API guide describes accession passport and trait-data query workflows and a token-based API. | Discover relevant accessions, origins, descriptors and available trait datasets; not automatically a usable yield dataset. | API access requires an authorized token; terms, attribution, institution-level permissions and material-transfer conditions apply. | **P1 germplasm discovery after access/terms review.** |
| DS-007 | [FAOSTAT](https://www.fao.org/faostat/). Country/region-level production, prices, inputs, trade and food-security statistics. | Context variables, plausible production/price baselines and food-system prioritization. | Public statistical portal; capture dataset vintage, series definition, geography, units and terms. Aggregated time series are not cultivar-level trial evidence. | **P0 contextual baseline only.** |
| DS-008 | [Plant Phenotyping Datasets](https://www.plant-phenotyping.org/datasets-home). Existing image/annotation benchmark resources for detection, localization, segmentation, leaf counting and related CV tasks. | Reproducible computer-vision baseline before new imaging models. | Each dataset's own license, annotation protocol and train/test conventions must be checked separately. | **P1 for CV work only.** |
| DS-009 | [Mycelix #3303 ProductiveLoopV1](https://github.com/Luminous-Dynamics/mycelix/issues/3303), [#3309 lettuce/DWC profile](https://github.com/Luminous-Dynamics/mycelix/issues/3309), [#3311 cultivar-admission profile](https://github.com/Luminous-Dynamics/mycelix/issues/3311), and [#3317 admission validator](https://github.com/Luminous-Dynamics/mycelix/issues/3317). | Reuse existing output/lot/cultivar semantics and negative controls for controlled-environment integration. | These are project design/implementation work items, not third-party biological validation datasets. Their references must keep claim ceilings intact. | **P0 integration; do not count as biological observations.** |

## First ingestion gate

Before a file enters a training or evaluation dataset, record:

- stable source URI/identifier, version, retrieval timestamp, content digest, and file inventory;
- license text and intended-use decision, including any no-derivative/non-commercial or third-party-material limits;
- dataset authorship/citation and relevant access/material-transfer terms;
- experiment ID, biological material/cultivar/accession, site, season, management, replication, randomization/block structure where recorded;
- trait identifiers mapped to Crop Ontology or a documented local term, including method, scale, unit and missingness;
- train/test reuse restrictions and any published benchmark split rules;
- provenance of normalization, conversion, exclusion and derived fields.

**Fail closed:** unresolved rights mean quarantine; missing genotype identity means no genotype-level claim; incompatible trait definitions mean no naïve pooling; absent site-year context means no site-year transfer claim. A public landing page alone is not permission to copy or train on every artifact linked from it.

## Recommended initial work order

1. Resolve DS-001 and DS-002 exact terms, download only if allowed, and inspect workbook structure manually/reproducibly. Do not merge them just because both are tomato data.
2. Build a trait/measurement crosswalk against MIAPPE + Crop Ontology, then determine whether one outcome is comparable enough for a first baseline.
3. Check sample count and grouping structure **after actual inspection**; if there are too few site-years or replicates, stop the predictive benchmark and use the data only for schema/benchmark plumbing.
4. Verify DS-003's data-specific license before any processing beyond public metadata review.
5. Open an ARC collaboration/access conversation for the indigenous-crop lane; choose the crop based on local relevance and actual evidence/seed access, not on a preconceived model target.
6. Only after the dataset inventory passes should we freeze the specific trait, model baselines, held-out split, metrics and evaluation receipt.

## Source notes

- The two LiveSeeding records were published in August 2026 and describe small, structured tomato trial workbooks. The record pages do not show a clearly filled license in the rendered metadata; this inventory intentionally does not assume reuse permission.
- The Horti-M3-Tomato paper describes multi-season, multimodal coverage, but article license and data-package license must be distinguished.
- ARC's program page documents an active research remit and local crop species; the existence of a research program is not proof of a public raw dataset.
- Genesys, FAOSTAT and the phenotype dataset portal serve different purposes. Germplasm discovery, macro-level agricultural context, imaging annotations and replicated field-trial outcomes must not be conflated.
