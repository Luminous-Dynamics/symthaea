# HEAVY-OIL-RESCUE-001 — B1 reference-data inventory

**Prepared:** 2026-10-22  
**Current status:** metadata and landing pages reviewed; raw archives have not yet been ingested, hashed, schema-checked or normalized. This inventory is a plan, not a claim that a dataset has passed validation.

## Highest-priority candidates

| Priority | Source | What it contributes | Main limits / intended use |
|---|---|---|---|
| P0 | [DOE/NREL Crude Oil Analysis (COA) Database](https://catalog.data.gov/dataset/crude-oil-analysis-coa-database) | Public metadata describes 9,076 crude-oil analyses collected during 1920–1983. Fields include specific gravity/API gravity, pour point, viscosity, sulfur, nitrogen, color and bulk properties of distillation cuts. Data.gov declares CC BY 4.0 and links the [archive ZIP](https://data.openei.org/files/178/coa.zip). | Good starting corpus for historical feedstock characterization and schema normalization. The data are old; location identifiers are often incomplete; it is not a linked dataset of modern upgrading experiments, catalyst lifetime, process economics or lifecycle emissions. Do not assume every record has every field. |
| P0 | [Alberta Bitumen Assay Program aggregate data](https://open.canada.ca/data/en/dataset/dcd6f538-020c-4033-b2a4-646ae14011ea) | Government of Alberta's Bitumen Assay Program aggregates bitumen assays by oil-sands region and recovery technology; the stated design protects individual operators' assay data. The landing record identifies the Open Government Licence — Alberta and links an XLSX resource. | Directly relevant to heavy bitumen composition, but aggregate strata are not individual independent assays. Preserve grouping and aggregation semantics; never expand one aggregate into fake sample-level records. Confirm access and licence terms at ingestion. |
| P1 | [USGS Energy Geochemistry Database v2.0](https://www.usgs.gov/data/energy-geochemistry-database) | Metadata reports 3,619,149 analytical results from 111,791 samples; the release is marked CC0 1.0. It combines laboratory, historical and literature/public-domain geochemical data. | Broad hydrocarbon/geochemistry corpus, not a petroleum-refinery upgrading dataset. Use only after the data dictionary confirms an analyte is directly relevant; retain sample/analysis provenance and avoid conflating geological potential with process-feed properties. |
| P1 | [NOAA ADIOS Oil Database](https://response.restoration.noaa.gov/adios-oil-database) | Public oil-assay records in structured JSON, including physical/chemical properties and sometimes laboratory methods; the purpose is oil-spill response. | Useful for checking property units and metadata conventions. NOAA notes that records vary in completeness and are undergoing source-fidelity checks. Not evidence for refinery yield, catalyst activity, or process economics; review reuse terms before mirroring. |

## Secondary sources requiring care

- [ExxonMobil crude-oil assays](https://corporate.exxonmobil.com/What-we-do/Energy-supply/Crude-trading/crude-oil-assays) publishes PDF/XLS assays, including listed heavy/sour grades. Its page includes a disclaimer placing responsibility for interpretation/use on the user. Record exact assay revision, download date and applicable terms; do not mirror or redistribute the files until the terms are reviewed.
- [NETL solvent-based in-situ heavy-oil upgrading project](https://www.netl.doe.gov/node/3838) and its [2009 final technical report](https://www.netl.doe.gov/files/oil-gas/NT42745_FinalReport.pdf) are public technical references for reservoir-scale solvent processes. This is an in-situ recovery/upgrading boundary, not interchangeable with refinery residue conversion. Extract a case only if feed, experimental conditions, measured outputs and uncertainty can be traced.
- [NETL Utah Heavy Oil Program final report](https://www.netl.doe.gov/files/oil-gas/NT15569_FinalReport.pdf) is a candidate public report corpus. Its presence alone does not mean a machine-readable or independently validated process dataset has been found.

## What we have—and have not—found

The initial research identifies openly accessible feedstock-characterization datasets and public, process-specific research reports. It has **not yet identified a single open dataset that consistently links**:

feedstock assay → process/catalyst conditions → full product/by-product/waste mass balance → energy/utility use → product-specification outcome → catalyst ageing/lifetime → costs → lifecycle impacts

That linked record is the critical gap for a fair end-to-end benchmark. The project must not fill it with fabricated correlations. Where only feedstock characterization is available, use it for characterization/coverage tests; where a report contains a reproducible experiment, preserve that case separately with its original boundary and evidence tier.

## Ingestion gate (before any row becomes benchmark evidence)

For each downloaded artifact, record:
1. source URL, publisher, access date, declared licence/terms, original filename and SHA-256 digest;
2. dataset/version, source report/page/table, sample/assay identity, sample date/location when disclosed and aggregation level;
3. property name, value, explicit unit, method/conditions, detection/quantification limit if applicable, and uncertainty or missingness reason;
4. whether the value is observed, derived, aggregated, inferred, simulated or forecast;
5. data transformation code/version and a raw-to-normalized lineage map;
6. a licence and redistribution decision, including whether derived aggregate data may be published.

Reject duplicate/conflicting samples, untraceable units, impossible values and implicit unit conversion. Do not interpolate missing chemistry merely to make a simulator run.

## Proposed B1 sequence

1. Inventory the COA archive's tables/fields and licence; implement a read-only normalization report, without committing the raw archive until size/licence and repository policy are checked.
2. Inspect the Alberta aggregate workbook and preserve its strata/aggregation semantics.
3. Select one heavy-feedstock characterization record/case with adequate traceability; then search the DOE/NETL reports for a fully bounded experiment that can be reproduced from cited tables.
4. Freeze the selected record(s), citations, units, applicability envelope and held-out split before model tuning.
5. Only then compare applicable routes and simulators. Use DWSIM where its property/unit-operation models support the selected case; model applicability must be explicit and simulator agreement is not empirical validation.

## Sources

- Data.gov, COA Database: https://catalog.data.gov/dataset/crude-oil-analysis-coa-database
- Government of Alberta, Bitumen Assay Program aggregate data: https://open.canada.ca/data/en/dataset/dcd6f538-020c-4033-b2a4-646ae14011ea
- USGS Energy Geochemistry Database: https://www.usgs.gov/data/energy-geochemistry-database
- NOAA ADIOS Oil Database: https://response.restoration.noaa.gov/adios-oil-database
- NETL solvent-based heavy-oil project: https://www.netl.doe.gov/node/3838
- DWSIM thermodynamic-model guide: https://dwsim.org/help/main/38-introduction.html
