# CMNI source-data access audit

**Snapshot:** 2026-11-20  
**Tracking:** https://github.com/Luminous-Dynamics/symthaea/issues/13023

## 1. Immediate finding

The research programme is **not blocked** pending the 2026 dataset release. A public archive associated with the 2025 Chen et al. experiment is visible now, and the article itself identifies the published paper and supplementary information as containing supporting data.

However, the public record needs accession reconciliation before we treat it as the same persistent object cited in the article.

## 2. 2025 Chen et al. source

Article:

- Chen et al., “Electrochemical loading enhances deuterium fusion rates in a metal target,” *Nature* 644, 640–645 (2025).
- DOI: https://doi.org/10.1038/s41586-025-09042-7
- Public article page: https://www.nature.com/articles/s41586-025-09042-7
- Public full-text copy: https://pmc.ncbi.nlm.nih.gov/articles/PMC12367529/

The article's data statement says supporting data are in the paper and its Supplementary Information. It says analysis code and CAD files are available at Zenodo DOI **10.5281/zenodo.13830012**.

The Zenodo search/catalog result currently visible for the title is:

- Record: https://zenodo.org/records/14752388
- DOI displayed on that record: **10.5281/zenodo.14752388**
- Title: “Electrochemically-enhanced nuclear fusion of deuterium”
- Version: v2
- Resource type: Computational notebook
- Description: source code and CAD files for the manuscript.
- Displayed file inventory:
  - `eCell_CAD.zip` — 14.8 MB; advertised MD5 `2a4c4dd05136adbe66d50d679e65d454`
  - `In-situ-XRD-cell_Air_CAD.zip` — 7.4 MB; advertised MD5 `46b090e96f5c897678d22fd62231159b`
  - `In-situ-XRD-cell_Vacuum_CAD.zip` — 12.8 MB; advertised MD5 `604bbf5ef2ac186e041dee988ac9e079`
  - `Plasma thruster_CADzip.zip` — 1.9 MB; advertised MD5 `28e40c5cd865e58d99aa9089857cff73`
  - `thunderbird_psd_v2.zip` — 14.7 GB; advertised MD5 `4f1f581738cb1b860fd3fafc54a0f63f`

The preview endpoint for the 14.7 GB ZIP reports “Too many files inside the ZIP file.” Its internal contents have **not** been inspected in this audit. Do not assume it contains raw waveforms, all analysis source, or a specific data schema until an archive manifest is inspected.

### Accession discrepancy

The article's availability statement cites `10.5281/zenodo.13830012`, while the currently indexed Zenodo record found by title lists `10.5281/zenodo.14752388`. Their relationship is not established by the page inspection recorded here. Before ingesting, resolve whether these identifiers are:
- an earlier record and its new version;
- related but distinct deposits;
- a citation/metadata error; or
- separate records containing different assets.

Never silently rewrite one DOI into the other.

## 3. Published quantitative anchors (not raw data)

The paper reports three campaign-wise neutron-rate changes with electrochemical deuterium loading: 15.1(1)%, 11.3(1)% and 18.5(2)%. It separately summarizes the average as 15(2)%.

The paper also gives rounded phase-I/phase-II saturation-rate values in the narrative. Do not independently derive the campaign percentages from rounded display values and replace the reported percentages: retain the exact reported values, the rounded rate pairs, their separate provenance, and a reconciliation note. The source-level time series and analysis code must determine the precise estimator and uncertainty propagation.

The paper explicitly describes beam-driven D–D fusion: a 30-keV D+ beam is used to bombard Pd, and electrochemical loading changes the target fuel state. The measured fusion output is about 10^-9 W against 15 W input. This is not accelerator-free cold fusion or net energy.

Useful source details to preserve for the 2025 track:
- three separate experimental campaigns;
- plasma immersion ion implantation / D+ beam excitation;
- nominal sheath voltage and beam current;
- electrochemical D2O vs H2O control;
- target D/Pd loading is inferred from related in-situ XRD measurements, not directly measured during each fusion run;
- neutron detector response and background characterization;
- calibration/transport simulations;
- sample anneal and history;
- time-series phase boundaries and steady-state definition.

## 4. 2026 Karahadian et al. source

Article:

- Karahadian et al., “Enhanced nuclear fusion in the sub-keV energy regime,” *Nature Communications* (2026).
- DOI: https://doi.org/10.1038/s41467-026-74421-1
- Zenodo record: https://zenodo.org/records/20435774
- The Zenodo record lists the data files as embargoed until **2026-12-01** as checked on 2026-11-20.

The 2026 track is therefore not ready for direct file-level source reconstruction today. Published equations, methods, figures, and reported summary values may be independently transcribed with explicit `published_summary` provenance, but they must not be passed off as raw-data reproduction.

The 2026 paper's lower-energy yield data are from the proton channel; neutron measurements cover a higher-energy range. Model and schema records must preserve channel availability, and not imply every energy point is independently measured by both channels.

## 5. Ingestion strategy

1. **Now:** freeze article and SI references; capture published summary values and their quoted uncertainties as `published_summary` records; never merge them with raw runs.
2. **Now:** inspect the 2025 Zenodo record's metadata, file list, licence, version history, checksums, and DOI relations. Resolve the 13830012/14752388 discrepancy.
3. **Now:** inspect the 2025 code/readme and archive manifest before choosing any download. The 14.7 GB file should not be fetched wholesale before confirming its contents and a selective-processing plan.
4. **Now:** implement the 2025 published-result reproduction as a benchmark with separate records for reported percentage changes and rounded rate pairs.
5. **On/after 2026-12-01:** revisit 2026 Zenodo access, verify files are genuinely public, record version and checksums, and begin raw-data reconstruction.
6. **After either source is ingested:** freeze source bytes, compute SHA-256, record the publisher MD5 where supplied, preserve raw and derived artifacts separately, and replay the source analysis before introducing new models.

## 6. Provenance labels

Use only:

- `published_summary`: number explicitly stated in a paper/figure/table.
- `supplementary_data`: a source-provided data table or supplementary file, with file/page/table coordinates.
- `source_archive`: an archive from a persistent repository, with a verified file digest and manifest.
- `raw_measurement`: raw instrument output with run and calibration context.
- `derived`: a result computed from other named inputs, with code revision and parameters.
- `digitized_plot`: a value transcribed from pixels; uncertainty includes digitization limits.
- `synthetic`: generated for testing, never physical evidence.

## References

- 2025 paper and data/code statement: https://doi.org/10.1038/s41586-025-09042-7
- Public Zenodo v2 record found by title: https://zenodo.org/records/14752388
- DOI cited by the 2025 article: https://doi.org/10.5281/zenodo.13830012
- 2026 paper: https://doi.org/10.1038/s41467-026-74421-1
- 2026 data record (embargoed until 2026-12-01): https://zenodo.org/records/20435774


## 7. Supplementary-information audit: 2025 neutron diagnostics and controls

The 10.9 MB Supplementary Information was inspected alongside the main paper:
https://assets-eu.researchsquare.com/files/rs-5046357/v1/267adeeb10150099b6fa9938.pdf

### Detector and signal processing

The Thunderbird neutron diagnostic is described as a single EJ-309 liquid-scintillator detector:
- 12.7 × 12.7 cm cylindrical scintillator and Hamamatsu R1250 PMT.
- CAEN DT5730S 14-bit / 500 MHz waveform digitizer with DPP-PSD firmware.
- Pulse-shape discrimination using tail/total charge-integral ratio.
- Gamma PSD distribution fitted in 5-keVee light-output slices; neutron-window lower boundary set at gamma mean + 5 standard deviations, upper boundary another 0.2 PSD units higher; 50-keVee lower light-output threshold.
- Pileup contribution reported below 2% after the specified rejection and geometry.
- Magnetic shielding reduces the detector-front field from about 40 G to below 0.3 G.
- Light-output calibration with 60Co, 137Cs and 152Eu.
- Detector response compared with MCNP6.2 + DRiFT and Geant4 simulations for 2.45-MeV D-D neutrons; spectrum unfolded with a Geant4 response matrix and GRAVEL.

The article reports exclusion of more than 99.9999% of gamma rays under its stated PSD method. Treat this as source-reported, setup-specific performance, not a universal false-positive guarantee or independent detector replication. The SI reports best response agreement mainly over 200–1000 keVee; the lower acquisition threshold truncates part of the response.

### Controls that must become first-class evidence

- **Background:** 0.21(9) neutrons/s; the article also reports stable background over 24 hours.
- **Repeatability:** seven beam-loading experiments using different Pd targets reached about 69.3(2) neutrons/s after 120 minutes at -30 kV / 0.2 mA. Keep this separate from the three electrochemical Phase I/II campaigns, which use different conditions.
- **Sheath-voltage ON/OFF:** turning off the -30 kV sheath after steady beam loading returned the neutron rate to about 0.2 neutrons/s; restoring voltage restored the signal. This is a beam-excitation control.
- **H2O isotope control:** 2 M K2CO3 in H2O instead of D2O produced an 88.3(3)% reduction in neutron production rate. This supports isotope-specific fuel loading in this apparatus; it does not exclude every possible artifact.
- **Electrochemical-cell OFF/ON:** with beam conditions otherwise maintained, the paper reports no more than 2% variation during the cell-off interval and a smaller increase (~6%) on reactivation than the initial (~20%) increase. This tests persistence of previously loaded fuel; it is not the same as sheath-voltage-off.
- **D/Pd measurement limitation:** the article explicitly states that D/Pd was not measured directly during the fusion runs. The ~0.7 ratio is inferred from separate in-situ XRD characterization and assumed to be similar in the fusion apparatus.

### Campaign-rate transcription cross-check

The full article text and Fig. 3 confirm Phase I saturation rates of 135.5(5), 142.9(5), and **138.6(3)** n/s, and Phase II rates of 156.7(4), 159.2(5), and 164.3(13) n/s. The third Phase I uncertainty is **0.3**, not 1.3. The published campaign percentage increases (15.1(1)%, 11.3(1)%, 18.5(2)%) remain separate source claims; direct division of the rounded plateau rates does not reproduce all three percentages. Preserve both and resolve the estimator from source time-series and analysis code.

### Archive limits

The public 14.7-GB archive remains uninspected. Do not assume it contains complete raw waveforms or infer its internal structure from its filename. Begin with the accessible published summaries and SI; inspect the archive manifest before selective download.
