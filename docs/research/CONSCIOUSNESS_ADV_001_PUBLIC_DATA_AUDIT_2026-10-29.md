# CONSCIOUSNESS-ADV-001 — public-data source and feasibility audit

**Audit date:** 2026-10-29  
**Status:** Public-source metadata audited; no participant data downloaded, hashed, or analyzed.  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/8971  
**Companion protocol:** [CONSCIOUSNESS_ADV_001_THEORY_BENCHMARK_2026-10-29.md](CONSCIOUSNESS_ADV_001_THEORY_BENCHMARK_2026-10-29.md)

## Result

The 2025 Cogitate adversarial-collaboration study is not blocked by a complete absence of public data or analysis code. The primary article links its preregistration, data and code availability statements. It reports that generated data are available under CC BY 4.0 and points to raw/BIDS data bundles and an XNAT repository. Public modality-separated analysis and task code are also available.

This clears the source-discovery gate, not the reproduction gate. No data bundle was downloaded in this audit, no binary artifact hash was captured, no environment was reproduced, and no empirical prediction was re-analyzed. A source link or license statement is not a replication receipt.

## Preregistration access caveat

The Nature article provides a detailed published summary of the three core theory comparisons (content decoding/task independence; temporal maintenance; interareal connectivity) and says the OSF record contains the full protocol, pass/fail criteria and agreed interpretation. However, the OSF page itself returned HTTP 403 to the research-browser open request during this audit. The exact OSF file/version and contents therefore remain to be captured directly before any analysis card is frozen. A link is identified; the preregistration content is not claimed as independently inspected here.

## Primary sources and provenance

| Asset | Official source | What is established | What remains to verify locally |
|---|---|---|---|
| Published adversarial collaboration | https://doi.org/10.1038/s41586-025-08888-1 | Nature paper published 2025-04-30; article links preregistration, data and code availability | Preserve the article version and supplementary materials in the analysis receipt |
| Preregistration | https://osf.io/92tbg/ | Linked by the paper as containing study design, theory predictions, expected outcomes, acquisition details and analyses | Capture exact preregistration version/content digest; compare amendments and deviations with the paper |
| Data bundles | https://www.arc-cogitate.com/data-bundles | Experiment 1 raw and BIDS bundles are listed for M-EEG, fMRI and iEEG | Accept portal terms; capture downloaded bundle identity, subject manifest, file sizes and SHA-256 digests |
| XNAT source | https://cogitate-data.ae.mpg.de/ | Portal identifies XNAT as an alternative for selecting specific data | Check metadata, subject/session inclusion, access requirements and exact export manifest |
| Data-release documentation | https://cogitate-consortium.github.io/cogitate-data/ | Documents directory layout, acquisition and modality-specific data structures | Treat as versioned documentation; resolve counts against the primary article and dataset manifests |
| Analysis code | https://github.com/Cogitate-consortium/cogitate-msp1 | Public repository has separate behaviour/eye tracking, M-EEG, fMRI and iEEG analyses; repository indicates MIT license | Pin commit SHA and modality-specific environment/configuration; identify primary-analysis entry point |
| Task/stimulus code | https://github.com/Cogitate-consortium/cogitate-experiment-code | Public task code repository; repository indicates MIT license | Pin relevant task-code commit and identify the exact task/stimulus version used for the cohort |

**Access note:** The portal says COGITATE data users must accept its Terms of Use and GDPR Requirements: https://www.arc-cogitate.com/terms-conditions. Do not infer that the paper's CC BY 4.0 statement removes those access and data-handling requirements.

## Experiment 1 bundle inventory

Sizes below are the bundle sizes currently listed by the source portal. They are planning values, not downloaded-artifact byte counts, and are not cryptographic identifiers.

| Modality | Format | Sample bundle | Full bundle | Dataset DOI |
|---|---|---:|---:|---|
| M-EEG | Raw | 29 GB | 740 GB | https://doi.org/10.17617/1.WQA3-WK71 |
| M-EEG | BIDS | 28 GB | 700 GB | https://doi.org/10.17617/1.NZWP-7J89 |
| iEEG | Raw | 13 GB | 77 GB | https://doi.org/10.17617/1.CQYN-9A87 |
| iEEG | BIDS | 16 GB | 98 GB | https://doi.org/10.17617/1.6QPG-GZ82 |
| fMRI | Raw | 9 GB | 538 GB | https://doi.org/10.17617/1.DHJJ-ER02 |
| fMRI | BIDS | 9 GB | 261 GB | https://doi.org/10.17617/1.8AX5-T263 |

The portal lists the Experiment 1 bundles under CC BY 4.0, in addition to requiring acceptance of the portal terms. Do not mix later Experiment 2 releases into a reproduction of the 2025 Experiment 1 paper.

## Sample-accounting discrepancy to resolve

The Nature article reports 256 analyzed participants across modalities: fMRI n=120, MEG n=102, and iEEG n=34. The data-release overview describes 256 participants with fMRI n=118, MEG n=100, and iEEG/ECoG n=38. Both totals are 256, but the modality allocations differ.

This audit cannot explain that discrepancy from landing-page metadata alone. Before scientific analysis, reconcile the released subject/session manifest, quality-control exclusions, per-modality analysis code and the article's final analysis samples. Do not silently treat every released participant as included in every primary analysis.

The paper reports that fMRI and MEG analyses used an optimization phase on roughly one-third of the sample followed by final testing on previously unexamined data; iEEG used a different strategy given its smaller sample. Reproduction must preserve the modality-specific strategy and preregistered split. A sample bundle is not a substitute for confirmatory evaluation.

## Recommended implementation order

1. **Freeze the source.** Record the Nature DOI/version, OSF preregistration version, analysis-code commit, task-code commit, data-release documentation version, and terms acceptance/access path in the run manifest.
2. **Resolve sample accounting.** Enumerate published subject/session IDs in the selected bundle, map them to the paper's included/excluded cohort, and record all non-matches with reasons.
3. **Use the fMRI BIDS sample bundle only as a plumbing smoke test.** The source lists its sample bundle as 9 GB and the full BIDS bundle as 261 GB. Validate directory layout, BIDS metadata, event labels, conditions, missingness and official pipeline startup. A sample-only run must be marked SMOKE_TEST, never REPRODUCED or CONFIRMATORY.
4. **Run one frozen preregistered fMRI prediction on the complete eligible cohort.** Use official analysis code where feasible, pin dependencies, and retain the consortium result as a reference baseline. Copy the exact primary contrast, ROI/window, and outcome criteria from the preregistration before opening confirmatory outputs.
5. **Keep exploratory and confirmatory results separate.** Any changed preprocessing, new ROI/window, subject exclusion, or model adjustment gets a new profile and remains exploratory unless prospectively justified.
6. **Add modalities only after the first pathway is auditable.** Use official M-EEG and iEEG pipelines and modality-specific inclusion/testing rules; do not average all modality results into one unqualified score.

## Minimum evidence receipt

A qualifying reproduction receipt should include:

- source URLs, DOI/version identifiers and retrieval timestamp;
- exact preregistration version and digest;
- accepted data-use requirements and data-release DOI;
- code commit hashes, dependency/environment lock and command line;
- subject/session manifest digest, exclusions and analysis split;
- input artifact SHA-256 values and derived-data transformation digests;
- preregistered target, observable, window/region, contrast, effect direction, statistical test, and multiplicity policy;
- output digest, effect estimate and uncertainty interval, runtime and resource use;
- comparison to the published/official reference result, with discrepancies and failed cases retained.

Raw human-subject data must not be committed to Symthaea or Symtropy. Only public metadata, code, manifests, non-sensitive receipts, and appropriately licensed artifacts should enter the repository. If the terms disallow redistribution, store a reference and digest instead of the file.

## Current qualification

- **Source discovery:** PASS — primary article, preregistration link, data bundles, data portal and analysis code identified.
- **License/access review:** PARTIAL — CC BY 4.0 is stated by the article and portal, but portal Terms of Use and GDPR requirements must also be accepted and followed.
- **Dataset identity:** BLOCKED — no files downloaded or hashed; subject/session mapping and modality-count discrepancy not yet resolved.
- **Analysis reproduction:** NOT RUN.
- **Scientific conclusion:** NONE from this audit.

This audit supports starting a careful reproduction workflow; it does not support any new conclusion about consciousness or a winner between theories.
