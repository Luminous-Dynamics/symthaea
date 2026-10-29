# Active-evidence source-derived drift registry (provisional)

**Last checked:** 2026-10-29  
**Status:** candidate-source triage only. This is not legal advice or a release approval. No external dataset payload is stored in this repository.

## Eligibility register

| Candidate source | Authoritative/source evidence | License/use status | Stream semantics | Gate before inclusion |
|---|---|---|---|---|
| Covtype / UCI Covertype | [UCI dataset 31](https://archive.ics.uci.edu/dataset/31/covertype), DOI [10.24432/C50K5N](https://doi.org/10.24432/C50K5N) | UCI states CC BY 4.0. Potential commercial-compatible candidate with attribution; *not yet exact-release cleared*. | Multivariate data; original row order must not be treated as time unless a source field proves chronology. | Pin the exact raw file, verify it is the same release used by the stream adapter, record SHA-256 and attribution, then test a source-order or explicitly seeded-order track. |
| Gas Sensor Array Drift | [UCI dataset 224](https://archive.ics.uci.edu/dataset/224/gas%2Bsensor%2Barray%2Bdrift%2Bdataset), DOI [10.24432/C5RP6W](https://doi.org/10.24432/C5RP6W) | UCI states research-only use and explicitly excludes commercial use. Use only in a separately marked research-only lane unless the licensor grants additional permission. | Time-varying sensor measurements; preserve batch/time sequence and original row order. | Verify the exact archive and batching semantics. A commercial-compatible manifest must reject this license class. |
| Electricity / normalized NSW electricity stream | [CapyMOA Electricity docs](https://capymoa.org/api/modules/capymoa.datasets.Electricity.html) point to the [SourceForge MOA archive](https://sourceforge.net/projects/moa-datastream/files/Datasets/Classification/elecNormNew.arff.zip/download/). | The checked documentation establishes the dataset description and download location but does not establish the dataset archive's license. **Not cleared for commercial-compatible use.** | Published stream has temporal dependencies; keep order; do not randomly shuffle. | Identify an authoritative license for the exact archive and record its evidence URL, version, and digest. |
| Posture | Named in the KDD '26 drift-evaluation stream list and its configuration | Exact upstream URL, file/version, license, and digest not yet verified. | Unknown until source audit. | Do not include until the exact source and license are verified. |
| Rialto | Named in the KDD '26 drift-evaluation stream list and its configuration | Exact upstream URL, file/version, license, and digest not yet verified. | Unknown until source audit. | Do not include until the exact source and license are verified. |
| NOAA | Named in the KDD '26 drift-evaluation stream list and its configuration | Exact product / station / release / license not yet verified; "NOAA" alone is not a source identity. | Likely temporal, but exact sampling, station, and row order remain unresolved. | Identify the actual NOAA product, station/site, release, and terms; preserve its chronology. |
| Asfault | Named in the KDD '26 drift-evaluation stream list and its configuration | Exact upstream URL, file/version, license, and digest not yet verified. | Unknown until source audit. | Do not include until the exact source and license are verified. |
| Gas sensor array low-concentration (optional supplemental source) | [UCI dataset 1081](https://archive.ics.uci.edu/dataset/1081/gas%2Bsensor%2Barray%2Blow-concentration), DOI [10.24432/C5CK6F](https://doi.org/10.24432/C5CK6F) | UCI states CC BY 4.0. Distinct dataset from the Gas Sensor Array Drift dataset above. | Only 90 examples with long sensor-response traces; treat as a small sensor stress set, not an interchangeable long stream. | Evaluate whether the experimental unit is a row, a sensor trace, or a time point before using it; pin preprocessing and avoid inflating effective sample size. |

## Research-baseline provenance

The named seven-stream list is reported by Cerqueira et al., *A Framework for Evaluating and Benchmarking Concept Drift Detection Methods* (KDD '26, DOI [10.1145/3770855.3819070](https://doi.org/10.1145/3770855.3819070)). Their open experiment code is at [vcerqueira/experiments-drift_evaluation](https://github.com/vcerqueira/experiments-drift_evaluation). It is a source for reproducing the benchmark design, not a license grant for the underlying dataset payloads.

The real-stream unsupervised survey by Lukats et al. compares seven detectors over eleven streams and reports that preferred methods depend on the metric ([DOI 10.1007/s41060-024-00620-y](https://doi.org/10.1007/s41060-024-00620-y)). Its data provenance must be checked separately before reusing any payload.

## Approval protocol

A source becomes approved only when the run manifest records:
1. exact upstream URL and file version/revision;
2. license identifier, license-evidence URL, attribution requirement, intended-use profile, and documented human review;
3. row order and temporal / non-temporal semantics;
4. SHA-256 of the exact acquired bytes and the transformation/preprocessing configuration digest;
5. whether redistributing transformed payloads is permitted.

The validator's intended-use guard is deliberately conservative but not a legal parser. It blocks obvious non-commercial identifiers from a commercial-compatible run; it cannot determine the scope of a license or whether a transformation creates a derivative work. Unknown or ambiguous sources remain blocked rather than guessed.
