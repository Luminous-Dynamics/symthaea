# Larsen Basin assessment reproduction — V0

**Evidence tier:** published-table transcription and consistency checks.  
**Not completed:** raw data-release ingestion or independent reproduction of the USGS geological resource model.

This workstream implements [LARSEN-BASIN-REPRO-001 #8425](https://github.com/Luminous-Dynamics/symthaea/issues/8425). Its purpose is to reproduce and audit the published result table without turning an uncertain estimate of undiscovered resources into a discovery, reserve, commercial project or extraction recommendation.

## Source authorities

- USGS, *Assessment of undiscovered conventional oil and gas resources of the Larsen Basin, Antarctica, 2025*, Fact Sheet 2026-3063, published 25 February 2026: https://pubs.usgs.gov/publication/fs20263063/full
- USGS result/input/boundary data release, DOI 10.5066/P14LT9DY, declared CC0 1.0 Universal: https://www.usgs.gov/data/usgs-national-and-global-oil-and-gas-assessment-project-larsen-basin-antarctica-assessment
- Protocol on Environmental Protection to the Antarctic Treaty, Article 7: https://www.antarctica.gov.au/about-antarctica/law-and-treaty/the-madrid-protocol/

The source JSON at `larsen_basin_assessment_v0.json` transcribes the published assessment table and essential metadata. It explicitly sets `raw_bytes_downloaded_by_this_workstream=false`, leaves `raw_sha256=null`, and states that the geological model and distributions have not been independently reproduced. A DOI, data-release landing page or table transcription is not evidence that the raw data were ingested.

## Published result summary

| Total undiscovered technically recoverable resources | F95 | F50 | F5 | Mean |
|---|---:|---:|---:|---:|
| Oil | 0 | 245 | 591 | 269 MMBO |
| Gas | 0 | 11,339 | 37,804 | 14,257 BCFG |
| Natural gas liquids | 0 | 350 | 1,160 | 439 MMBNGL |

The source describes fully risked estimates for one named assessment unit, with an assessment-unit probability of 0.9. The source's quantile notation is exceedance-oriented: F95 is the amount with a 95% chance of at least that amount, F50 is the 50% exceedance amount, and F5 is the 5% exceedance amount. Do not relabel these as conventional lower/median/upper percentiles without explaining that convention.

The assessment is geological, not an economic assessment. “Technically recoverable” does not mean economically recoverable. The result is about undiscovered accumulations and is not evidence of discoveries or proved reserves. The report's inputs for source-rock adequacy and burial history are uncertain, and the burial-history model gives only a general indication of thermal maturity. The assessment-unit probability is already represented in the fully risked table; do not multiply by 0.9 again.

A zero F95 value alone does not give the exact probability of zero resources. The resource-volume distributions and the chance of an adequate petroleum system are different objects.

## What V0 checks

The dependency-free audit script checks:
- schema/status, source identifiers, AU probability and the declared evidence limits;
- published table values, units and F95/F50/F5 ordering;
- consistency between the source's displayed oil/gas accumulation rows and displayed total row, by reported fractile and mean;
- resource-category semantics: no discoveries, proved reserves, economic reserves or commerciality claim;
- source-ingestion status: raw data are not claimed as downloaded or hashed;
- frozen adversarial cases that mutate values, units, quantile ordering, risk flags and ingestion metadata.

Those checks protect a transcription and its semantics. They **do not** independently reproduce the geological model, Monte Carlo calculation, source-rock assumptions, burial history or the probability distributions from raw inputs. A test that confirms a transcribed number matches an expected literal is not independent scientific replication.

## Reproduce locally

From a checkout of this repository:

```sh
bash scripts/check-larsen-basin-assessment.sh
```

Requires `jq`. The path-scoped workflow is `.github/workflows/larsen-basin-assessment.yml`.

## Next evidence tier

1. Retrieve the original USGS CC0 release archive, then preserve the exact bytes, filename, download date, URL, licence and SHA-256 digest. Do not set the ingestion flag until this is done.
2. Match the raw table and assessment-unit boundary records to the publication's table definitions and units.
3. Reproduce the published output from raw inputs only to the degree the released methodology and input files specify a reproducible calculation. Preserve any undocumented or inaccessible modelling choices as an explicit reproduction gap.
4. Independently check the resource-category/quantile outputs and sensitivity to documented geological assumptions. Never infer well locations, specific accumulation locations, extraction economics or a development plan.
5. Keep all results inside the scientific resource-assessment boundary. Article 7 of the Environmental Protocol prohibits mineral-resource activities other than scientific research; this package neither evaluates extraction methods nor recommends exploitation.

Queued, skipped, absent or stale CI is not a pass. A green software check only validates this transcription/audit contract, not the physical truth or economic significance of the geological estimate.
