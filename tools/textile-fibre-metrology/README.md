# Textile Fibre Metrology v0.1

A standalone Rust contract for fibre-length distributions that preserves method scope, sampling design, weighting basis, calibration, uncertainty, tails and lot lineage.

## Why method-aware metrology

A result in micrometres is not enough to establish comparability. Different methods can measure different populations, weightings and parts of the fibre-length distribution.

Primary method references:
- ASTM D1440-07(2024), Standard Test Method for Length and Length Distribution of Cotton Fibers (Array Method): https://store.astm.org/d1440-07.html
- ASTM D1447-26, Standard Test Method for Length and Length Uniformity of Cotton Fibers by Photoelectric Measurement: https://www.intertekinform.com/en-au/standards/astm-d-1447-2026-141222_saig_astm_astm_4059938/
- ISO 6989:1981, Textile fibres — Determination of length and length distribution of staple fibres (single-fibre measurement): https://www.iso.org/cms/live/live/en/sites/isoorg/contents/data/standard/01/35/13551.html

ASTM's published scope is material- and process-specific. D1447-26 excludes fibres recovered from cotton yarns/fabrics, cotton blends and fully synthetic fibres. A method profile must represent the permitted scope; the software does not infer scope from the method name. A standard citation is not proof that a laboratory followed that standard.

## Contract rules

- Method ID and version, permitted fibre material/process state, resolution and supported weighting basis are explicit.
- Recovered-material states outside the formal method scope require a separate local validation evidence reference.
- Mass-weighted and number-weighted distributions are not interchangeable. Receiver mass-fraction criteria reject number-weighted input.
- Independent subsamples and repeated acquisitions are separate. Repeated scans of one subsample do not increase independent sample count.
- Underflow, overflow, contiguous bins and unclassified fraction are preserved. The full distribution must close to 1,000,000 ppm.
- A bin that straddles the receiver's minimum length receives no conservative qualifying credit. The lower bound subtracts expanded uncertainty.
- Calibration must match method/version and instrument and be valid at acquisition time.
- Cross-method comparison is unavailable unless a paired bridge study records the same physical specimens, both method-specific results and a preregistered equivalence margin.
- Process comparison binds the output distribution to the exact parent lot, process-trial ID and output-lot ID.
- The quality-adjusted output metric is a length-qualified mass fraction of incoming lot mass. It is **not** yarn quality, spinning success or receiver acceptance.

## Receiver thresholds

The receiver provides the exact specification/version, fibre-length threshold, required qualifying mass fraction and maximum uncertainty. No universal 12.7 mm criterion is hardcoded. The 12.7 mm threshold appears in specific cotton research/industry practice, but is not universal across fibres, yarn systems or receiving processes.

## Usage

~~~sh
cargo test --manifest-path tools/textile-fibre-metrology/Cargo.toml

cargo run --manifest-path tools/textile-fibre-metrology/Cargo.toml -- \
  verify --input measurement-bundle.json
~~~

The input schema is `bundle.schema.json`. The synthetic regression corpus is `fixtures/metrology-cases-v0.1.json`. No raw dataset, physical measurement or receiver decision is included.

## Claim ceiling

This crate validates measurement declarations, method scope, internal distribution consistency and conservative threshold calculations. It does not authenticate issuer identity, inspect instrument calibration certificates, determine whether a laboratory actually followed the named standard, validate sampling representativeness in the field, prove fibre spinnability or certify yarn/fabric acceptance.
