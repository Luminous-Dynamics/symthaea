# Marine shallow-water site qualification audit

A standalone Rust preflight evidence auditor for full-scale ship underwater-radiated-noise campaigns. It treats bathymetry, seabed, CTD/current, hydrophone geometry/calibration, vessel condition and reciprocal run state as part of the measurement experiment.

## Run

~~~sh
cargo test --manifest-path tools/marine-shallow-water-site-audit/Cargo.toml
cargo run --manifest-path tools/marine-shallow-water-site-audit/Cargo.toml -- \
  docs/research/marine/fixtures/synthetic-marine-shallow-water-site-v0.1.0.json
~~~

The synthetic fixture must return INCOMPLETE as physical site qualification. Exit codes:
- 0: site evidence eligible for the declared measurement, or conditional on advisory array practice;
- 2: invalid input;
- 3: incomplete evidence;
- 64: invocation or JSON decoding failure.

## What v0 checks

- Deep/shallow classification boundary: deeper than max(150 m, 1.5 × overall ship length) is classified deep; less is shallow. Exact boundary is deliberately unresolved.
- Bathymetry coverage along track/across array footprint and declared depth-uniformity tolerance.
- Seabed substrate observations, local sound-speed samples, source/method/digest and declared spatial-uniformity tolerance.
- CTD sound-speed profile with instrument, method, calibration and evidence lineage; current direction/speed and ambient background measurement.
- Hydrophone positions, depth/orientation, calibration frequency coverage and certificate digest; pre-deployment and post-recovery in-situ checks.
- IACS recommendation for three hydrophones is advisory, not a universal hard requirement: one/two channels can be CONDITIONAL when other gates pass.
- Vessel overall/acoustic ship length, acoustic reference point, GPS offset, propulsion/service/loading identity, draft, hull and propeller condition.
- Reciprocal port/starboard runs with speed through water, shaft RPM, propulsion power, heading/aspect, CPA, data-window duration, timebase, operating point and background evidence.
- IACS Recommendation 181's CPA recommendation (greater of 100 m or acoustic ship length) and 10 s minimum data window as advisory campaign gates.
- Per-band site cutoff, hydrophone calibrated frequency span, propagation method/revision, expanded uncertainty and distinct quantity IDs.

## Quantity semantics and claim boundary

The tool preserves received SPL, radiated-noise level (RNL), source level (SL) and deep-water-equivalent RNL as different quantities. It never relabels one as another and does not compute any ISO 17208-3 propagation/source-level equation. No generic 20 log10(r) correction, learned seabed correction or inferred site transfer is applied.

When two campaigns have different depth category, seabed substrate or seabed sound-speed class, the comparison is INCOMPARABLE unless a separately reviewed, per-band propagation-transfer receipt is supplied. A supplied receipt is only referenced; this audit does not independently validate its physics.

## Interpretation

A passing result means that the declared site/campaign evidence meets this preflight contract. It is not ISO compliance, a source-level calculation, propagation-model validation, environmental exposure, ecological benefit or quiet-ship certification. Hash values are identifiers, not authentication. The contract does not replace ISO 17208-3, qualified measurement practitioners or validated propagation software.
