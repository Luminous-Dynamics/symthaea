# LL-009AF — Single-scan RMS stress campaign with intervention witnesses

LL-009AF hardens LL-009AE without changing its scientific question:

> How sensitive are the existing Site01 terrain-horizon and visibility conclusions to a declared multiplicative stress on the Product90 far-field RMS model?

AF preserves AE's semantics class:

`model_conditional_rms_stress_sensitivity_frontier`.

It does **not** infer a calibration multiplier, add a distributional model, or strengthen the probability interpretation.

## Why AF exists

AE V1 is deliberately simple: for every declared `lambda`, it reruns the exact LL-009Y materializer and intercepts only the far RMS solver input. That makes the intervention easy to audit, but a real NASA campaign would rescan the same Product90 raster population once per stress point.

AF removes that redundant I/O while adding stronger provenance.

The governing execution theorem is:

1. execute the exact LL-009Y materializer once at `lambda = 1`;
2. intercept every exact per-bin admitted record population immediately before `Y.solve_members_for_bin()`;
3. copy those exact structured records to AF scratch storage without mutation;
4. require the resulting Y pack to reproduce the supplied baseline Y pack byte-for-byte;
5. verify each captured bin's count and complete structured-record SHA-256 against the exact LL-009X input-population digest;
6. for each non-baseline `lambda`, create one ephemeral in-memory copy of one captured bin;
7. change only `rms_m -> float64(lambda * rms_m)`;
8. call the unchanged Y memberwise geometry-aware Markov solver once;
9. discard the stressed copy;
10. reconstruct the counterfactual Y pack from the exact baseline template, then run the exact LL-009S visibility engine.

The source elevation, Product90 RMS and effective-resolution rasters are therefore scanned only by the one exact baseline Y execution.

## Baseline identity

`lambda = 1` is not recomputed through an alternative AF path.

The one real Y scan is itself the baseline Y computation. AF captures records through a non-mutating wrapper around `Y.solve_members_for_bin()`, delegates immediately to the original solver, and then requires:

- exact baseline Y bytes;
- exact baseline S bytes.

This is stronger than comparing only horizon arrays.

It proves the capture layer did not change source admission, record order, solver behavior, serialization, visibility geometry or any other baseline state.

## Exact X population binding

Y already requires each reconstructed far bin to reproduce LL-009X's:

- admitted pixel count;
- `input_population_digest_sha256`.

AF checks those identities again on the captured post-admission records.

The counterfactual experiment therefore starts from the exact same structured population that X and Y already qualified.

## Intervention witness

For each ordered `(lambda, bin)` pair AF records:

- admitted record count;
- SHA-256 over every non-RMS geometry/identity field:
  - `c`;
  - `radius_m`;
  - `row`;
  - `col`;
  - `nominal_elevation_deg`;
- baseline RMS byte digest;
- stressed RMS byte digest;
- maximum absolute error versus exact binary64 `lambda * baseline_rms`;
- baseline RMS squared L2 sum;
- stressed RMS squared L2 sum;
- expected `lambda^2 * baseline_l2_sq`;
- L2 identity error.

The non-RMS digest must be identical before and after the intervention.

Per-element stressed RMS must be bit-identical to the chosen little-endian IEEE-754 binary64 multiplication result.

AF also rechecks the global identity across all bins:

`sum(stressed_rms^2) ~= lambda^2 * sum(baseline_rms^2)`

under the explicitly declared numerical tolerance.

The complete ordered witness sequence lives in a separate immutable manifest. The scientific AF receipt commits both to the manifest-body self-hash and to the exact serialized manifest-file SHA-256.

No modified NASA raster is ever authored.

## Solver-call completeness

AF treats missing or duplicated solver execution as evidence corruption.

For each declared stress point and every exact azimuth bin there must be exactly one inherited Y solver call.

The baseline `lambda=1` calls are the solver calls made by the one exact Y scan itself. Non-baseline lambdas use the captured population.

## Reconstructing counterfactual Y

AF does not invent a new terrain-horizon schema.

For every non-baseline lambda it deep-copies the exact baseline Y pack and changes only the fields that the RMS perturbation can legitimately affect:

- per-member far scenario horizon;
- per-member far Markov bound;
- per-member full near/far maximum;
- per-bin winning branch;
- per-bin empirical summary;
- selected K horizon;
- memberwise solver digest;
- final receipt self-hash.

All source, frame, epoch, site, Q, V, W, X, R and L lineage remains inherited from the exact baseline Y evidence.

These counterfactual Y objects remain internal AF numerical intermediates. They are not promoted as independent Y evidence receipts.

## AE equivalence gate

AF does not require a full real-data AE rerun on every production execution, because that would destroy the single-scan performance gain.

Instead the implementation carries a direct synthetic equivalence gate.

For a structured-record adversarial case it compares:

- AE's exact `_scaled_solver()` path;
- AF's explicit ephemeral-copy intervention;

using the unchanged `Y.solve_members_for_bin()` implementation at:

- `lambda < 1`;
- `lambda = 1`;
- multiple `lambda > 1`.

The synthetic composition includes:

- a far-dominant bin that changes under RMS stress;
- a near-dominant bin that must remain unchanged;
- a visibility threshold chosen between baseline and high-stress far horizons.

The exact LL-009S engine is then run on matched AE/AF synthetic K vectors and the metric payloads must agree while the declared threshold crossing is preserved.

That gate qualifies implementation equivalence. Every real run still independently requires byte-exact lambda=1 Y/S identity.

## Monotonicity

AF reuses AE's complete frontier checks.

As `lambda` increases:

- every final conservative Y horizon bin must be nondecreasing within the declared tolerance;
- visibility/availability fractions may not improve;
- occlusion/outage durations may not improve.

The check is applied after memberwise near/far composition and the finite-Q empirical statistic, not merely to the far solver in isolation.

## Output separation

AF emits three products:

1. **scientific frontier receipt** — immutable, self-hashed, contains lineage, execution theorem, baseline identity and final horizon/visibility frontier;
2. **intervention witness manifest** — immutable, commits every lambda/bin admitted-record intervention;
3. **performance diagnostic** — replaceable and explicitly non-authoritative.

The performance diagnostic reports:

- source-raster pass count;
- baseline scan/Y wall time;
- stress-solver wall time;
- maximum captured bin bytes;
- maximum ephemeral stressed-copy bytes;
- Y member-batch and pixel-chunk sizes.

Wall-clock measurements never affect scientific authority or the AF receipt hash.

## Probability boundary

Nothing in AF changes the statistical interpretation established by V/W/X/Y:

- Product90 `ADJ_ERR` remains an RMS/second-moment model input;
- the far theorem remains distribution-free Markov plus union bound;
- Q remains a finite published empirical clone ensemble;
- no Q × far joint probability distribution is created;
- no Gaussian assumption is added;
- no pixel independence or covariance model is added;
- `lambda` remains a declared sensitivity coordinate, not a confidence level or calibration estimate.

## Spatial boundary

LL-009R remains authoritative for unresolved physical terrain between represented support points.

AF improves counterfactual execution and intervention provenance. It does not turn sampled terrain support into a continuous hard physical bound.

## Promotion boundary

A real Site01 AF receipt requires all exact real inputs required by AE/Y/S plus the NASA raster bytes.

Until that real execution exists, AF is implementation and synthetic-equivalence evidence only. It makes no claim about where Site01 solar or DTE performance actually changes on the stress ladder.
