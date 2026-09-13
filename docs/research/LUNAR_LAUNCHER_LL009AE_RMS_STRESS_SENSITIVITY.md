# LL-009AE — Product90 RMS stress-sensitivity frontier

LL-009AE asks a deliberately narrower question than calibration:

> How much can the exact Product90 far-field RMS model be stressed before the existing Site01 terrain-horizon and visibility conclusions materially change?

It does **not** infer a true calibration multiplier from LL-009AC, LL-009AD, the SDEM, or any other cross-method comparison.

## Intervention

For each declared stress factor `lambda`, AE changes exactly one numerical input to the inherited LL-009Y far-field solver:

`rms_m -> lambda * rms_m`.

The transformation occurs on the per-bin structured population **before** Y evaluates its geometry-aware Markov bounds. The original records are copied and remain unchanged.

Everything else is frozen:

- exact Product90 elevation/RMS/effective-resolution source bytes;
- exact admitted support population and azimuth bins;
- exact whole-sky and per-bin exceedance budgets;
- exact Q member ordering and same-realization observer heights;
- exact R spatial-support margins;
- exact Y near/far memberwise-max composition and empirical quantile;
- exact S target samples, time grid, apparent angular radii and LOS margin.

The checked-in Site01 ladder is:

`lambda = {0.75, 1.0, 1.25, 1.5, 2.0, 3.0}`.

`lambda < 1` is diagnostic and explicitly non-conservative.

## Reuse rather than reimplementation

AE imports the existing production materializers:

- `materialize_ll009y_memberwise_hybrid_horizon.py`;
- `generate_ll009s_semantic_visibility.py`.

For each lambda it temporarily wraps only Y's `solve_members_for_bin()` entry point, multiplies the copied `rms_m` field, and then delegates back to the exact inherited solver. The wrapper is restored after every run, including exception paths.

The counterfactual Y and S objects are numeric intermediates internal to AE. They are **not** promoted as new Y/S evidence receipts because their numeric values intentionally differ from the exact source-bound baseline semantics.

## Baseline identity theorem

`lambda = 1.0` is an executable identity test, not merely a spot check.

AE requires:

1. the regenerated Y K pack to be byte-for-byte identical to the supplied exact baseline Y pack;
2. the regenerated S visibility receipt to be byte-for-byte identical to the supplied exact baseline S receipt.

If either differs, no sensitivity frontier is emitted.

This catches source drift, solver drift, changed ordering, changed JSON serialization, target-sampling changes, changed margins and hidden state that would otherwise make the stress comparison non-like-for-like.

## Monotonicity theorem

For increasing lambda, AE requires every final conservative horizon bin to be nondecreasing within the configured numerical tolerance.

It then re-runs the exact S geometry. Metrics whose geometric direction is known are checked:

- `*_visibility_fraction` and `*_availability_fraction` must be nonincreasing;
- occlusion and outage duration metrics must be nondecreasing.

A larger terrain uncertainty envelope may make a conclusion worse or leave it unchanged. It may not improve geometric visibility.

This is an especially useful regression property because the final Y horizon is a memberwise near/far maximum followed by an empirical Q statistic; the check therefore covers the composed result rather than only the far-field solver in isolation.

## Decision-threshold brackets

AE can report the first declared lambda at which an independently supplied engineering threshold is crossed. The result is a **stress-ladder bracket**, not a calibration estimate.

The checked-in Site01 policy deliberately contains no engineering thresholds yet. A solar-availability or DTE-outage cutoff should only be added after an architecture or mission requirement establishes it independently of the observed AE curve.

This avoids choosing a threshold after seeing the answer.

## Probability semantics

AE preserves the earlier statistical boundary.

Product90 `ADJ_ERR` remains an RMS/second-moment model input. The inherited far theorem remains distribution-free Markov plus union bound. AE adds no Gaussian model, no pixel independence, no covariance model and no joint Q-clone × far-error probability distribution.

Scaling RMS by lambda is a counterfactual model stress. It does not mean:

- `lambda` is a confidence level;
- `lambda` is the true error multiplier;
- the SDEM calibrated Product90;
- the Q empirical quantile and far alpha can be multiplied into a joint confidence statement.

## Spatial-support boundary

LL-009R remains authoritative for unresolved physical terrain between represented raster support points. Stressing Product90 RMS changes the vertical error model at represented points; it does not create continuous terrain completeness.

LL-009AB/AC/AD remain complementary:

- AB asks what independent SfS reconstruction adds at native 5 m support;
- AC tests Product90 RMS at matched 80 m support;
- AD decomposes the matched discrepancy by terrain/product ancestry;
- AE asks whether architecture conclusions are robust over a declared family of RMS scales.

None substitutes for another.

## Local executed logic evidence

The exact production implementation committed in this branch has Git blob:

`9d41c893089c4a9cc581a02956abb4732fe74b4b`.

That byte-identical artifact passed local Python compilation and its self-test. The campaign verifies:

- valid/strict stress-ladder grammar with exact lambda=1 inclusion;
- horizon monotonicity rejection;
- visibility/outage monotonicity checks;
- deterministic first-threshold ladder bracketing;
- an adversarial structured-record intervention proving `[1, 2, 4] m` becomes `[1.5, 3, 6] m` before the inherited solver at `lambda=1.5`;
- baseline structured records remain unmodified;
- the temporary Y solver wrapper is restored after the intervention scope.

The self-test is logic evidence only. It does not constitute a real Site01 AE receipt.

## Real-data promotion boundary

A real AE receipt requires the exact real Q/V/W/X/R/L lineage, exact passing Y K pack, exact S config/O receipt/baseline S receipt, and all exact NASA raster bytes required by Y.

The real run must additionally pass byte-exact lambda=1 Y and S reproduction before any stressed result can be interpreted.

Until that run exists, no claim is made about where Site01 solar or DTE conclusions actually change on the declared stress ladder.
