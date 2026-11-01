# PPK2 metrology uncertainty and conservative energy comparison v0.1

Tracking: #9744  
Implementation: `symthaea-architecture-engineering::metrology_uncertainty`  
Depends on: #9740 / PR #9743 raw timestamp-aware trace receipts.

## Motivation

A low observed energy point estimate is not automatically a meaningful energy improvement. Instrument accuracy, calibration, current range, repeatability, trigger placement, board state and supply voltage can make the uncertainty wider than the observed difference between candidates.

The PPK2's published current accuracy is range-dependent. A manufacturer accuracy bound is not automatically a standard uncertainty. Do not convert it to a Gaussian standard deviation by dividing by a coverage factor unless an explicit distribution/coverage argument supports that conversion.

This module requires a declared *relative standard uncertainty* from a calibration/metrology budget with a basis and coverage factor. The certificate digest is bound into the manifest; its authenticity and validity must be checked by a separate trust/acquisition policy.

## Evidence model

### Metrology manifest

The content-addressed manifest binds:
- instrument model and serial;
- current measurement range;
- calibration certificate SHA-256;
- Type B relative standard uncertainty and its basis;
- coverage factor;
- replicate-protocol digest;
- explicit declaration that replicate segments are independent.

The `replicates_independent` declaration is a protocol assertion, not something proved by this module. The run design must justify independence (for example, reset and thermal/warm-up blocking) before enabling it.

### Type A and Type B terms

Type A repeatability uses the sample standard deviation (s) of segment-level energy values and estimates the standard uncertainty of their mean as:

`u_A = s / sqrt(n)`

Type B is supplied as a calibration-derived relative standard uncertainty (u_{B,rel}):

`u_B = |mean_energy| × u_{B,rel}`

The v0.1 independent-term approximation combines them by root-sum-square:

`u_c = sqrt(u_A² + u_B²)`

Expanded uncertainty:

`U = k × u_c`

The coverage factor (k), uncertainty distribution and coverage interpretation must be supplied by the metrology authority/protocol; v0.1 does not label every interval as a 95% confidence interval.

If calibration/systematic terms are correlated between candidate measurements, the RSS formula and independent-interval comparison may be conservative or inappropriate depending on covariance. v0.1 does not assume those terms cancel. A future paired-covariance model must bind the shared calibration/session basis and be reviewed before use.

## Conservative energy comparison

Given candidate and reference intervals ([L_C,U_C]) and ([L_R,U_R]):

- `CANDIDATE_STRICTLY_LOWER` only when (U_C < L_R);
- `CANDIDATE_STRICTLY_HIGHER` only when (L_C > U_R);
- otherwise `INDETERMINATE_INTERVALS_OVERLAP`.

An indeterminate result is not a tie and is not evidence of equivalence. It means the current evidence cannot support strict ordering. The search can acquire more independent replicates under a preregistered sequential plan, or retain candidates as incomparable.

Both the upstream trace receipt and the derived uncertainty receipt are revalidated before values are used. The PPK2 receipt itself can be deserialized and strictly validated, and the source-replay verifier can rebuild it from the exact CSV if that source is retained.

## Formula and units

Energy is measured in microjoules, relative standard uncertainty is dimensionless, and coverage factor is dimensionless. Type A, Type B, combined and expanded uncertainty are all in microjoules. The lower/upper bounds are in microjoules.

The input PPK2 trace receipt is produced by #9740 and includes exact supplied-CSV digest, voltage, sampling policy, segment inventory and timestamp-aware energy integrals.

## Authored tests

Synthetic tests cover:
- Type A/B combination and expected standard error;
- refusal of a single replicate;
- refusal of a protocol that does not support independent replicates;
- invalid Type B uncertainty and malformed calibration digest;
- tampered trace-receipt rejection;
- indeterminate overlapping intervals;
- strict win only for disjoint intervals;
- tampered uncertainty-receipt rejection;
- uncertainty-receipt serialization round-trip and unknown-field rejection;
- rehashed Type A term substitution rejection.

Tests are authored but have not been compiled or executed in this environment. No actual PPK2 certificate, raw trace or physical replicate has been evaluated.

## Search integration (next increment)

Feed uncertainty receipts into Pareto-regret / value-of-information scheduling:
- estimate the probability that another replicate changes candidate ordering or Pareto membership;
- charge target-device time, instrument occupancy, energy, money and verification effort in the existing vector resource ledger;
- promote only on a preregistered evidence rule;
- never collect extra samples selectively after inspecting favourable interim values unless the sequential stopping rule was frozen first;
- use blocked/randomized candidate order, reset boundaries and matched instrument/board conditions when a paired comparison is scientifically valid.

The module currently provides uncertainty calculation and pairwise interval decisions; it does not yet schedule acquisitions or model paired covariance.

## Claim ceiling

- A calibration digest is not certificate authentication.
- PPK2 datasheet accuracy is not a calibrated Type B standard uncertainty.
- A coverage interval is not automatically a confidence interval or a 95% interval.
- Replicate independence is a protocol assumption that must be justified.
- Disjoint intervals under this v0.1 budget do not prove arbitrary future-device or environmental superiority.
- No empirical uncertainty-reduction factor or SOTA improvement is claimed.

## Sources

- Nordic PPK2 user guide: https://docs.nordicsemi.com/r/bundle/ug_ppk2/page/ug/ppk/ppk_user_guide_intro.html
- PPK2 measurement accuracy: https://docs.nordicsemi.com/r/bundle/ug_ppk2/page/ug/ppk/ppk_measure_accuracy.html
- Apicella et al., measurement methodology (Computer Standards & Interfaces, 2026): https://doi.org/10.1016/j.csi.2025.104120
- Bartoli et al., energy/latency phase accounting for TinyML (IJCNN 2025): https://doi.org/10.1109/IJCNN64981.2025.11228997
