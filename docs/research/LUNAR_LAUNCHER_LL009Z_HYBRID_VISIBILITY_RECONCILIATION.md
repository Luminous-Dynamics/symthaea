# LL-009Z — hybrid scenario-sampled visibility reconciliation

LL-009Z is an additive semantic layer after LL-009S. It does **not** change LL-009S geometry, interpolation, target samples, visibility intervals, outage intervals, or any other numeric result.

Its purpose is to recognize when an LL-009S visibility receipt consumed the exact LL-009Y K-pack produced from the stronger terrain-evidence chain:

`Q finite Site01 scenarios → V Product 90 RMS semantics → W distribution-free familywise theorem → X geometry-aware far envelope → Y memberwise hybrid K horizon → S visibility geometry`.

## New evidence class

A passing Z receipt emits:

`hybrid_scenario_sampled_visibility`

This is stronger than generic descriptive geometry because the numeric horizon used by S is itself bound to the exact Q/V/W/X/R/L evidence lineage. It is deliberately weaker than either `risk_qualified_visibility` or `deterministic_visibility_bound`.

## Numerical immutability theorem

Z is not another visibility calculator.

It requires `S.k_horizon_pack_sha256 == SHA256(Y)` and copies the exact S `metrics` and `targets` into its output. It computes a canonical SHA-256 over those two structures and declares S the numerical authority. Z owns only the evidence classification.

If S's numbers change without a valid S self-hash, Z fails. If S points at different Y bytes, Z fails.

## Exact lineage checks

Z verifies:

- S and Y self-hashes;
- exact S → Y K-pack SHA-256;
- Y `statistical_horizon_binding.status = bound`;
- exact Y mode `empirical_q_of_scenario_parameterized_distribution_free_far_envelopes`;
- Y's memberwise `max`-before-quantile theorem;
- exact Q, V, W, X, R and L file hashes carried by Y;
- Q/V/W/X/R self-hashes;
- Q and V exact L-config binding;
- W exact V/L binding;
- X exact W/L binding;
- Q remains `empirical_ensemble`;
- V remains `rms_error`;
- S/Y study, site, frame and azimuth-bin agreement;
- Y/R layer-by-layer spatial-support agreement.

## Probability boundary

Z does not create a probability theorem that LL-009Y deliberately refused to claim.

The Q statistic remains a finite empirical scenario statistic. The Product 90 far-field alpha remains a separate distribution-free RMS familywise theorem. Z therefore records:

`joint_probability_interpretation = prohibited_without_joint_coupling_theorem`

It does not multiply the Q empirical quantile by `1-alpha`, condition the Product 90 RMS model on a Q member, assume independence, or introduce covariance/Gaussian structure.

## Spatial support

LL-009R remains authoritative for spatial support. Z copies the exact Y/R support classes into its receipt. In particular, current `sample_points_only`, `resolution_qualified`, or empirical support cannot be promoted to continuous physical-terrain completeness by Z.

## Capability result

A passing Z receipt may assert:

- `hybrid_scenario_sampled_visibility_eligible = true`.

It must keep false:

- `risk_qualified_visibility_eligible`;
- `deterministic_visibility_bound_eligible`;
- `joint_probability_visibility_eligible`.

Continuous physical-terrain completeness is separately reported from LL-009R and does not by itself strengthen the Q/RMS probability semantics.

## Local logic evidence

The exact committed script passed Python compilation and its deterministic synthetic self-test. The campaign verifies:

- deterministic replay;
- exact S metric/target preservation;
- S self-hash tamper rejection;
- exact S → Y byte binding;
- exact Y mode enforcement;
- exact Q/V/W/X/R/L lineage;
- current incomplete spatial-support disclosure;
- stronger risk/deterministic capability remains blocked.

This is logic evidence only. No real NASA Z receipt is claimed until the exact upstream NASA byte-acquisition and Q/V/W/X/Y/S execution chain exists.

## Non-claims

LL-009Z is not an RF link budget, solar-array delivered-power model, continuous-terrain theorem, site qualification, mission-safety decision, architecture authority, or operations authority.
