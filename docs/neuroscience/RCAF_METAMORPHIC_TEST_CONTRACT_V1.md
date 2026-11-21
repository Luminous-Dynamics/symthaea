# RCAF Metamorphic Test Contract V1

**Status:** Implemented and focused-workflow validated on exact head `24c481252b64b8033b853408124b99a0321d6bd6`; broad repository CI and biological validation are not part of this result.
**Date:** 2026-11-21
**Scope:** Matrix comparison and edge-transition behavior only. No biological inference.

## Why metamorphic tests

Known-truth simulation measures whether an analysis recovers a declared signal under selected data-generating conditions. It does not, by itself, show that the analysis respects its own mathematical contract when inputs are represented differently.

Metamorphic tests provide a second axis of qualification: apply a transformation with a declared mathematical relation to the input, then verify that the output is invariant or changes in a predicted way. This can expose hidden dependence on ROI ordering, argument order, or accidental sign erasure before any empirical analysis is attempted.

The contract is transformation-specific. It does not assume that all preprocessing transformations should preserve connectivity or that every estimator is scale, offset, or polarity invariant.

## Implemented properties

### 1. Joint ROI permutation invariance

If rows, columns and ROI labels are permuted together in both matrices, the labeled comparison should return the same similarity. This verifies that edge enumeration depends on the represented labeled graph, not the arbitrary array index used to store it.

The test uses a nontrivial four-ROI permutation and checks all four RCAF weighted-Jaccard modes: absolute magnitude, shared positive edges, shared negative edges, and signed shift.

A permutation that is not accompanied by the same label ordering should still fail closed with the existing ROI-order mismatch error. The new test does not relax that guard.

### 2. Symmetric argument exchange and identity

For the matrix similarity definitions, exchanging left and right operands should preserve the score. Comparing a matrix with itself should return one, provided the mode has at least one qualifying edge.

These are exact properties of the symmetric weighted-Jaccard construction, not empirical claims. The underlying nonnegative weighted-Jaccard vector kernel is also tested for invariance to a shared positive scaling of both vectors and to a joint edge permutation.

### 3. Global polarity inversion: invariant versus non-invariant behavior

When all off-diagonal correlations in both matrices are sign-inverted:
- Absolute WJ should remain invariant because it discards edge polarity by definition.
- Signed-shift WJ should generally change because it maps each edge from r + 1 to 1 - r, preserving polarity information in a different geometry.

The test explicitly checks that the absolute score is unchanged and that signed-shift score changes on a fixture selected to demonstrate the distinction. This prevents an implementation from silently treating signed and unsigned comparisons as interchangeable.

### 4. Edge-transition reversal

Swapping the before/after matrices should preserve total polarity-reversal count and total sign-state-change count, while directional transitions swap:
- positive→negative ↔ negative→positive;
- positive→neutral ↔ neutral→positive;
- negative→neutral ↔ neutral→negative.

This ensures that symmetric counts remain symmetric without losing the direction of individual transitions.

## Exact-head validation record

RCAF focused workflow [44950930997](https://github.com/Luminous-Dynamics/symthaea/actions/runs/44950930997) passed on source commit `24c481252b64b8033b853408124b99a0321d6bd6`: 36/36 Rust tests, rustfmt, Clippy, report/schema checks, EIG and multiverse checks, all five synthetic simulation/report stages, content-addressed receipt generation and artifact upload.

Artifact SHA-256: `005a984e00220474e98414cdbb50d9a25a5492bc8901b47cd3e1828c9d61d0e4`.
Receipt source-tree SHA-256: `0581db6a1f6e66501197ffc516492dd7c5c7ee5a1760a8fe620d87d4bb5d51f2`.

This is a focused RCAF research-workflow pass only. Broad repository CI and PR Governance were skipped. The workflow runs the entire focused simulation stack as an integration check; the metamorphic contract itself is limited to the Rust function fixtures described above.
## Not assumed or currently tested

- Global gain/offset invariance is endpoint-specific and is not asserted for correlation matrices in general.
- Resampling, duration changes, filtering, censoring, nuisance regression and site harmonization can alter the estimand or its uncertainty; they require separate intervention contracts.
- A permutation of input arrays without relabeling is not an invariance: it changes the meaning of ROI indices and should be rejected or explicitly remapped.
- Passing these tests validates only the tested Rust function behavior and fixtures. It does not establish that upstream connectivity construction, preprocessing, atlas registration or biological interpretation is valid.

## Next metamorphic expansion

The next tests should add:
1. Participant-record reorder invariance for point summaries while keeping bootstrap randomization/CI semantics explicit.
2. Independent ROI-label mismatch negative controls for comparison and edge transitions.
3. Connectivity constructor-specific transformations on synthetic time series, including rank-preserving monotone transforms for Spearman and controlled gain shifts for Pearson/covariance.
4. Serialization/replay metamorphics: same frozen data/config/seed/source should reproduce identical report bytes and receipt digests.
5. Explicit non-invariance tests for transformations that legitimately change an estimand, such as time censoring or nonstationary filtering.

## References within CETB/RCAF

- RCAF Counterfactual Generalization Benchmark V1.
- RCAF Adversarial Instrument Calibration Protocol V1.
- RCAF direction-aware and signal-plus-nuisance calibration addendum.