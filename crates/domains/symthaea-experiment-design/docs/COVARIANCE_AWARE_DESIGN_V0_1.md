# Covariance-aware experiment design v0.1 — superseded

This initial draft described a caller-supplied covariance matrix for every unordered pair of rival hypotheses. A later mathematical review found that a double-centering "matrix variogram" consistency test is not valid for arbitrary vector-valued outputs: it can reject legitimate joint models whose cross-observable blocks contain admissible antisymmetric components.

The package now uses a stronger v0.2 contract: one PSD **joint covariance of the stacked hypothesis-output vector**, from which all pairwise difference covariances are derived. This guarantees common-model coherence without relying on the invalid pairwise matrix completion test.

Use the current specification instead:

- [Joint-covariance experiment design v0.2](COVARIANCE_AWARE_DESIGN_V0_2.md)
- [Joint covariance JSON schema](../fixtures/covariance-design.schema.json)
- [Synthetic redundancy fixture](../fixtures/covariance-redundancy-v0.1.json)

The original pairwise-covariance implementation must not be used for vector-output experiments. The v0.2 code retains the singular-support quarantine, PSD validation, digest binding, resource caps, covariance-aware Mahalanobis screening and exact duplicate-observable controls, but changes the input model and covariance validation boundary.

This history is preserved so that the earlier contract is not mistaken for the current implementation.
