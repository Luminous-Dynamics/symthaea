# Particle-measure normalization and topology audit — v0.1

This fixture studies support dependence in the per-sample coordinate transform used by NEOPIC, while separating the coordinate normalization from the later Fourier transform and the physical boundary topology.

## Exact paper and code convention

The NEOPIC paper (§4.2, equation 13) writes the torus coordinate as

`zeta = -pi + 2*pi*xi`, with `zeta in [-pi, pi)`.

Appendix A.1 clarifies that its implementation uses

`xi = (x - min(X)) / (max(X) - min(X)) in [0,1]`

and evaluates Fourier characters `exp(2*pi*i*k*xi)`. These are equivalent periodic coordinates up to an origin shift. The learned linear lifting, however, is applied to `xi) itself rather than to a periodic function, which the paper notes can create a discontinuity when a particle crosses the periodic seam.

Pinned code source:
- Repository: https://github.com/srikrrish/operator_learning
- Tag: `neopic-arxiv-v1`
- Annotated tag object: `8e872e5fa520843a9c9db4a68d43036a66355c19`
- Peeled commit: `2cf6a15e0ddb6c408a4ef6d162c31faf1deb9431`
- `operator_learning/data/pic_dataset.py`, blob `dcb046656830dd8c5707e98bdd1d611c8376b412`
- `normalize_per_sample`: lines 129–139
- Position-channel application: lines 291–298

The helper takes unweighted particle-position arrays and maps each sample's minima/maxima to [0,1]. The paper reports equal particle charges in its experiments and discusses general positive weights for importance-sampled initial data. The coordinate-map witness below therefore separates the native equal-weight sampling case from a weighted-measure extension.

## Case A: source-faithful unweighted particle sample

The base sample has five equally weighted positions [0.1, 0.3, 0.5, 0.7, 0.9], giving normalized positions [0, 0.25, 0.5, 0.75, 1]. Add a sixth, equally weighted position at x=0. The six normalized positions become [0, 1/9, 1/3, 5/9, 7/9, 1].

For the five original particles, the coordinate shifts are [1/9, 1/12, 1/18, 1/36, 0]. Maximum shift is 1/9; mean shift is 1/18. This is a source-faithful unweighted sample-count/support perturbation. The empirical measure also changes because the new point is an additional equally weighted particle.

## Case B: weighted-measure extension

The fixture also tests a weighted empirical-measure extension: add x=0 with weight 1e-12 while preserving essentially all original inlier mass. Since the position normalizer does not consume particle weights, it produces the same coordinate remapping as Case A even though the weighted measure changes negligibly.

This weighted extension is not the paper's native equal-charge input experiment. It is included because later particle-measure systems may use unequal macro-particle weights, importance sampling or split/merge operations.

## Topology warning: normalization is not a periodic lift

The paper writes the same normalized torus coordinate as `zeta = -pi + 2*pi*xi`, and its Fourier characters are periodic in `xi). But Appendix A.1 states that the linear lifting acts directly on the normalized coordinate and is not periodic. Thus a particle crossing from `xi≈1` to `xi≈0` can experience a discontinuous change in lifted features even though the points are neighbours on the physical torus.

This motivates the separate topology-aware lifting programme [#13447](https://github.com/Luminous-Dynamics/symthaea/issues/13447): compare periodic trigonometric / character lifts against the source-faithful linear lift on periodic cohorts, while preserving non-periodic geometry on open-boundary cohorts.

## Interpretation and limits

The two cases demonstrate that a sample-support change can move normalized coordinates of all existing particles (except the upper endpoint), not just the new extreme point. They do not imply that the trained field operator necessarily predicts an inaccurate field; no model is executed here.

The source code's optional DSE Vandermonde helper scales coordinates for Fourier evaluation using an approximate 6.28 factor. That is a transform implementation detail, distinct from the [0,1] coordinates supplied by the data preprocessor and the periodic coordinate convention stated in the paper. The current fixture targets the normalized coordinates supplied to the learned lifting.

The tagged implementation uses a denominator-one fallback for degenerate support, mapping a constant sample to zeros. The local Rust `MinMaxCoordinateMap` instead rejects degenerate support as UNKNOWN / unqualified. That is an intentional fail-closed policy, not an exact reproduction of this edge-case fallback.

Use this fixture with [#13416](https://github.com/Luminous-Dynamics/symthaea/issues/13416) and the full NEOPIC reproduction [#12952](https://github.com/Luminous-Dynamics/symthaea/issues/12952).