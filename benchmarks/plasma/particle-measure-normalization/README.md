# Particle-measure normalization audit — v0.1

This fixture studies support dependence in a per-sample min–max coordinate transform.

## Exact NEOPIC arXiv v1 transform

The pinned NEOPIC arXiv v1 source implements:

`z(x) = (x - min(X)) / (max(X) - min(X))`

It maps each sample's observed particle-position coordinates to [0, 1], independently for each spatial coordinate channel. The extrema are computed from positions only; the function does not consume particle weights.

Source pin:
- Repository: https://github.com/srikrrish/operator_learning
- Tag: `neopic-arxiv-v1`
- Annotated tag object: `8e872e5fa520843a9c9db4a68d43036a66355c19`
- Peeled commit: `2cf6a15e0ddb6c408a4ef6d162c31faf1deb9431`
- File: `operator_learning/data/pic_dataset.py`
- Blob SHA: `dcb046656830dd8c5707e98bdd1d611c8376b412`
- Function `normalize_per_sample`: lines 129–139; Torch equivalent 141–152
- Applied to coordinate channels at lines 415–423

## Case A: source-faithful unweighted particle sample

The base sample has five equally weighted positions [0.1, 0.3, 0.5, 0.7, 0.9], giving normalized positions [0, 0.25, 0.5, 0.75, 1]. Add a sixth, equally weighted position at x=0. The six normalized positions become [0, 1/9, 1/3, 5/9, 7/9, 1].

For the five original particles, the coordinate shifts are [1/9, 1/12, 1/18, 1/36, 0]. Maximum shift is 1/9; mean shift is 1/18. This is a source-faithful unweighted sample-count/support perturbation. The empirical measure also changes because the new point is an additional equally weighted sample; this case does not claim negligible physical-mass change.

## Case B: weighted-measure extension

The fixture also tests a weighted empirical-measure extension: add x=0 with weight 1e-12 while preserving essentially all original inlier mass. Since the position normalizer does not consume weights, it produces the same coordinate remapping as Case A even though the weighted measure changes negligibly.

This weighted extension is **not** the paper's native unweighted particle-input interface. It is included because future particle-measure systems may use unequal macro-particle weights or split/merge operations. It tests the general risk of a position-only support map that ignores the measure weights.

## Interpretation and limits

Both cases demonstrate that a sample-support change can move the normalized coordinates of all existing particles (except the upper endpoint), not just the new extreme point. This does not imply that NEOPIC's trained field operator necessarily produces an inaccurate output; no model is executed here.

The tagged implementation's degenerate-support fallback uses a denominator of one, producing zeros for a constant sample. The local Rust `MinMaxCoordinateMap` instead rejects degenerate support as UNKNOWN / unqualified. That is an intentional fail-closed policy, not an exact reproduction of this edge-case fallback.

Where the physical domain and origin are known, fixed-domain normalization is a direct control. For unknown/moving domains, support-aware or robust alternatives must retain explicit tail-coverage metadata: quantile clipping can discard real rare-tail information.

Use this fixture with [#13416](https://github.com/Luminous-Dynamics/symthaea/issues/13416) and the full NEOPIC reproduction [#12952](https://github.com/Luminous-Dynamics/symthaea/issues/12952).