# Particle-measure normalization audit — v0.1

This fixture isolates one effect in per-sample min–max coordinate normalization.

Given a particle cloud X, the transform maps its observed minimum and maximum to [0, 1]:

`z(x) = (x-min(X))/(max(X)-min(X))`.

The base cloud has five inlier positions on the known unit interval. The perturbed cloud preserves those inliers and adds one particle at x=0 with weight 1e-12. The empirical measure therefore changes by at most the tiny outlier mass, but the coordinate-map minimum changes from 0.1 to 0.0. As a result, all inlier normalized coordinates except the upper endpoint move.

The JSON fixture records exact expected values:
- maximum absolute inlier coordinate displacement: 1/9 of the normalized interval;
- mean absolute inlier coordinate displacement: 1/18 of the normalized interval.

This is a pure synthetic normalization witness using the exact affine [0,1] sample transform from the pinned NEOPIC arXiv v1 tag. It does not run NEOPIC or measure a learned-field error, and it does not establish that the published field operator fails. It isolates the claim that the coordinate transform can move all existing particles when a very low-weight outlier changes the bounding box.

The physical domain and origin are known in this fixture, so fixed-domain normalization is the direct control. For unknown or moving domains, test robust support maps only with explicit tail coverage / residual metadata: quantile clipping can remove genuine rare-tail information.


## Exact source anchor

This fixture now uses the actual NEOPIC arXiv v1 per-sample min–max map from:
- repository: https://github.com/srikrrish/operator_learning
- tag: `neopic-arxiv-v1`
- annotated tag object: `8e872e5fa520843a9c9db4a68d43036a66355c19`
- peeled source commit: `2cf6a15e0ddb6c408a4ef6d162c31faf1deb9431`
- source file: `operator_learning/data/pic_dataset.py`, blob `dcb046656830dd8c5707e98bdd1d611c8376b412`
- `normalize_per_sample`: lines 129–139 (and Torch equivalent 141–152)
- application to each particle-coordinate channel: lines 415–423.

The tagged code independently normalizes each sample to [0,1] using the unweighted minimum and maximum over the particle-position samples; the same code applies it to each spatial coordinate channel. The fixture is therefore source-faithful to the transform, but remains a synthetic one-coordinate example—not a NEOPIC field-output or kinetic simulation reproduction.

For the five inliers [0.1,0.3,0.5,0.7,0.9], normalized values change from [0,0.25,0.5,0.75,1] to [1/9,1/3,5/9,7/9,1] after adding x=0 with weight 1e-12. Maximum inlier shift is 1/9 and mean shift is 1/18 of the normalized interval.

Use this fixture with [#13416](https://github.com/Luminous-Dynamics/symthaea/issues/13416) and the source-faithful NEOPIC reproduction [#12952](https://github.com/Luminous-Dynamics/symthaea/issues/12952).
