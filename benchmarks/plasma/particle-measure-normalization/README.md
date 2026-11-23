# Particle-measure normalization audit — v0.1

This fixture isolates one effect in per-sample min–max coordinate normalization.

Given a particle cloud X, the transform maps its observed minimum and maximum to [-pi, pi]:

\`z(x) = -pi + 2*pi*(x-min(X))/(max(X)-min(X))\`.

The base cloud has five inlier positions on the known unit interval. The perturbed cloud preserves those inliers and adds one particle at x=0 with weight 1e-12. The empirical measure therefore changes by at most the tiny outlier mass, but the coordinate-map minimum changes from 0.1 to 0.0. As a result, all inlier normalized coordinates except the upper endpoint move.

The JSON fixture records exact expected values:
- maximum absolute inlier coordinate displacement: 0.6981317007977319 rad;
- mean absolute inlier coordinate displacement: 0.34906585039886584 rad.

This is a pure normalization witness. It does not run NEOPIC, does not measure a learned-field error, and does not establish that min–max normalization is universally inappropriate. It is an exact synthetic test for the claim that the coordinate transform can move all existing particles when a very low-weight outlier changes the bounding box.

The physical domain and origin are known in this fixture, so fixed-domain normalization is the direct control. For unknown or moving domains, test robust support maps only with explicit tail coverage / residual metadata: quantile clipping can remove genuine rare-tail information.

Use this fixture with [#13416](https://github.com/Luminous-Dynamics/symthaea/issues/13416) and the source-faithful NEOPIC reproduction [#12952](https://github.com/Luminous-Dynamics/symthaea/issues/12952).
