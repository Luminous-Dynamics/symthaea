# LQCD-020O — independent correlated lattice-potential fit-family oracle

Authority: independent standard-library Python numerical oracle. No Symthaea/Rust implementation is imported.

Exact executed subject SHA-256: `f436bdea286bf11d4ec5ea6eee870958c8e5ade00c1bac7a121a75cb92e32ae9`

Frozen stdout SHA-256: `df3e7932f14c23fc29f587882f304a098db05541e4aed6309eddf402637a095e`

Canonical result SHA-256: `6457691fa9cfabc6273870191540acefe5392f73bfcaaa3f962e026ea070da3d`

## Theorem exercised

The same correlated potential data are fit under three separately declared long-distance/static-potential models:

1. free four-parameter lattice-artifact model
   `V(r) = V0 + sigma*r - e*[1/r]_lat + l*([1/r]_lat - 1/r)`;
2. three-parameter model with `e = pi/12` fixed and `l` free;
3. two-parameter long-distance model with `e = pi/12` and `l = 0` fixed.

No model chooser exists. The purpose is to expose model dependence and sigma stability under explicitly declared constraints.

The frozen fixture injects `(V0, sigma, e, l) = (0.7, 0.18, 0.25, 0.04)` over six on/off-axis vectors using the LQCD-020G lattice-Coulomb values and a full correlated covariance matrix.

Frozen sigma estimates:

- free 4-parameter: `0.1847670004239007`, chi2/dof `0.13566139440388655`;
- fixed-e 3-parameter: `0.17693500020385283`, chi2/dof `0.6665391335487374`;
- fixed-e, l=0 2-parameter: `0.17607757425274784`, chi2/dof `1.1223735883704522`.

All remain within `0.006` of the injected sigma, while the most constrained model shows the expected degradation in correlated fit quality.

## Scientific boundary

This is algebra/statistics qualification only. It does not authorize automatic model selection, choose a physical r-range, establish equilibrium, determine a physical string tension, set the lattice spacing, or establish a continuum result. A real campaign must preregister the fit-family members and r-ranges before final analysis and preserve full covariance plus lattice-Coulomb provenance.
