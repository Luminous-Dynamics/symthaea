# LQCD-021A — β=6.0 static-potential measurement design

This independently executed subject freezes the **measurement and operator design only** for a future Edwards–Heller–Klassen β=6.0 Wilson-action reproduction campaign. It intentionally does not freeze production burn-in, measurement stride, seed commitments, retained configuration count, final effective-potential plateau windows, or final `r`-fit ranges.

Exact executed subject SHA-256:

`fb8ea28b82af95707164d436cdf4bb23531f63ebc1dbbe859d85a914faf9b387`

Canonical design/result SHA-256:

`df92cdfbbf882a8f3c37eb911476eb961bbc7790d6d6adae60bccc1f8c999c72`

## Frozen benchmark and lattice target

- external benchmark: LQCD-020Q / PR #2488;
- external benchmark canonical extraction: `cc7a16813bd175b56cc9242425ec4c1f235a9f54483efe9dc173b87fbd3b28b2`;
- Wilson pure-gauge action;
- β = `6.0`;
- periodic `16^3 × 32` lattice;
- quoted comparison targets: `a sqrt(sigma)=0.2189(9)`, `r0/a=5.369(9)`, `r4/a=8.831(21)`, `r6/a=10.89(3)`.

## Frozen transition/operator intent

- sampler identity: `cm_heatbath_or_v1:force=staple:or_sweeps=3:max_attempts=256`;
- transition kernel revision: `56c115b46920fc8b34daa8897051cff59c61a0bf`;
- sampler stable-ID/trace revision: `ce6cb274e563289ae0001a49babba5f2025cb9ae`;
- spatial APE convention: `spatial_ape_ehk_polar_v1`;
- APE `alpha=0.7`, `iterations=19`, giving the EHK proxy `epsilon≈1/(4+alpha)=0.2127659574` and `epsilon*n≈4.04255319`;
- APE implementation revision: `1c7c039d4e9e90b394108254dc9220211eec9371`;
- bounded Wilson convention: `generalized_bresenham_cubic_ape_spatial_unsmeared_temporal_wilson_v1`;
- bounded Wilson implementation revision: `98a1d4d7874ccf9860d85651309f34eaf5e42506`;
- independent Bresenham geometry oracle: PR #2520, subject `c270fe43785b1c45cfd7e03acb4ac583782eb5e83193bf1fa977a6358242fdb9`, result `884142f2645067e312380be738b618e85bc8bb727cbf6a3b0ea675858852c285`.

## Frozen measurement geometry

The design measures 24 representative displacement vectors drawn from the four EHK base-vector families:

- `n(1,0,0)`, `n=1..7`;
- `n(1,1,0)`, `n=1..7`;
- `n(1,1,1)`, `n=1..7`;
- `n(2,1,0)`, `n=1..3`.

Every representative is averaged over its signed-permutation cubic orbit using the bounded generalized-Bresenham operator. Every spatial component is at most `7`, so the design stays strictly below the ambiguous half-box separation on a spatial extent of `16`. The largest Manhattan path is `21` links/orientation and the largest cubic orbit has `24` orientations.

Each representative retains Wilson loops for temporal extents `T=1..8`. This is measurement coverage, **not a frozen plateau choice**.

The largest Euclidean radius is `|r|=sqrt(147)=12.124...`, which extends beyond the quoted benchmark `r6/a=10.89` while preserving nonwrapping geometry.

## Frozen analysis implementations

- effective-potential analysis revision: `781b0818a9c5daf879ca1458cf032bc6f89f281e`;
- tree-level lattice Coulomb revision: `24f81937b9a7e5221051a79b86cf7b22495d6f16`;
- three-member correlated potential fit family revision: `ff4a5b2614c2cee78f51d80ae082d5a57957431d`;
- covariance-aware Sommer-scale revision: `ecd482f7ca250c938a5cff232b1b4081d2715b3a`;
- scale targets `c={1.65,4,6}` for `r0,r4,r6`.

## Deliberately unfrozen until a disjoint target-volume pilot

The following are **not authorized by this design** and must be frozen later from a pre-production target-volume pilot whose configurations are permanently excluded from the final benchmark sample:

- production burn-in;
- production measurement stride;
- final retained configuration count;
- seed commitments;
- primary/diagnostic `V_eff` plateau windows;
- final static-potential `r_min/r_max` fit ranges.

No final production data may be used to choose those values.

## Authority boundary

This is a reproducible measurement-design artifact, not a campaign execution manifest. It does not authorize Markov-chain production, prove equilibrium/topological mobility, select a plateau or fit range, reproduce the EHK benchmark, determine a string tension or Sommer scale, or establish finite-volume/continuum physics. Every referenced Rust subject still requires its own passed exact-head CI before the later production campaign can be authorized.
