# LQCD-016E independent SU(2) heat-bath density oracle

## Executed subject

`scripts/lqcd-su2-heatbath-density-oracle.py --self-test`

The standard-library Python oracle was executed independently before commit. It imports no Symthaea/Rust code and deliberately does **not** implement Kennedy-Pendleton; it uses a generic exact rejection sampler so a later Kennedy-Pendleton implementation can be tested against a different algorithm.

Local executed copy SHA-256:

`a072e0600e968435037a6a8d106befa32529ed542a5666e6d272184a16f1cd41`

## Target distribution

For an SU(2) heat-bath update with effective coupling `alpha >= 0`, the scalar quaternion component is distributed as

`p(a0) da0 proportional to sqrt(1-a0^2) exp(alpha a0) da0`, `a0 in [-1,1]`.

Conditional on `a0`, the three-vector direction is uniform on `S^2`, giving a unit quaternion `(a0,a1,a2,a3)`.

The oracle samples `a0` by proposing uniformly on `[-1,1]` and accepting with

`sqrt(1-a0^2) exp(alpha (a0-1))`,

which is bounded by one and has exactly the desired density. Deterministic midpoint quadrature provides an independent numerical reference for the first two moments and a 20-bin marginal shape.

## Executed results

```text
ok
haar_reference_mean=-3.1110907490364605e-17
haar_reference_second=0.25000000260027166
alpha_1p5_reference_mean=0.34414401056444055
alpha_1p5_sample_mean=0.34345477101209826
alpha_1p5_mean_z=0.49535862109518686
alpha_1p5_reference_second=0.31171199015127166
alpha_1p5_sample_second=0.31155920890741229
alpha_1p5_second_z_bound=0.10431972842212162
alpha_1p5_max_bin_z=1.6529610897272149
alpha_1p5_mean_attempts=4.3589200000000003
alpha_1p5_vector_means=-0.0012015444636992573,0.0019773974985404638,-0.0026008649208549405
alpha_1p5_max_norm_error=5.5511151231257827e-16
alpha_5p0_reference_mean=0.71934058877995022
alpha_5p0_sample_mean=0.71891784449475837
alpha_5p0_mean_z=0.59169431276456352
alpha_5p0_reference_second=0.56839566258899776
alpha_5p0_sample_second=0.56788874779420317
alpha_5p0_second_z_bound=0.32359779014441137
alpha_5p0_max_bin_z=1.4979872904016414
alpha_5p0_mean_attempts=19.399529999999999
alpha_5p0_vector_means=0.00098296164932483812,-0.00016657151729614409,-0.00054757811230596678
alpha_5p0_max_norm_error=4.4408920985006262e-16
```

`100,000` accepted samples were used per nonzero-alpha fixture.

## Semantics established

- the scalar conditional density used by the SU(2) heat-bath program is explicit and independently testable;
- the Haar-measure scalar limit at `alpha=0` reproduces `E[a0]=0` and `E[a0^2]=1/4` to quadrature precision;
- generic rejection samples agree with numerical quadrature for the mean, second moment, and binned marginal shape at moderate and stronger effective coupling;
- conditional vector directions are consistent with isotropy on the frozen fixtures;
- all produced quaternions remain unit length to floating-point precision.

## Non-claims

This is a distribution oracle, not a production sampler. SplitMix64 is used only to freeze a deterministic independent qualification fixture. The result does not qualify Kennedy-Pendleton, Cabibbo-Marinari SU(3) composition, an equilibrium lattice ensemble, thermalization, autocorrelation, topology, continuum scaling, or any physical observable.

## References

Kennedy and Pendleton (Phys. Lett. B 156, 393-399, 1985) introduced an efficient sampler for this SU(2) heat-bath distribution. Cabibbo and Marinari use SU(2) subgroup updates to construct the standard SU(3) pseudo-heatbath strategy. The next Symthaea gate should require a Kennedy-Pendleton implementation to match this distribution oracle before embedding it into SU(3).
