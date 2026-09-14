# LQCD-021F — independent target-isolated synthetic analysis oracle

Independent standard-library execution for the statistical-analysis and target-isolation semantics tracked by #2715. The subject is stacked on the frozen #2535 preproduction authority protocol and imports no Symthaea/Rust implementation.

Exact executed subject SHA-256:

`48afb52c9cce92044c7d8638093787af20128df3b1e3996b1a5fe9bfe2c615b4`

Canonical result SHA-256:

`e60c557005a81407cf93390f76393c8ba275b26583c4a0aa5cf9e500d4a13cf4`

Sealed synthetic-analysis digest:

`a6e9235bd001fe4b90d736ec6e502c0d92eb7e89dbae5b57618077c8c94fe3b7`

## Synthetic theorem

The fixture contains 64 configurations with two orientations per `(r,T)`. Orientations are averaged inside each configuration before any resampling. Eight contiguous blocks of eight configurations are the jackknife units. A deliberately slow deterministic mode has lag-1 correlation `0.858921312009`, so the subject is not an IID toy.

Wilson means contain explicit excited-state contamination. The frozen late effective-potential window `T=5,6,7` recovers injected `sigma=0.052` as `0.052065535323`, while an early `T=1,2,3` window yields `0.053791363001`; the oracle requires the early-window bias to be detectably larger.

## Correlated potential analysis

Six frozen synthetic displacement points feed a fully correlated generalized-least-squares fit. The independent oracle executes all three frozen families:

- `free_v0_sigma_e_l_v1`;
- `fixed_e_pi_over_12_free_l_v1`;
- `fixed_e_pi_over_12_l0_v1`.

The injected truth deliberately has nonzero `l=0.035`. The first two families recover it, while the forced-`l=0` family produces a very large mismatch (`chi2 ≈ 2297.59`), proving the model-family sensitivity test is active rather than decorative.

The primary synthetic family is `fixed_e_pi_over_12_free_l_v1`. It produces:

- `a*sqrt(sigma)=0.228178735476`;
- `r0=5.163580437350` with block-jackknife SE `0.001017988705`;
- `r4=8.473369264395` with SE `0.001670506407`;
- `r6=10.498148088743` with SE `0.002069687169`.

The injected Sommer truths are `5.166833229285`, `8.478707054236`, and `10.504761385824`; the oracle requires each recovered value to lie within `0.01` of its independent injected truth.

## Target isolation

The sealed result is produced before any benchmark object is supplied. Two deliberately different synthetic benchmark objects are then compared downstream. Both comparison receipts bind the exact same sealed digest, proving benchmark replacement cannot alter the already-produced estimator output.

No EHK benchmark central values or uncertainties are inputs to this oracle.

A malformed three-orientation record is rejected rather than treated as additional Monte Carlo evidence.

## Scientific boundary

This subject qualifies synthetic analysis semantics, configuration-block resampling, correlated fitting, nonlinear Sommer propagation, model-family sensitivity, excited-state detectability, and target isolation. It does not establish correctness of a real beta=6.0 ensemble, production Rust analysis parity, or agreement with EHK.
