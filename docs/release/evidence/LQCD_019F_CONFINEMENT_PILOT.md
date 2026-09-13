# LQCD-019F — contractible Wilson-loop / Creutz-ratio pilot

## Scope

Independent standard-library Python campaign-development evidence for pure SU(3) with the Wilson action. This is deliberately **not** a physical string-tension result.

The first isotropic `3^4` attempt exceeded the local execution ceiling before producing retained data. No result from that timed-out attempt is used. The frozen subject instead uses a non-degenerate anisotropic `3x2x2x3` torus, which preserves contractible `R,T in {1,2}` rectangles in spatial direction 0 and temporal direction 3 while reducing execution cost.

## Exact executed generation subject

- subject: `scripts/lqcd-019f-confinement-pilot.py`
- SHA-256: `7e54b7fac9ad2ea4878e03b78e8743fcfccbc31946535c0425a8217a9a455761`
- canonical result SHA-256: `89e1cb4f598b93518656f21969cac98e5ad715b96de629ee9692d72c24e94099`
- stdout SHA-256: `d3093e015976fa93c464871ce128ae84885085ef709bd467231e2a860b627d47`
- retained-history SHA-256: `82a3766d5d3f5c9abb471a5d15384f99763a6e3dfb89c901e67172a0894da573`

Campaign:
- dimensions `3x2x2x3`;
- beta `5.7`;
- independent cold + disordered starts;
- direct-staple Cabibbo-Marinari heat-bath + one overrelaxation sweep per cycle;
- burn-in `12` cycles;
- stride `1`;
- `8` retained measurements per chain;
- one spatial orientation only (`mu=0`) because only that spatial extent is 3;
- measured `W11`, `W21`, `W12`, `W22`, plaquette and `|P|`.

Pre-run direct-staple / five-probe force parity error:

`4.440892098500626e-16`.

## Frozen observable result

Pooled loop means:

- `W11 = 0.549860869756739`
- `W21 = 0.326609198843753`
- `W12 = 0.3252679346633963`
- `W22 = 0.11904407041905947`

The development Creutz estimator is formed from expectation-value estimates, not by averaging per-configuration logarithms:

`chi(2,2) = -log[ <W22><W11> / (<W21><W12>) ]`

Frozen pooled value:

`chi(2,2) = 0.4842545562936156`.

Per-chain values differ materially:

- cold: `0.41418563002664316`
- disordered: `0.5656727402620224`

Cross-chain diagnostics are also not adequate for promotion:

- plaquette max rank/folded R-hat `1.2333812753460875`;
- `|P|` max rank/folded R-hat `1.345097377827937`.

The individual Wilson-loop scalar diagnostics look numerically closer, but they do not override the broader ensemble failure.

## Exact joint jackknife replay

- subject: `scripts/lqcd-019f-creutz-jackknife.py`
- SHA-256: `15dc3a1b2b087601f3b8d5e6331d4ce0e487cf778606475584e32733f9991a64`
- canonical result SHA-256: `679ced05098d7f0f596e34eb88c714e6371eadf8e424b70959aa9d460a827ebe`
- stdout SHA-256: `35481ebf46727d5f327759032290c04913b80ac2a77b865d5b28455fd516d3f4`

The jackknife deletes complete chain-local blocks across the full correlated `(W11,W21,W12,W22)` vector and recomputes the nonlinear Creutz estimator inside every replicate.

Standard errors:

- block size 1: `0.07214355068976241` from 16 replicates;
- block size 2: `0.06393690677014327` from 8 replicates;
- block size 4: `0.09304428470122127` from only 4 replicates.

This does not show a stable uncertainty plateau. The largest admissible block has very little resampling support and a larger uncertainty than the two smaller block choices.

## External-reference boundary

Edwards, Heller and Klassen (`hep-lat/9711003`) are an appropriate later coarse-Wilson-action benchmark: they studied the SU(3) Wilson action for `5.54 <= beta <= 6.0`, including a `beta=5.7` ensemble on `16^3 x 32`, with several thousand effectively independent configurations, smeared Wilson loops and static-potential analysis. Their methodology and volume are not equivalent to this tiny anisotropic unsmeared pilot, so no direct string-tension comparison is claimed here.

## Scientific conclusion

LQCD-019F establishes only that the independently implemented HB+OR stack can generate positive contractible `1x1`, `2x1`, `1x2`, `2x2` Wilson-loop expectation estimates and a finite Creutz-ratio development observable on a non-degenerate lattice.

It **fails** the conditions for a physical confinement claim because:
- the two chain estimates disagree materially;
- plaquette and Polyakov-magnitude diagnostics are weak;
- the nonlinear uncertainty does not show block-size stability;
- the volume is tiny and anisotropic;
- only one spatial orientation is contractible at `R=2`;
- there is no finite-volume study;
- there is no smearing / static-potential plateau analysis;
- there is no external benchmark agreement;
- there is no continuum extrapolation.

The next confinement campaign should therefore be compiled and isotropic, with at least `3^4` only as a development floor and preferably substantially larger volumes, multiple independent chains, full spatial-orientation averaging, block-stable correlated uncertainty, and a preregistered external-reference comparison.
