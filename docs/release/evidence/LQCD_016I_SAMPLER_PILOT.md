# LQCD-016I — Independent Metropolis vs HB+OR tiny-lattice pilot

## Status

Qualification evidence only. **This is not a physical lattice-QCD result and does not establish equilibrium, continuum validity, topology sampling, or sampler superiority.**

## Independent subject

`scripts/lqcd-hbor-sampler-oracle.py`

The subject is standard-library Python and imports no Symthaea or Rust code. Before the run it self-checks its ChaCha8 implementation against the published all-zero 256-bit-key ChaCha8 reference block.

The subject independently implements:

- periodic `2 x 2 x 2 x 2` SU(3) Wilson lattice semantics;
- canonical touching-plaquette de-duplication;
- Cabibbo-Marinari random-walk Metropolis;
- five-probe local affine subgroup force reconstruction `T(a) = c + q.a`;
- Kennedy-Pendleton SU(2) conditional sampling;
- local-force orientation with explicit `q.a / |q| == b0` checks;
- deterministic equal-action subgroup overrelaxation with local trace-drift checks;
- distinct initialization and transition ChaCha8 streams;
- independent cold and disordered transition replicas;
- classical split-R-hat as a descriptive pilot diagnostic.

## Frozen pilot configuration

- lattice: `2 x 2 x 2 x 2`
- Wilson beta: `5.7`
- burn-in: `20` transition cycles
- retained measurement stride: `2` cycles
- retained measurements per chain: `10`
- initial states: cold identity vs independently disordered
- Metropolis proposal width: `0.5`
- HB+OR schedule: one heat-bath sweep + one overrelaxation sweep per cycle
- heat-bath local force: five-probe reference construction
- transition replicas: cold `0`, disordered `1`
- ensemble slot: `29`

One transition cycle is deliberately **not** assumed to have equal compute cost between samplers.

## Executed exact-subject output

The exact local subject that was subsequently checked into the branch executed successfully and printed:

```text
ok
metropolis
  cold_mean=0.5871001711606786
  disordered_mean=0.40110614655623167
  split_rhat=8.803643775228498
  cold_uniform_draws=30720
  disordered_uniform_draws=30720
  cold_scalar_attempts=0
  disordered_scalar_attempts=0
hbor
  cold_mean=0.5924613796816569
  disordered_mean=0.6002560138617226
  split_rhat=0.9757806716162685
  cold_uniform_draws=46992
  disordered_uniform_draws=46924
  cold_scalar_attempts=7908
  disordered_scalar_attempts=7891
```

The script freezes these values and fails if they drift beyond the declared numerical tolerances or if the uniform-draw counts change.

## Interpretation

The pilot provides a useful **algorithm-development signal**:

- under this deliberately short fixture, random-walk Metropolis retains a severe cold/disordered start separation;
- HB+1OR removes that obvious separation much more rapidly for the average plaquette;
- the HB+OR cold/disordered retained plaquette means are close on this fixture.

However, the HB+OR classical split-R-hat value below one is **not a convergence certificate**. Only ten retained points per chain are present, the diagnostic is the classical form rather than the full rank/folded analysis in LQCD-003, and the run does not assess topology.

It is therefore invalid to infer from this pilot that:

- the HB+OR chain is equilibrated;
- HB+OR has the correct stationary distribution in production;
- HB+OR is more compute-efficient;
- the topology mixes adequately;
- any physical observable is established.

## Next evidence gates

Before sampler promotion:

1. exact-head Rust CI for the shared force, composed HB+OR cycle, and sampler-neutral trace;
2. longer independent multi-chain campaigns with LQCD-003 rank-normalized/folded R-hat, autocorrelation and ESS;
3. wall-clock/environment-qualified ESS-per-compute comparison;
4. topology/slow-mode qualification from LQCD-017/#1743;
5. agreement between samplers on stationary plaquette, Polyakov-loop and Wilson-loop observables;
6. reproduction of trusted external pure-SU(3) reference data.

The random-walk Metropolis path remains the independent baseline even if HB+OR eventually proves more efficient.
