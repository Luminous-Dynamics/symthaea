# LQCD-001A — independent SU(3) Wilson-action oracle

## Exact scope

This evidence slice freezes the semantics of a tiny periodic four-dimensional pure-gauge SU(3) lattice independently of Symthaea's Rust implementation.

The checked-in Python oracle implements only:

- 4D periodic indexing;
- 3x3 complex matrix multiplication / Hermitian conjugation / determinant;
- deterministic diagonal SU(3) matrices;
- unitary + determinant-one validation;
- oriented plaquettes;
- Wilson pure-gauge action;
- average plaquette;
- temporal Polyakov loop;
- local gauge transformation `U_mu(x) -> G(x) U_mu(x) G^dagger(x+mu)`.

It intentionally imports no Symthaea crate or lattice-QCD implementation.

## Independent execution

The exact script was executed locally before commit with Python 3.

Command:

`python scripts/lqcd_wilson_su3_oracle.py`

Result:

`ok`

The deterministic fixture mode was also executed:

`python scripts/lqcd_wilson_su3_oracle.py --emit-fixture`

Key results for a `2 x 2 x 1 x 1` lattice at `beta = 6.0`:

| case | Wilson action | average plaquette |
| --- | ---: | ---: |
| identity links | 0.0 | 1.0 |
| one localized diagonal SU(3) link | 0.2783710710205054 | 0.9980668675623576 |
| local gauge transform of that field | 0.2783710710205074 | 0.9980668675623576 |

Identity-field temporal Polyakov loop: `1 + 0i`.

The action difference after the deterministic local gauge transformation is approximately `2.0e-15`, below the oracle's `1e-12` invariance threshold.

Periodic-boundary self-tests also verify wraparound independently in spatial and temporal directions.

## What this establishes

For these synthetic fixtures, the reference implementation has the intended algebraic semantics for Wilson plaquettes/action and local gauge invariance.

This can serve as an independent parity target for a later Rust production implementation.

## What this does **not** establish

This is **not** a lattice-QCD physics result and establishes none of the following:

- equilibrium gauge-field generation;
- detailed balance or ergodicity;
- HMC/heat-bath/Metropolis correctness;
- autocorrelation or effective sample size;
- continuum or infinite-volume limits;
- string tension;
- deconfinement temperature;
- glueball masses;
- QCD with dynamical fermions;
- any physical prediction.

Those require later LQCD gates under master issue #1621.
