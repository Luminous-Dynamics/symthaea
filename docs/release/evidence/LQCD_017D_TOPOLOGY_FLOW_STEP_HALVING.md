# LQCD-017D fixed-flow-time step-halving qualification

## Subject

`scripts/lqcd-topology-flow-step-halving.py`

Executed-subject SHA-256:

`95d9a3ccdb669d84d2b8204cc7a725dbee52f84f4c4cd8e477a7796f7a20b5a2`

The subject is Python-standard-library only and imports no Symthaea/Rust implementation code.

## Frozen experiment

The same explicit 40-link `2^4` SU(3) fixture used by the LQCD-017D topology oracle is evolved to the **same total flow time**

`t = 0.004`

with four Lie-Euler step sizes:

- `dt=0.002`, 2 steps;
- `dt=0.001`, 4 steps;
- `dt=0.0005`, 8 steps;
- `dt=0.00025`, 16 steps.

The flow generator is the independently derived analytic six-staple Wilson-action gradient already parity-checked against the finite-difference oracle.

## Executed result

```text
ok
total_flow_time=0.0040000000000000001
dt=0.002 steps=2 action=7.865198193066977 q=-0.0003964908549886643 action_drop=-0.35716299800162331
dt=0.001 steps=4 action=7.8661416702008197 q=-0.00039653225241561587 action_drop=-0.35621952086778058
dt=0.00050000000000000001 steps=8 action=7.8666108474186598 q=-0.00039655287463448922 action_drop=-0.35575034364994046
dt=0.00025000000000000001 steps=16 action=7.8668448008926157 q=-0.00039656316670853977 action_drop=-0.35551639017598458
action_step_halving_ratios=2.0109131141970018,2.0054071014532575
q_step_halving_ratios=2.007420866231691,2.0036929281405781
```

The complete stdout is preserved in `LQCD_017D_TOPOLOGY_FLOW_STEP_HALVING_RESULT.txt`.

## What this establishes

At fixed total flow time, successive differences in both the Wilson action and clover topological charge shrink by approximately a factor of two when `dt` is halved. This is the expected global first-order discretization behavior of the Lie-Euler integrator.

Every individual flow step in the frozen subject also lowers the Wilson action within the declared numerical tolerance.

This is stronger than merely showing that a smaller step produces a similar answer: the observed convergence order itself matches the algorithm's expected order on an independently executed fixture.

## What this does not establish

This result does **not** establish:

- a production flow step size;
- third-order or higher-order integration accuracy;
- `t0` or `w0`;
- a physical topological charge or susceptibility;
- adequate topological tunnelling;
- continuum scaling;
- that first-order Lie-Euler should be the production integrator.

The intended next gate is a separately implemented Lie-group third-order Runge-Kutta Wilson-flow integrator, checked against the same semantic Wilson-flow generator and required to exhibit its own higher-order fixed-flow-time convergence before promotion.
