# Superconductivity Phase-Network Benchmark

**Contract:** SC-PHASE-NETWORK-0.1  
**Status:** research specification; no simulations or material measurements are claimed  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/12755

## Scientific question

At fixed pairing amplitude, node count, physical dimension and total phase-coupling budget, can network topology change the emergence and robustness of global phase coherence?

This is a test of an effective collective-phase model. It is not a microscopic theory of pair formation, and a graph metric is not a superconductivity observable.

## Model definition

The first qualified model is a classical phase-only XY/Josephson network:

\[
H(\{\theta\}) = -\sum_{(i,j)\in E} J_{ij}
\cos(\theta_i-\theta_j-A_{ij}),
\qquad J_{ij}\ge 0.
\]

- \(\theta_i\) is a compact phase at node \(i\).
- \(J_{ij}\) is a declared Josephson/phase-coupling energy.
- \(A_{ij}\) is a gauge-link phase. The zero-field baseline has \(A_{ij}=0\); nonzero loop flux/frustration must be explicit and gauge-consistent.
- The phase-only model holds the local pair amplitude fixed. It cannot claim to calculate a pairing gap, pair lifetime, microscopic \(T_c\), or material stability.
- Set \(k_B=1\) for dimensionless simulations and publish the conversion convention. Do not map an effective-model temperature to kelvin without an independently calibrated microscopic/material mapping.

### Resource normalization

For every graph report:

\[
B_J = \frac{1}{N}\sum_{e\in E}J_e
\]

as the extensive coupling budget per node, along with node count, edge count, mean degree, coupling histogram, spatial embedding and any nonlocal-edge cost.

Use two comparison tiers:

1. **Topology-isolated tier:** match node count, dimension, degree sequence, coupling multiset, total coupling budget and disorder realization as closely as graph rewiring permits.
2. **Architecture-cost tier:** permit degree/connectivity changes, but constrain the total coupling budget and separately account for edge length, coordination, interface count and nonlocal-coupling cost.

A dense graph must not win merely because it receives more total coupling energy. Do not treat equal mean coupling as proof that two graphs have equal spectral or energetic resources.

## Analytic and numerical reference controls

All constants below are reference checks for specified textbook models, not predictions for a real superconductor.

| Control | Reference behaviour | Required gate |
|---|---|---|
| 1D finite-range XY chain | No finite-temperature long-range phase order in the thermodynamic limit | A finite-size crossover must not be reported as a nonzero thermodynamic transition |
| 2D square-lattice classical XY, nearest-neighbour \(J=1\) | Best precision reference located: \(T_{BKT}/J=0.892943(2)\) from tensor-network level spectroscopy; HOTRG reports \(0.89290(5)\) | Reproduce published precision; compare TNR/level spectroscopy, HOTRG and Monte Carlo, reporting finite-bond-dimension and finite-size systematics |
| 3D simple-cubic classical XY, nearest-neighbour \(J=1\) | \(K_c=J/(k_BT_c)=0.45420(2)\) (Gottlob & Hasenbusch, 1993), equivalently \(k_BT_c/J\approx2.2017\) | Recover critical coupling with finite-size scaling and controlled corrections to scaling |

The 2D helicity-modulus jump is \(\Upsilon(T_{BKT}^{-})=2k_BT_{BKT}/\pi\), using the conventional normalization. The implementation must document the twist direction and geometric normalization used for \(\Upsilon\).

Reference sources:
- Ueda & Oshikawa, *Resolving the Berezinskii-Kosterlitz-Thouless transition in the two-dimensional XY model with tensor-network-based level spectroscopy*, Phys. Rev. B 104, 165132 (2021), https://doi.org/10.1103/PhysRevB.104.165132. Reports \(0.892943(2)\) and removes leading logarithmic corrections through level spectroscopy.
- Yu et al., *Tensor renormalization group study of the classical XY model on the square lattice*, Phys. Rev. E 89, 013308 (2014), https://doi.org/10.1103/PhysRevE.89.013308. An independent HOTRG comparison reports \(0.8921(19)\).
- Gottlob & Hasenbusch, *Critical behaviour of the 3D XY-model: a Monte Carlo study*, Physica A 201 (1993), https://doi.org/10.1016/0378-4371(93)90131-M. Reports \(K_c=0.45420(2)\) using single-cluster Monte Carlo.
- Hasenbusch, *Eliminating leading and subleading corrections to scaling in the three-dimensional XY universality class*, Phys. Rev. B 112, 184512 (2025), https://doi.org/10.1103/1d4g-483z. A modern correction-to-scaling methodology comparator, not the same nearest-neighbour model.

## Topology and disorder families

The benchmark compares regular lattices, layered/anisotropic systems, bond-diluted networks, random weak links, correlated/clustered disorder, hierarchical/modular systems, loop-redundant networks, and physically embedded rewired networks.

Hyperuniform or anticorrelated disorder is an exploratory family, not an assumed improvement.

For each instance preserve:
- node coordinates / physical embedding and boundary conditions;
- ordered edge identities and endpoints;
- positive coupling values and their multiset;
- gauge-link phases and gauge-invariant loop fluxes;
- disorder-generation family, seed and spatial correlation specification;
- missing-bond fraction, degree and loop statistics;
- total coupling budget and declared architecture cost.

Candidate explanatory descriptors include weighted Laplacian spectral gap, minimum-cut conductance, edge-disjoint path counts, loop redundancy, degree/coupling heterogeneity, disorder correlation length and gauge-invariant frustration. They must be tested for incremental predictive value beyond dimensionality, mean coupling and disorder moments.

## Observables

Do not collapse these into one score:

- helicity modulus / stiffness tensor from a declared boundary twist;
- phase correlation function and correlation length;
- BKT or thermodynamic critical estimate with method, finite-size analysis and uncertainty;
- vortex density and winding-sector statistics where defined;
- bond-removal / weak-link / correlated-disorder robustness;
- equilibration diagnostics, autocorrelation time and effective independent sample count;
- compute cost and exact run identity.

A finite-size crossover, a large graph spectral gap, or a small resistance in a classical proxy is not a measured superconducting transition.

## Fair benchmark protocol

- Freeze graph generators, seeds, boundary conditions, coupling normalization, temperature grids, finite-size sequence, solvers and convergence tolerances.
- Use complete topology-family and disorder-family holdouts. Randomly splitting graph instances from the same generator is not an OOD test.
- Compare analytical / mean-field, regular-lattice XY, matched random-network, percolation/connectivity proxies, graph-only surrogates and the full XY/Josephson Monte Carlo model.
- Report uncertainty intervals, failed equilibration, finite-size drift, seed variance and wall-clock/compute budget.
- Use an independent implementation or analytic check for at least one reference case before promoting the harness.

## Claim levels

1. **Schema-valid instance:** input contract passes; no physics is established.
2. **Reference-validated solver:** analytic benchmark cases pass; no topology benefit is established.
3. **Matched numerical effect:** a topology family changes a declared coherence observable at fixed resource budget, with uncertainty and topology-family holdouts.
4. **Experimental hypothesis:** a measurable architecture and observable mapping are proposed; still not a material result.
5. **Experimental validation:** requires same-specimen, source-linked pairing/coherence/bulk observables under the existing #12677 gates.

The issue's 20% coherence-robustness target is a preregistered research target, not an achieved result or a forecast.

## Related programme

- Ambient-pressure and translation strategy: https://github.com/Luminous-Dynamics/symthaea/issues/12617
- Pair formation versus global coherence: https://github.com/Luminous-Dynamics/symthaea/issues/12677
- Discovery tournament and prospective yield: https://github.com/Luminous-Dynamics/symthaea/issues/12634
- Topology/disorder phase-network hypothesis: https://github.com/Luminous-Dynamics/symthaea/issues/12755
