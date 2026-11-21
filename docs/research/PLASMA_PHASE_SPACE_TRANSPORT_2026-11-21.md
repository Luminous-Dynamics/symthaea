# Phase-space transport reference for nonnegative measures

**Date:** 2026-11-21  
**Status:** source authored; exact-head Rust validation pending  
**Tracking:** [Issue #13053](https://github.com/Luminous-Dynamics/symthaea/issues/13053)

## Motivation: two distinct distribution geometries

The phase-space MMD work measures the difference between kernel mean embeddings. It is label-independent and characteristic on the declared cylinder. It is not a literal physical relocation cost. A transport metric asks a complementary question: how much phase-space mass must move, and how far, to transform one empirical distribution into another?

Keep both as separate observables:
- characteristic MMD: smooth, characteristic kernel discrepancy that is convenient for spectral approximations and signed finite measures;
- Wasserstein transport: geometric relocation cost for nonnegative measures under a frozen ground metric.

Do not combine them into one score without a validated calibration and do not claim one subsumes the other.

## Domain and ground metric

For periodic position and unbounded velocity, define shortest-arc spatial separation:

\[
d_{S^1_L}(x,x')=\min(\delta_x,L-\delta_x),\qquad
\delta_x=|x-x'|\bmod L.
\]

Use the dimensionless product ground metric:

\[
d((x,v),(x',v'))=
\sqrt{
\left(\frac{d_{S^1_L}(x,x')}{\ell_x}\right)^2+
\left(\frac{v-v'}{\ell_v}\right)^2
}.
\]

The position and velocity scales are cohort inputs. They must be frozen from physical normalization or training/calibration data and must not be tuned against held-out trajectories.

## Measure cohorts

### Balanced nonnegative measures

The implemented small-N reference is for nonnegative atomic measures with equal total mass (up to a strict configured relative tolerance). Co-located positive weights are aggregated before transport. Candidate/reference masses are retained in the result, and the normalized distributions are transported under the declared metric.

### Unequal mass

A separate unbalanced-OT extension is required for nonnegative measures with varying mass. It must choose and freeze the mass-creation/destruction penalty (e.g. KL marginal penalties), report transported/destroyed/created mass, and disclose entropy-regularization bias. This PR does not claim unbalanced transport.

### Signed delta-f

Standard balanced W1 is not defined for arbitrary signed empirical weights. The implementation rejects negative weights. Do not take absolute values or split signs into two positive transports and call that the same signed-measure metric. Signed delta-f diagnostics remain on the characteristic-MMD lane until a separate signed-transport theory is established.

## Exact small-N implementation

The crate builds the dense bipartite transportation network and solves its linear program via successive shortest augmenting paths. Residual reverse arcs have negative costs, so the shortest path is found with Bellman-Ford rather than applying Dijkstra without valid potentials. The result reports the normalized W1 cost and algorithm work counters.

This is a small-N correctness reference. It is not a production Sinkhorn implementation, GPU solver or NUFFT-accelerated method.

## Validation suite

The authored tests cover identity, permutation, positive co-located split/merge, shortest-arc seam behavior, velocity translation, joint position/velocity correlation changes, strict mass mismatch, signed-weight rejection, empty measures, invalid inputs and residual rerouting.

The exact-head Rust 1.96 workflow will run format, unit tests and strict Clippy. No pass is claimed until the latest exact head completes those steps.

## Research / industry baseline

Lakshmanan & Pichler (Journal of Scientific Computing, 2024) study connections between unbalanced optimal transport and MMD, and describe NUFFT-accelerated radial-kernel convolution schemes with O(n log n) arithmetic in structured settings. This is a relevant future baseline for the unbalanced/accelerated lane, not a claim that the current reference implementation has that complexity.

Source: [Unbalanced Optimal Transport and Maximum Mean Discrepancies: Interconnections and Rapid Evaluation](https://link.springer.com/article/10.1007/s10915-024-02586-2).

Next comparators:
- exact small-N LP reference (this crate);
- entropy-regularized Sinkhorn, with regularization bias quantified;
- unbalanced Sinkhorn for unequal positive masses;
- characteristic MMD exact/spectral (PR #13033 / #13045);
- one-dimensional marginal W1 / histogram refinement.

## Tournament use

Add W1 only to positive equal-mass phase-space cohorts. Preserve separate results for:
- W1 geometric relocation;
- MMD characteristic discrepancy;
- field error and particle-force error;
- in-loop trajectory/distribution error;
- field/noise spectra;
- physical invariants and discrete residuals;
- convergence and complete end-to-end cost.

No cross-cohort aggregate should mix signed delta-f with probability-measure transport. The transport metric does not certify the governing equations or the solver.
