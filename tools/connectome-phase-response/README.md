# Connectome phase-response transfer

Standalone Rust 2024 crate for exact fixed-point circular phase-reset algebra:

`cargo test --manifest-path tools/connectome-phase-response/Cargo.toml --locked`

This crate is deliberately outside the root Cargo workspace. It has no third-party dependencies and does not simulate a fly, estimate a neural mechanism, or authorize physical interventions.

## Measurand

For sender leg j, recipient leg i, perturbation phase phi, perturbation class u, context c and delay tau:

`G[i,j](phi,u,c,tau) = wrap(theta_i_perturbed(t0+tau) - theta_i_matched_control(t0+tau))`

Each phase sample stores an integer cycle index and a signed fractional phase in microturns, with 1,000,000 units per cycle and canonical fractional range [-500,000, 500,000). This avoids floating point in phase wrapping and preserves whole-cycle advances/delays that modulo-only phase would alias to zero. It is a quantized phase representation; source phase-estimator uncertainty must be carried by any biological adapter.

The output keeps two separate vectors: wrapped fractional phase displacement and integer cycle slips. They are never collapsed into one scalar. Cycle slips are computed as the residual between the unwrapped phase difference and the wrapped fractional difference, divided by one full cycle. This prevents a small phase advance across the fractional branch boundary from being mislabeled as a full-cycle slip.

Model separation is reported per probe and across a joint campaign portfolio. The joint metric concatenates the registered probe responses before pairwise distance, so a probe that is uninformative by itself can complement another probe.

A measured row has paired six-leg control and perturbed phases. The output vector preserves canonical axis order:
1. left foreleg
2. left middle leg
3. left hindleg
4. right foreleg
5. right middle leg
6. right hindleg

The local reset is the sender-axis component; off-diagonal transfer is the sum of absolute circular phase reset in the other axes. For a global descending input without a single sender leg, all six axes are retained.

Descending-command probe keys require the exact source identity DNg100, DNg97 or DNb08. A generic descending-command row is invalid; this prevents distinct command pathways from being pooled in the same transfer surface.

## Fail-closed semantics

No response, failed assay, missing track, out-of-support and not-assessed are explicit states. They are not encoded as six zero phase shifts. Probe keys preserve context, perturbation class, sender leg or exact descending-neuron identity, phase bin, perturbation magnitude bin and response delay. Magnitude bins are bounded to 16 in V1; the perturbation receipt digest commits the physical dose/strength mapping.

This crate is an adapter, not a phase oscillator or optimiser. Symtropy #3926 supplies phase-network traces; #3454 owns noise-aware predictive separation; #3345 owns physical contact feedback; #3478 hands frozen predictions to Symthaea EIG/Fisher/EVSI objectives. Duplicate rows, missing grid cells, invalid phase bins and mismatched model probe grids are rejected.

Model-separation diagnostics compare exact circular L1 distances over the same registered probe cells. They are deterministic phase-unit diagnostics, not likelihoods, probabilities, causal effects or biological effect sizes. A positive distance only distinguishes supplied finite model signatures on the supplied design.

## Synthetic limitation

The unit tests demonstrate fixed-point phase arithmetic, local versus off-diagonal indexing, non-response preservation and a synthetic example in which two models share the same mean frequency and steady-state PLV but differ in their perturbation response. No biological input is loaded and no biological mechanism is validated.
