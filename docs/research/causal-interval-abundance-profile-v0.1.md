# Causal-set interval-abundance profiles v0.1

Date: 2026-11-08  
Status: executable statistical profile primitive; no continuum-limit or physical closeness claim  
Parent: [CAUSAL-GEOMETRY-001 #11129](https://github.com/Luminous-Dynamics/symthaea/issues/11129)  
Performance follow-up: [CAUSAL-INTERVAL-PERF-001 #11035](https://github.com/Luminous-Dynamics/symthaea/issues/11035)  
Programme: [CTGP #10802](https://github.com/Luminous-Dynamics/symthaea/pull/10802)

## Executable primitive

`symthaea-causal-set-lab::IntervalAbundanceProfile` is built from the existing exact open-interval histogram:

- `histogram[k]`: number of unordered related endpoint pairs with exactly k strict interior events;
- `related_pairs`: sum of the histogram bins;
- `incomparable_pairs`: total unordered endpoint pairs minus related pairs;
- `related_pair_fraction`: related-pair share of all endpoint pairs.

This preserves two different statistical facts: the interval-cardinality distribution conditional on comparability, and the fraction of pairs that are comparable. A conditional histogram alone would discard the second fact and could make a sparse random order look like a dense causal geometry.

## Fixed-width normalized profile

For a causal set with N events, open interval cardinality k is mapped to a normalized size coordinate

    s = k / max(N - 2, 1)

and linearly interpolated onto a fixed number of profile bins. Incomparable-pair probability is stored in a separate final bin. Probabilities are normalized by the total number of unordered endpoint pairs.

This is a declared finite-size comparison convention, not a universal continuum scaling law. Reports should retain the raw histogram, event count, sprinkling density, region volume and boundary/window convention alongside normalized profiles.

The implementation caps profile bins at 4,096 and fails closed on malformed histogram length, overflow, histogram counts exceeding total pair count, zero-pair samples and invalid bin requests.

## Statistical distances

The profile currently supplies:
- total-variation distance;
- Hellinger distance;
- Jensen–Shannon distance (square root of base-2 divergence).

These are generic distances between finite probability vectors. They are **not** the Surya 2026 Lorentzian closeness function, and they are not Lorentzian Gromov–Hausdorff distances.

## Tests

The module tests:
- total-order interval counts;
- antichain/incomparable mass;
- normalization to unit probability;
- zero distance for identical profiles;
- symmetry and boundedness of all three distances;
- empty/singleton causal sets;
- malformed histograms and overflow;
- agreement between 1+1D and dimension-generic profile builders on the same order.

## Next validation

1. Compare profiles across fixed-volume sprinklings at increasing N and across dimensions 2–4.
2. Include Lorentz boosts and order-preserving relabellings as invariance controls.
3. Compare flat versus curved reference causal sets and random-DAG controls on held-out seeds.
4. Report bootstrap/Monte Carlo confidence intervals, not only point distances.
5. Implement the paper's closeness function verbatim from its exact definition as a separate function, with independent reference fixtures.
6. Measure whether the full abundance profile adds predictive discrimination beyond Myrheim–Meyer ordering fraction and longest-chain scaling.

## Claim ceiling

A passing unit test establishes the profile arithmetic and its finite causal-set invariances for the declared fixtures. It does not prove spacetime discreteness, a continuum limit, quantum gravity, or a physical metric on the space of Lorentzian geometries.
