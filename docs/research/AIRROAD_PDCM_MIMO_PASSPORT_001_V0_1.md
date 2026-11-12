# AIRROAD-PDCM-MIMO-PASSPORT-001 — V0.1 complete impedance matrix evidence contract

**Status:** Rust evidence validator and exact synthetic known-answer fixture  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11764  
**MIMO reference:** https://github.com/Luminous-Dynamics/symthaea/pull/11761

## 1. Why a matrix passport is necessary

A multi-port impedance matrix is not just a rectangular array of complex values. Every column is produced by a distinct current injection at one input port, while every output-port voltage must be observed under the same operating point and frequency. The matrix can only be reconstructed when the port set, directions, excitation records and output observations are complete and consistent.

Missing cross-port data are not zero. A zero is valid only when the corresponding port-pair response was actually recorded. Interpolated or extrapolated values outside recorded knots are not measurements.

This is the evidence boundary needed before consuming measured matrices in the synthetic 2×2 MIMO stability model.

## 2. V0.1 data model

The authoritative small-signal convention is:

```
Z_ij(jw) = V_i(jw) / I_j(jw)
```

The input passport contains:

- schema version, configuration digest, operating-point digest and sweep digest;
- ordered port descriptors, each with unique ID, voltage-positive reference and current-positive direction;
- one excitation column for every driven current port on every frequency knot;
- complex injected-current amplitude, excitation digest and column payload digest;
- one complex voltage response and absolute error bound for every observed port in every column;
- evidence origin, with required provenance depending on whether it is measured, simulated or an analytical reference.

For measured origin, instrument ID, calibration-certificate digest, channel-map digest and synchronization digest are mandatory. Simulated origin requires simulator identity/version and model digest. Analytical origin requires a derivation digest. These origins are separate labels and cannot be silently converted into one another.

## 3. Fail-closed validation

The Rust validator checks:

1. Every identity and direction descriptor is non-empty, and port IDs are unique.
2. Frequency knots are finite, non-negative and strictly increasing.
3. Every knot has exactly one excitation column per port.
4. Every column has a finite, non-zero complex current injection, a finite non-negative current error bound, and non-empty excitation/payload digests.
5. Every column observes each output port exactly once.
6. Every complex voltage response is finite and has a finite non-negative error bound.
7. The evidence origin has the corresponding mandatory provenance fields.
8. Querying a matrix is permitted only at an exact recorded frequency knot.

Any missing or duplicate port-pair response returns a typed NOT EVALUABLE error. The validator never fills missing matrix elements with zero.

## 4. Output and uncertainty boundary

After complete structural validation, the module computes each point estimate (Z_{ij}=V_i/I_j). The output is explicitly named a point estimate. Raw voltage and current error bounds remain attached to the source sweep but are **not** propagated here into a formally bounded complex impedance set.

This matters: division of uncertain complex quantities requires a qualified enclosure method, including non-zero denominator margin and correlation treatment. Until that is implemented by the upstream interval/uncertainty owners, the passport must not claim a hard-bound impedance result.

## 5. Frozen reference

The machine-readable fixture is:

```
docs/research/fixtures/airroad-pdcm-mimo-impedance-passport-v0.1.json
```

It contains a complete synthetic 2×2 two-frequency sweep. At each frequency the matrix point estimate is:

```
Z = [ 2 + j       1 + 2j ]
    [ 1/2 - j     3 - j  ] ohm
```

The independent Python standard-library Fraction verifier recomputes all four ratios at both frequency knots and checks the negative controls. No production Python runtime is added.

## 6. Industry comparison and differentiation

PLECS 5.0 already supports AC and frequency-response analysis. Its standalone frequency-response workflow reaches a steady-state operating point, injects discrete sinusoidal perturbations and extracts the corresponding response frequency. This passport is not a claim to invent frequency sweeps.

The useful differentiation is that every matrix column and row is bound to stable port identities, current/voltage directions, operating-point/configuration digests, calibration/channel/synchronization provenance, uncertainty inputs and raw result digests. The result can be replayed and joined to the full matrix stability model without silently losing cross-port terms.

## 7. Benchmarks

- 100% complete matrix validation for all frozen complete cases.
- 100% exact rational agreement for every (Z_{ij}) on the known-answer sweep.
- Zero missing matrix elements accepted as zero.
- Zero duplicate driven ports or output observations accepted.
- Zero measured sweeps accepted without calibration, channel-map and synchronization identity.
- Zero off-knot interpolation/extrapolation returned as measured data.
- Separate reporting of structural completeness, evidence origin, frequency coverage and uncertainty availability.

Next tier: propagate measurement uncertainty into complex impedance enclosures with explicit correlations, then compare matched matrices against ngspice/PLECS and calibrated PHIL data.

## 8. Claim ceiling

The frozen fixture is synthetic analytical evidence only. It qualifies the validator structure and arithmetic. It does not validate actual measurements, converter impedance, generalized-Nyquist analysis, aircraft stability, PHIL/HIL, flight safety, airworthiness or certification.
