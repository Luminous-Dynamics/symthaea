# Kinetic Information Return (KIRR) pulse graph contract v0.1

This is the first machine-readable protocol component for #13659. It freezes ordered pulse paths and observation horizons before any physical response run.

The fixture contains three synthetic paths:
- a no-intervention baseline;
- a two-pulse echo-anchor path;
- a three-pulse cascade path.

Each pulse records mode, phase, non-negative kick amplitude and application time. Pulse order is preserved. The Rust `KineticPulseSequence` identity binds the frozen physical pulse-family fingerprint, observation horizon and ordered canonical pulse digests. Validated zero-amplitude no-op pulses are excluded from the physical path identity. Out-of-order pulses, pulse times at/after the observation horizon, excessive path lengths and invalid horizons fail closed.

The schema constrains shape and path cardinality. The focused Rust tests instantiate the same fixture against the frozen pulse-family contract, so the semantic validator additionally checks the admissible mode, phase, amplitude, time and cohort horizon.

This contract does **not** calculate echo timing or response, run Vlasov–Poisson, identify echo branches, or establish nonlinear memory retention. It supplies an immutable intervention path for subsequent solver/reference comparisons.


## Path-conditioned response signatures

The Rust prototype also exposes `KineticPulsePathResponseSignature`, which binds a KRS complex response signature to exactly one immutable sequence digest.

Its semantic checks:
- the signature's sole intervention identity must equal the sequence digest;
- response sample times must lie at/after the final active pulse (or at/after zero for a baseline) and no later than the declared observation horizon;
- missing complex responses and unresolved branch identifiers remain explicit and can be queried as incomplete coverage;
- candidate/reference comparisons are allowed only for the same sequence digest and then retain KRS's QoI scaling, time axes, branch mismatch and structure-class diagnostics.

This prevents accidental comparison of two different pulse paths under a shared label. It still does not infer echo branches or compute a physical response; the KRS signature must be populated from an independently executed candidate/reference pair.
