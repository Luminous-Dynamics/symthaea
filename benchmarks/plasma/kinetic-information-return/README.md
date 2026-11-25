# Kinetic Information Return (KIRR) pulse graph contract v0.1

This is the first machine-readable protocol component for #13659. It freezes ordered pulse paths and observation horizons before any physical response run.

The fixture contains three synthetic paths:
- a no-intervention baseline;
- a two-pulse echo-anchor path;
- a three-pulse cascade path.

Each pulse records mode, phase, non-negative kick amplitude and application time. Pulse order is preserved. The Rust `KineticPulseSequence` identity binds the frozen physical pulse-family fingerprint, observation horizon and ordered canonical pulse digests. Validated zero-amplitude no-op pulses are excluded from the physical path identity. Out-of-order pulses, pulse times at/after the observation horizon, excessive path lengths and invalid horizons fail closed.

The schema constrains shape and path cardinality. The focused Rust tests instantiate the same fixture against the frozen pulse-family contract, so the semantic validator additionally checks the admissible mode, phase, amplitude, time and cohort horizon.

This contract does **not** calculate echo timing or response, run Vlasov–Poisson, identify echo branches, or establish nonlinear memory retention. It supplies an immutable intervention path for subsequent solver/reference comparisons.
