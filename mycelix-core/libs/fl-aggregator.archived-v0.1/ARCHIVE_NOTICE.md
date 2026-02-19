# fl-aggregator — ARCHIVED (February 2026)

This crate has been archived. Its unique features have been ported to the
production FL stack:

| Feature | Ported To |
|---------|-----------|
| MultiKrum, GeometricMedian | `mycelix-fl-core/src/aggregation.rs` |
| Adaptive Defense | `mycelix-fl-core/src/adaptive_defense.rs` |
| Replay Detection | `mycelix-fl-core/src/replay_detection.rs` |
| Shapley Values | `mycelix-fl-core/src/shapley.rs` |
| Ensemble Defense | `mycelix-fl-core/src/ensemble_defense.rs` |
| Top-k Compression | `mycelix-fl-core/src/compression.rs` |
| HDC Aggregation | `mycelix-fl/src/hdc_aggregation.rs` |
| PoGQ-v4.1 Lite | `mycelix-fl/src/pogq.rs` |
| Phi Time-Series | `mycelix-fl/src/phi_series.rs` |
| Attack Simulation | `mycelix-fl/src/attacks.rs` |
| HyperFeel Compression | `mycelix-fl/src/compression.rs` |
| Proof Types + Field Ops | `mycelix-fl-proofs/` (scaffold) |

Production FL stack: `mycelix-workspace/crates/mycelix-fl-core/` and
`mycelix-workspace/crates/mycelix-fl/`

Do NOT add new code here. This directory is preserved for reference only.
