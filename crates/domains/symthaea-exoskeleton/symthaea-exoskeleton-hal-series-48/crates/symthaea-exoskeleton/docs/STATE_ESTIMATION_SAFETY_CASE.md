# State Estimation Safety Case

This document defines the bounded claim made by `state_estimation.rs`.

## Claim

The estimator provides a deterministic, fixed-allocation state estimate with explicit freshness, continuity, innovation rejection, and confidence outputs. It does **not** prove the true biomechanical state and does not authorize assistance by itself.

## Invariants

- Non-finite, future-dated, stale, discontinuous, or over-period frames return zero authority.
- Initialization is explicitly derated.
- Position innovations above the configured joint-specific bounds are rejected rather than absorbed.
- Rejected channels are represented in a channel mask.
- The estimator's authority ceiling may only reduce the authority allowed by independent supervisors.

## Required evidence before human-worn use

- Encoder and IMU latency distributions under maximum bus load.
- Innovation bounds justified across anthropometries and gait speeds.
- Fault-injection evidence for freezes, jumps, polarity reversal, quantization, and timestamp corruption.
- Comparison against an independent motion-capture or calibrated reference system.
- Demonstration that estimator rejection always reaches the final actuator disable path within the system deadline.
