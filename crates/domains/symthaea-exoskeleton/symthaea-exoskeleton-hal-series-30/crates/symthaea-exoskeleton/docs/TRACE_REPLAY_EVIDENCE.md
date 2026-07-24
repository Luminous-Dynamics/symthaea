# Control Trace and Replay Evidence

Each control tick may be recorded into a fixed-capacity ring without allocation.
The trace captures the state fingerprint, authority input quality, governed
assistance mode, authority and safety reasons, final authority ceiling, and the
command that actually passed the safety kernel.

## T0 — chronological export

Ring wraparound must preserve chronological order and report how many older
records were overwritten.

## T1 — replay invariants

Offline verification checks finite values, exact sequence continuity,
monotonic timestamps, normalized torque bounds, and the zero-authority invariant
for torque, stiffness, and damping.

## T2 — deterministic fingerprint

The same ordered trace produces the same FNV-1a evidence fingerprint. This is a
reproducibility checksum only. Publication bundles requiring authenticity must
sign the exported evidence with the repository's approved evidence-signing
profile.

## T3 — privacy boundary

Raw biometric streams and high-dimensional cognitive intent are deliberately
excluded from the default trace. Evidence export should contain only the minimum
fields needed to reproduce actuator authority and safety decisions.
