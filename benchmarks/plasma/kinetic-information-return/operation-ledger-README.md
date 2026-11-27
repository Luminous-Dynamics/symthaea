# KIRR physical/numerical operation ledger v0.1

This ledger separates three channels that must never be collapsed into one irreversibility scalar:

1. **Physical collision operator:** allowed only under a declared collisional model; carries a named physical entropy-production quantity and non-negative production under that convention.
2. **Numerical operation:** filter, spectral truncation, hypercollision, projection, compression, basis transition, coordinate remap or fallback. Numerical entropy changes and discarded-effect measures are explicitly labelled as numerical quantities.
3. **Representation / state evidence:** each event binds state-before and state-after digests, operator identity and physical time.

Lossy numerical operations require a declared effect metric and non-negative discarded-effect measure. The ledger sums only events with the same metric ID; it never adds unlike norms, spectral energies or entropy quantities.

A collisionless Vlasov ledger rejects physical collision events. A numerical filter or hypercollision cannot populate the physical entropy-production channel. A declared reversible transition must report zero discarded-effect measure in the chosen transfer metric.

The included fixture contains one synthetic collisionless ledger (filter, projection, exact basis transition) and one synthetic collisional ledger (physical Landau collision event). These are wire-contract examples only; no solver or physical entropy production was measured.

The Rust implementation is `KineticOperationLedger`. It validates bounded event count, unique IDs, chronological application times, state digest shape, event/horizon validity, physical/numerical entropy-channel separation and per-metric loss attribution. It does not infer the truth of an operator's declared measurement; independent receipts and reference comparisons are required.
