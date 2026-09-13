# symthaea-assurance-reference-integrity

Reference-integrity evaluation for the TPM measured-boot assurance chain.

This crate deliberately separates four claims:

1. a fresh TPM quote was independently verified;
2. the measured-boot event log replays to the quoted PCR state;
3. normalized per-event measurements were independently extracted from that exact replayed log; and
4. those measurements are evaluated against an exact signed/versioned reference-integrity manifest.

A result is `Approved`, `Rejected`, or `Incomplete`. Explicit known-bad measurements and unexpected critical measurements are rejected. Missing required reference measurements or unresolved unknown non-critical measurements remain incomplete unless a reviewed policy explicitly permits those unknowns.

The normalized manifest model is not a parser for any one external RIM serialization. Exact source RIM bytes, signature verification, and normalized-content mapping remain independently evidenced and content-addressed. This follows the TCG distinction between attestation evidence and reference assertions without treating a parser or manifest string as a trust decision.

This crate never grants physical authority and contains no TPM mutation, targeting, interception, firing, jamming, spoofing, weapon-control, or engagement-optimization logic.
