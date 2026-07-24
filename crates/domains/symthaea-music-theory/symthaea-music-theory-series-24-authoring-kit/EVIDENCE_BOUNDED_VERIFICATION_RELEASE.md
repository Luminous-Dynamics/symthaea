# Bounded Verification Release Contract

A verifier may claim bounded hostile-artifact handling only when:

1. Every untrusted entrypoint requires or derives a trusted local limit policy.
2. All relevant byte, count, depth, lineage, external-call, archive, and output dimensions are bounded.
3. Cheap checks precede expensive authentication and lineage work where semantics allow.
4. Timeout, cancellation, crash, and limit exhaustion cannot commit an accepted result.
5. Archive verification is streaming and extraction is capability-confined.
6. Rust and an independent verifier agree on frozen limit fixtures and failure codes.
7. Frozen malicious cases execute within the CI resource envelope.
8. Valid threshold-edge bundles pass and have measured resource reports.

This contract does not claim immunity to every implementation bug or suitability of one default limit profile for every deployment.
