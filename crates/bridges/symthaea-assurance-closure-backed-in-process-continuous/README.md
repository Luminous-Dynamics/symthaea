# Closure-backed in-process continuous verifier

Composes the exact Nix-closure-backed in-process TPM verifier runtime policy with one opaque `ContinuousVerifierExecution` from the existing measured-launch/runtime-lineage theorem.

Qualification requires the continuous execution to name the exact runtime-policy digest and verifier role already bound to the current in-process TPM+IMA provider host and verified Nix closure. The child does not re-run, reinterpret, or weaken either parent theorem.

The result establishes continuous execution under the exact closure-backed in-process runtime policy. It still does not establish computation-time Nix co-observation, mapped-memory identity, atomic shared-library pinning, Nix database currentness/non-equivocation, trusted time, or physical authority.
