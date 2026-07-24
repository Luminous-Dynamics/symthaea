# Cumulative Integration Release Contract

A Series 23 release is demonstrated only when all of the following are true for the same exact source tree:

1. The cumulative numbered lineage replays from its pinned baseline without manual edits.
2. The replayed and authored final Git trees are identical.
3. The declared Cargo target and feature matrix passes.
4. The declared Nix lane passes from a clean source export.
5. Rust and an independent verifier agree on the frozen conformance corpus.
6. Every public archive and manifest rebuilds byte-for-byte.
7. Mandatory negative controls fail at their expected stages.
8. The public claim matrix is generated from these evidence records.

A failure in any required dimension blocks the corresponding claim. Series 23 does not permit an aggregate green badge to conceal a red or unavailable component.
