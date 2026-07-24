# Applying Symthaea Music Theory Patch Series 24

1. Verify the exact base tree: the exact implemented and verified Series 23 final tree derived from the Series 21 baseline.
2. Verify all prerequisite archive manifests and recorded tree identities.
3. Apply patches in `PATCH_ORDER.md` without manual edits.
4. Compare the replayed final tree with the authored final-tree identity.
5. Run formatting, all-target/all-feature Cargo checks, tests, and Clippy.
6. Run the canonical Nix and independent-verifier lanes.
7. Execute every adversarial and transaction matrix case.
8. Reproduce source, patch, and evidence archives deterministically.
9. Publish only claims supported by retained evidence.

Do not resolve a branch, verifier, authority, or transaction disagreement by majority vote, newest timestamp, or artifact-supplied policy.
