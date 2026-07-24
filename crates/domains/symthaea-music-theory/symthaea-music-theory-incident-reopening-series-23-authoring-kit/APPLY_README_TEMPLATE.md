# Applying Symthaea Music Theory Patch Series 23

## Expected base

the exact demonstrated Series 22 final tree produced from the Series 21 base.

## Procedure

1. Verify the exact base tree and all prerequisite archive manifests.
2. Apply patches in `PATCH_ORDER.md` without manual edits.
3. Compare the resulting Git tree with the authored final-tree identity.
4. Run formatting, all-target/all-feature Cargo checks, tests, and Clippy.
5. Run the canonical Nix lane and independent-verifier fixtures.
6. Run every adversarial and transactional failure case.
7. Rebuild public evidence and patch archives deterministically.
8. Publish exact observed limitations and unexecuted gates.

Do not resolve a verifier, branch, authority, or transaction disagreement by majority vote or by selecting the newest artifact.
