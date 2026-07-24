# Applying Grounded Patch Series 26

1. Verify the exact expected base: the exact Series 21 final tree plus reviewed consolidation of the grounded Series 22–25 plans.
2. Verify prerequisite archive checksums and manifests.
3. Apply patches in `PATCH_ORDER.md` without manual edits.
4. Compare the replayed final tree with the authored final-tree identity.
5. Run formatting, all target/feature Cargo checks, tests, and Clippy.
6. Run Nix, independent-verifier, transaction, resource, privacy, and reproducibility lanes.
7. Generate the implementation and claim matrices from observed evidence.
8. Publish deterministic source, patch, and evidence archives.

A semantically similar tree is not an exact replay. A green dashboard is not verification evidence.
