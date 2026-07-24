# Applying Patch Series 23

1. Verify the external digest of the exact Series 22 source archive.
2. Verify and record the exact baseline Git tree in `BASELINE_REQUIRED.txt`.
3. Run the cumulative input-ledger audit.
4. Apply every mail patch in `PATCH_ORDER.md` with `git am` in a clean repository.
5. Run the complete release-truth workflow and mandatory negative controls.
6. Compare the resulting final tree with the advertised Series 23 tree.
7. Verify the deterministic release archive and its externally stored digest.

Do not repair application failures manually. A mismatch means the baseline, patch chain, or advertised identity is wrong.
