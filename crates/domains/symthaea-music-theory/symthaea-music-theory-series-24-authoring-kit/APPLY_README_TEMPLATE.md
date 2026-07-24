# Applying Patch Series 24

1. Verify the exact Series 23 source tree and cumulative integration evidence.
2. Fill `BASELINE_REQUIRED.txt` with the independently verified identities.
3. Apply patches in `PATCH_ORDER.md` without manual edits.
4. Run all existing Series 16–23 tests first, then the abuse corpus, fuzz-seed replay, archive-safety tests, independent-verifier limit fixtures, and worst-case-valid benchmarks.
5. Re-run the complete Series 23 clean-room release workflow.
6. Verify the deterministic Series 24 evidence archive and external digest.

Do not accept a limit-related failure by merely increasing defaults. First establish whether the input is valid, whether the profile is appropriate, and whether the algorithm performs avoidable work.
