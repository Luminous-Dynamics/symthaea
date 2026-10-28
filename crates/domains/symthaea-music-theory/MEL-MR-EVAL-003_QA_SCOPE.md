# MEL-MR-EVAL-003 fixture QA scope

- Dataset: MEL-MR-COMPOSITION-SMOKE-1.0
- Cases: 108 (60 accept, 48 reject)
- Motif families: 12; family-held-out splits: development 6 families / 54 cases, calibration 2 / 18, test 4 / 36
- Structural checks: case IDs unique; all targets match selection lengths; no family crosses a split; expected verdict agrees with actual-vs-claimed descriptors and declared perturbation.
- Status: PASS for synthetic fixture structure and oracle consistency.
- Explicitly NOT established: Rust verifier correctness, integration, compile/test status, mutation adequacy, human recognition, commercial competitiveness, or generalization.

The corpus is smoke-scale. Do not use it as the sole evidence for a release or publication claim.

## Follow-on Rust prototype status

The PR includes a Rust unit test that replays the committed corpus and asserts all 108 expected verdicts, plus adversarial and metamorphic unit tests. These tests are **authored but not executed**: local cargo/rustc/rustfmt are unavailable and the current exact-head GitHub workflow records are skipped. A separate local Python simulation of the intended alignment and verdict logic reported 0/108 expected-verdict mismatches. That result is not interchangeable with execution of the Rust implementation.


## Evidence receipt contract

Each verification result records the descriptor and threshold config, target offset, ordered match pairs with independent residuals, unmatched source/target indices, aggregate axis evidence and failure reasons. The receipt is JSON round-trip tested by the authored corpus replay test. This supports audit/replay but is not a cryptographic provenance receipt: callers still need to bind source/target content digests and the exact verifier build SHA when persisting evidence.


## Additional source-level regression controls

- Constant-pitch motifs cannot establish interval inversion; the Rust API returns `UnsupportedTransform` and has a dedicated regression test.
- A bounded deletion test checks the original source index recorded in the receipt.
- A fragment test checks that selected-subsequence indices map back to the original source sequence.
- Local JSON Schema validation succeeded for all 108 corpus records.
- A standalone Python reference simulation of the DP/verdict equations recorded 0/108 verdict mismatches, plus passing one-insert and one-delete controls at the configured coverage/gap boundaries. This remains an algorithm-level simulation, not execution of the Rust source.
