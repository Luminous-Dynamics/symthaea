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
