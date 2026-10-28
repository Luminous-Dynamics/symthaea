# MEL-MR-EVAL-003 fixture QA scope

- Dataset: MEL-MR-COMPOSITION-SMOKE-1.0
- Cases: 108 (60 accept, 48 reject)
- Motif families: 12; family-held-out splits: development 6 families / 54 cases, calibration 2 / 18, test 4 / 36
- Structural checks: case IDs unique; all targets match selection lengths; no family crosses a split; expected verdict agrees with actual-vs-claimed descriptors and declared perturbation.
- Status: PASS for synthetic fixture structure and oracle consistency.
- Explicitly NOT established: Rust verifier correctness, integration, compile/test status, mutation adequacy, human recognition, commercial competitiveness, or generalization.

The corpus is smoke-scale. Do not use it as the sole evidence for a release or publication claim.
