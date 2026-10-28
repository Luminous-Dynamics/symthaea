# MEL-MR-EVAL-003 fixture QA scope

- Dataset: MEL-MR-COMPOSITION-SMOKE-1.0
- Cases: 120 (60 accept, 60 reject)
- Motif families: 12; family-held-out splits: development 6 families / 60 cases, calibration 2 / 20, test 4 / 40
- Structural checks: case IDs unique; all targets match selection lengths; no family crosses a split; expected verdict agrees with actual-vs-claimed descriptors and declared perturbation.
- Status: PASS for synthetic fixture structure and oracle consistency.
- Explicitly NOT established: Rust verifier correctness, integration, compile/test status, mutation adequacy, human recognition, commercial competitiveness, or generalization.

The corpus is smoke-scale. Do not use it as the sole evidence for a release or publication claim.

## Follow-on Rust prototype status

The PR includes a Rust unit test that replays the committed corpus and asserts all 120 expected verdicts, plus adversarial and metamorphic unit tests. These tests are **authored but not executed**: local cargo/rustc/rustfmt are unavailable and the current exact-head GitHub workflow records are skipped. A separate local Python simulation of the intended alignment and verdict logic reported 0/120 expected-verdict mismatches. That result is not interchangeable with execution of the Rust implementation.


## Evidence receipt contract

Each verification result records the descriptor and threshold config, target offset, ordered match pairs with independent residuals, unmatched source/target indices, aggregate axis evidence and failure reasons. The receipt is JSON round-trip tested by the authored corpus replay test. This supports audit/replay but is not a cryptographic provenance receipt: callers still need to bind source/target content digests and the exact verifier build SHA when persisting evidence.


## Additional source-level regression controls

- Constant-pitch motifs cannot establish interval inversion; the Rust API returns `UnsupportedTransform` and has a dedicated regression test.
- A bounded deletion test checks the original source index recorded in the receipt.
- A fragment test checks that selected-subsequence indices map back to the original source sequence.
- Local JSON Schema validation succeeded for all 120 corpus records.
- A standalone Python reference simulation of the DP/verdict equations recorded 0/120 verdict mismatches, plus long-motif substitution-vs-gap controls that yielded 19 aligned events, one deletion and one insertion (95% coverage, 5% gap fraction). The new contract rejects these because selected-source deletions fail closed. This remains an algorithm-level simulation, not execution of the Rust source.


## Alignment gap-evasion regression (2026-10-28)

- Added 12 long-motif negative controls, one per existing motif family (20 events each). Each target has a single pitch substitution with the correct time origin.
- Independent simulator aligns 19/20 events and reports one source deletion plus one target insertion; all matched-axis residuals are within tolerance, coverage is 0.95 and gap fraction is 0.05. This demonstrates why coverage/gap thresholds alone are insufficient.
- Verification now rejects any unmatched source event. Target insertions are rejected by default and require explicit `allow_target_insertions=true` opt-in.
- Revised suite totals: 120 records, 60 accept / 60 reject; split counts development 60, calibration 20, test 40; 12 families remain split-isolated.
