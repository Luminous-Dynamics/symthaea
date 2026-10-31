# HARDSEC-020 — Workload Context Assignment Auditor

**Status:** Rust assignment auditor, versioned policy/stream/truth schemas and synthetic known-answer fixtures authored; no real workload classifier or platform metadata is authenticated  
**Issue:** [#9484](https://github.com/Luminous-Dynamics/symthaea/issues/9484)  
**Sequential integration:** [HARDSEC-019 / PR #9478](https://github.com/Luminous-Dynamics/symthaea/pull/9478)

## 1. Why context is a security boundary

HARDSEC-019's DKW-CCTM uses a separate frozen null reference for each workload context. Its statistical argument is conditional on the score distribution and context assignment being correct. Routing a memory-heavy window to the compute reference can generate false alarms or conceal a real distribution shift. A context label cannot simply be trusted because it appears in the input JSON.

Context assignment is now a separate, score-independent evidence product with its own artifact digest, confidence and margin thresholds, held-out truth evaluation and fail-closed runtime mapping.

## 2. Inputs

- **Policy:** fixed context vocabulary, assignment mode, assignment-artifact SHA-256, anomaly-score-artifact SHA-256, independent training/calibration group IDs, confidence/margin thresholds and pre-evaluation freeze declaration.
- **Assignments:** one ordered record per telemetry window, with candidate context probabilities, top-1 context/confidence, top-1/top-2 margin, sensor state and the declared accept/abstain decision.
- **Truth oracle:** held-out, offline context labels. These are never inputs to assignment decisions and their per-window labels are not copied into the runtime receipt.

The assignment artifact must have a distinct digest from the anomaly score artifact. The policy explicitly disallows deriving contexts from the same anomaly score or from CCTM wealth. This checks the declared contract; it does not prove how a real classifier was trained or executed.

## 3. Fail-closed rules

A record is accepted only when:
- sensor state is valid;
- the highest-probability context belongs to the preregistered vocabulary;
- top-1 confidence meets the frozen threshold;
- top-1/top-2 margin meets the frozen threshold;
- the supplied decision status, reason, selected label and candidate scores agree exactly.

Otherwise it must abstain:
- invalid/missing/stale/misaligned/multiplexed/untrusted sensor → `invalid_sensor`;
- top-1 label outside the declared vocabulary → `unknown_context`;
- confidence or margin below threshold → `ambiguous_assignment`.

The evaluator rejects incorrect top-1 labels, fabricated confidence/margin fields, candidate probabilities not summing to one, duplicate windows, missing/extra truth records, non-contiguous sequence identities and train/calibration/evaluation group overlap.

## 4. Coverage-aware evaluation

The offline truth oracle is joined by exact sequence/window identity and used only after assignment decisions are fixed. The receipt reports:
- assignment count, accepted and abstained counts;
- accepted coverage;
- accuracy on accepted assignments, with its explicit denominator;
- end-to-end correct-context coverage over *all* windows;
- per-true-context recall and a worst-declared-context recall;
- confusion matrix and abstention reasons.

The synthetic fixture has 19 windows: 16 accepted assignments, 3 abstentions, 100% accuracy among accepted assignments, and 16/19 end-to-end correct-context coverage (84.21%). These are fixture mechanics, not evidence of a real classifier's accuracy.

## 5. Runtime mapping receipt

The auditor emits `hardsec-context-assignment-receipt-v1`. Its `runtime_mapping.assignments` contains only sequence index, window ID, decision status/reason and selected context. It explicitly contains no truth labels. It binds policy, assignment stream and truth-oracle bytes by SHA-256 and includes the aggregate offline evaluation.

HARDSEC-019 now requires this receipt as `--context-receipt`. Before CCTM accumulates any score evidence it verifies exact policy ID, assignment artifact, stream/evaluation identity and exact sequence/window mapping. Accepted assignments must match the context in the CCTM score stream. Abstained assignments cannot update CCTM wealth, even if a caller supplies a known context string.

The runtime mapping is unsigned and caller-declared. It is not hardware attestation or proof that the actual classifier produced the probabilities. HARDSEC-019 now checks the mapping's exact policy/artifact/stream identities, each sequence/window pair, accepted selected context, truth-free status and aggregate counts before updating CCTM wealth.

## 6. Run the auditor

```bash
cargo test --locked \
  --manifest-path tools/hardsec-adaptive-evasion-audit-harness/Cargo.toml \
  --features hardsec-context-assignment-audit \
  --example hardsec_context_assignment_audit

cargo run --locked \
  --manifest-path tools/hardsec-adaptive-evasion-audit-harness/Cargo.toml \
  --features hardsec-context-assignment-audit \
  --example hardsec_context_assignment_audit -- \
  --policy tests/fixtures/hardsec/context-assignment-v1/policy.json \
  --assignments tests/fixtures/hardsec/context-assignment-v1/assignments.json \
  --truth tests/fixtures/hardsec/context-assignment-v1/truth-oracle.json \
  --receipt /tmp/hardsec-020-context-receipt.json
```

Thirteen authored Rust unit tests cover the known answer, score-artifact separation, train/calibration/evaluation leakage, candidate-score arithmetic, confidence/margin mismatch, invalid-sensor acceptance, duplicate windows and truth-oracle coverage/order errors.

## 7. Claim ceiling

A valid receipt establishes consistency of the declared policy, assignment decisions, exact stream identity and offline truth join. It does not authenticate the classifier binary, training process, metadata source or truth oracle; prove context exogeneity; establish within-context i.i.d. PMU scores; or prove detector security. A high accepted accuracy must always be reported beside coverage and end-to-end correct-context coverage.
