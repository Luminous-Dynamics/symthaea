# IIC-Eval v0.2 Rust evaluator

This evaluator compares system submissions with the separately stored development oracle. It is intentionally narrower than a semantic adjudicator.

## Run

```sh
cargo test --manifest-path benchmarks/iic-eval/v0.2/harness/Cargo.toml
cargo run --manifest-path benchmarks/iic-eval/v0.2/harness/Cargo.toml -- \
  benchmarks/iic-eval/v0.2/cases.json \
  benchmarks/iic-eval/v0.2/oracle.json \
  submission.json
```

## Submission contract

The submission must contain one result for every case. Each candidate cites a source document and its version, an exact UTF-8 byte range `[span_start_byte, span_end_byte)`, the quoted source text, modality, conditions and adjudication state. Byte offsets are used deliberately: UTF-8 code-point offsets and byte offsets are not interchangeable.

## Metrics

The evaluator reports source-span precision/recall/F1, valid UTF-8 span count, source-version binding, exact modality/condition/state agreement on matched spans, false mandatory promotions, explicit-prohibition preservation and duplicate candidates.

The evaluator rejects duplicate case IDs, missing cases, duplicate source document IDs, and oracle references to absent sources. Candidate-level wrong spans are scored as unmatched rather than crashing the run.

## Limits

Exact quotation is not semantic equivalence. The oracle is a development seed without independent adjudication. The evaluator does not score clarification quality or reason semantically about prohibited inferences; those require blinded expert review. It does not implement legal interpretation or authority adjudication.
