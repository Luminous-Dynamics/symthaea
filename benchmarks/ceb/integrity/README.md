# CEB Benchmark Integrity Corpus

This directory contains the **design-only** adversarial and positive-control catalog for the CEB benchmark-integrity lane.

- [Attack catalog V0](attack-catalog.v0.json): 36 negative/adversarial and 10 positive/valid-state controls.
- [Protocol and targets](../../../docs/research/CEB_BENCHMARK_INTEGRITY_PROTOCOL_V0_2026-11-04.md): threat model, metrics, confidence bounds, lifecycle and qualification protocol.
- [Task CEB-VER-001](../catalog.v0.json): benchmark integrity task in the main CEB catalog.
- [Tracking issue #10138](https://github.com/Luminous-Dynamics/symthaea/issues/10138).

**No execution is claimed.** All entries are `design_only_not_executed`. Candidate-controlled code, task build scripts, tests and attack payloads must not run until #10064 is qualified. Static verifier tests may be authored separately, but are not evidence of runner isolation.
The CEB Contract Validation V0 workflow also runs `scripts/audit_doc_command_claims.py` to check tracked Markdown Cargo manifest/binary references without executing any documented command.

## Measurement and comparative claims

The V4V extension explicitly tests replay, repeatability, provenance, same-input label consistency, trivial verifiers and mutation sensitivity.

The CEB-wide [measurement matrix](../measurement-matrix.v0.json) and [comparative claim protocol](../../../docs/research/CEB_MEASUREMENT_CLAIM_PROTOCOL_V0_2026-11-05.md) define the primary metrics, external comparators, baseline arms, evidence authorities, hard gates and holdout design across 13 CEB lanes. The matrix is design-only; it does not contain CEB performance results.
