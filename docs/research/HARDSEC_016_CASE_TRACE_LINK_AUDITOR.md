# HARDSEC-016 — Adaptive Case / Trace / Window Link Auditor

**Status:** executable Rust linker and synthetic known-answer fixture authored; exact-head CI is not yet observed  
**Issue:** [#9463](https://github.com/Luminous-Dynamics/symthaea/issues/9463)  
**Prerequisites:** [HARDSEC-012 PR #9410](https://github.com/Luminous-Dynamics/symthaea/pull/9410) and [HARDSEC-014 PR #9416](https://github.com/Luminous-Dynamics/symthaea/pull/9416)

## Purpose

HARDSEC-012 evaluates adaptive-evasion outcomes against declared program, execution, device, attack-family and adaptation-family groups. HARDSEC-014 audits trace/window ancestry, duplicate content, source offsets and split lineage. HARDSEC-016 makes the join explicit: every adaptive case must resolve to a named trace and one or more named windows, and the trace metadata must agree exactly with that case's manifest metadata.

This avoids the invalid inference that a correctly shaped case manifest automatically proves the provenance of its rows.

## Input contract

1. The HARDSEC-012 adaptive case manifest.
2. The exact HARDSEC-014 trace lineage manifest.
3. The HARDSEC-014 auditor receipt generated from those exact trace-manifest bytes.
4. A versioned case-link sidecar with exact case ID, trace ID and one-or-more window IDs.

The sidecar schema is `schemas/hardsec-adaptive-case-trace-links-v1.schema.json`. Every adaptive case, including train and validation metadata, must appear exactly once. Extra or missing case IDs are rejected.

For each link, the linker checks exact equality of split, program group/program ID, execution group/execution ID, device configuration, attack family and adaptation family. Unknown trace origins cannot be linked. A global window ID cannot be reused. Identical window-content hashes cannot be assigned to different adaptive cases. A trace may supply multiple cases only through distinct windows and identical declared trace-level metadata.

## Trace-receipt reconciliation

The supplied HARDSEC-014 receipt must have the expected schema, `result=audited`, the expected claim scope and dataset ID. Its `inputs.manifest_sha256` must equal the SHA-256 of the exact trace-manifest bytes supplied to HARDSEC-016. Source-corpus digest, trace/window counts, origin counts, duplicate-content groups, split-window counts and unique group counts are recomputed and compared.

The receipt is unsigned. Digest matching binds its declared input; it does not authenticate the receipt author or prove that a particular binary generated it. The linker is a consistency gate, not a cryptographic attestation system.

## Run

First run the HARDSEC-014 auditor from its own branch/checkout (or from the repository after PR #9416 is merged) against the exact trace manifest used below. The HARDSEC-014 source and harness are intentionally not copied into this stacked PR. The focused workflow here uses a fixed synthetic receipt only to test the receipt/linkage contract:

```bash
cargo run --locked \
  --manifest-path tools/hardsec-trace-lineage-audit-harness/Cargo.toml \
  --features hardsec-trace-lineage-audit \
  --example hardsec_trace_lineage_audit -- \
  --manifest path/to/trace-lineage.json \
  --receipt /tmp/trace-lineage-receipt.json
```

Then run the linker from the adaptive-evasion harness:

```bash
cargo run --locked \
  --manifest-path tools/hardsec-adaptive-evasion-audit-harness/Cargo.toml \
  --features hardsec-case-trace-link-audit \
  --example hardsec_case_trace_link_audit -- \
  --adaptive-manifest path/to/case-manifest.json \
  --trace-manifest path/to/trace-lineage.json \
  --trace-receipt /tmp/trace-lineage-receipt.json \
  --case-links path/to/case-trace-links.json \
  --receipt /tmp/case-trace-link-receipt.json
```

The receipt is deterministic, unsigned, input-digest-bound and created with no-overwrite semantics. Put it outside every input directory. For a real campaign, do not substitute the synthetic receipt: generate the receipt with the pinned HARDSEC-014 auditor from the exact same trace-manifest bytes.

## Synthetic known-answer fixture

The fixture contains 11 adaptive cases (one train, one validation, nine test), 12 trace records and 12 windows. One repeated-iteration trace is a child of the original test trace; it reuses a window-content digest, but is not mapped to a second case. The expected linkage is 11/11 cases, 11 mapped traces, 11 mapped windows. The lineage side reports one repeated trace, one duplicate-content group and one extra duplicate occurrence.

These are fixture mechanics only. They are not HARPY-V results, detector results or evidence of physical/simulator performance.

## Claim ceiling

The output leaves `effective_sample_size_estimate=null` and `independence_status=not_established`. Linkage counts do not prove independent groups, source-file authenticity, oracle correctness, causal attack effect, cross-platform transfer or security. Repeated/replayed traces remain within their source lineage and do not create independent programs or executions.
