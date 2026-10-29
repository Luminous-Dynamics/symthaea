# Verified Legacy Engineering Benchmark (VLEB)

VLEB is a draft research harness for evaluating legacy-code repair, migration and transaction integrity through reproducible engineering episodes. It is not a production banking certification or a measured Symthaea model score.

## Entry points

- [Research protocol and statistical plan](../../docs/research/VERIFIED_LEGACY_ENGINEERING_BENCHMARK_2026-10-24.md)
- [Industry benchmark comparison and next frontier](../../docs/research/VLEB_INDUSTRY_COMPARISON_2026-10-28.md)
- [Task contract schema](task-contract.schema.json)
- [Requirement-to-oracle traceability protocol](../../docs/research/VLEB_REQUIREMENT_ORACLE_TRACEABILITY_2026-10-28.md)
- [Pinned admission-cohort schema](admission-cohort.schema.json)
- [Admission-cohort rollup tool](summarize_admission_cohort.py)
- [Admission-cohort adversarial tests](test_summarize_admission_cohort.py)
- [Auditor guardrail mutation campaign](run_auditor_mutation_campaign.py)
- [Two-sided oracle-control replay verifier](verify_oracle_controls.py)
- [Two-sided oracle-control integration/adversarial tests](test_verify_oracle_controls.py)
- [Two-sided oracle replay protocol](../../docs/research/VLEB_TWO_SIDED_ORACLE_REPLAY_2026-10-29.md)
- [Paired task-cluster comparison analyzer](compare_paired_runs.py)
- [Paired comparison analyzer adversarial tests](test_compare_paired_runs.py)
- [Paired comparison and uncertainty protocol](../../docs/research/VLEB_PAIRED_COMPARISON_PROTOCOL_2026-10-29.md)
- [Admission-cohort rollup protocol](../../docs/research/VLEB_ADMISSION_COHORT_ROLLUP_2026-10-29.md)
- [Evidence receipt schema](evidence-receipt.schema.json)
- [Trusted run-manifest schema](run-manifest.schema.json)
- [Run-manifest validator and confirmatory attestation verification](verify_run_manifest.py)
- [Run-manifest schema and adversarial tests](test_run_manifest.py)
- [Run-manifest verifier tests](test_verify_run_manifest.py)
- [Confirmatory-run provenance procedure](CONFIRMATORY_RUNS.md)
- [Language capability matrix](language-capability-matrix.json)
- [Language capability claim-boundary tests](test_language_capability_matrix.py)
- [Read-only fixed-format COBOL indexer](cobol-index/README.md)
- [COBOL index output schema](cobol-index/index-output.schema.json)
- [COBOL index output integration tests](test_cobol_index_output.py)
- [Atomic legacy claim evidence schema](legacy-claim-evidence.schema.json)
- [Atomic legacy claim evidence negative controls](test_legacy_claim_evidence.py)
- [Legacy claim source/witness auditor](audit_legacy_claim.py)
- [Legacy claim auditor adversarial tests](test_audit_legacy_claim.py)
- [Synthetic ledger task contract](examples/ledger-reconciliation/task.json)
- [Rust reference implementation and test instructions](ledger-reconcile/README.md)
- [Fixture integrity gate](check-fixture-integrity.sh)
- [Independent receipt auditor](audit_receipt.py)
- [Receipt auditor adversarial tests](test_receipt_auditor.py)
- [Registered mutation catalog](mutation-catalog.json)
- [Mutation campaign runner](run_mutation_campaign.py)
- [Mutation-generator unit tests](test_mutation_campaign.py)
- [Evidence-qualified scorecard](scorecard.json)
- [Scorecard invariant tests](test_scorecard.py)

## Current validation state

The latest completed baseline VLEB workflow before the run-manifest additions passed at head `6386bcfb45d28dc431fd9285bf5afb0f35ffd3c2`, run [40657774817](https://github.com/Luminous-Dynamics/symthaea/actions/runs/40657774817). It passed formatting, fixture integrity, receipt-schema conformance, independent auditor tests, 20 Rust unit tests, 4 CLI integration tests, strict Clippy, mutation-generator unit tests, scorecard invariants, and the semantic mutation campaign. Four registered semantic mutants were compiled and rejected by their preregistered target tests (0 survivors, 0 unresolved). The raw mutation artifact is [12802810670](https://github.com/Luminous-Dynamics/symthaea/actions/runs/40657774817/artifacts/12802810670), archive SHA-256 `3a4b44a9f4b1bf5d3ec7027fb4266def6381d69d5d50c74975071f500c7c8f74`.

The run-manifest schema, verifier and its tests were added after that passing baseline; their CI must pass at the exact current head before the new provenance path is called validated. The current verifier additionally requires out-of-band digests for its own implementation, schema, permitted receipt-auditor source and receipt-verifier binary, checks the signed certificate's immutable repository/workflow/run claims, and audits only files extracted from the attested ZIP. These new controls remain unvalidated until the current-head workflow completes successfully.

The paired comparison analyzer is a new analysis-only tool; its synthetic tests and bootstrap calculations are not benchmark results. The two-sided oracle-control replay is a separate tool that executes the pinned empty baseline and reference patch on isolated shallow checkouts with no source refs or shared Git object store; it still requires a hardened runner for network/resource isolation and signed provenance. Its exact-head workflow is required before the code path is described as CI-validated. The scorecard deliberately keeps the Legacy-Bench model score, AASSR, security correctness and cost per independently verified fix at **not measured**. The public ledger task is a development fixture only. Its 4/4 mutation result is an oracle smoke test, not the 90% broad mutation target, and is not evidence of production banking or mainframe equivalence.

When follow-on commits change files under `tools/vleb/`, check the VLEB workflow at the exact new PR head. A passed result on an ancestor commit does not automatically validate the current head.

## Legacy claim evidence

The atomic legacy claim contract separates textual anchors, static structure, and runtime behavior. The follow-up gate in [PR #8707](https://github.com/Luminous-Dynamics/symthaea/pull/8707) wires evaluator-pinned claim inventories into confirmatory run-manifest score eligibility: the manifest pins the claim auditor/schema and every admitted witness-verifier kind out of band, the verifier audits only files extracted from the attested ZIP, and confirmatory eligibility now requires both the independent receipt audit and claim-evidence audit to pass. It reports supported, contradicted, and unresolved claim counts separately. A static index witness cannot certify a transaction or numeric-behavior claim; those require an independent execution witness.

**Validation boundary:** the claim auditor recomputes source-file/source-span hashes, verifies witness-artifact bytes, and compares witness verifier digests to external pins. It does not independently rerun the witness verifier or transaction scenario, and the next hardening step is evaluator-produced witness receipts bound to task, source, run attempt, command, inputs/outputs and runtime identity (tracked in [issue #8702](https://github.com/Luminous-Dynamics/symthaea/issues/8702)). Keep this draft gate unmerged until exact-head tests pass; do not interpret a declared `observed_result` or a valid hash chain as a semantic correctness certificate.
