# Verified Legacy Engineering Benchmark (VLEB)

VLEB is a draft research harness for evaluating legacy-code repair, migration and transaction integrity through reproducible engineering episodes. It is not a production banking certification or a measured Symthaea model score.

## Entry points

- [Research protocol and statistical plan](../../docs/research/VERIFIED_LEGACY_ENGINEERING_BENCHMARK_2026-10-24.md)
- [Task contract schema](task-contract.schema.json)
- [Evidence receipt schema](evidence-receipt.schema.json)
- [Trusted run-manifest schema](run-manifest.schema.json)
- [Run-manifest validator and confirmatory attestation verification](verify_run_manifest.py)
- [Run-manifest schema and adversarial tests](test_run_manifest.py)
- [Run-manifest verifier tests](test_verify_run_manifest.py)
- [Confirmatory-run provenance procedure](CONFIRMATORY_RUNS.md)
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

The scorecard deliberately keeps the Legacy-Bench model score, AASSR, security correctness and cost per independently verified fix at **not measured**. The public ledger task is a development fixture only. Its 4/4 mutation result is an oracle smoke test, not the 90% broad mutation target, and is not evidence of production banking or mainframe equivalence.

When follow-on commits change files under `tools/vleb/`, check the VLEB workflow at the exact new PR head. A passed result on an ancestor commit does not automatically validate the current head.
