# Verified Legacy Engineering Benchmark (VLEB)

VLEB is a draft research harness for evaluating legacy-code repair, migration and transaction integrity through reproducible engineering episodes. It is not a production banking certification or a measured Symthaea model score.

## Entry points

- [Research protocol and statistical plan](../../docs/research/VERIFIED_LEGACY_ENGINEERING_BENCHMARK_2026-10-24.md)
- [Task contract schema](task-contract.schema.json)
- [Evidence receipt schema](evidence-receipt.schema.json)
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

The latest completed VLEB workflow on head `b5556a6d457d42d3e155b1785108708e6641e6f0` passed formatting, fixture integrity, receipt-schema conformance, independent auditor tests, 20 Rust unit tests, 4 CLI integration tests, strict Clippy, mutation-generator unit tests, scorecard invariants, and the executable mutation campaign. Four registered semantic mutants were compiled and rejected by their preregistered target tests (0 survivors, 0 unresolved). The [run and artifact](https://github.com/Luminous-Dynamics/symthaea/actions/runs/40657264711) preserve the CI record and raw mutation evidence.

The scorecard deliberately keeps the Legacy-Bench model score, AASSR, security correctness and cost per independently verified fix at **not measured**. The public ledger task is a development fixture only. Its 4/4 mutation result is an oracle smoke test, not the 90% broad mutation target, and is not evidence of production banking or mainframe equivalence.

When follow-on commits change files under `tools/vleb/`, check the VLEB workflow at the exact new PR head. A passed result on an ancestor commit does not automatically validate the current head.
