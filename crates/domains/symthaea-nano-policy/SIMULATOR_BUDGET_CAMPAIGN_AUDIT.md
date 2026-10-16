# Nanophotonics Campaign Seed and Budget Reconciliation

**Status:** Rust campaign-audit implementation and synthetic smoke fixture; not a validation of the historical MEEP experiments.  
**Tracking:** [NANO-006: Reconcile actual MEEP call budgets and seed-count provenance](https://github.com/Luminous-Dynamics/symthaea/issues/7586)  
**Single-run ledger:** [SIMULATOR_BUDGET_LEDGER.md](./SIMULATOR_BUDGET_LEDGER.md)  
**Manifest schema:** [simulator-campaign-manifest.schema.json](./schema/simulator-campaign-manifest.schema.json)  
**Report schema:** [simulator-campaign-audit.schema.json](./schema/simulator-campaign-audit.schema.json)

## 1. What the campaign auditor adds

The single-run ledger can prove only that one frozen run's proposal records and simulator-call records reconcile according to the declared contract. It cannot prove that a comparison included the same complete seed set for all policy arms.

The new **nano-policy-campaign-audit** CLI reads a frozen campaign manifest and per-run ledger files under a declared root. It verifies:
- the explicit expected seed roster;
- exactly one ledger reference for each expected seed in each policy arm;
- no missing, extra, or duplicated seed references;
- raw-byte SHA-256 of each per-run ledger against the campaign manifest;
- benchmark, arm, policy ID/revision, exact seed, call budget, proposal-attempt cap, cost unit, task contract, evaluation manifest and simulator recipe identity;
- successful single-run actual-call budget reconciliation for every ledger;
- duplicate run IDs across the campaign.

The campaign report includes expected/referenced/verified run counts, per-arm missing seed references, missing verified seeds, duplicates, unexpected seeds and per-ledger blockers. This makes a discrepancy such as “the overview says six replications but detailed results list five” a machine-checkable evidence gap **when the actual run ledgers and expected roster are available**.

## 2. Example (synthetic fixture only)

~~~bash
cargo run --locked -p symthaea-nano-policy --bin nano-policy-campaign-audit -- \
  --manifest crates/domains/symthaea-nano-policy/fixtures/synthetic-sim-budget-campaign.json \
  --ledger-root crates/domains/symthaea-nano-policy/fixtures \
  > simulator-campaign-audit.json
~~~

The fixture deliberately has two policy arms and one seed, seed 42. It is a plumbing test, not enough replication for a confirmatory scientific claim. The files are pinned in the manifest by raw-byte SHA-256, and each per-run ledger is independently audited first.

CLI exit codes:
- **0**: all arms contain exactly one verified ledger for every expected seed;
- **3**: manifest is valid but a ledger, identity, seed-set, digest, or call-budget gate fails;
- **2**: malformed manifest, unsafe path, or invalid contract.

The CLI rejects absolute paths, path traversal, symbolic-link ledger files, path escapes from the declared root, ledgers over 64 MiB and combined run-ledger data above 512 MiB. The library repeats the aggregate byte cap so its callers cannot bypass the resource limit.

## 3. Using it for real MEEP studies

1. Obtain the actual per-run raw output/receipt manifests from the original study authors or an immutable artifact repository. Do not infer the missing sixth seed or regenerate it.
2. Create one run ledger per policy/seed that records actual simulator starts, terminal states, exact seed, simulator budget and all identities.
3. Freeze the intended seed roster in the expected-seeds field, and include exactly one relative path and raw-byte SHA-256 per run.
4. Use the same expected seed set and actual simulator-call budget for every arm. Preserve proposal failures, preflight rejections and simulator failures; do not trim the denominator to finite results.
5. Resolve the README/detailed-results five-versus-six discrepancy by reference to raw run receipts or source-author evidence. If the original records cannot be obtained, report the historic count as unresolved and run a new properly preregistered comparison rather than claiming the old set was reconstructed.
6. For a confirmatory comparison, require at least 20 paired seeds **or a documented power analysis agreed before unblinding**. The campaign auditor checks roster completeness; it does not claim 20 seeds always suffice statistically.

## 4. Evidence boundaries

A passing campaign report means per-run ledger files, their expected seeds, policy identities and actual-call budget audits match the frozen campaign contract. It does not independently open the simulator start/terminal/result artifacts referenced inside those per-run ledgers, prove MEEP ran, validate the optical physics, guarantee independent seeds, or prove policy quality. Receipt digests inside each per-run ledger are still only shape-checked in the current structural auditor; a separate artifact-root verifier remains future work.

The audit is a provenance and denominator gate. It does not resolve the historical study's discrepancy without raw records, and it does not create or report a scientific benchmark result.
