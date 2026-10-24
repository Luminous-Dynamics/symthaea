# Synthetic Ledger Reconciliation Fixture

**Purpose:** a small, fully synthetic contract fixture for developing VLEB's task-admission and differential-comparison machinery. It is not an executed benchmark and is not eligible for an agent score yet.

## Required behavior

Read the CSV input. Every row has a unique transaction ID, account ID, side (CREDIT or DEBIT), and positive integer amount in minor units.

1. Exact duplicate transaction rows are idempotent and applied once.
2. Reusing a transaction ID with different content is a hard conflict: reject the input, return a nonzero status, and apply no partial ledger changes.
3. Use integer minor units and checked integer arithmetic; do not use binary floating-point for financial amounts.
4. Report per-account credit, debit, and net totals in lexicographic account order, along with global totals, unique transactions applied, and exact duplicate rows ignored.
5. Unknown side, non-positive amount, malformed row, arithmetic overflow, or conflicting duplicate must fail closed.
6. The reported net total must equal total credits minus total debits, and all account totals must sum to the corresponding global totals.

## Fixture files

- input.csv: normal ledger input containing one exact duplicate row.
- expected.json: expected canonical output for that input.
- conflicting-duplicate.csv: negative control reusing an ID with a different amount. Expected behavior is rejection with no partial effects.
- task.json: machine-readable VLEB task-contract blueprint.

## Expected normal-case summary

- Unique transactions applied: 4
- Exact duplicate rows ignored: 1
- Credits: 2049 minor units
- Debits: 350 minor units
- Net: 1699 minor units

All amounts are integer minor units; no currency-specific interpretation is intended.

## Admission status

This is a **contract/fixture prototype only**. Before it can be a scored task, add a real implementation target, independent reference implementation, baseline/empty-patch and gold/reference executions, actual toolchain/container identity, hashes of all evidence, and reviewer-validated behavior/mutation tests. Do not present static fixture review as compilation or runtime evidence.
