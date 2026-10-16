# Nanophotonics Simulator-Call Budget Ledger

**Status:** Contract and structural auditor prototype; it has not been run against real MEEP output.  
**Tracking:** [NANO-006: Reconcile actual MEEP call budgets and seed-count provenance](https://github.com/Luminous-Dynamics/symthaea/issues/7586)  
**Kernel:** [symthaea-nano-policy](./README.md)  
**Audit schema:** [simulator-call-ledger.schema.json](./schema/simulator-call-ledger.schema.json)  
**Report schema:** [simulator-call-audit.schema.json](./schema/simulator-call-audit.schema.json)

## 1. Why this is required

A configured loop count, candidate proposal, passing preflight, simulator process invocation, finite output, and usable in-spec candidate are different quantities. They must have different counters. In a simulation-budget comparison, **every actual simulator invocation consumes one unit of the simulator-call budget**, including failed, timed-out, interrupted, and non-finite calls. A candidate rejected by a pre-simulation heuristic consumes a proposal attempt but not a simulator call.

The pinned public study [pberlizov/nanophotonics-inverse-design at `f2475999c0ad24f8b5ae2f26982c0ad19b257236`](https://github.com/pberlizov/nanophotonics-inverse-design/tree/f2475999c0ad24f8b5ae2f26982c0ad19b257236) is a useful case study, not a claim that the study is invalid:
- Its README describes a six-seed B=100 pilot, with 15.0 ± 3.2 in-spec for surrogate ranking and 10.6 ± 4.8 for sigma-only.
- `docs/SIM_BUDGET_REPLICATION_RESULTS.md` reports five replicates/seeds (3026, 4026, 5026, 6026, 7026).
- In `scripts/run_sim_budget_study.py`, random and sigma policies reject heuristic-DRC failures before the MEEP call, while sigma's Optuna loop is configured by trial count. A configured trial is therefore not necessarily a MEEP invocation.
- `src/nano_inv/sim_budget.py` counts only finite split values as `n_meep` in its summary. A failed or non-finite MEEP response can consume a call but be omitted from that counter.

These are denominator/provenance reconciliation questions. They do not establish that the study's conclusions are wrong. The source repository's README also says result data are gitignored, so raw replication manifests and invocation-level receipts are needed to resolve the seed and call-count discrepancy.

Pinned files:
- [README.md](https://github.com/pberlizov/nanophotonics-inverse-design/blob/f2475999c0ad24f8b5ae2f26982c0ad19b257236/README.md)
- [Replication results](https://github.com/pberlizov/nanophotonics-inverse-design/blob/f2475999c0ad24f8b5ae2f26982c0ad19b257236/docs/SIM_BUDGET_REPLICATION_RESULTS.md)
- [Simulation-budget runner](https://github.com/pberlizov/nanophotonics-inverse-design/blob/f2475999c0ad24f8b5ae2f26982c0ad19b257236/scripts/run_sim_budget_study.py)
- [Existing summary helper](https://github.com/pberlizov/nanophotonics-inverse-design/blob/f2475999c0ad24f8b5ae2f26982c0ad19b257236/src/nano_inv/sim_budget.py)

## 2. Ledger contract

The Rust module `sim_budget` consumes a frozen `nano-simulator-call-ledger-v1` JSON document and emits an auditable summary.

Each per-run ledger records the exact integer `seed` alongside the run ID and policy arm. This preserves per-run seed provenance; a campaign-level comparison must still check the declared seed sets across arms rather than infer pairing from aggregate tables.

Each proposal-attempt row records:
- one-based contiguous sequence and stable attempt identity;
- whether generation succeeded, candidate ID and canonical candidate digest;
- preflight state and rejection reason;
- generation/preflight cost and elapsed time;
- either a started simulator invocation record or an explicit reason why an eligible candidate was not invoked.

A simulator invocation record binds:
- invocation ID;
- start-receipt digest;
- terminal-receipt digest;
- terminal status (`returned_finite_result`, `returned_nonfinite_result`, `failed`, `timed_out`, or `interrupted`);
- result-artifact digest when a response was returned;
- simulator cost and elapsed time.

**Counting rule:** the presence of an invocation record means one actual simulator-call budget unit was consumed. A start followed by a crash, timeout, interrupted run, or non-finite result still counts as a call. A DRC reject or generation failure without an invocation record does not.

Every policy arm gets the same expected actual-call budget (B) and a separate proposal-attempt cap. If the attempt cap is hit before (B) calls are started, the run is incomplete; do not silently compare it as a smaller budget. Attempts after the (B)-th simulator call are retained for audit and prevent an exact-protocol completion claim.

The ledger also records the call count claimed by the previous run summary and the exact per-run seed. The audit report preserves that seed; NANO-006 still requires raw manifests/receipts to reconcile the reported replicate count. The auditor compares it with the count recomputed from invocation records. This directly exposes summary semantics such as “finite outputs counted as MEEP calls.”

## 3. Example

Run against the checked-in **synthetic fixture only**:

```bash
cargo run --locked -p symthaea-nano-policy --bin nano-policy-budget-audit -- \
  --input crates/domains/symthaea-nano-policy/fixtures/synthetic-sim-budget-ledger.json \
  > simulator-budget-audit.json
```

The example deliberately reports one simulator call while recording two actual invocations: one finite result and one failed call. The auditor produces a JSON report with:
- `expected_simulator_calls = 2`;
- `reported_simulator_calls = 1`;
- `observed_simulator_calls = 2`;
- one preflight rejection;
- one finite result and one simulator failure;
- `reported_count_matches_observed = false`;
- a non-passing audit disposition.

The CLI exits 3 for a valid but non-reconciling/incomplete ledger, 2 for malformed input, and 0 only when the run claims completion, the exact actual-call budget is met with terminal receipts, and the original reported call count matches. A second synthetic fixture with reported count 2 exercises the passing call-accounting path. These are software fixtures, not MEEP or physical results.

## 4. Audit ladder and hard thresholds

| Gate | Requirement | Failure disposition |
|---|---|---|
| Source identity | Pin exact run, policy, data/source, evaluation, task and simulator-recipe manifests | Block; no benchmark claim |
| Proposal accounting | Every generation/proposal attempt represented; attempts do not exceed the separate cap | Invalid/incomplete |
| Preflight accounting | Rejections are counted as proposals but do not inflate simulator calls | Invalid accounting |
| Invocation accounting | Each actual start has a unique invocation ID and a start plus terminal receipt digest | Incomplete if unresolved; block if missing |
| Call budget | Exactly B actual invocations per arm | Under-budget incomplete or over-budget |
| Output denominator | Finite results, non-finite responses, failures, timeouts and interrupted calls reported separately | Invalid denominator if excluded |
| Seed provenance | Per-run seed IDs reconcile with raw manifests and aggregates | Block claims on unresolved n |
| Cost accounting | Proposal and simulator costs use one declared unit; failure costs remain included | No end-to-end cost claim |
| Metric recomputation | Independent metrics reconstructed from raw post-outcome observations and receipts | No performance claim until passing |

For the photonic splitter example, the external study declares the frozen objective as
`abs(split_ratio_upper - 0.5) <= 0.05` at 1550 nm under `phase0_v1`. Insertion loss remains a separate metric/constraint; it must not disappear into an unlabelled scalar. The call ledger does not evaluate this objective—that belongs to the separate post-outcome metrics evaluator.

**Confirmatory design target:** where practical, use at least 20 paired seeds for the next confirmatory comparison or a documented power analysis agreed before exposing held-out results. This is the acceptance proposal in NANO-006, not proof that 20 is always sufficient. Retain the five-vs-six-seed discrepancy until raw per-run manifests resolve it.

## 5. Claim boundaries

This module validates ledger shape, counter semantics and run-completion consistency. It does **not** dereference receipt digests, run MEEP, prove that a process actually ran, validate electromagnetic physics, or establish material/device performance. A future artifact verifier should hash the referenced start/terminal/result files from a frozen artifact root. An equal-budget pass is only an accounting gate; it does not establish an equal-cost comparison, quality improvement, fabrication robustness, or scientific validity.

No historical MEEP numbers have been recomputed by this tool. No raw external run receipts were available in the source files reviewed during this implementation.
