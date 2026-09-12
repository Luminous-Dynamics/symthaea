# symthaea-energy-benchmark-zero-suite

Five-fold evidence suite for Energy Discovery Benchmark Zero.

This crate composes the leakage qualification (#1769), fixed composition baseline (#1781), and property-blind controls (#1794) across all five Matbench v0.1 positional folds.

## Folds are partitions, not replications

The most important rule in this crate is semantic:

**five folds != five independent scientific replications**

They are disjoint evaluation partitions of one source dataset. Fold-level consistency is useful evidence about sensitivity to partitioning, but it must not be converted into an independent-replication count or a significance claim.

Every suite receipt carries this disclosure.

## Suite plan

`BenchmarkSuitePlan` binds one shared:

- target interval;
- top-k;
- exact baseline method version;
- exact blind-control method version;
- 256-control ensemble contract;
- five exact fold-plan SHA-256 values.

The suite plan can be generated before the benchmark artifact is read and receives its own SHA-256 for external preregistration if desired.

As in #1794, a digest proves content identity, not registration chronology.

## Partition integrity

Execution requires folds 0 through 4 exactly.

For each fold the suite independently obtains the leakage-qualified truth and requires:

- canonical fold population matches the frozen Matbench procedure;
- controlled receipt reproduces the same leakage-qualification identity;
- every retained candidate id appears in exactly one cleaned fold.

A candidate-partition SHA-256 binds the five qualification identities and every retained candidate id in fold order.

## Aggregate metrics

The suite reports:

- total retained candidate count;
- total excluded training-overlap count;
- total qualifying truth count;
- total top-k hit count;
- micro top-k precision;
- micro top-k recall;
- candidate-count-weighted baseline MAE;
- macro mean and maximum target regret;
- blind-control expected top-k hit count across folds;
- blind expected micro precision;
- baseline hit and precision lift over the blind mean;
- mean descriptive control fractions across folds;
- number of folds where baseline hits meet/exceed blind mean;
- number of folds where baseline regret is at/below blind median.

These aggregates remain measurement summaries. They are not inferential significance statistics.

## Evidence identity

A suite result content-addresses:

- suite plan;
- optional external registration evidence reference;
- five fold qualification/comparison receipts;
- exact cleaned candidate partition;
- aggregate metrics;
- final suite SHA-256.

The full fold receipts remain embedded so an aggregate cannot erase contrary fold-level evidence.

## CLI

Plan without reading benchmark truth:

`energy-benchmark-zero-suite plan <target-min-eV> <target-max-eV> <top-k>`

Execute all five folds:

`energy-benchmark-zero-suite run <matbench_expt_gap.json.gz> <target-min-eV> <target-max-eV> <top-k> [registration-evidence-ref]`

The CLI performs no network fetch and omits host-local artifact paths from evidence output.

## Deliberate non-claims

The suite does not establish:

- five independent replications;
- p-values or statistical significance;
- an official Matbench leaderboard result;
- historical blindness of the baseline;
- calibrated uncertainty;
- material novelty, synthesizability, or device performance;
- experimental validation;
- deployment superiority;
- material certification or physical authority.

The stack remains draft/unqualified until Cargo lock reconciliation and exact-head check/test/strict-Clippy execute under the pinned toolchain.
