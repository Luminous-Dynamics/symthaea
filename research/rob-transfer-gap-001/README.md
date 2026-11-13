# ROB-TRANSFER-GAP-001

Status: first executable descriptive comparison between familiar-source reference episodes and whole-source-group hidden transfer episodes.

Parent: [ROB-BENCHMARK-PROVENANCE-HOLDOUT-001 #11830](https://github.com/Luminous-Dynamics/symthaea/issues/11830). Compose with:
- #10814 for frozen dataset and split receipts;
- [#11863](https://github.com/Luminous-Dynamics/symthaea/pull/11863) for source-group holdout validation;
- [#11807](https://github.com/Luminous-Dynamics/symthaea/pull/11807) for paired independent-cluster inference;
- SafeVLA-style separate task-success and safety axes.

## Why this exists

A hidden-only benchmark reports performance on the hidden source distribution. It does not tell whether the policy was already weak on familiar sources or whether it collapses under source transfer.

This reference adds a **fresh familiar-source evaluation regime**: new cases/scenarios from source groups represented in training, with a separate evaluation manifest. The source group is familiar; the evaluated episodes are held out. It is compared against a hidden evaluation regime whose complete source groups are disjoint from training and calibration.

The input binds the training, calibration, familiar-reference and hidden-confirmatory manifests plus a split-separation receipt. The crate cannot prove caller-supplied training exclusion, truthful source labels, or that the upstream custodian actually withheld the cases. Those remain dataset-owner obligations.

## Source split semantics

For each required source dimension:
- familiar-source reference cases must belong to source groups assigned to Training;
- hidden-source transfer cases must belong to source groups assigned to HiddenConfirmatory;
- Calibration groups remain distinct and cannot be used as either evaluation regime;
- a whole source group cannot be assigned to more than one partition;
- familiar and hidden evaluation manifests must be distinct from training/calibration and from one another.

The reference requires at least two source groups and two independent clusters per regime (frozen minimums may be raised). Multiple episodes within one cluster do not increase the independent-cluster count.

## Outcome contract

Each scheduled episode has exactly one terminal outcome:
- safe-success;
- successful-but-unsafe;
- safe failure;
- unsafe failure;
- Unknown/censored.

The report preserves three separate axes:
- native task success = safe-success + successful-but-unsafe;
- safe-success;
- successful-but-unsafe (SBU).

Unknown episodes remain in the scheduled denominator and make the affected dimension Inconclusive. They are not silently discarded or treated as resolved failures.

## Transfer-gap vector

Rates are calculated within each independent cluster. Cluster rates are then equally weighted in the descriptive mean so a high-volume source cannot dominate the result with correlated episodes.

For outcome axis y:

    familiar_rate = mean(cluster rates from familiar source groups)
    hidden_rate   = mean(cluster rates from hidden source groups)
    transfer_gap  = hidden_rate - familiar_rate

Candidate improvement relative to baseline:

    difference_in_differences
      = candidate_transfer_gap - baseline_transfer_gap

Positive difference-in-differences indicates a less negative observed source-transfer gap for the candidate under this frozen scope. This is descriptive only; it is not a causal claim, confidence interval or p-value.

The report includes per-cluster rates, source group counts, cluster counts, episodes, min/max rates, per-axis familiar/hidden rates, transfer gaps and difference-in-differences.

## Statistical boundary

The crate does not run statistical inference. Source groups, independent clusters, episodes and dimensions are not interchangeable statistical units. Multiple dimension views of the same mission may be dependent. Use #11807 and preregistered cluster construction for confirmatory uncertainty.

## Running the reference

The package is dependency-free and isolated from the Symthaea root Cargo workspace:

    cargo fmt --manifest-path research/rob-transfer-gap-001/reference/Cargo.toml -- --check
    cargo test --locked --manifest-path research/rob-transfer-gap-001/reference/Cargo.toml
    cargo clippy --locked --manifest-path research/rob-transfer-gap-001/reference/Cargo.toml --all-targets -- -D warnings

The dedicated exact-head Rust 1.96 workflow is the qualification gate. Passing these synthetic fixtures does not prove physical generalization or benchmark superiority.
