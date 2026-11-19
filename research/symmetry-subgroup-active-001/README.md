# Active finite-subgroup identification reference v0.1

First executable reference for issue #12883, reusing the exact adaptive and exact static design algorithms in PR #12670.

## Paradigm

The unknown task-preserving set is not modeled as an arbitrary subset of a finite group. Under a valid action and complete task semantics, the stabilizer is a subgroup. V0 therefore enumerates the exact subgroup lattice of a known finite ambient group G, then asks which subgroup hypothesis H ≤ G is supported by exact element-membership checks.

A probe asks whether one group element preserves the complete frozen task in one declared context. The task checker is external to this crate; its result carries a checker ID, task-contract ID and evidence/counterexample reference. The planner chooses which element to check next.

## Implemented

- Complete finite-group table validation: range, Latin rows/columns, identity, two-sided inverses and associativity.
- Exhaustive subgroup enumeration with identity, inverse and product closure.
- Explicit group-order, subgroup-count and work budgets; no partial lattice is returned as complete.
- Candidate subgroup hypotheses compiled into the existing exact adaptive minimax planner.
- Exact minimum-cost static probe-basis baseline on bounded probe families.
- Explicit unresolved-pair frontier when available probes cannot separate all subgroup hypotheses.
- Evidence-bound membership observations for Preserved, Violated and Unknown.
- Unknown observations pause the policy and preserve the entire current candidate frontier.
- A membership pattern inconsistent with every subgroup returns Inconsistent; no nearest subgroup is selected.
- Exact policy, observation and receipt replay.

## Frozen group-lattice controls

- C4: 3 subgroups.
- V4: 5 subgroups.
- S3: 6 subgroups.
- D4: 10 subgroups.

These counts are checked against independent small-group tables. The D4 fixture replays every possible target subgroup through the adaptive policy and confirms singleton identification.

The D4 adversarial control declares {e, r, s} preserving and rejects r*s. Since that set is not a subgroup, the policy must terminate with an empty compatible frontier and explicit model inconsistency.

## Cost metrics

The policy receipt reports the exact adaptive worst-case query cost. The static baseline is the exhaustive minimum-cost element-probe set when its subset budget permits. Full-scan cost is the sum of all admissible probe costs. These are design metrics, not measured physical experiment throughput.

The prospective target is at least 1.5x reduction against a full all-element scan on at least one nontrivial frozen group, while retaining 100% subgroup identification and zero false identifications. No broad speedup is claimed from these fixtures alone.

## Reproduction

    cargo fmt --manifest-path research/symmetry-subgroup-active-001/Cargo.toml --all -- --check
    cargo test --locked --manifest-path research/symmetry-subgroup-active-001/Cargo.toml
    cargo clippy --locked --manifest-path research/symmetry-subgroup-active-001/Cargo.toml --all-targets -- -D warnings

## Evidence boundary

This crate verifies abstract group tables, subgroup closure, policy construction and consistency of membership observations with the subgroup lattice. It does not verify the external evidence references or prove that a real system obeys the declared group action. The receipt's pruning authority is always none. The downstream quotient compiler must still verify the finite action, all task semantics, orbit witnesses, complete domain coverage and solution lifts before hard pruning.

Statistical or learned symmetry scores are not exact membership evidence. General noisy ODT/ASRN is not reimplemented; unknown observations pause, and bounded/persistent uncertainty belongs in the separate active-noise reference.
