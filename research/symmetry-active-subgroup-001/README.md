# Active task-preserving subgroup identification v0.1

Implementation of issue #12883, reusing the exact adaptive experiment planner in PR #12670.

## Paradigm shift

Do not query an unconstrained list of candidate subsets. First enumerate the mathematically valid subgroup lattice of the known finite ambient group. Then use element-membership tests to identify which subgroup is consistent with the observed task-preservation evidence.

The stabilizer of a task under a valid group action is a subgroup. Therefore a membership pattern that is not closed under the group law is a model/checker/evidence inconsistency—not a nearby subgroup to be guessed.

## V0 implementation

- Validate group identity, inverses, multiplication outputs, associativity and element IDs.
- Exhaustively enumerate all subsets containing the identity and retain only those closed under inverses and multiplication.
- Bound group order to 16, subgroup hypotheses to 16, membership probes to 64, and enumeration/planning work by explicit budgets.
- Convert element-membership predictions to 0 = violated and 1 = preserved outcomes and reuse the exact adaptive minimax planner.
- Also compute the exact minimum-cost static probe basis.
- Return explicit incomplete identification when the admissible probes cannot distinguish every subgroup pair.
- Bind each observation to the task-contract ID, checker ID and evidence reference.
- Unknown observations do not eliminate hypotheses. Contradictory repeated outcomes and non-subgroup preservation patterns return model inconsistency.
- Replan the next exact adaptive probe on the current candidate frontier.
- The certificate and replay receipt explicitly set pruning_authorized = false.

## Frozen lattice oracle

| Group | Order | Expected number of subgroups |
|---|---:|---:|
| C4 | 4 | 3 |
| V4 | 4 | 5 |
| S3 | 6 | 6 |
| D4 | 8 | 10 |

The test suite also checks a D4 non-subgroup pattern {e, r, s} where r*s = rs is not preserved; the replay must return an empty candidate set and ModelInconsistent, never snap to the nearest subgroup.



## Probe-cost benchmark fixtures

The current frozen fixtures exclude the identity membership test because it is true for every subgroup and carries no information.

| Fixture | Full admissible scan | Exact adaptive worst case | Exact static basis |
|---|---:|---:|---:|
| C4 (three non-identity probes, unit cost) | 3 | 2 | 2 |
| D4 (seven non-identity probes, costs 1,1,1,2,2,2,2) | 11 | 9 | 10 |

These are exact model-cost fixtures. They are not wall-clock speedups or evidence of fewer physical measurements in a real system. The certificate also reports the summed per-hypothesis path cost under a uniform subgroup prior, so the assumed prior is visible rather than implicit. The equality assertions are authored regression tests and remain pending the exact-head Rust 1.96 run.

## Reproduction

    cargo fmt --manifest-path research/symmetry-active-subgroup-001/Cargo.toml --all -- --check
    cargo test --locked --manifest-path research/symmetry-active-subgroup-001/Cargo.toml
    cargo clippy --locked --manifest-path research/symmetry-active-subgroup-001/Cargo.toml --all-targets -- -D warnings

## Comparison and metrics

Compare exact adaptive minimax against:
- full scan of every admissible group-element membership probe;
- exact minimum-cost static probe basis;
- greedy worst-case probe choice;
- entropy choice under a separately stated prior (non-authoritative);
- theorem-matched COLT 2026 statistical subgroup discovery, only within its data/identifiability assumptions.

Report subgroup-lattice exactness, invalid subset rejection, adaptive worst-case cost, static basis cost, full-scan cost, expected cost under declared priors, observation count, evidence/checker cost, work units, runtime, memory, and unresolved cases.

## Claim boundary

This identifies a candidate task-preserving subgroup from exact finite membership evidence under the caller-declared context. It does not prove the caller's checker is correct, infer an unknown ambient group, handle noisy/statistical observations, establish physical symmetry, or authorize search quotient pruning. The downstream quotient compiler must still verify the full action, task semantics, orbit coverage and lift witnesses.
