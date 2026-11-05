# CEB Rust Pilot Balanced Sampling Plan V0 — 2026-11-05

**Status:** Design-only; six repository placeholders, zero actual repository/task assignments.  
**Related issue:** [#10614](https://github.com/Luminous-Dynamics/symthaea/issues/10614)  
**Existing pilot:** [CEB Rust repository-repair pilot](./CEB_RUST_PILOT_PREREGISTRATION_V0_2026-11-05.md)  
**Machine-readable plan:** [`benchmarks/ceb/rust-balanced-sampling.v0.json`](../../benchmarks/ceb/rust-balanced-sampling.v0.json)  
**Schema:** [`benchmarks/ceb/rust-balanced-sampling.schema.json`](../../benchmarks/ceb/rust-balanced-sampling.schema.json)  
**Validator:** [`scripts/validate_ceb_balanced_sampling.py`](../../scripts/validate_ceb_balanced_sampling.py)

## 1. Why balance is necessary

A 48-task benchmark can produce a misleading result if repository identity, fault family, task difficulty and holdout membership are confounded. For example, a model may appear weak at security repair only because all security tasks came from one difficult repository.

SWE-Bench Pro V2 uses 642 tasks across 11 repositories and removes invalid/contradictory tasks through expert review. Rust-SWE-bench uses 500 tasks across 34 repositories and filters issues through provenance, test-change, fail-to-pass and manual leakage/quality review. These support a cross-repository design, but do not prove that CEB's 48 tasks are representative. Sources: https://labs.scale.com/leaderboard/swe_bench_pro_public_v2 and https://arxiv.org/abs/2602.22764

## 2. Balanced assignment

The plan reserves six repository placeholders and six fault families. The 48 slots are formed by:
- **36 base cells:** one task slot in every repository × fault-family cell.
- **12 extra slots:** two per repository, assigned to its own family and the next cyclic family.

This yields exactly eight slots per repository and eight per fault family.

The holdout assignment is balanced:
- 16 holdout and 32 development slots.
- Each repository receives two or three holdout slots.
- Each fault family receives two or three holdout slots.

The deterministic slot plan does not choose real repositories, issues or tasks. The six repository identities remain placeholders until license, provenance and task admission review.

## 3. Difficulty and task representativeness

Each slot has a planned low/medium/high stratum; the plan allocates 16 slots to each. Actual difficulty remains null until blinded reviewers assess the task before candidate results are available.

The review protocol records:
- crates and files touched by the reference repair;
- dependency fan-out and downstream consumers;
- feature/target configuration breadth;
- baseline test runtime and resource envelope;
- diagnostic clarity and issue-description ambiguity;
- security criticality and invariant count;
- reference patch semantic scope.

Difficulty labels may not be changed after observing candidate success, failure, cost, refusal or timeout. Any task that is invalid, rejected, unfilled or withdrawn remains in the attrition audit; no silent denominator shrinkage is allowed.

## 4. Statistical use and limitations

The balance reduces repository/family confounding but does not make 48 tasks representative of all Rust software engineering. Report task-level outcomes by repository, family, partition and difficulty. Use repository/family clusters in uncertainty analysis. The pilot remains a feasibility and variance-estimation study; a confirmatory sample size must be determined from observed paired discordance and cluster variance.

## 5. Current state

- Planned slots: 48.
- Repository placeholders: 6.
- Fault families: 6.
- Development slots: 32.
- Holdout slots: 16.
- Actual repository identities: 0.
- Actual task identities: 0.
- Admitted tasks: 0.

The existing CEB task-admission registry remains authoritative. This slot plan is not a task manifest and cannot admit a task by itself.
