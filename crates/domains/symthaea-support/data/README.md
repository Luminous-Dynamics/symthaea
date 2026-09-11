# IT Golden Incident Data

This directory contains **public benchmark fixtures**, never private grading oracles.

`it_golden_incidents_public_v1.json` preserves the original V1 contract. V1 public fixtures may
contain symptoms, normalized evidence summaries, technology/applicability tags, qualification
thresholds, and diagnostic actions. They must not contain ground-truth root causes, private action
outcomes, required findings, accepted/prohibited remediations, or verification oracles.

`it_golden_incidents_public_v2.json` adds multi-fault/currentness-aware cases and distinguishes
**public repository visibility** from **solver prompt visibility**. The complete V2 fixture may
contain evaluator-maintenance metadata such as domain, competency level, adversarial conditions,
evidence class, high-stakes status, and qualification thresholds. A normal benchmark harness
should pass only `GoldenIncidentCaseV2::solver_view()` (or the corpus-level `solver_view()`) to the
solver. That projection contains the incident narrative, evidence/currentness, and available
diagnostic actions but omits evaluator metadata such as `MultipleFaults` and scoring thresholds.

This prompt projection is a benchmark-hardening measure, not a secrecy boundary against a solver
that is deliberately given arbitrary repository access. The stronger secrecy boundary is the
oracle boundary below.

Real evaluation oracles belong in a separate benchmark artifact or private harness input that is
not present in the solver-visible repository/worktree. Keeping an oracle under a repository path
such as `data/oracle/` is not considered isolation for repo-aware evaluation. Neither V1 nor V2 may
contain root-cause truth, causal-chain truth, action outcomes, required findings, accepted or
prohibited remediation, or verification oracles.

V1 is loaded by `src/golden_incidents.rs`. V2 is loaded by `src/golden_incidents_v2.rs`, with the
minimal solver-facing projection implemented in `src/golden_solver_view_v2.rs`. Both versions
project stable case identity/revision into the same IT qualification matrix so results bind the
exact public case definition while private grading remains outside the support crate.
