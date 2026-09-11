# IT Golden Incident Data

This directory contains **solver-visible** benchmark fixtures only.

`it_golden_incidents_public_v1.json` may contain symptoms, normalized evidence summaries,
technology/applicability tags, qualification thresholds, and the diagnostic actions a solver may
request or propose. It must not contain ground-truth root causes, private action outcomes,
required findings, accepted/prohibited remediations, or verification oracles.

Real evaluation oracles belong in a separate benchmark artifact or private harness input that is
not present in the solver-visible repository/worktree. Keeping an oracle under a repository path
such as `data/oracle/` is not considered isolation for repo-aware evaluation.

The public schema is loaded by `src/golden_incidents.rs`. Case identity/revision is projected into
the IT qualification matrix so benchmark results can bind the exact public case definition while
private grading remains outside the support crate.
