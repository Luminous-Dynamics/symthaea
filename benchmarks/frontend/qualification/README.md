# SFEB oracle qualification

## Why this exists

A benchmark can be internally consistent and still be wrong. A task schema can
validate while a missing assertion lets a broken interface pass; a screenshot
can look plausible while keyboard access, role enforcement or network isolation
is broken. SFEB therefore qualifies its evaluator separately from its agents.

## Campaign contents

- `mutation-campaign-v1.json`: 20 planned seeded negative controls across
  functional behavior, accessibility, responsive layout, security, evidence
  integrity and benchmark gaming.
- Five positive controls: harmless source formatting/renaming, semantics-neutral
  DOM structure, equivalent design tokens and alternate supported input modality.
- Five metamorphic relations: pointer/keyboard equivalence, deep-link equivalence,
  responsive continuity, semantics-preserving refactors and receipt identity.

This campaign is currently **designed, not executed**. The checked-in cases are
public-pilot development data and do not measure hidden-holdout performance.

## Qualification metrics

For a class (c):

`mutation\_kill\_rate_c = rejected\_mutants_c / executed\_mutants_c`

A mutant is killed only when the independent evaluator rejects the expected
failure mode. Unrelated build errors, timeouts, agent self-reports or incomplete
receipts are not credited as a successful oracle detection.

Targets (not results):

| Gate | Minimum |
|---|---:|
| Critical mutation kill rate | 100% |
| Security mutation kill rate | 100% |
| Evidence-integrity mutation kill rate | 100% |
| High-severity mutation kill rate | 95% |
| Overall mutation kill rate | 90% |
| False rejection of positive controls | 0% |

Report raw counts, denominators, per-class and per-severity results. With only
20 seeded mutants these are qualification probes, not precise estimates of
real-world defect detection; expand the mutant bank and report binomial
confidence intervals before using the metric as an assurance claim.

## Execution order

1. Freeze the task brief, interface contract, fixtures and independent oracle.
2. Generate a baseline run and verify that all required evidence artifacts exist.
3. Apply each mutant independently in a disposable fixture workspace.
4. Replay the exact same browser journeys and viewport matrix.
5. Record the oracle that rejected each mutant and preserve the evidence receipt.
6. Apply the positive controls and require all contract-defined outcomes to remain
   accepted.
7. Execute the metamorphic relations on at least the same pinned browser build.
8. Publish per-case evidence; never convert unexecuted cases into passes.

Security and evidence-integrity mutants are hard gates. A single accepted critical
mutant blocks oracle qualification. A mutation plan is not a mutation runner: the
Rust-owned runner, isolation, artifact retention and signed result schema remain
follow-on implementation work.

## Relationship to external SOTA

This suite measures whether SFEB's acceptance machinery is trustworthy. It is
not a website-generation leaderboard and its mutation score must never be
compared to Vision2Web, Design2Code or browser-agent scores. Those require
separate matched evaluations.

## Structural plan validation

The campaign is also checked by the standalone Rust contract checker:

```bash
cargo run --manifest-path benchmarks/frontend/contract-check/Cargo.toml -- --mutations
```

This checks schema conformance, unique case IDs, reject/accept dispositions, the
presence of critical/security/evidence-integrity mutants, minimum case counts and
hard 100% targets for the critical classes. It does not execute any mutation or
browser test. Run the checker and the actual campaign separately; only retained
per-case evidence can establish an executed qualification result.
