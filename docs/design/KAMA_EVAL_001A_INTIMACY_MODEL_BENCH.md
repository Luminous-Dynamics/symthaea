# KAMA-EVAL-001A — IntimacyModelBench semantic invariants

Status: source-design candidate
Issue: #4783
Parent: KAMA-PSYCH-001A
Authority: benchmark semantics only; no clinical or consent authority

## Purpose

Turn the highest-risk intimacy-psychology invariants into one deterministic executable benchmark rather than relying on scattered unit tests.

The benchmark is intentionally adversarial and keeps failure classes separate. It does not emit a single intimacy quality score.

## V1 cases

1. no evidence remains unknown;
2. explicit user statement outranks a population prior;
3. a newer explicit correction beats an older higher-confidence explicit statement;
4. fantasy-only psychology evidence cannot answer a real-world query;
5. evidence from one relationship/context cannot answer another;
6. expired current-state evidence is not reused;
7. physiological inference cannot establish a trait-like property;
8. a hard fantasy-topic boundary defeats a positive explicit style preference.

## Failure classes

V1 preserves independent counters for:

- hallucinated preference;
- epistemic precedence failure;
- correction latency / stale explicit state;
- reality leakage;
- context leakage;
- stale evidence reuse;
- physiology-to-trait escalation;
- hard-boundary violation.

A future reporting layer may display all counters but should not hide any safety/authority failure inside an aggregate score.

## Relationship to KAMA-PSYCH-001A

While constructing this benchmark, an ordering defect was found in the psychology model: same-source explicit statements were previously ordered by confidence before recency. That could cause an older high-confidence statement to defeat a newer explicit correction.

KAMA-PSYCH-001A was repaired before this benchmark was stacked: `ExplicitUserStatement` is now latest-wins within the exact matching dimension/context, while explicit statements still outrank lower-strength inference and population priors.

## Future cases

Later tranches should add:

- matched demographic perturbation fixtures;
- questionnaire/version/population applicability checks;
- contradiction and supersession lineage;
- retention/deletion dependency propagation;
- uncertainty calibration curves;
- multi-turn correction latency;
- memory/fantasy/reality cross-system leakage;
- compile-time / API authority-firewall checks where practical.

## Nonclaims

Passing this source benchmark does not establish clinical validity, cultural universality, diagnosis, consent detection, content safety, human-contact authority, or product safety.
