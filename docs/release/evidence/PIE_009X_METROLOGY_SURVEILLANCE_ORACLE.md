# PIE-009X — Independent Metrology Surveillance Oracle

## Purpose

This reference defines deterministic surveillance-coverage semantics for long-duration planetary metrology.

It answers a narrower question than PIE-009V/W:

> Even if independent references and intercomparison procedures exist, can the settlement sustain enough fresh, independent comparison evidence over time to keep decision-critical references under surveillance?

The model is intentionally structural. It does not infer physical drift rates or failure probabilities.

## Core invariant

A critical reference is covered at opening step `t` only when there exists a three-reference witness:

- the target reference;
- two active peers;
- all three pairwise comparison channels;
- every comparison qualified;
- every comparison still within the declared maximum evidence age;
- no shared declared comparison failure group among the three channels.

Comparison work executed in step `N` becomes usable only at opening step `N+1`.

Therefore a comparison scheduled after an opening-state lapse cannot retroactively repair that lapse.

## Why a triangle?

PIE-009V permits relative-outlier isolation only when two peers agree with each other and both independently disagree with a suspect.

A surveillance triangle preserves the structural evidence needed for that style of future diagnosis:

```
      A
     / \
   AB   AC
   /     \
  B---BC---C
```

Fresh `AB` and `AC` without fresh `BC` are not enough to maintain the same isolation witness.

## Finite service capacity

The campaign has an explicit number of comparison slots per step.

A schedule that attempts more work than the available slots is malformed and fails closed.

The reference does not silently reorder or prioritize work by ID.

## Evidence age

For comparison evidence whose result became available at opening step `s`, its age at opening step `t` is:

`age = t - s`

The evidence is fresh only when:

`0 <= age <= max_evidence_age`

The campaign reports the worst evidence age used by any covered critical reference.

This is a structural freshness measure. It is not a physical drift-rate estimate, stochastic detection latency, or guarantee that a fault would be detected within that many real-world hours.

## Independent execution fixture

The synthetic fixture uses three critical references `A`, `B`, and `C`, three pairwise comparison channels, one comparison slot per step, maximum evidence age 2, and a rotating schedule:

```
step 0: AB
step 1: AC
step 2: BC
step 3: AB
step 4: AC
step 5: BC
...
```

Opening evidence exists for all three comparisons at step 0.

The final candidate self-test was executed locally with Python 3 on 2026-09-13 and returned `ok`.

The fixture proves:

1. the one-slot rotating schedule maintains continuous critical surveillance;
2. the worst evidence age reaches exactly 2;
3. delaying `AC` creates a visible opening-step lapse;
4. performing `AC` during that lapsed step cannot repair the opening state;
5. over-capacity schedules fail closed;
6. a shared comparison failure domain defeats a numerically fresh triangle;
7. an unqualified comparison cannot support surveillance;
8. an unmonitored noncritical reference does not invalidate otherwise continuous critical coverage;
9. reversed-duplicate comparison pairs and unknown schedule IDs fail closed.

## Relationship to adjacent PIE layers

PIE-009T asks whether a measurement has a sufficiently precise traceability path.

PIE-009U asks whether apparently redundant measurements have genuinely independent metrology lineages.

PIE-009V asks whether qualified intercomparisons support disagreement detection or relative-outlier isolation.

PIE-009W defines quarantine, remediation generations, and conservative re-entry.

PIE-009X adds the temporal service question:

> Can the settlement keep enough of that evidence fresh continuously?

## Non-claims

This reference does not establish actual reference drift behavior, physical comparison uncertainty, optimal scheduling, worker or robotic labor requirements, energy consumption, economic cost, cyber authenticity, calibration certification, probability of missing a fault, or physical hardware authority.
