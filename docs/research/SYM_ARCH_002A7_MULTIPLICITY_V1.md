# SYM-ARCH-002A7 — Multiplicity-Safe Confirmatory Families v1

**Status:** statistical analysis-plan infrastructure; no architecture result

**Tracks:** #55

**Base:** SYM-ARCH-002A2 hierarchical statistics and power (#58)

## Why this exists

SYM-ARCH-002 intentionally measures several phenomena and may compare more than one architecture/control. Without a frozen hypothesis family and multiplicity correction, a later analysis could accidentally select the most favorable comparison or metric and report an ordinary nominal p-value as though it were the only test performed.

A7 makes the claim-bearing hypothesis family explicit before outcomes are interpreted.

It does **not** generate effect estimates or raw p-values. It only freezes family membership/tails and applies family-wise correction to inferential outputs produced by separately preregistered tests.

## Frozen hypothesis family

`MultiplicityPlan` records:

- schema/version;
- normalized `family_id`;
- family-wise alpha;
- exact hypothesis ids;
- frozen tail for every hypothesis (`greater`, `less`, or `two_sided`);
- primary/secondary role for every hypothesis.

Hypothesis ids must be unique. The plan receives a domain-separated BLAKE3 digest that is independent of input ordering but changes when family membership, role, tail, alpha, or family identity changes.

A claim-bearing analysis must bind the exact plan digest before observing CONFIRM outcomes.

## Raw inferential inputs

`RawHypothesisPValue` records:

- frozen hypothesis id;
- tail actually used by the raw test;
- raw p-value.

A7 fails closed unless:

- the raw family contains exactly the frozen hypothesis ids;
- no hypothesis appears more than once;
- every p-value is finite and in `[0,1]`;
- every submitted tail exactly matches the tail frozen in the plan.

This prevents a one-sided/two-sided or direction switch from being hidden inside the multiplicity layer.

A7 cannot verify that the upstream raw test itself was implemented correctly. Test statistic, resampling method, pairing unit, nuisance topology, and seed policy remain part of the separately frozen A2/experiment analysis contract.

## Holm family-wise correction

`apply_holm` implements Holm's step-down family-wise correction:

1. sort raw p-values ascending;
2. deterministic tie-break by frozen hypothesis id;
3. multiply each ordered p-value by the number of hypotheses remaining;
4. cap at `1.0`;
5. take the running maximum so adjusted p-values are monotone;
6. compare adjusted p-values to the frozen family alpha.

The returned result is canonicalized by hypothesis id rather than significance rank.

This follows the same correction pattern already used by Symthaea's Muse confirmatory-study analysis, but A7 is implemented independently in psych-bench so architecture research is not coupled to Muse-specific endpoint types.

## Simultaneous confidence-interval support

A7 also exposes:

- Bonferroni per-comparison alpha = `family_alpha / m`;
- corresponding confidence level = `1 - family_alpha/m`.

This permits a separately implemented interval estimator to request conservative simultaneous confidence intervals for a frozen family.

Holm-adjusted p-values and Bonferroni simultaneous intervals are **two reporting tools**, not ingredients of a single blended score.

A later analysis must preregister which inferential path supports each claim. It must not choose between nominal, Holm, or Bonferroni reporting after seeing which yields the preferred conclusion.

## Relationship to SESOI / practical effects

Multiplicity control and practical-effect control answer different questions:

- multiplicity: how much false-positive risk is created by a family of tests?
- SESOI: is the estimated effect large enough to matter?

A statistically significant but practically tiny effect does not pass the architecture claim gate merely because Holm rejects it.

Likewise, a large point estimate does not become confirmatory evidence if its multiplicity-safe inferential gate fails.

Claim-bearing architecture results should therefore preserve both dimensions separately in the ClaimLedger/evidence record.

## Primary vs secondary hypotheses

`HypothesisRole` records whether a hypothesis was frozen as primary or secondary.

The role is part of the plan digest. A hypothesis cannot be relabeled primary after outcomes are observed without producing a different plan identity and therefore a new/post-hoc analysis specification.

A7 does not impose one universal rule about whether primary and secondary hypotheses belong in the same family. The preregistration must define the family structure in advance and justify it. Splitting one scientific family into several smaller families after observing results is not permitted.

## What must be frozen before CONFIRM

At minimum:

- exact family id(s);
- exact hypothesis ids;
- hypothesis role;
- direction/tail;
- family alpha;
- upstream raw test for each hypothesis;
- pairing/generalization unit;
- nuisance topology;
- metric and comparator;
- SESOI/practical-effect rule;
- multiplicity method;
- interval method/alpha if simultaneous intervals are used;
- code/analysis revision and plan digest.

DEV may be used to choose this structure. CONFIRM/REPL may not change it after behavioral outcomes are observed.

## Acceptance tests

The exact PR head must demonstrate:

1. known Holm step-down example produces expected adjusted p-values;
2. plan digest is independent of hypothesis serialization order;
3. changing a frozen tail changes plan identity;
4. Bonferroni family alpha is computed correctly;
5. raw hypothesis ids must match the frozen family exactly;
6. raw test tail must match the frozen tail;
7. duplicate hypothesis ids fail closed;
8. NaN/out-of-range p-values fail closed;
9. deterministic tie handling is stable;
10. psych-bench library compiles.

## Claim ceiling

Merging A7 supports only:

> Symthaea psych-bench can freeze claim-bearing hypothesis families and apply deterministic family-wise multiplicity control to separately preregistered inferential outputs.

It does **not** support:

- a Symthaea performance claim;
- a claim that any raw p-value is valid merely because A7 accepts its numeric range;
- a practical-effect claim without SESOI/effect-size evidence;
- post-hoc family splitting or tail switching;
- a claim that Holm and Bonferroni results should be averaged or collapsed into one score.

## Next use

After A2/A7 and the rest of the A-series are executable and green, freeze the exact primary hypothesis family for a DEV dry run, validate the complete analysis path, and only then commit the untouched CONFIRM family/plan digest.
