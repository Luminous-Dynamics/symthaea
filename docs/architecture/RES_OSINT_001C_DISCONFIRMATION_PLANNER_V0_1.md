# RES-OSINT-001C — Deterministic Disconfirmation and Next-Information Planning v0.1

Status: architecture/test-vector subject only. **NOT EXECUTED / NOT QUALIFIED / NOT PASS.**

Parent program: `#5417`.

This subject freezes the first deterministic planning theorem above bounded investigation semantics without depending on a model, crawler, live network, persistence layer, or action system.

## Purpose

A useful investigator must do more than ask for information that agrees with its current leading explanation. It must retain alternatives, identify observations that could weaken the currently preferred hypothesis, and distinguish epistemic value from permission to collect.

The governing laws are:

```text
preferred hypothesis under profile
!= true

failed falsifier search
!= confirmation

high information value
!= permission to collect

Pareto-nondominated
!= authorized

blocked request
!= useless analytic information

no available disconfirmation candidate
!= support for preferred hypothesis
```

## Authority ceiling

The maximum output authority is:

```text
CandidateAnalysisOnly
```

Every next-information request remains:

```text
ProposalOnly
```

This subject creates no HTTP/browser/tool capability, OPSEC permit, target admission, durable lease, Mycelix mutation, scientific disposition, or other effect authority.

## Input model

The deterministic planner consumes only already-explicit planning state:

- a finite live hypothesis set including an insufficiency alternative;
- an optional `PreferredWithinProfile` hypothesis;
- discriminating-observation candidates and their expected relation vectors;
- explicit unknowns and limitations;
- dependency/search/coverage coordinates;
- multidimensional next-information values/burdens;
- privacy/OPSEC/policy disposition;
- exact frontier/profile/derivation references.

It does not infer source truth, source dependency, semantic entailment, or search completeness.

## Disconfirmation obligation

When one hypothesis is currently preferred under a profile, the planner must surface candidate observations that could weaken it or strengthen a live alternative when such candidates are present.

For preferred `H1`, a planning vector such as:

```text
H1 -> StronglyDiscriminatesAgainst
H2 -> StronglyDiscriminatesFor
```

is a disconfirmation candidate.

If none exists, the planner records:

```text
NoAvailableDisconfirmationCandidate
```

rather than increasing H1 confidence.

A previous zero-result falsifier search under `UnknownCoverage` remains unresolved. It does not reduce the need for the candidate solely because the search returned nothing.

## Policy partition before Pareto analysis

Candidates are first partitioned by explicit policy disposition:

```text
EligibleForExternalAuthorizationCandidate
AnalyticallyUsefulButPrivacyBlocked
AnalyticallyUsefulButOpsecBlocked
AnalyticallyUsefulButPolicyBlocked
```

Blocked candidates remain visible in the analysis record because they may explain what information would be useful, but they cannot enter the eligible execution-oriented Pareto set.

```text
VeryHigh information value
+ privacy/OPSEC block
-> analytically useful
-> still ProposalOnly
-> not executable
```

## Named Pareto profile

V0.1 freezes:

```text
symthaea:next-information:pareto-front:v1
```

Benefit coordinates, higher-is-better:

- discriminating power;
- contradiction/falsifier value;
- dependency reduction;
- currentness-gap reduction;
- coverage-gap reduction;
- expected reproducibility.

Burden coordinates, lower-is-better:

- expected collection cost;
- privacy sensitivity;
- OPSEC disclosure cost.

Ordinal values are:

```text
None < Low < Medium < High < VeryHigh
```

For two policy-eligible candidates A and B:

```text
A dominates B
iff
A is no worse than B on every declared coordinate
and
A is strictly better than B on at least one coordinate
```

There is no hidden weighted scalar. Incomparable candidates remain together on the non-dominated set.

## Dependency questions are legitimate information goals

A proposal may be valuable because it resolves source lineage even when it neither directly supports nor contradicts a factual hypothesis.

For example:

```text
Are A2 and A3 independent observations,
or are both downstream publications of A1?
```

The answer changes corroboration interpretation without itself establishing the underlying real-world claim.

```text
dependency-reduction value
!= factual support
```

## Frozen synthetic fixture

Machine-readable fixture:

`docs/architecture/fixtures/RES_OSINT_001C_PLANNER_V0_1.json`

Git blob identity at authoring:

`09331ce91386f2151e3681f66eb0c341f04aac89`

The fixture is synthetic. It performs no network I/O and requires no LLM.

At frontier `F2`:

```text
H1 = PreferredWithinProfile
H2 = Live
H3 = Live
HU = InsufficientEvidence
```

Candidate observations are:

- **D1** — an independent second gauge reports no corresponding level change; a strong H1 challenge;
- **D2** — a controlled calibration retest demonstrates sensor drift; a strong H1 challenge and H2 discriminator;
- **D3** — an exact publication-lineage record resolves whether downstream reports derive from the station report; primarily dependency-reduction value;
- **D4** — protected technician device telemetry could be highly discriminating but is explicitly privacy-blocked;
- **D5** — another repeat of the same station report with no new observation lineage; intentionally low-value/redundant.

A prior D2 search has:

```text
result_count = 0
coverage = UnknownCoverage
finding = NoMatchObservedUnderSearchProfile
```

Its only permitted interpretation in this fixture is:

```text
UnresolvedDueToUnknownCoverage
```

The forbidden interpretation is:

```text
SupportsPreferredHypothesis
```

## Required deterministic output

The fixture freezes:

```text
disconfirmation candidates for H1 = {D1, D2, D4}
policy-eligible disconfirmation    = {D1, D2}
eligible Pareto front             = {D1, D2, D3}
privacy-blocked analytic candidate = {D4}
dominated eligible candidate      = D5, witnessed by D2
```

D1, D2, and D3 are deliberately incomparable under the named profile, so the planner must preserve all three rather than force a scalar winner.

D4 remains visible but cannot enter the eligible Pareto front.

D5 is dominated under the exact frozen coordinates. A future coordinate change must invalidate that domination witness rather than retaining a stale conclusion.

## Metamorphic obligations

A future independent qualifier must derive altered fixtures and show at least:

1. removing D1, D2, and D4 produces `NoAvailableDisconfirmationCandidate`, not stronger H1 confidence;
2. changing D4 to policy-eligible allows it into Pareto consideration but still does not authorize execution;
3. changing the prior D2 coverage from unknown to exhaustive finite-corpus coverage changes only the scoped negative-evidence interpretation and still does not support H1 by absence;
4. improving D5 coordinates to equal D2 invalidates the original D2-dominates-D5 witness;
5. changing a blocked proposal's information value cannot make it executable while its block remains;
6. removing HU violates the parent investigation profile rather than forcing a binary answer.

## OPSEC composition

Any later external side effect remains downstream of independent authority systems:

```text
NextInformationPlanCandidate
 -> SearchPlanCandidate
 -> exact OPSEC disclosure intent
 -> current disclosure authority
 + target admission
 + durable Started authority
 + confinement
 -> bounded connector side effect
```

The planner is not an authority shortcut.

## Historical/frontier discipline

Planner output binds the exact input frontier. New evidence creates a new planning result. It does not rewrite an older result to appear as though later evidence was available earlier.

```text
F2 -> plan P2
F3 -> plan P3

P3 != mutation of P2 history
```

## Nonclaims

This subject does not establish factual truth, source authenticity, source independence, complete search coverage, model correctness, legal authority, safe deanonymization, privacy compliance, network anonymity, scientific proof, or action authority.

It freezes only a deterministic method for keeping disconfirmation, source-lineage uncertainty, multidimensional information value, and privacy/OPSEC constraints visible at the same time.
