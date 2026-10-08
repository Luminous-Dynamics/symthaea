# Source Credibility and Institutional Expectation Updating

**Status:** research design v0  
**Related:** #7064, #7061, #7062, #7058

## Purpose

Separate signal content, source identity, source reputation, and agent-specific trust in expectation formation.

Recent work on credibility shows that public beliefs about commitment can update from the relationship between announced plans and realized actions; fiscal-credibility evidence likewise finds that institutional strength and policy track record affect how announcements re-anchor expectations. Experimental work on misinformation finds that source reliability can materially affect belief updating. These results support treating credibility as a dynamic evidence state rather than a fixed confidence parameter.

Sources:
- https://www.sciencedirect.com/science/article/pii/S0022053125001085
- https://www.elibrary.imf.org/view/journals/001/2022/036/article-A001-en.xml
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4907730

## Core separation

~~~text
signal content
!=
source identity
!=
source reliability history
!=
agent-specific trust
!=
objective correctness
~~~

A credible source may be wrong. An unreliable source may be right.

## Source state

Where reputation is modeled, record:

- source identity;
- source role/authority;
- public history;
- fulfilled commitments;
- broken commitments;
- independently verified past actions;
- reputation-update rule;
- observer-specific trust;
- public/common reputation;
- reputation observation time.

## Deterministic treatments

### C0 — source-blind

Signal content is processed without source differentiation.

### C1 — fixed public reliability

Sources have explicitly declared reliability profiles.

### C2 — reputation from observed commitments

Credibility changes according to a frozen deterministic rule based on verified fulfilled/broken commitments.

### C3 — heterogeneous trust

Agents can maintain different trust states over the same source.

### C4 — strategic source

Source behavior can depend on the anticipated response of recipients.

C4 should be a later treatment because it introduces a strategic sender problem.

## Required fixtures

- identical signal from reliable vs unreliable source;
- reliable source breaks one commitment;
- unreliable source fulfills one commitment;
- truthful signal from distrusted source;
- false signal from trusted source;
- new accurate source vs entrenched trusted source;
- delayed verification;
- source identity spoofing;
- source role changes while identity persists;
- reputation disagreement across communities;
- correlated source coalition;
- corrected public information after a false signal.

## Reputation path dependence

Compare two identical sources with different histories.

If downstream expectations differ despite identical current statements, the divergence is attributable to source-history state only when all other inputs are held fixed.

## Self-reinforcing credibility

Test:

~~~text
low credibility
-> lower anticipatory cooperation
-> weaker realized outcome
-> lower realized credibility
~~~

against:

~~~text
high credibility
-> higher anticipatory cooperation
-> stronger realized outcome
-> maintained credibility
~~~

This should be analyzed as a possible self-reinforcing equilibrium, not automatically as proof of source quality.

## Accuracy-vs-credibility crossover

Introduce a new source that has higher objective accuracy but lower accumulated credibility.

Measure:

- time to adoption;
- persistence of incumbent trust;
- information loss from delayed adoption;
- conditions under which accurate sources displace trusted sources.

## Credibility attack

Permit a source to strategically make promises that are attractive but difficult to fulfill.

Measure the tradeoff between:

- immediate belief shift;
- later reputation loss;
- long-run source influence.

Do not permit the evaluator to label a promise dishonest unless dishonesty is part of the explicit agent model.

## Qualification boundary

A PASS establishes only the declared source/reputation state transitions and their effect on the synthetic expectation-update process.

It does not establish that simulated credibility corresponds to human trust, that a source is objectively truthful, or that one communication policy is optimal.