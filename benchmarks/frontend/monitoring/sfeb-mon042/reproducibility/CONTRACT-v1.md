# MON-042 — Dependence-Aware Reproducibility Contract v1

Profile: mon042-dependence-aware-input-v1. Status: standalone offline evaluator and synthetic tests; signed dependency inventory and runtime adapters pending.

## Research question

When repeated recovery replays share hosts, storage, network, clocks, identity, orchestrators, control planes or external services, what is the defensible experimental unit for a reproducibility claim?

MON-040 computes a one-sided Clopper–Pearson interval over replay observations. That interval's Bernoulli independence assumption cannot be established by distinct replay IDs or seeds. MON-042 adds a conservative cluster-level estimand.

## Estimand

A declared execution unit represents one deployment-level experimental unit with:
- a stable unit ID and one preregistered stratum;
- a complete dependency-domain inventory;
- an exact replay count;
- one outcome receipt per replay.

Two units are connected when they share an identifier in any preregistered critical failure-domain class. Connected components of this graph form dependence clusters (transitive closure applies). A cluster succeeds only when every required replay in every member unit is complete, blocked, and has the exact target failure-signature digest.

The reported one-sided Clopper–Pearson lower bound is over the proportion of independently sampled dependency clusters that complete the entire preregistered replay schedule successfully. It is not a per-replay probability, and it is not a direct replacement for MON-040's per-replay estimand.

Incomplete or missing replay observations make the relevant cluster a non-success. They never reduce the independent-cluster denominator. A missing unit from the preregistered frame makes the campaign indeterminate; it is not silently dropped.

## Dependency graph

The required-domain list is fixed before evaluation. Typical classes include:
- executor, process host and hypervisor;
- physical or cloud storage failure domain;
- network path / fabric / availability zone;
- clock and time-source infrastructure;
- workload identity, credential issuer and key-management service;
- orchestrator and cluster control plane;
- region, zone, provider control plane;
- external services used by the recovery test.

The exact list is workload-specific. Omitting a domain from the preregistration means the evaluator cannot detect sharing in that domain. Domain IDs must be stable, scoped, and obtained from independent infrastructure inventory. Identifiers that merely differ syntactically while resolving to one shared failure domain do not establish independence.

If a control plane or external service is shared across all units, the graph intentionally collapses them into one cluster. Do not relabel or omit that domain to manufacture a larger N.

## Statistical procedure

1. Preregister the expected unit count, required runs per unit, required independent clusters, target reliability, confidence level, target signature, strata and required domain classes.
2. Verify every unit's inventory completeness and digest binding.
3. Validate replay identities, seeds, exact scope, verdicts, signatures and full-record receipt digests.
4. Build an undirected overlap graph: units share an edge when any required domain class has the same declared domain ID.
5. Compute connected components; each component is one dependency cluster.
6. Mark a cluster successful only when all its preregistered observations are complete exact matches. Incomplete observations count as cluster non-success, not as missing denominator rows.
7. Apply one-sided Clopper–Pearson to the cluster-success count only when the declared cluster count and unit frame match preregistration.
8. Report the exact cluster membership, shared domain IDs, cluster outcome, denominator and confidence bound.

Mixed strata are rejected by this v1 single-cohort evaluator; analyze each preregistered stratum separately. Do not pool heterogeneous strata into one iid binomial estimate.

## Verdicts and limitations

- blocked: invalid profile/parameters, duplicate IDs/seeds, receipt tampering, manifest digest mismatch, mixed strata, extra units or other known protocol violation.
- indeterminate: missing/partial domain inventory, missing observations, too few independent clusters, incomplete unit frame or any unknown evidence.
- qualified statistical result: the bound meets the threshold for the declared cluster-level estimand under the stated sampling assumptions.
- campaign verdict: always indeterminate in v1. Hashes bind bytes; they do not authenticate inventory origin, signer authority, revocation state or independence.

The connected-component algorithm detects only the overlaps represented in the declared inventory. It cannot prove the absence of undeclared common-cause dependencies. MON-041 custody must authenticate dependency manifests and observation receipts before the result can inform a live campaign. Even then, independence, representativeness and the sampling frame remain experiment-design assumptions requiring independent review.

## Known-answer mathematics

At 95% one-sided confidence:
- 29/29 independent successful clusters gives a lower bound of approximately 0.901855.
- 28/29 gives approximately 0.846608 and does not meet a 0.90 target.

These values do not apply to 29 replay observations if those observations collapse into fewer shared dependency clusters. For example, 29 execution units sharing one identity or control-plane domain form one cluster, not 29 independent units.

## Safety boundary

The evaluator consumes caller-supplied domain IDs and digest-bound receipts. It does not authenticate the domain inventory, prove real physical separation, query infrastructure APIs, issue keys, maintain a durable nonce ledger, inject faults or authorize production use. It never sets production_action_permitted to true.
