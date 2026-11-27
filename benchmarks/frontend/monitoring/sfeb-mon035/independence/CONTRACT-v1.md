# MON-035 — Observer Common-Cause Independence Contract v1

**Profile:** `mon035-observer-independence-input-v1`  
**Status:** offline dependency-closure evaluator and adversarial tests implemented; signed inventory adapters and deployment attestation pending.

## Research question

Do two nominally independent recovery-evidence observers remain independent when their transitive dependencies are inspected? Different observer IDs, process names or public keys do not establish independence when both observers rely on the same operator, identity provider, control plane, host, storage, network, time source or signer.

MON-024 requires distinct observer-control roots and at least two observer roots, while warning that string inequality does not establish administrative separation, tenancy or common-cause independence. MON-035 makes that gap executable.

## Model

Each observer has a root node in a directed acyclic dependency graph. Edges `A -> B` mean observer component A depends on B; the transitive closure includes every reachable dependency.

The evaluator requires each observer's closure to represent eight critical failure-domain classes:

1. `operator`
2. `identity_provider`
3. `control_plane`
4. `compute_host`
5. `storage`
6. `network`
7. `time_source`
8. `signer`

Each failure-domain node carries both a class and a stable domain identifier. Two distinct nodes with the same class/domain identifier are one common-cause dependency. Domain identifiers must be canonical and consistently assigned by an independently trusted inventory authority.

A pair qualifies only when both inventories are complete, fresh, runtime-independently graded, graph-complete and acyclic, and the transitive closures share no identifier in any critical class. A shared root or any shared critical failure domain blocks the pair. Missing domains, edges, attestation or freshness evidence remain indeterminate.

## Verdict semantics

- `qualified`: exactly two eligible observers are proven disjoint across all eight critical failure-domain classes.
- `blocked`: shared critical failure domain/root, graph cycle, identity contradiction, malformed/unsupported domain or resource-cap violation.
- `indeterminate`: missing or stale inventory/attestation/time/root/edge/domain, or non-runtime-independent evidence.

The evaluator never infers independence from different observer names alone.

## Resource and trust boundaries

- v1 accepts exactly two observers; additional quorum topologies require a separately versioned policy.
- Graph caps: 50,000 nodes and 250,000 edges.
- Inventory SHA-256 fields bind an external inventory artifact but are not authenticated by this semantic evaluator.
- The evaluator cannot establish that an operator's inventory is truthful or complete. A separately trusted collection/attestation process must establish dependency coverage, canonical domain identity and current authority before MON-024 accepts the result.
- This is not a full probabilistic reliability model or a minimal-cut-set solver. It detects shared single failure-domain dependencies represented in the declared closure. Multi-node common-cause combinations and correlated-but-unrepresented domains remain outside v1.

## Adversarial tests

The suite covers independent positive controls; shared identity-provider, control-plane, operator, time-source and signer domains; incomplete/missing dependencies; cycles; missing critical classes; invalid classes; duplicate roots; future/expired evidence; and resource-cap violations.

## Benchmarks and prospective acceptance

The benchmark evaluates two 1,000-node observer dependency closures and reports latency and allocations.

Prospective targets:
- 100% acceptance of valid independent controls.
- 100% detection of each shared critical domain in the eight-class mutation matrix.
- zero false-independent outcomes in 10,000 generated unsafe dependency graphs.
- zero silent unknowns: incomplete inventories never qualify.
- p95 < 5 ms for two 1,000-node closures on a pinned 2-vCPU reference host (target, not measured).

At 0 false-independent outcomes in 10,000 independent unsafe trials, the one-sided 95% upper bound is still about 0.03%; zero observed failures is not proof of zero risk.

## Industry comparison

This complements rather than replaces common-cause failure analysis, fault-tree analysis, N-version diversity and independent verification. The differentiation hypothesis is evidence-observer independence across operational control dependencies, not a new database or backup engine.

The first comparison should hold observers and workload fixed while varying only infrastructure topology: fully separated domains, shared IAM, shared cloud account/control plane, shared host/hypervisor, shared storage, shared network, shared time source and shared signing root. Report false-independence, false-blocking, abstention, dependency coverage and evaluation cost separately.

## Limits

No live infrastructure inventory, production observer separation, cloud tenancy, KMS/HSM or administrative control plane is inspected here. A qualified result means only that the supplied dependency inventory contains no shared identifier in the specified classes. It is not a certification that actual operators or infrastructure are independent.
