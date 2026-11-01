# Canonical Proof-Frontier Adapter V1

**Status:** executable research baseline under issue [#9699](https://github.com/Luminous-Dynamics/symthaea/issues/9699). Not production state attestation, not a proof verifier, and not a learned cost predictor.

## Objective

Derive the nine cost-to-go features from the actual typed graph instead of accepting caller-authored counts. Inputs are the planner-facing `pod_meta_lab::graph::PlannerCase`, its matching POD-Meta `Obligation` states, and an explicit execution context for facts not represented in the graph.

V1 requires exact ID alignment: each reachable claim node maps to an obligation with the same ID. Ambiguous mappings are not guessed.

## Traversal semantics

- Start at unresolved target claims; supported targets are terminal and contribute no remaining work.
- Traverse claim dependencies reachable through alternatives of unresolved claims. Supported dependencies are recorded as discharged leaves and are not expanded.
- Requirements within one alternative are ANDed; alternatives on one claim are ORed.
- Count unresolved reachable claim nodes, unresolved target obligations, dependency edges, maximum DAG depth, and locally admissible alternatives.
- Mark a direct evidence requirement unavailable when its metadata fails temporal, scope, authority, integrity, lifecycle, subject-claim, or independence checks.
- Count distinct evidence families only among currently admissible evidence referenced by the live frontier.
- Obtain operation-family IDs and mandatory verifier-operation counts only from the explicit pre-action execution context. Do not infer these from graph structure.

A locally admissible alternative means only that its referenced evidence metadata and local independence minimum pass and its claim dependencies resolve structurally. It does not mean the alternative's conclusion is true, dependent claims are discharged, or the full path is independently verified.

## Fail-closed conditions

Duplicate claim/evidence/obligation/target IDs, duplicate alternatives or requirements, absent targets, absent claim dependencies, missing obligation mappings, unknown evidence IDs, evidence attached to the wrong claim, missing state identity for unresolved work, cycles, counter overflow, and inconsistent empty-frontier context are rejected.

Traversal and longest-path calculation are iterative; no recursive graph walk is used. Canonical ordering is independent of insertion order. Valid but unreachable claims and unreferenced evidence do not alter frontier features or fingerprint.

## Canonical identity

The byte encoding is versioned and length-prefixed. It binds the case and query profile, sorted targets, explicit operation-family and verifier context, target-reachable claim/obligation states, sorted alternatives and requirements, and all evidence metadata referenced by that frontier.

The V1 32-byte fingerprint is four seeded FNV-1a lanes. This is a stable identity checksum for deterministic tests only. It is not cryptographic, collision-resistant, authenticating, signed, or anti-tamper. Production integration must replace it with a cryptographic digest such as SHA-256 over the exact canonical bytes and bind that digest to independently captured, signed or attested pre-action state.

## Tests and benchmark matrix

The module tests hand-computed features, input permutation, unreachable-node invariance, dependency-depth metamorphism, OR-alternative changes, evidence revocation, completed targets, missing references, duplicates, cycles, and a 10,000-node chain.

The qualification executable emits deterministic rows at 8, 32, 64, 256, 1,000, 10,000, and 100,000 nodes. Rows report edge count, maximum depth, locally admissible alternatives, and canonical identity checksum. These rows establish scale and deterministic extraction, not latency guarantees.

## Next evaluation

Run paired held-out tasks with identical case graphs and outcome receipts under (1) manual caller-authored feature counts, (2) this canonical graph-derived adapter, and (3) graph-derived features with evidence metadata removed or corrupted as negative controls.

Report downstream six-resource point error, joint envelope coverage and width, false-feasible rate, abstention, feature-extraction time, peak memory, and per-task-family calibration. Keep task families and time partitions disjoint. Preserve censored episodes for survival/IPCW analysis; do not convert them into completed-case conformal residuals. Do not claim forecasting gains until the adapter improves held-out outcomes without weakening proof verification or hard-budget enforcement.
