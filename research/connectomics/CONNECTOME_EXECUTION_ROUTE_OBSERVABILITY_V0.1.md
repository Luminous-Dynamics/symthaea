# Connectome Execution-Route Observability V0.1

**Status:** research protocol; source inventory and canary qualification required  
**Issue:** [#10447 — SCI-CONNECTOMICS-075](https://github.com/Luminous-Dynamics/symthaea/issues/10447)  
**Prerequisite:** [#10433 — fusion-frame closure](https://github.com/Luminous-Dynamics/symthaea/issues/10433)  
**Downstream:** [#10414 — provenance-completeness sampling](https://github.com/Luminous-Dynamics/symthaea/issues/10414), [#9255 — query-scoped certificates](https://github.com/Luminous-Dynamics/symthaea/issues/9255)  
**Date:** 2026-11-04  
**Claim ceiling:** canary coverage of an enumerated build-bound execution surface; no proof that all deployed or real-world operations are observable

## 1. Why ledger agreement is not enough

A fusion-frame closure audit can reconcile two event exports perfectly while both exports share the same blind spot. For example, a public API wrapper may emit to a common event bus, while a batch importer, plugin, replay path or internal function calls the claim engine directly and bypasses that wrapper. Both ledgers agree because neither sees the omitted operation.

This is not a sampling defect. The omitted path has zero probability of appearing in either export.

V0.1 therefore adds a separate precondition: qualify the in-scope execution routes that can consume evidence or issue a claim/certificate, and demonstrate that each route is observed by the event-recording chain.

## 2. Target execution surface

The inventory must cover every supported path capable of:
- consuming evidence in an operation that contributes to a scientific claim;
- constructing or modifying a fusion evidence bundle;
- issuing, replacing, revoking or replaying a query certificate;
- importing/backfilling annotations or evidence that can be consumed by a claim;
- invoking an administrative or recovery override that can bypass normal claim/fusion instrumentation.

Potential paths include public API handlers, internal Rust entrypoints, RPC/IPC handlers, command-line tools, scheduled jobs, batch importers, replay/recovery paths, plugin/dynamic dispatch, feature-gated modules and direct internal function calls.

The target is the actual build and configuration under qualification, not a handwritten list of routes believed to exist.

## 3. Build-bound route inventory

Every inventory receipt binds:
- repository and exact source tree commit;
- exact build recipe/toolchain/dependency lock digests;
- binary or deployable artifact digest where available;
- feature/configuration set and target triple;
- route-discovery tool and version digest;
- build-time registered dispatch table digest;
- source-level entrypoint inventory digest;
- unresolved dynamic dispatch / reflection / FFI findings;
- inventory generator and exclusions.

A route record binds:
- stable route ID;
- semantic route class;
- entrypoint symbol and source file/range;
- exact implementation/build identity;
- supported query types, claim types and evidence roles;
- instrumentation contract and event schema version;
- expected event sources/observers;
- route owner/custodian;
- status ACTIVE, DISABLED_WITH_PROOF or OUT_OF_SCOPE_WITH_RATIONALE;
- reachability, dispatch and feature evidence.

A route status cannot be excluded based only on comments or self-declaration. Disabled paths require build/configuration evidence and a negative invocation check. Dynamic or unresolved dispatch prevents a complete inventory unless its reachable targets are enumerated.

## 4. Discovery is an evidence chain, not one grep

The inventory generator should reconcile at least:
1. source-level entrypoint discovery;
2. build-time route/handler registration;
3. feature and configuration resolution for the exact build;
4. deployment/API/CLI/job registration where the project uses those surfaces.

Each discovery channel binds its input and output digests, tool version, scope and exclusions. A generated manifest must report disagreements and unresolved paths rather than silently selecting one source.

V0.1 does not claim that static code discovery proves absence of hidden routes in dependencies, compiler/runtime, host or deployment infrastructure. It provides a falsifiable and reproducible inventory boundary.

## 5. Synthetic canary contract

Each ACTIVE route receives at least one route-specific canary. A canary:
- uses a reserved synthetic namespace and unique nonce;
- carries synthetic evidence IDs, a synthetic dependence-bundle digest and a synthetic claim/query ID;
- traverses the actual route dispatch and the same claim/fusion instrumentation points as production;
- is incapable of becoming a biological claim or authoritative annotation;
- runs with credentials unable to write biological annotation state;
- records the exact build, route, feature set, policy, verifier and event schema identities;
- is observed in the expected execution and claim/certificate ledgers;
- is checked against the independent observation/export chain.

The canary harness must prove that it exercised the actual handler, not merely a wrapper that normally calls the handler. Route-specific coverage receipts bind the call/trace identity to the handler's implementation digest.

Canaries must not enter training corpora, evaluation ground truth, production annotations or biological evidence stores. Isolation is checked by explicit namespace allowlists and negative contamination queries.

## 6. Negative controls and fault injection

At minimum, exercise:
- suppress one event-emission hook while the main operation succeeds;
- bypass the standard wrapper and call an internal handler directly;
- disable an observer or event sink;
- drop an event between two sinks;
- invoke a feature-gated route;
- invoke a plugin/dynamic target not present in the route inventory;
- replay or backfill an operation;
- invoke certificate issuance through an alternate route;
- use an old route-coverage receipt against a changed binary;
- omit a required canary field or alter the route/policy/build digest;
- attempt to persist a canary as a biological annotation/training example.

A fault mutation passes only when the harness detects the exact missing or mismatched observation and the route/query disposition is downgraded. Merely recording that a fault injection was attempted is not sufficient.

## 7. Route dispositions

Each route receives one of:
- ROUTE_COVERED_BY_CANARY;
- ROUTE_NOT_EXERCISED;
- OBSERVATION_MISSING;
- OBSERVATION_MISMATCH;
- ROUTE_INVENTORY_INCOMPLETE;
- DISABLED_ROUTE_VERIFIED;
- NOT_ESTIMABLE.

A route can be covered only if all expected event sources contain a semantically matching receipt with the exact route/build/policy/evidence identity. A route with an unknown observer, unresolved target or missing canary remains uncovered.

A query-specific route-coverage receipt evaluates the transitive set of routes that can contribute to its query/claim/certificate dependency closure. A global coverage percentage cannot hide an uncovered claim-issuance, import or high-materiality fusion route.

## 8. Promotion gate and invalidation

Route coverage is a conjunctive prerequisite to #10433 frame closure. The exact chain is:

1. build-bound route inventory is complete within declared discovery scope;
2. every applicable ACTIVE route is canary-covered and every disabled route is independently verified disabled;
3. expected observations match across qualified sinks;
4. negative controls demonstrate detection of enumerated bypass/drop faults;
5. contamination checks pass;
6. the route-coverage receipt binds the exact build, route inventory and observer identities;
7. #10433 may reconcile evidence-use exports;
8. #10414/#10416 may audit a closed assertion frame.

Any source code, build, feature, route registry, dynamic target, instrumentation library, event schema, observer, retention policy or deployment configuration change invalidates affected route-coverage receipts. Stale receipts cannot be reused merely because the branch name or package version is unchanged.

## 9. Adversarial known-answer families

1. Internal direct call bypasses public instrumentation.
2. Feature-gated route emits no event.
3. Plugin target is unresolved or unregistered.
4. Batch/backfill importer omits use receipt.
5. Certificate issuance bypasses the normal fusion executor.
6. Retry path creates duplicate physical events but loses semantic-use idempotency.
7. Logger drops events while operation succeeds.
8. Two sinks share a dropped-event queue.
9. Canary exercises only a wrapper rather than the real handler.
10. Canary evidence contaminates biological training/evaluation data.
11. New route is absent from a stale inventory.
12. Disabled route remains reachable through a secondary entrypoint.
13. A receipt from build A is reused for build B.
14. Aggregate coverage hides one uncovered claim-issuance route.
15. Event observed with a wrong policy, evidence set, build or schema digest.
16. Static discovery and runtime registration disagree.
17. FFI/reflection/dynamic dispatch has an unresolved target.
18. Observer retention does not cover the canary window.
19. A source route emits both records through one unqualified shared library.
20. Canary result is replayed or re-used across a different query/release scope.

Evaluator-only injection truth must be unavailable to route discovery and canary adjudication until receipts are frozen.

## 10. Metrics and qualification goals

Report separately:
- route inventory completeness within declared discovery scope;
- fraction of active routes canary exercised;
- fraction of canaries observed by each required sink;
- exact semantic receipt-match rate;
- unresolved dynamic target count;
- detection rate for each bypass/drop mutation;
- canary contamination count;
- coverage by route class, claim type, query type, feature set and build;
- time and expert/engineering effort to discover and close route gaps;
- invalidation latency after source/build/observer changes.

Proposed goals, not measured results:
- 100% of applicable ACTIVE routes exercised by a route-specific canary;
- zero promotion with an uncovered or unresolved applicable route;
- 100% detection of the enumerated event-bypass and stale-build mutations;
- zero canary contamination of biological data, annotations or evaluation labels;
- 100% invalidation after relevant route/build/instrumentation changes.

A high route coverage percentage achieved by excluding difficult or unresolved paths is not a pass.

## 11. Comparator matrix

- Handwritten route list only (unsafe baseline).
- Static source scan only.
- Build-time registry only.
- One runtime canary through the public API only.
- Multi-source route inventory plus per-route canaries.
- The same canary harness with injected logger/sink failures.
- Full reachable-route census for a small synthetic application, where feasible.

Compare route discovery recall, canary route coverage, bypass detection, contamination, false promotion and operational cost.

## 12. Deliverables and integration

- Versioned route-inventory, discovery receipt and canary receipt schemas.
- Synthetic active/disabled/unresolved route fixture.
- A route-discovery/reconciliation tool pinned to exact source/build inputs.
- Route-specific canary harness and fault-injection suite.
- Independent verifier for route, build, observer, event and invalidation bindings.
- Query-specific adapter consumed by #10433 and #9255.

#10406 remains responsible for dependence-graph structure. #10433 remains responsible for event/frame reconciliation. #10414/#10416 remain responsible for sampling and statistical claims. This work qualifies whether the software execution surface that feeds those ledgers is itself covered.

## 13. Claim ceiling

Passing V0.1 establishes canary coverage for an enumerated, build-bound execution surface, under named discovery channels, configurations, observers and synthetic probes. It does not prove that hidden code paths do not exist in dependencies, compiler/runtime, host or deployment infrastructure, nor that every real-world production execution is observed. It does not establish evidence independence, biological truth, or causal identification.
