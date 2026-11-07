# CEB causal-debugging task bank blueprints V0 — 2026-11-04

**Status:** 24 non-executable task blueprints; not executable fixtures, a validated benchmark set, or a performance result.  
**Parent:** [CEB-CAUSE-001 #10366](https://github.com/Luminous-Dynamics/symthaea/issues/10366)  
**Integration:** [CEB causal ledger architecture](https://github.com/Luminous-Dynamics/symthaea/blob/research/ceb-causal-debugging-2026-11-04/docs/research/CEB_CAUSAL_INTEGRATION_ARCHITECTURE_V0_2026-11-04.md)  
**Isolation:** candidate builds, tests and task-controlled programs remain blocked by [#10064](https://github.com/Luminous-Dynamics/symthaea/issues/10064).

## 1. Purpose and status

This file turns the six CEB causal-debugging fault families into 24 concrete authoring blueprints (four per family). Each blueprint defines the capability being tested, a confounder class, a safe discriminating action class, and the expected evaluation boundary.

These are not yet runnable tasks. They intentionally do not include the exact injected root-cause diff, hidden grader, reference patch, private expected observations, or sealed answer key. Those belong in evaluator-only custody and must not be committed to the agent-visible repository. The blueprints can guide fixture authoring, split planning and preregistration now, without executing untrusted code.

A blueprint is not counted as a qualified benchmark task until the task has a frozen revision, exact source/configuration digests, valid reference and empty-patch controls, an independently reviewed grader, a pristine regrade path, and qualified isolation.

## 2. Common task contract

Every eventual task instance must bind:

- stable task and blueprint IDs, version and authoring provenance;
- repository template family and source-tree digest;
- exact language/toolchain/dependency lock, feature/target/configuration world and execution image;
- agent-visible initial evidence and allowed tools;
- hidden fault manifest, ground-truth causal graph and admissible repair set kept outside the agent-visible workspace;
- competing hypotheses, including null/no-fault and observationally equivalent alternatives;
- experiment/action catalog with cost, risk, reversibility, blast radius and permission;
- expected observation distributions committed before action;
- hidden behavioral invariants, regression suite, mutation controls and independent grader revision;
- reference patch and empty patch for two-sided task qualification;
- primary and pristine-regrade receipts with complete task-level dispositions.

Do not define task difficulty solely by lines changed or number of files. Include semantic coupling, evidence ambiguity, interaction structure, cost to disambiguate, and consequences of a wrong action.

### 2.1 Required evidence and answer custody

**Agent-visible:** initial failure/symptom, source/configuration permitted by the task, visible logs/traces, allowed read-only probes, and the declared resource budget.

**Evaluator-only:** exact injected cause, causal mechanism graph, expected intervention outcomes, reference solution, hidden tests, admissible-edit rubric, should-abstain label, and split membership.

The task author must not place evaluator-only facts in comments, generated names, file paths, commit history, benchmark metadata returned to the agent, or test failure text. Public development examples must never double as sealed confirmation items.

### 2.2 Common paired controls

Each blueprint should be authored as a family, not a single item:

- a normal actionable instance;
- a near-identical should-abstain instance with one minimal blocking change;
- a misleading-symptom instance with at least one plausible but false hypothesis;
- where appropriate, a multi-fault instance requiring interaction-aware diagnosis;
- a null/simple instance where no elaborate investigation is useful.

At least six blueprints should have multi-fault compositions; at least six should have intermittent or noisy observations; and at least six should contain an explicit non-identifiability or should-abstain branch. These counts overlap and are not independent tasks.

## 3. Rust semantics and API evolution

### CEB-RUST-001 — Trait-bound regression with a misleading local error

**Capability:** distinguish a direct type error from the upstream public-trait or generic-bound change that made the implementation invalid.

**Visible challenge:** a multi-crate Rust workspace reports a compile failure in a downstream implementation after an apparently unrelated trait/API change. A local annotation or clone may silence the immediate diagnostic without restoring the intended abstraction.

**Confounders:** stale compiler output, a blanket implementation that compiles under one feature set, and a nearby unrelated warning.

**Discriminating action:** inspect the exact compiler-resolved trait relation and compare the pinned supported feature/configuration worlds. A source-text search alone is insufficient.

**Evaluation boundary:** correct root-cause family, supported compiler evidence, compatibility preservation for hidden implementors, and no broad bound relaxation that weakens the API contract.

### CEB-RUST-002 — Module relocation and public re-export drift

**Capability:** reason about symbol identity and change impact across modules, re-exports and external call sites.

**Visible challenge:** a public type has moved and a subset of internal tests still pass. Downstream users and macro-expanded references may use a different path than the visible in-repository call sites.

**Confounders:** stale docs, private imports, generated code and one unaffected facade path.

**Discriminating action:** construct a typed dependency/call-site impact set from compiler or language-server evidence where available; retain unresolved macro-generated edges as unresolved.

**Evaluation boundary:** preserve the public compatibility promise, update only justified references, and report unsupported external paths rather than inventing certainty.

### CEB-RUST-003 — Generic overlap and blanket-implementation interaction

**Capability:** identify interactions among generic constraints, coherence and blanket implementations.

**Visible challenge:** a change resolves one concrete type but creates ambiguity or overlap for another class of types. The first compiler diagnostic points to a downstream impl rather than the changed generic contract.

**Confounders:** overlapping test helpers, feature-gated impls and an apparently equivalent where-clause simplification.

**Discriminating action:** enumerate the relevant impl/feature matrix and use compiler-resolved coherence diagnostics from the pinned toolchain.

**Evaluation boundary:** all protected type families remain accepted/rejected as specified; no unsafe or overly broad blanket implementation.

### CEB-RUST-004 — Async lifetime / Send-Sync boundary

**Capability:** distinguish lifetime, ownership and thread-safety constraints across an asynchronous boundary.

**Visible challenge:** a future or task fails to satisfy the required lifetime or Send/Sync property after a refactor. A tempting move or clone changes ownership behavior and can introduce unnecessary allocation or lifetime extension.

**Confounders:** target-specific executor bounds, feature-gated Send behavior, and a test that never crosses the failing concurrency boundary.

**Discriminating action:** trace the exact future type and capture set under the frozen executor/configuration, then test the declared boundary in the future isolated fixture.

**Evaluation boundary:** preserve cancellation and ownership semantics, not merely compile the visible example.

## 4. Cargo features and build configuration

### CEB-CARGO-001 — Feature interaction hidden by default features

**Capability:** reason about the resolved feature graph rather than a single default build.

**Visible challenge:** a change passes the default workspace configuration but fails when a supported optional feature is selected with a second feature.

**Confounders:** feature unification, an optional dependency activated transitively, and a misleading stale build artifact.

**Discriminating action:** inspect resolved Cargo metadata and compare the frozen supported feature matrix; do not execute an untrusted build script in the current host environment.

**Evaluation boundary:** all declared supported feature worlds satisfy their contracts, with unsupported combinations explicitly rejected or documented.

### CEB-CARGO-002 — Cross-target dependency mismatch

**Capability:** detect target-specific dependencies and cfg paths that are absent from the developer's host build.

**Visible challenge:** a crate compiles on the default host but fails or loses required behavior for a declared second architecture or operating-system target.

**Confounders:** target-specific optional dependency, cfg-gated module declaration and a host-only unit test suite.

**Discriminating action:** statically resolve target-specific dependency and cfg conditions, then later use a pinned cross-target toolchain inside qualified isolation.

**Evaluation boundary:** report target coverage honestly; a host pass is not evidence for an untested target.

### CEB-CARGO-003 — Default-feature drift and compatibility trap

**Capability:** diagnose a regression caused by changing default features or a compatibility facade.

**Visible challenge:** the new default path is faster or smaller but silently removes a behavior that downstream consumers previously received without opting in.

**Confounders:** explicit-feature CI jobs still pass, while a minimal downstream consumer uses only defaults.

**Discriminating action:** compare old and new resolved feature graphs and a minimal consumer contract with the default feature set.

**Evaluation boundary:** preserve declared compatibility or make a deliberate breaking change with explicit migration evidence; do not redefine a regression away by weakening the contract.

### CEB-CARGO-004 — Workspace membership / lockfile resolution discrepancy

**Capability:** understand workspace graph changes, package selection and dependency resolution as distinct from source-level module changes.

**Visible challenge:** a workspace edit changes which packages or dependency versions are resolved; a narrow package test does not expose the impact on the documented workspace surface.

**Confounders:** excluded standalone crate, default-members selection, target-specific dependencies and stale lockfile state.

**Discriminating action:** compare workspace/package metadata and lockfile digests against the declared supported commands. Build scripts remain unexecuted until isolation is qualified.

**Evaluation boundary:** exact package-selection semantics are preserved; no false claim that an uninvoked package or configuration was tested.

## 5. Behavioral and state-machine regressions

### CEB-BEH-001 — Boundary transition in a lifecycle state machine

**Capability:** identify an incorrect transition and its invariant impact rather than patching the first failing assertion.

**Visible challenge:** a record behaves correctly for ordinary states but becomes invalid at a boundary such as expiry, revocation, finalization or retry exhaustion.

**Confounders:** stale cached status, inclusive/exclusive time boundary and an unrelated validation error.

**Discriminating action:** build a transition table from the declared state contract and compare read-only event histories around the boundary.

**Evaluation boundary:** legal transitions remain legal, prohibited transitions remain prohibited, and terminal-state invariants survive replay.

### CEB-BEH-002 — Cache invalidation / stale derived fact

**Capability:** trace a stale value through source-of-truth state, derived projection and cache invalidation.

**Visible challenge:** a state change is committed but one read path continues returning an old derived value.

**Confounders:** eventual propagation delay, independent cache lifetime and an unrelated stale replica.

**Discriminating action:** compare authoritative record/version, projection revision and cache key lineage without mutating production state.

**Evaluation boundary:** invalidate or version the correct derived facts; do not solve the symptom by globally disabling caching unless the budgeted contract supports that tradeoff.

### CEB-BEH-003 — Multi-fault invariant interaction with an incomplete test oracle

**Capability:** distinguish multiple simultaneous defects and avoid optimizing only for the visible test.

**Visible challenge:** an operation appears successful, but two independent invariants are broken under a less common sequence. The visible test exercises one invariant only.

**Confounders:** one repair masks the second defect; a broad rewrite makes all visible tests pass while altering a protected behavior.

**Discriminating action:** derive an invariant matrix and add an independently specified counterexample/probe for each candidate hypothesis.

**Evaluation boundary:** recover all seeded faults, retain protected behavior and report bug recall separately from edit precision.

### CEB-BEH-004 — Concurrency race / time-of-check-to-time-of-use

**Capability:** reason about temporal ordering and shared state, not only deterministic single-thread traces.

**Visible challenge:** the failure is intermittent and depends on an operation interleaving; serial replay appears correct.

**Confounders:** scheduler noise, logging that changes timing, and a lock that removes the symptom while deadlocking another path.

**Discriminating action:** inspect synchronization and event order; later use a deterministic scheduler/model-checking or controlled stress harness in isolation.

**Evaluation boundary:** reproduce the interleaving or state the remaining uncertainty; preserve liveness as well as safety.

## 6. Distributed delivery, retries and recovery

### CEB-DIST-001 — Duplicate delivery and idempotency-key scope

**Capability:** distinguish duplicate transport delivery from duplicate logical operation.

**Visible challenge:** retries create a duplicate state transition or duplicate external effect after an apparent timeout.

**Confounders:** idempotency key reused across distinct principals/tenants, key retention expiry and a deduplication projection that lags the authoritative record.

**Discriminating action:** map the operation identity, authorization context, idempotency scope and durable state transition; inspect event lineage across retries.

**Evaluation boundary:** at-least-once delivery cannot create more than one authorized logical effect; deduplication must not collapse distinct operations.

### CEB-DIST-002 — Crash between durable commit and acknowledgement

**Capability:** identify the durability/acknowledgement boundary and recover correctly from process death.

**Visible challenge:** after a crash, the caller retries an operation whose durable write may have committed even though acknowledgement was never observed.

**Confounders:** stale read replica, delayed message replay and a process-local cache that disappears on restart.

**Discriminating action:** compare durable commit evidence, transaction/receipt identity and replay policy; do not infer commit or non-commit from a missing response.

**Evaluation boundary:** replay is idempotent, recovery preserves the durable state, and ambiguity remains explicit where commit status cannot be established.

### CEB-DIST-003 — Reordering, tombstones and stale resurrection

**Capability:** reason about causal ordering, deletion semantics and stale replicas.

**Visible challenge:** an older update arrives after a newer deletion or revocation and appears to restore the deleted object.

**Confounders:** clock skew, non-monotonic event timestamps and independent replication lag.

**Discriminating action:** inspect logical version/causal metadata, tombstone retention and merge rules; wall-clock ordering alone is not an acceptable causal order.

**Evaluation boundary:** deletion/revocation cannot be undone by stale delivery; legitimate later recreation remains distinguishable from resurrection.

### CEB-DIST-004 — Partitioned commit with uncertain quorum outcome

**Capability:** distinguish safety from liveness under partition and recover without inventing certainty.

**Visible challenge:** two components observe different availability and the client receives an ambiguous result around a quorum/commit boundary.

**Confounders:** stale quorum view, delayed retry and local success before durable cluster commit.

**Discriminating action:** map quorum/commit receipts, epoch/term, fencing token and authorization state; avoid speculative retry where it could duplicate an external effect.

**Evaluation boundary:** no conflicting committed state, no unauthorized action, bounded recovery when assumptions permit, and correct abstention when commit status is unknowable.

## 7. Performance regressions and optimization

### CEB-PERF-001 — Algorithmic complexity hidden by small fixtures

**Capability:** identify asymptotic or data-distribution-sensitive regressions.

**Visible challenge:** a path passes correctness tests and is fast on a small sample but degrades sharply at production-shaped scale.

**Confounders:** cache warmth, input skew, allocator noise and a microbenchmark that does not exercise the real call path.

**Discriminating action:** inspect workload shape and call graph first; later run pinned, paired workload measurements with randomized order and a declared warm-up policy.

**Evaluation boundary:** semantic equivalence is mandatory; report scaling curve, uncertainty and absolute runtime, not only one speedup number.

### CEB-PERF-002 — Lock contention and throughput/tail-latency reversal

**Capability:** identify contention and queueing effects that average throughput hides.

**Visible challenge:** a change improves single-thread throughput but worsens p95/p99 latency or throughput under concurrent load.

**Confounders:** CPU oversubscription, lock convoying, scheduler variance and unrelated background work.

**Discriminating action:** correlate lock/wait telemetry with controlled concurrency strata and validate that instrumentation overhead is bounded.

**Evaluation boundary:** do not trade away tail latency or starvation constraints to improve the mean unless the task contract explicitly permits it.

### CEB-PERF-003 — Allocation / copy churn in a semantically equivalent pipeline

**Capability:** locate repeated allocation, cloning or serialization costs across function boundaries.

**Visible challenge:** the hot function appears inexpensive in isolation, while the end-to-end workload allocates or copies the same data repeatedly.

**Confounders:** arena reuse, allocator caching and a benchmark whose input lifetime differs from production.

**Discriminating action:** trace allocation ownership and call boundaries; compare allocation counts/bytes with the exact workload and preserve lifetime/safety invariants.

**Evaluation boundary:** lower resource use without aliasing bugs, use-after-free, data corruption or hidden retention growth.

### CEB-PERF-004 — Cache optimization with a memory or invalidation tradeoff

**Capability:** diagnose whether a cache improves the actual workload after accounting for invalidation, memory and consistency.

**Visible challenge:** a cache improves a hot benchmark but creates memory growth, stale results or a miss-path tail-latency regression.

**Confounders:** skewed key distribution, too-short test duration and a warm-cache-only comparison.

**Discriminating action:** compare hit/miss distributions, invalidation events, memory pressure and cold/warm behavior under fixed inputs.

**Evaluation boundary:** correctness and freshness are hard gates; report throughput, tail latency, memory and invalidation cost separately.

## 8. Security and configuration failures

### CEB-SEC-001 — Path traversal / symlink race across a trust boundary

**Capability:** identify the difference between lexical path validation and safe filesystem resolution.

**Visible challenge:** a path appears to remain within an allowed root but can resolve through traversal or a symlink change between check and use.

**Confounders:** Unicode normalization, platform-specific separator behavior and a benign-looking path prefix.

**Discriminating action:** statically trace path construction and filesystem authority; later test adversarial paths in a disposable isolated filesystem, not on the host.

**Evaluation boundary:** no access outside the declared root and no check/use race; reject ambiguous or unsupported path semantics.

### CEB-SEC-002 — Authorization confused deputy / stale projection

**Capability:** distinguish the authoritative authorization decision from application state, cache or replicated projection.

**Visible challenge:** an action is accepted based on a stale permission snapshot after revocation or principal/context change.

**Confounders:** cached role membership, tenant crossing and a DHT projection that is eventually consistent.

**Discriminating action:** trace authority source, principal, resource, action, policy version and revocation epoch; use a read-only decision trace before any side effect.

**Evaluation boundary:** deny by default when authoritative authorization is unavailable or stale; no app-state or projection-only grant.

### CEB-SEC-003 — “Fix” one security property by weakening another

**Capability:** detect compensating security regressions and preserve multiple invariants simultaneously.

**Visible challenge:** a proposed fix closes one exposed path but broadens permissions, disables a validation layer or changes the secure default.

**Confounders:** visible exploit test passes, compatibility tests pass, but a separate privilege boundary is weakened.

**Discriminating action:** construct an invariant/attacker-capability matrix and use negative controls that exercise the protected neighboring boundary.

**Evaluation boundary:** all security invariants hold together; local exploit closure alone is not accepted.

### CEB-SEC-004 — Secret disclosure through diagnostics or error propagation

**Capability:** trace data lineage from secret-bearing input to logs, errors, telemetry and persisted diagnostic artifacts.

**Visible challenge:** the primary API response is redacted, but a retry/error or debug path serializes a secret-bearing structure.

**Confounders:** structured logging fields, panic/debug formatting, nested error sources and sampling-only telemetry.

**Discriminating action:** perform static data-flow review and inspect sink allowlists; later use synthetic canary secrets in isolated tests.

**Evaluation boundary:** no canary disclosure in any declared sink; logging must remain useful without serializing secret-bearing payloads.

## 9. Cross-family composition matrix

The 24 blueprints are base cases. The benchmark needs a controlled composition layer:

| Composition | Families combined | Diagnostic question |
|---|---|---|
| CEB-X-01 | Rust semantics + Cargo features | Is the failure caused by a real type/API relation or only a configuration world? |
| CEB-X-02 | Behavioral + distributed | Does the visible state-machine defect originate in local transition logic or delivery/replay order? |
| CEB-X-03 | Performance + concurrency | Does the throughput result survive contention and tail-latency constraints? |
| CEB-X-04 | Security + distributed revocation | Does retry/replication preserve the authoritative revocation boundary? |
| CEB-X-05 | Cargo configuration + security | Does a feature/target fix accidentally disable a protection on one supported build? |
| CEB-X-06 | Behavioral + performance | Is a speedup genuine when the original semantic invariant and input distribution are restored? |

These are composition recipes, not six additional qualified tasks. The evaluator must verify that the constituent faults are independently identifiable where intended and that the composition does not accidentally create an unlabelled third cause.

## 10. Split and statistical design

- Use at least two independently authored repository templates per family. Hold out complete repository templates and fault-mechanism families, not just random seeds.
- Keep public pilot, calibration, development and sealed confirmatory sets distinct.
- Pair each should-act case with a minimally perturbed should-abstain/clarify case. Verify that the perturbation changes only the declared blocker.
- Randomize task order and treatment assignment. Pair methods on the same initial snapshot and resource envelope.
- Use repeated runs where the agent/harness is stochastic. Report pass@k separately from pass-all-k reliability.
- Cluster uncertainty by repository template and injected fault family. Multiple seeds of one template are not independent repository samples.
- Pre-register primary outcomes, practical effect margin, task/sample-size reasoning, exclusions, missingness and stop rules before sealed evaluation.
- Report the full disposition denominator: pass, fail, abstain, unsupported, tool failure, timeout, ungraded and not attempted.
- Human review adjudicates task validity and ambiguous repair quality; an LLM judge alone is not the correctness oracle.

## 11. Qualification gates before public scoring

A task blueprint becomes a task only after all of the following:

1. A pinned fixture and source/configuration digest exist.
2. The reference patch passes and the empty patch fails.
3. Known-bad and regression-inducing patches fail for the intended reason.
4. The hidden grader cannot be modified or read by the agent.
5. The grader's test coverage and mutation sensitivity are independently reviewed.
6. The candidate is regraded from a pristine environment.
7. The task has no hidden network or external-state dependency unless explicitly declared.
8. The runner satisfies #10064; host-opt-in execution is not accepted.
9. The run receipt and regrade evidence bind exact task, source, patch, grader, model, harness and environment identities.
10. A separate reviewer confirms the task is not answer-leaking, under-specified beyond its intended ambiguity, or solved by a cheap shortcut.

## 12. Metrics and target discipline

Report this vector, not an opaque aggregate:

- root-cause top-1/top-3 and false confident-singleton rate;
- causal relation precision/recall by evidence authority;
- forecast Brier/log score and observed-versus-predicted calibration;
- experiment yield, decision change, risk, cost and time;
- bug-level recall and controlled-cohort edit precision;
- independently accepted repair rate and regression rate;
- cost and latency per accepted repair;
- expert-normalized performance speedup where measured;
- paired act/abstain accuracy, question precision/blocker recall and selective risk;
- transfer to held-out repositories, languages, targets, fault families and environments.

Use the proposed internal qualification gates from #10366/#10385 only as preregistered targets. They are not external standards or achieved performance. Do not publish a SOTA claim until the task set, isolation, grader, receipts and regrade are independently qualified.

## 13. Immediate no-runner work

Safe now:

- review and freeze these blueprint definitions;
- author inert task metadata and hidden answer manifests under separate custody;
- write pure Rust ledger/evaluator data types and validation over synthetic values;
- define scoring functions over already recorded, inert task-level results;
- build source/configuration graph fixtures that do not invoke builds or scripts;
- audit every benchmark command and executable path cited in documentation.

Not safe/qualified yet:

- run fixture build scripts, tests, candidate code, benchmarks or agent-generated executables;
- treat a host container or a signed receipt as isolation proof;
- train on sealed labels or use public task blueprints as confirmation data;
- claim external benchmark reproduction or comparative SOTA from blueprint review.

The next implementation slice is the pure Rust CEB ledger model and deterministic synthetic-record validator described in the [integration architecture](https://github.com/Luminous-Dynamics/symthaea/blob/research/ceb-causal-debugging-2026-11-04/docs/research/CEB_CAUSAL_INTEGRATION_ARCHITECTURE_V0_2026-11-04.md), with all candidate execution still held behind #10064.
