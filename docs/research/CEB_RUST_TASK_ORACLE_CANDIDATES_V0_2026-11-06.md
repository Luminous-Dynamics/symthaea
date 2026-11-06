# CEB Rust Task and Oracle Candidate Dossier V0 — 2026-11-06

**Status:** Public, design-only development material. Not an admitted task set, executable fixture, sealed holdout, qualified oracle, or performance result.  
**Tracking issue:** [#10701 — author six reviewer-ready Rust task and oracle candidates](https://github.com/Luminous-Dynamics/symthaea/issues/10701)  
**Prerequisites:** [#10609 task admission and oracle qualification](https://github.com/Luminous-Dynamics/symthaea/pull/10609), [#10617 balanced sampling plan](https://github.com/Luminous-Dynamics/symthaea/pull/10617), [#10064 execution isolation](https://github.com/Luminous-Dynamics/symthaea/issues/10064), [#10369 signer trust and independent runner attestation](https://github.com/Luminous-Dynamics/symthaea/issues/10369).

## 1. Purpose and claim boundary

CEB already has a 48-slot sampling plan and a task-admission contract, but the slot plan is capacity planning and the admission registry currently contains zero admitted real tasks. This dossier moves one step closer to usable task inventory by turning six broad Rust-engineering domains into reviewable candidate specifications with:

- stable, normative requirement IDs;
- explicit assumptions, scope and non-requirements;
- concrete synthetic workspace outlines;
- more than one acceptable implementation strategy where the contract permits alternatives;
- invalid near-miss behaviors;
- requirement-to-test traceability;
- explicit oracle polarity and verifier risk notes;
- unresolved provenance, fixture, environment and execution dependencies.

These are **candidate blueprints**, not actual task cards or runnable repositories. No task ID is assigned, no sampling slot is filled, and no record is promoted in `benchmarks/ceb/catalog.v0.json`. The eventual author must use the canonical #10609 task schema and admission state machine. No parallel task registry, run receipt, oracle authority, grader store or metric is introduced here.

All material in this public repository is exposed development material. It is permanently ineligible to be called a sealed holdout.

No candidate-controlled code, Cargo build script, procedural macro, unit test, integration test, grader, fuzzer or attack payload was executed to prepare this document. Nothing here qualifies the host runner, candidate execution, an oracle, signer custody or independent regrading.

## 2. How to use these candidates

The six domains are now mapped to the exact six root-cause family IDs in the canonical #10617 sampling and preregistration artifacts. This is a **family mapping only**: no repository identity, slot, task ID, partition, or difficulty label is assigned. The canonical plan remains authoritative for all sampling decisions.

A compact machine-readable requirement/oracle crosswalk is maintained in [`CEB_RUST_CANDIDATE_ORACLE_TRACEABILITY_V0_2026-11-06.json`](CEB_RUST_CANDIDATE_ORACLE_TRACEABILITY_V0_2026-11-06.json). It contains requirement IDs, case IDs, expected dispositions, family mappings and fixture-validity controls. It is an authoring aid only—not a task registry or task manifest.

A reviewer should classify each candidate:

- **Ready for static review:** requirements and oracle logic are sufficiently precise to review without running candidate code.
- **Revise:** a requirement, alternative implementation, negative control, or expected disposition is ambiguous or unsupported.
- **Reject:** the task is underspecified, unbounded, contaminated, not independently gradeable, or has no defensible oracle.

A candidate is not `fixture_ready` merely because its specification is detailed. It remains design-only until the concrete source snapshot, source/license provenance, reference patch, grader, environment and all required custody fields exist and have been reviewed under #10609.

## 3. Candidate summary

| Candidate theme | Canonical #10617 family ID | Mapping rationale | Main engineering challenge |
|---|---|---|---|
| Cross-crate API and trait migration | `multi-crate-api-regression` | Public API changes must preserve all in-scope downstream crate consumers and compatibility behavior. | Preserve public semantics across crate boundaries. |
| Async trait/API bounds and object-safe adapter | `rust-types-traits` | The decisive correctness surface is trait-object compatibility, associated types, lifetimes and `Send`/`Sync` bounds. | Preserve type-level and trait-level contracts through an async abstraction. |
| Deterministic reproduction of an event-processing defect | `issue-reproduction-debugging` | The task is admitted only if the defect is reproducibly triggered, diagnosed and fixed from a bounded event schedule. | Reproduce a stateful failure, isolate its cause and demonstrate a durable repair. |
| Workspace command and path boundary | `security-correctness` | The core acceptance property is preventing unauthorized command execution and filesystem escape. | Prevent command injection and path escape. |
| Cargo feature/target composition | `feature-target-configuration` | The contract is the declared matrix of supported feature combinations and compilation targets. | Make supported feature sets coherent without environment leakage. |
| Streaming top-k under bounded resources | `performance-correctness` | The task requires a measured resource improvement without semantic or workload-generalization regression. | Exact ranking, updates and deterministic ties at bounded memory. |

**Mapping provenance:** `benchmarks/ceb/rust-balanced-sampling.v0.json` and `benchmarks/ceb/rust-pilot-preregistration.v0.json` on the #10617 branch define the canonical six family IDs. This dossier maps one development candidate theme to each family but does not assign any `repo-slot-*` cell, fill a planned slot, or select a development/holdout partition.

These candidates are not a claim that six tasks are sufficient for the 48-task pilot or a confirmatory study. They are a reviewer-calibration tranche.

---

## 4. Candidate A — Cross-crate API and trait migration

**Canonical family:** `multi-crate-api-regression`.  
**Provisional theme:** Rust API compatibility and type-system correctness.  
**Provisional difficulty:** Medium.  
**Synthetic workspace outline:**

```text
api-migration/
  Cargo.toml                 # workspace root
  crates/
    domain-types/
    record-parser/
    storage-adapter/
    cli/
  docs/
    compatibility-contract.md
```

### Normative requirements

- **API-01 — Typed failure:** malformed input must be represented by a typed error result; the implementation must not panic or silently fabricate a value.
- **API-02 — Compatibility surface:** the declared public compatibility adapter remains callable with its documented input/output semantics.
- **API-03 — Cross-crate coherence:** every in-scope consumer uses the new typed contract; no crate may retain a stale signature or duplicate an incompatible local error type.
- **API-04 — Error identity:** callers can distinguish the normative error classes declared by the specification without parsing human-readable strings.
- **API-05 — Behavior preservation:** valid legacy inputs preserve their specified semantic result.
- **API-06 — No scope deletion:** removing a public item, consumer, test, feature or documented use is not an acceptable way to make the migration compile.

### Valid implementation alternatives

1. Define one shared error enum in the domain-types crate and expose it through the parser crate.
2. Keep parser-internal errors private and translate them through a stable public adapter type, provided the adapter preserves all required error distinctions.
3. Use a compatibility wrapper around the new API rather than retaining two independent parser implementations.

Exact type names and module placement are not normative unless later frozen in the task specification.

### Negative near-misses

- Change the signature in the producer crate but leave one downstream crate using the stale API.
- Convert all errors to `None`, a generic string or a panic.
- Remove the compatibility adapter or the documented consumer instead of migrating it.
- Make the workspace compile only with default features while the specification requires an additional declared feature combination.
- Duplicate error enums across crates so downstream code cannot reliably distinguish the required classes.

### Oracle traceability

| Oracle case | Requirement | Expected disposition |
|---|---|---|
| Valid input through new API and compatibility adapter | API-02, API-05 | Accept |
| Each declared malformed-input class | API-01, API-04 | Accept typed error; reject panic/fabrication |
| Compile/type-check every in-scope consumer | API-03 | Accept only coherent cross-crate surface |
| Remove one consumer or public adapter | API-06 | Reject |
| Semantically equivalent shared or translated error representation | API-04 | Accept |

**Composition case:** a valid legacy request must still succeed through the compatibility adapter after the producer API changes, while a malformed request must produce the declared typed error through both entry points.

**Oracle warning:** compiler success is necessary but insufficient. A checker that compares exact function names or enum layout can reject legitimate wrappers; a checker that only runs `cargo check` can miss deleted behavior and error collapsing.

**Unresolved before fixture creation:** freeze the exact public items, feature combinations, compatibility window, Rust toolchain and which semantic error distinctions are observable.

---

## 5. Candidate B — Async trait/API bounds and object-safe adapter

**Canonical family:** `rust-types-traits`.  
**Provisional difficulty:** Medium–high.  
**Synthetic workspace outline:**

```text
async-trait-boundary/
  Cargo.toml
  crates/
    domain-types/
    async-contract/
    worker-adapter/
  tests/
    # proposed type-level and behavior cases only; not yet implemented or run
```

### Normative requirements

- **TRAIT-01 — Object-capable contract:** the public async service contract must be callable through the trait-object form explicitly required by the specification.
- **TRAIT-02 — Sendable future:** the returned future must satisfy the declared `Send` bound so it can cross the target executor's task boundary.
- **TRAIT-03 — Borrowed lifetime:** the call may borrow the service instance for the future's lifetime; the implementation must not impose an undocumented `'static` requirement.
- **TRAIT-04 — Typed associated contract:** success and error types preserve the declared associated-type and error semantics; stringification or erasure of required error distinctions is prohibited.
- **TRAIT-05 — Consumer compatibility:** all in-scope generic and trait-object consumers compile against the same normative contract.
- **TRAIT-06 — No semantic escape:** boxing, pinning or macro expansion may change representation, but may not change cancellation, error or result semantics stated by the contract.

### Valid implementation alternatives

1. An explicit object-safe method returning `Pin<Box<dyn Future<Output = Result<Receipt, ServiceError>> + Send + 'a>>`.
2. An approved async-trait transformation that preserves the same object-safe, `Send`, borrowed-lifetime and error contract.
3. A generic adapter plus a narrow erased boundary, provided the required trait-object consumer remains supported and observable semantics remain identical.

The specification must pin the public observable contract, not a preferred macro or exact source spelling. Whether a macro dependency is permitted must be stated before fixture freeze.

### Negative near-misses

- Return a future that is not `Send` when the declared executor boundary requires `Send`.
- Force `'static` and reject otherwise valid borrowed service implementations.
- Make the trait usable only through static dispatch when the public trait-object form is required.
- Erase associated error distinctions into a generic string or panic.
- Update the trait declaration but leave one in-scope implementor or consumer on the stale signature.
- Change cancellation or error behavior while preserving the same Rust type signature.

### Oracle traceability

| Oracle case | Requirement | Expected disposition |
|---|---|---|
| Explicit boxed-future implementation with required bounds | TRAIT-01–TRAIT-05 | Accept |
| Approved macro-based implementation with identical observable contract | TRAIT-01–TRAIT-06 | Accept |
| Non-`Send` future crossing the required executor boundary | TRAIT-02 | Reject |
| Borrowed implementation rejected solely by an invented `'static` bound | TRAIT-03 | Reject |
| Generic consumer works but required trait-object consumer does not | TRAIT-01, TRAIT-05 | Reject |
| Same error/result contract through generic and erased call paths | TRAIT-04, TRAIT-06 | Accept |

**Composition case:** a borrowed service object is invoked through the required trait-object path on a `Send` executor; the future must satisfy the bound, preserve the typed error and allow the service borrow to end at the correct lifetime.

**Oracle warning:** a textual check for `async_trait`, `Pin<Box<...>>` or one particular trait shape is invalid as a sole oracle. The correct oracle combines type checking against the public contract with observable result/error semantics. Type-check/build execution remains gated by #10064.

**Unresolved before fixture creation:** exact public trait signature, target executor, whether macro dependencies are allowed, MSRV, object-safety form and permitted lifetime relationships.

## 6. Candidate C — Deterministic reproduction of an event-processing defect

**Canonical family:** `issue-reproduction-debugging`.  
**Provisional theme:** reproduce a stateful defect through a bounded retry/replay schedule, isolate its root cause and verify the repair. The idempotence and crash/replay requirements below are the defect domain, not a new family label.  
**Provisional difficulty:** High.  
**Synthetic workspace outline:**

```text
event-ledger/
  Cargo.toml
  crates/
    event-contract/
    event-store/
    projector/
    replay-cli/
  fixtures/
    # public deterministic event traces only
```

### Normative requirements

- **EVENT-01 — Stable identity:** every event has a stable identity independent of delivery attempt.
- **EVENT-02 — Duplicate idempotence:** redelivery of the same identity and same payload produces no additional committed effect.
- **EVENT-03 — Identity conflict:** the same identity with a different payload is a conflict, not a duplicate.
- **EVENT-04 — Commit before acknowledgement:** an event is acknowledged only after the required durable state transition is committed.
- **EVENT-05 — Replay recovery:** after restart, replaying already committed events does not duplicate effects.
- **EVENT-06 — Order contract:** out-of-order events follow the explicit contract (buffer, reject or defer); no silent reorder or drop is permitted.
- **EVENT-07 — Atomic state:** deduplication identity and the business effect cannot diverge across a crash boundary.

### Valid implementation alternatives

1. Transactional insert-once deduplication and state update in one database transaction.
2. An append-only event log with a deterministic projection checkpoint, if checkpoint and projection semantics satisfy EVENT-04 through EVENT-07.
3. A transactional outbox/inbox arrangement with explicit identity conflict handling.

The benchmark must not require a specific database or storage architecture if multiple approaches satisfy the contract.

### Negative near-misses

- Deduplicate only in process memory so restart replays an already committed event.
- Acknowledge before durable commit, then lose the effect on crash.
- Treat same ID/different payload as an ordinary duplicate.
- Commit the business effect but fail to persist the deduplication record (or vice versa).
- Drop an out-of-order event instead of following the specified defer/reject policy.
- Use timestamp order when the contract specifies a sequence number or causal predecessor.

### Oracle traceability

| Oracle case | Requirement | Expected disposition |
|---|---|---|
| Deliver the same event twice without restart | EVENT-01, EVENT-02 | One committed effect |
| Replay the same event after restart | EVENT-02, EVENT-05 | No additional effect |
| Reuse ID with modified payload | EVENT-03 | Explicit conflict |
| Crash between effect and acknowledgement boundary | EVENT-04, EVENT-07 | No acknowledged-but-lost effect |
| Reverse two events with declared dependency | EVENT-06 | Specified defer/reject behavior |
| Inject a duplicate across concurrent workers | EVENT-02, EVENT-07 | One committed effect |

**Composition case:** a duplicate event is concurrently delivered while the process restarts after commit but before acknowledgement; the durable state must show one effect and a replay-safe identity record.

**Oracle warning:** a test that only compares final row counts can miss an incorrect intermediate acknowledgement or event-order violation. The oracle must observe durable state and the acknowledgement contract independently. If crash consistency cannot be tested within the eventual qualified runner, mark that requirement unresolved rather than silently downgrading it.

**Unresolved before fixture creation:** persistence backend, transaction isolation assumptions, crash injection mechanism, sequence contract and durability guarantees.

---

## 7. Candidate D — Workspace command and path boundary

**Canonical family:** `security-correctness`.  
**Provisional theme:** security correctness in Rust tooling.  
**Provisional difficulty:** Medium–high.  
**Synthetic workspace outline:**

```text
workspace-tool/
  Cargo.toml
  crates/
    workspace-path/
    command-runner/
    tool-cli/
  fixtures/
    # synthetic directory trees and adversarial path names
```

### Normative requirements

- **SEC-01 — No shell interpolation:** untrusted path or project content must never be concatenated into a shell command string.
- **SEC-02 — Argument separation:** executable selection and arguments are passed as separate values to the process API.
- **SEC-03 — Root containment:** filesystem operations are confined to the declared workspace root.
- **SEC-04 — Traversal rejection:** absolute paths, parent traversal and disallowed symlink escapes are rejected according to the declared path grammar.
- **SEC-05 — Error preservation:** invalid paths and process failures produce explicit errors; they are not silently mapped to success.
- **SEC-06 — Least authority:** the implementation must not broaden permissions or expose additional commands to bypass a failing check.

### Valid implementation alternatives

1. A strict workspace-relative path type that validates lexical components plus descriptor-relative filesystem access.
2. A platform-specific safe-open primitive (for example, Linux `openat2` with an explicit resolution policy) behind a narrow portable interface.
3. A command runner using direct `std::process::Command` argument vectors, with a restricted executable allowlist and no shell.

The chosen fixture must state its OS and filesystem assumptions. Do not imply that a string-prefix check is a race-free filesystem boundary.

### Negative near-misses

- Build `sh -c` or equivalent command text using untrusted input.
- Pass an untrusted path as part of an executable string.
- Use a naive `starts_with(root)` test that accepts sibling-prefix paths such as `/workspace-evil`.
- Canonicalize, check, then reopen a path under a threat model that permits symlink replacement between check and use.
- Follow a symlink outside the workspace when the contract requires rejection.
- Catch every error and return a successful empty result.

### Oracle traceability

| Oracle case | Requirement | Expected disposition |
|---|---|---|
| Path contains shell metacharacters | SEC-01, SEC-02 | Treat as one data argument or reject; never execute shell syntax |
| Absolute and parent-traversal paths | SEC-03, SEC-04 | Reject |
| Sibling-prefix and symlink escape | SEC-03, SEC-04 | Reject |
| Valid nested workspace path | SEC-03 | Accept |
| Executable outside allowlist | SEC-06 | Reject |
| Command failure | SEC-05 | Surface typed failure |

**Composition case:** a valid in-root path containing spaces and shell metacharacters must be handled as data, while a symlink path resolving outside the root must be rejected. This distinguishes command quoting from path authority.

**Oracle warning:** a static grep for `Command::new` is neither sufficient nor necessary. The grader must establish observable behavior under the stated platform threat model. The task must explicitly say whether concurrent symlink mutation is in scope.

**Unresolved before fixture creation:** OS/runtime image, allowed executable policy, symlink threat model, path normalization rules and whether descriptor-relative open is required.

---

## 8. Candidate E — Cargo feature and target composition

**Canonical family:** `feature-target-configuration`.  
**Provisional theme:** build-system and conditional-compilation correctness.  
**Provisional difficulty:** Medium–high.  
**Synthetic workspace outline:**

```text
feature-matrix/
  Cargo.toml
  crates/
    protocol/
    storage/
    client/
    cli/
  docs/
    supported-build-matrix.md
```

### Normative requirements

- **BUILD-01 — Declared feature surface:** every supported feature combination is explicitly listed in the task specification.
- **BUILD-02 — No accidental default reliance:** each supported no-default-features configuration builds without importing APIs that only exist under defaults.
- **BUILD-03 — Target correctness:** platform-specific dependencies and modules are gated for the targets declared as supported.
- **BUILD-04 — Feature coherence:** mutually exclusive features fail clearly or select the documented behavior; they must not silently select a different backend.
- **BUILD-05 — Public behavior:** the supported feature combinations preserve the same protocol/API contract where the specification requires equivalence.
- **BUILD-06 — No environment leakage:** the build must not pass only because of undeclared host packages, environment variables or locally cached artifacts.

### Valid implementation alternatives

1. Explicit optional dependency features with target-specific dependency declarations.
2. A small backend-selection module behind a stable public trait.
3. A build-time compile error for explicitly unsupported feature combinations, if the specification declares them unsupported.

### Negative near-misses

- Verify only the default feature set.
- Import a feature-gated symbol in unconditional code.
- Declare a feature but fail to wire it to the intended optional dependency.
- Allow two mutually exclusive backends to compile while silently choosing one.
- Depend on a host-installed library not declared by the pinned environment.
- Mark an unsupported target as supported without the required implementation.

### Oracle traceability

| Oracle case | Requirement | Expected disposition |
|---|---|---|
| Default feature set | BUILD-01, BUILD-05 | Accept if declared behavior is preserved |
| Each supported no-default feature set | BUILD-02 | Build and preserve required API |
| Each declared target | BUILD-03 | Accept only supported, correctly gated path |
| Mutually exclusive backend selection | BUILD-04 | Explicit specified result; no silent fallback |
| Clean pinned environment | BUILD-06 | No undeclared host dependency |

**Composition case:** a supported no-default feature build on a non-primary target must work without importing a default-only symbol or activating the wrong backend.

**Oracle warning:** cargo metadata or one successful workspace build cannot prove all declared target/feature combinations. The oracle must derive its matrix from the frozen specification rather than testing arbitrary undocumented combinations.

**Unresolved before fixture creation:** exact Rust toolchain, target triple set, feature matrix, optional dependency versions and available cross-compilation targets.

---

## 9. Candidate F — Exact streaming top-k under bounded resources

**Canonical family:** `performance-correctness`.  
**Provisional theme:** algorithmic correctness, state updates and performance contracts.  
**Provisional difficulty:** Medium–high.  
**Synthetic workspace outline:**

```text
streaming-topk/
  Cargo.toml
  crates/
    topk-core/
    topk-adapter/
  workloads/
    # public development workloads; not a secret speed test
```

### Normative requirements

- **TOPK-01 — Exact ranking:** results equal the top-k elements under the declared total ordering after every accepted update.
- **TOPK-02 — Deterministic ties:** equal scores use the specified stable tie-breaker.
- **TOPK-03 — Update semantics:** insert, update and remove operations affect the ranking exactly once and remove stale entries.
- **TOPK-04 — Boundary behavior:** (k=0), empty input, fewer than (k) elements and duplicate keys have specified results.
- **TOPK-05 — Numeric domain:** NaN, infinities and signed zero are either explicitly handled or explicitly excluded by the contract; no implementation-defined comparison is accepted.
- **TOPK-06 — Resource bound:** memory use and time complexity satisfy the declared workload and bound; the task must define whether bound is asymptotic, measured, or both.
- **TOPK-07 — Measurement integrity:** benchmark inputs and harness behavior may not be inspected or special-cased beyond the public contract; caching is allowed only when it preserves semantics for unseen in-scope inputs and the invalidation contract.

### Valid implementation alternatives

1. An indexed binary heap plus a key-to-index map.
2. A `BTreeSet` or equivalent ordered structure with a key index.
3. A lazy heap with versioned entries and bounded/periodic compaction, provided stale entries cannot change the result and the resource contract is met.

### Negative near-misses

- Sort a fixed visible input and return a precomputed answer.
- Use an unstable tie order that changes across runs.
- Retain stale heap entries after updates or removals.
- Drop equal-score items inconsistently at the (k)-boundary.
- Fail for (k=0) or (k) larger than the stream.
- Return a fast but approximate result when the contract says exact.
- Claim speedup from a benchmark-specific cache that fails on unseen in-scope values or invalidation.

### Oracle traceability

| Oracle case | Requirement | Expected disposition |
|---|---|---|
| Empty and boundary (k) values | TOPK-04 | Exact specified result |
| Duplicate keys and repeated updates | TOPK-03, TOPK-04 | Latest declared state reflected once |
| Equal-score boundary ties | TOPK-02 | Deterministic declared order |
| NaN/infinity/signed-zero cases | TOPK-05 | Exact declared policy |
| Randomized small streams against a simple reference model | TOPK-01, TOPK-03 | Exact equality |
| Unseen in-scope stream and cache invalidation | TOPK-07 | Same semantics without special-casing |
| Declared resource workload | TOPK-06 | Measure only after environment qualification |

**Composition case:** a key at the (k)-boundary is updated, another key is removed, and two remaining entries have equal scores. The result must reflect all three operations and the deterministic tie-breaker.

**Oracle warning:** a slow simple model is appropriate as a reference for small generated traces, not as the performance judge. Raw speed, semantic correctness, and integrity adjudication must remain separate.

**Unresolved before fixture creation:** key type, score domain, tie-breaker, workload distribution, memory ceiling, benchmark hardware and measurement protocol.

---

## 10. Shared oracle contract

The following is common to all six candidates and should be completed using #10609, #10680/#10681, #10685/#10686 and #9431 rather than creating another grader architecture.

### 10.1 Two-sided discrimination

For every proposed candidate:

- A reference implementation must pass the requirements it claims to implement.
- The empty patch and known-invalid near-miss variants must fail at least one cited requirement.
- Valid semantic alternatives must pass; exact source matching is prohibited as the sole oracle.
- False accepts and false rejects use separate denominators and uncertainty.
- An unresolved requirement or unobservable behavior is not silently converted into a candidate failure.

### 10.2 Specification-derived tests

Each test row must include:

- requirement ID(s);
- public or evaluator-only classification;
- input/state construction;
- expected observable behavior;
- rationale for why the behavior follows from the frozen specification;
- one or more valid alternative implementations it must permit;
- invalid near-miss behavior it must reject;
- authority for the expected label;
- reviewer and source lineage.

An evaluator-only test may be hidden from the candidate, but its requirement must be disclosed in the frozen specification before task freeze. Hiddenness is custody, not permission to invent requirements.

### 10.3 Oracle independence

Reusing the general judge-ensemble work in #9431 does not mean that different implementations automatically count as independent. Record, at minimum:

- specification interpretation lineage;
- reference-solution author and lineage;
- test author and fixture origin;
- oracle implementation lineage;
- gold-label reviewer lineage;
- adjudication and tie-resolution path.

A reference implementation, its tests and its grader that all derive from one unchallenged solution are one common-origin evidence chain, not three independent confirmations. Shared lineage must be disclosed; a simple majority cannot resolve a shared false premise.

### 10.4 Metamorphic and mutation controls

For each task, propose mutations that should change the verdict and transformations that should preserve it:

- **Necessary-condition mutation:** remove a required behavior; expected rejection.
- **Irrelevant-change mutation:** change an out-of-scope detail; expected disposition unchanged.
- **Alternative-representation control:** preserve semantics through a different valid design; expected acceptance.
- **Near-miss control:** violate exactly one normative requirement; expected rejection mapped to that requirement.
- **Composition control:** combine two or more independently specified features where the task contract requires their interaction.
- **Label-consistency control:** identical normalized inputs must not receive contradictory golden labels without a recorded scope/authority difference.

Mutation controls are designed here, not executed. Mutation yield is not a substitute for semantic review.

---

## 11. Cross-candidate reviewer checklist

Before any candidate can proceed beyond static review, reviewers should answer:

1. Is every required behavior normative, observable and bounded?
2. Does every proposed test map to an explicit requirement?
3. Are valid semantic alternatives represented, not just the reference patch?
4. Are invalid near-misses specific enough to diagnose without over-constraining implementation?
5. Does the composition case test a required interaction rather than an invented hidden rule?
6. Can two independent reviewers recover the expected disposition from the specification without seeing the reference implementation?
7. Are source, license, fixture, oracle, label and reviewer lineages separately recorded?
8. Are missing data, infrastructure failures, timeouts and unresolved requirements preserved as such?
9. Are candidate-visible artifacts separated from evaluator-only materials before any future execution?
10. Are all budget, environment, toolchain and platform assumptions explicit?
11. Does the proposed task map to a unique, not-yet-filled slot under #10617?
12. Does it meet the #10609 state transition requirements without assuming #10064 or #10369 has passed?

Any "unknown" on critical specification, oracle, license, custody or authority fields is a blocker to admission, not a reason to guess.

## 12. Suggested no-run work sequence

1. **Static reviewer calibration:** have two reviewers independently classify the six public candidate dossiers; compare disagreements by requirement and revise ambiguous language.
2. **Freeze public development specs:** record a version/digest after review. Public fixtures remain exposed and cannot later become sealed holdouts.
3. **Create synthetic source snapshots:** implement minimal Rust workspace fixtures for the candidates selected for further development; record all source files and provenance.
4. **Author positive alternatives and negative near-misses:** do this independently of the reference implementation where practical.
5. **Build independent oracles:** use requirement traceability, semantic alternatives, mutation controls and provenance; do not run them against candidate code until the execution boundary qualifies.
6. **Static task admission review:** populate the canonical #10609 contract only after the required fields and independent reviewers exist; keep status design-only/static-reviewed as allowed by the admission workflow.
7. **Execution and regrade qualification:** wait for #10064 and #10369 before candidate-controlled execution or any independent-grade claim.

## 13. Ownership and non-duplication

- **#10609 / #10607:** canonical task schema, admission state machine, source/license provenance, two-sided oracle admission.
- **#10617 / #10614:** 48-slot sampling design, repository/fault-family quotas and pre-score difficulty assignment.
- **#10436:** source-only agent view and evaluator-oracle custody.
- **#10680 / #10681:** verifier-of-verifiers controls and qualification.
- **#10685 / #10686:** specification-required composition and hidden-test fidelity.
- **#9431:** general judge ensembles, correlated judge errors and dependence-aware aggregation.
- **#10626 / #10634:** contamination transparency and task/source/fix exposure.
- **#10314 / #10370:** canonical receipt, evidence accounting and attestation convergence.
- **#10064:** OS-level isolation; **#10369:** signer trust and independent runner attestation.

No new CEB lane, primary metric, score, receipt, task schema, task registry, trust root, runner or grader authority is created by this dossier.

## 14. Current status

| Item | Status |
|---|---|
| Six candidate specifications authored | Design-only in this document |
| Exact mapping to #10617 fault-family labels | Pending canonical plan review |
| Actual source fixture workspaces | Not created |
| Reference patches | Not created |
| Independent grader implementations | Not created |
| Positive/negative oracle cases | Designed at specification level; not implemented or executed |
| Source/license provenance and contamination cards | Not supplied |
| Pinned environment and resource budgets | Not supplied |
| Task registry entries / admitted tasks | Zero claimed by this dossier |
| Candidate execution | Not performed; blocked by #10064 |
| Independent regrade qualification | Not performed; blocked by #10369 |
| Performance or SOTA comparison | None |

**Honest claim:** this dossier makes six Rust candidate domains reviewable and exposes their specification/oracle obligations. It does not yet provide executable fixtures, qualified oracles, admitted tasks, candidate results, a sealed holdout, or evidence of superiority.
