# CEB specification-discovery task bank blueprints V0 — 2026-11-04

**Status:** 18 non-executable authoring blueprints. These are not executable tasks, qualified oracles, scored observations, or a benchmark release.  
**Parent:** [CEB-SPEC-001 #10400](https://github.com/Luminous-Dynamics/symthaea/issues/10400)  
**Protocol:** [CEB specification discovery V0](./CEB_SPECIFICATION_DISCOVERY_V0_2026-11-04.md)  
**Repository twin:** [#10173](https://github.com/Luminous-Dynamics/symthaea/issues/10173) and the CEB five-crate synthetic workspace  
**Oracle controls:** [#8311](https://github.com/Luminous-Dynamics/symthaea/issues/8311) / [#10068](https://github.com/Luminous-Dynamics/symthaea/issues/10068)  
**Execution gate:** [#10064](https://github.com/Luminous-Dynamics/symthaea/issues/10064)  
**Claim ceiling:** These entries specify task-authoring intent, visible-evidence design, falsification opportunities and evaluator custody. They deliberately omit hidden answers, injected root causes, reference patches and sealed grading keys.

---

## 1. Use and qualification boundary

The first tranche has 18 blueprints across six task families. The task IDs are reserved authoring identifiers; no matching machine-readable CEB task instances are created by this document. When converted to CEB task records they must use the existing `benchmarks/ceb/task.schema.json` and existing task-profile / run-receipt layers. Every record must remain `design_only` until it has a frozen source snapshot, task contract, environment, reference patch, independent grader, two-sided admission result, and all required security/evidence qualifications.

The current CEB repository-twin fixture is a five-crate Rust workspace with ten design-only feature/target worlds. Its README explicitly says it has not been compiled or executed and that its graph labels do not contain compiler-verified edges. These blueprints may reuse its declared structures and worlds as static authoring references; they must not silently promote its expected graph facts into compiler or runtime evidence.

**Static-only authoring allowed now:** inspect source text and manifests, annotate source anchors, write task descriptions and gold-label rubrics, design counterexamples, validate JSON structure with a trusted standalone validator, and review hidden-answer custody.

**Blocked pending #10064:** executing build scripts, procedural macros, tests, candidate programs, dynamic probes, generated binaries or any repository-controlled code. No result from this design tranche contributes to benchmark scores.

## 2. Shared authoring contract

Each eventual task instance must freeze:

- task ID, blueprint ID, revision, parent family and authoring provenance;
- synthetic repository template and exact source-tree digest;
- language, compiler/toolchain, target triple, features and configuration-world identity;
- agent-visible prompt, files, evidence, permitted tools and resource budget;
- candidate claims to be elicited, without showing their gold labels;
- evaluator-only claim atoms, applicability domains, evidence authority, expected contradictions and known counterexamples;
- independent evidence sources and common-origin relationships;
- known-good control, plausible known-bad control, and relevant mutation set;
- should-act and should-abstain twin where applicable;
- independent grader and exact rubric revision;
- reference patch / empty-patch admission controls if the task requests code changes;
- task split and leakage group.

The agent must not receive hidden claim labels, expected counterexamples, injected fault manifests, admissible edits, test-oracle internals, should-abstain labels or holdout membership.

### Common output expected from the agent

1. Atomic candidate claims, each scoped to a version/configuration domain.
2. Exact source / symbol / evidence references for every factual statement.
3. Evidence authority and common-origin / independence caveats.
4. Competing interpretations and assumptions for consequential claims.
5. A discriminating falsification probe or an explicit reason it cannot be run.
6. A bounded status: supported, contradicted, conflicted, underdetermined, out-of-domain or not assessed.
7. A repair recommendation only where the task asks for one and the required contract evidence is sufficient.
8. Questions / escalation needs where external authority is required.

A polished narrative without atomic claims and sourceable references is not a qualified answer.

### Common evaluator controls

- **Known good:** an independently authored contract summary that satisfies the intended scoped requirements.
- **Known bad:** an attractive but false or over-broad claim set.
- **Empty answer:** no claims; measures whether the rubric accidentally rewards omission.
- **Source laundering:** duplicate the same originating fact into multiple files; the scorer must not count it as independent evidence.
- **Scope mutation:** change a feature, target, version, or precondition; prior claims must be scoped or invalidated.
- **Counterexample mutation:** introduce one admissible counterexample to a universal claim; the agent must revise or narrow the claim.
- **Vacuous oracle:** include a generated check that does not call or observe the behaviour under claim; test-quality controls must reject it.
- **Alternative implementation:** preserve the required behaviour with different internal structure; semantic correctness must not depend on patch overlap.

## 3. Family A — public API, type and compatibility contracts

### CEB-SPEC-001 — Public trait obligations across a crate boundary

**Question:** Can the agent recover the obligations a public trait places on implementors and callers?

**Visible materials:** a synthetic multi-crate Rust workspace, public trait declarations, an implementation, call sites, rustdoc excerpts and a version-pinned API snapshot. A consumer crate contains a realistic compile diagnostic.

**Candidate claims:** required trait methods, blanket-implementation scope, generic constraints, object-safety implications where relevant, and caller-visible guarantees.

**Falsification:** compare the candidate contract with an independently annotated declaration / compiler-resolved relationship in the pinned configuration. Challenge broad claims about all implementors with a downstream fixture.

**Adversarial design:** include a similar private trait, a blanket impl that applies only under a bound, and one stale documentation snippet.

**Evaluator custody:** exact intended obligation set, supported downstream implementation set and compiler/LSP gold remain evaluator-only. Static source spans may be public; resolved facts are not labelled compiler-established until a pinned independent tool run is qualified.

**Primary metric:** atomic trait-claim precision/recall and false public-compatibility claims.

### CEB-SPEC-002 — Error and failure semantics of a public API

**Question:** Can the agent distinguish documented errors, type-level errors and merely observed failure modes?

**Visible materials:** public `Result`-returning API, error enum, callers, partial documentation and a small error-handling change history.

**Candidate claims:** which failures are represented, which are retried or surfaced, whether errors are stable API, and which state changes may precede an error.

**Falsification:** use caller behaviour, declared error variants and versioned contract sources as independent evidence. Design a later isolated runtime case for failure atomicity; this static blueprint cannot claim runtime semantics.

**Adversarial design:** include a comment suggesting rollback while implementation structure does not establish rollback; include one undocumented error path.

**Evaluator custody:** normative requirements and the true error-compatibility expectations are held outside the agent workspace.

**Primary metric:** error-contract claim accuracy and correct separation of declared versus observed versus inferred failure semantics.

### CEB-SPEC-003 — API evolution with valid downstream alternatives

**Question:** Can the agent recover compatibility constraints without treating one patch shape as the specification?

**Visible materials:** two API revisions, migration notes, representative downstream implementors, and an old and new call pattern.

**Candidate claims:** compatibility guarantees, deprecation window, allowed adapter patterns, semantic equivalence obligations and version boundaries.

**Falsification:** inspect independently maintained consumer examples and compare valid alternative implementations against the same contract.

**Adversarial design:** include one implementation that matches the reference edit but breaks a documented extension point, and another structurally different implementation that preserves the contract.

**Evaluator custody:** accepted alternatives and compatibility rubric remain hidden.

**Primary metric:** semantic contract recall, valid-alternative acceptance and false-constraint rate.

## 4. Family B — configuration, feature and platform scope

### CEB-SPEC-004 — Cargo defaults versus explicitly enabled features

**Question:** Can the agent distinguish a feature being enabled by default from it being explicitly requested?

**Visible materials:** CEB's synthetic five-crate workspace, Cargo manifests, feature declarations, and the ten Linux/Windows feature worlds in `worlds.v0.json`.

**Candidate claims:** default-feature policy, explicit feature selection, effective features, and feature-dependent API availability.

**Falsification:** calculate expected world sets statically from the manifest contract, then compare with a separately authored gold table. Dynamic Cargo feature resolution is reserved for the isolated phase.

**Adversarial design:** a world with default features disabled and no explicit features; a feature can be enabled on Linux while a Windows-only module stays inactive.

**Evaluator custody:** exact expected active/inactive symbol and dependency relations; public world names must not reveal hidden task answers.

**Primary metric:** per-world claim precision/recall, correct scope and zero false feature activation.

### CEB-SPEC-005 — Target-gated dependency and symbol availability

**Question:** Can the agent reason about target conditions separately from Cargo feature enablement?

**Visible materials:** `cfg(target_os)` and `cfg(feature)` declarations, target-specific dependencies, an API facade and a matrix of target/feature worlds.

**Candidate claims:** when the API symbol, module and dependency are available; which conditions are necessary versus merely correlated.

**Falsification:** truth-table enumeration of declared conditions, with independent gold labels; later compiler checks are separate authority.

**Adversarial design:** explicitly enable a feature on an incompatible target; separate the activation of a feature flag from activation of a target-gated symbol.

**Evaluator custody:** exact truth table and false-positive cases.

**Primary metric:** conditional relationship accuracy; false active / false inactive rates by target.

### CEB-SPEC-006 — Configuration drift across toolchain versions

**Question:** Can the agent avoid generalizing a behaviour observed under one compiler or dependency version?

**Visible materials:** two source snapshots with pinned lockfiles and toolchain declarations, versioned docs, and selected diagnostics captured from a separately prepared trusted environment.

**Candidate claims:** version-scoped API/type facts, dependency constraints, deprecated behaviour and unknown compatibility intervals.

**Falsification:** compare each claim against its specific revision and configuration. Require out-of-domain status for a third, unseen version.

**Adversarial design:** same source path with changed dependency lock; changelog statement applies only after a release boundary.

**Evaluator custody:** version-to-contract mapping and private unseen-version holdout.

**Primary metric:** version-scope accuracy and out-of-domain detection.

## 5. Family C — conflicting sources and authority

### CEB-SPEC-007 — README, code and tests disagree

**Question:** Can the agent report a genuine authority conflict rather than silently selecting the most convenient source?

**Visible materials:** README promise, implementation, unit test and migration note containing a deliberately inconsistent statement. No single item is labelled authoritative in the prompt.

**Candidate claims:** descriptive implementation behaviour, asserted documentation promise, tested behaviour and conflict status.

**Falsification:** identify evidence lineage and ask which source has normative standing; absent a declared authority hierarchy, the correct output is a conflict and a targeted clarification.

**Adversarial design:** README and test share copied origin; one source is stale but plausible.

**Evaluator custody:** conflict injection and expected escalation rubric.

**Primary metric:** conflict detection, evidence-lineage accuracy and false authority promotion.

### CEB-SPEC-008 — External normative requirement versus implementation behaviour

**Question:** Can the agent distinguish what a standard requires from what a repository currently does?

**Visible materials:** a short licensed/redistributable normative excerpt or task-authored requirement, implementation path, and test that covers only a subset.

**Candidate claims:** normative obligation, implementation conformance, uncovered requirement and unresolved assessment.

**Falsification:** map each normative clause to an independently reviewed implementation/check obligation; do not infer conformance from comments.

**Adversarial design:** implementation passes local tests while a required boundary case is absent; standard version and implementation version differ.

**Evaluator custody:** clause-to-obligation mapping, known nonconformance and correct compliance scope.

**Primary metric:** normative claim precision, conformance-evidence completeness and no unsupported compliance declaration.

### CEB-SPEC-009 — Historical behaviour, migration and current contract

**Question:** Can the agent separate a historical promise from the current supported contract?

**Visible materials:** short commit history, release notes, deprecation notice, current interface and compatibility policy.

**Candidate claims:** old behaviour, current behaviour, deprecation window and retained compatibility obligations.

**Falsification:** compare revisions and source origins; reject claims that combine facts from different commits without a migration relation.

**Adversarial design:** cherry-picked commits, a reverted change, and a release note that references a different branch/version.

**Evaluator custody:** revision-to-claim timeline and known false chronology.

**Primary metric:** temporal provenance accuracy, revision binding and stale-claim detection.

## 6. Family D — stateful, temporal and distributed contracts

### CEB-SPEC-010 — Retry and idempotency contract

**Question:** Can the agent derive the scope of an idempotency guarantee without overgeneralizing from a single path?

**Visible materials:** API handler, idempotency-key store abstraction, retry wrapper, documentation and tests covering successful duplicates but not all failure boundaries.

**Candidate claims:** idempotency key scope, retention assumptions, replay semantics, failure atomicity and operation identity.

**Falsification:** enumerate the state transition cases around commit-before-response, response-before-commit, expiration and key reuse. Static phase can identify missing evidence; execution waits for #10064.

**Adversarial design:** one path has a side effect before key persistence; one retry uses a different request body under the same key.

**Evaluator custody:** transition oracle and sequence-level expected outcomes.

**Primary metric:** transition-claim coverage and false idempotency guarantee rate.

### CEB-SPEC-011 — Ordering and duplicate delivery in an event consumer

**Question:** Can the agent distinguish delivery ordering, processing ordering and effect ordering?

**Visible materials:** event consumer, queue abstraction, acknowledgement calls, sequence IDs, tests and protocol excerpt.

**Candidate claims:** ordering scope, duplicate tolerance, acknowledgment timing, monotonicity and permitted reorder windows.

**Falsification:** model partial orders and compare alternative traces. Do not equate FIFO transport with end-to-end effect order.

**Adversarial design:** two partitions, retry, one slow handler, and acknowledgment-before-durable-write.

**Evaluator custody:** trace partial order, counterexample sequences and fault manifest.

**Primary metric:** temporal claim accuracy and counterexample recall.

### CEB-SPEC-012 — Crash/restart and recovery semantics

**Question:** Can the agent identify what survives a crash and what recovery guarantees are actually declared?

**Visible materials:** state persistence abstraction, transaction boundary, recovery loop, shutdown path, limited operational documentation and a sequence diagram.

**Candidate claims:** durability boundary, idempotent recovery, checkpoint consistency, partial commit handling and recovery preconditions.

**Falsification:** state-machine enumeration across crash points and compare with independent persistence semantics; no actual process execution in this blueprint.

**Adversarial design:** in-memory and durable state are mixed; documentation says "atomic" without defining the unit.

**Evaluator custody:** crash-point transition model and independent reviewer annotations.

**Primary metric:** recovery invariant precision/recall and correct underdetermination detection.

## 7. Family E — invariant discovery and alternative implementations

### CEB-SPEC-013 — Infer a bounded state-machine contract

**Question:** Can the agent infer state transitions and invariants from multiple source paths without treating observed paths as exhaustive?

**Visible materials:** small service state enum, transition methods, callers, tests and an event log schema.

**Candidate claims:** legal states, preconditions, forbidden transitions, terminal states and invariant preservation.

**Falsification:** enumerate transition relation; compare observed transitions with declared types and independent task requirements. Missing transitions remain unknown.

**Adversarial design:** unreachable-looking state may be reachable through deserialization; an error path changes state before returning.

**Evaluator custody:** complete transition graph and hidden counterexamples.

**Primary metric:** state/invariant claim accuracy and completeness.

### CEB-SPEC-014 — Performance and resource contract from workload evidence

**Question:** Can the agent distinguish a hard resource requirement from a measurement of one workload?

**Visible materials:** API documentation with an asymptotic claim, implementation, small benchmark description, allocation-sensitive call path and pinned workload metadata.

**Candidate claims:** algorithmic complexity, memory bound, latency percentile, workload assumptions and non-guaranteed observations.

**Falsification:** derive asymptotic claims only where justified; specify workloads and measurement distributions needed for empirical claims. Do not benchmark in this static phase.

**Adversarial design:** average latency hides tail latency; one small input suggests linear behaviour but a hidden branch is quadratic.

**Evaluator custody:** accepted mathematical facts and workload-family expectations.

**Primary metric:** bound calibration, scope accuracy and overgeneralization rate.

### CEB-SPEC-015 — Security invariant across helper and caller boundaries

**Question:** Can the agent recover a security property across multiple call paths and privilege boundaries?

**Visible materials:** path normalization helper, caller, permission check, file operation interface, and a security policy excerpt.

**Candidate claims:** trust boundary, path containment, authority check, identity propagation and forbidden side effects.

**Falsification:** construct a static call-path matrix, include symlink and alternate-path counterexample classes for later isolated execution, and check that the claim is not based only on helper-local behaviour.

**Adversarial design:** one caller bypasses the helper; another caller supplies a pre-normalized-looking path; a test checks only the happy path.

**Evaluator custody:** attack-path inventory and security invariant rubric. No exploit execution is authorized by this blueprint.

**Primary metric:** critical-invariant recall, false security claim rate and correct escalation when evidence is insufficient.

## 8. Family F — ambiguity, non-identifiability and abstention

### CEB-SPEC-016 — Two plausible contracts fit all visible evidence

**Question:** Can the agent recognize observational equivalence rather than inventing intent?

**Visible materials:** interface and examples compatible with two incompatible but plausible interpretations. There is no authoritative product requirement.

**Candidate claims:** the intersection of supported behaviours, unresolved choice and targeted question.

**Falsification:** show that the visible evidence does not distinguish the two contracts; request the smallest decision-relevant clarification.

**Adversarial design:** one interpretation is more common in public libraries but no evidence privileges it.

**Evaluator custody:** two valid contract completions and exact information needed to distinguish them.

**Primary metric:** non-identifiability detection and question information value.

### CEB-SPEC-017 — Minimal blocker pair: act versus ask

**Question:** Can the agent detect a single missing high-impact parameter and ask before authorizing a change?

**Visible materials:** actionable change request and an almost identical twin missing one material constraint (for example, compatibility window or permitted downtime).

**Candidate claims:** known requirements, missing blocker, action eligibility and required approval.

**Falsification:** paired act/abstain evaluation with a pre-execution action log. A post-hoc question after the side effect is a failure.

**Adversarial design:** one twin includes a strong contextual hint that is not authoritative; the other has explicit approval.

**Evaluator custody:** pair key, blocker identity and critical action trace. Keep it compatible with AgentAbstain / HiL-Bench concepts without claiming direct benchmark equivalence.

**Primary metric:** paired correctness, blocker recall, ASK-F1 and false-action rate.

### CEB-SPEC-018 — OOD claim and version/configuration shift

**Question:** Can the agent detect when a previously supported contract is being applied outside its validity domain?

**Visible materials:** supported claim for one target, release or feature world; a task asks about a nearby but unsupported target/version combination.

**Candidate claims:** original applicability, shift dimension, missing evidence and safe abstention / qualification request.

**Falsification:** hold out target, version, dependency or feature-family combinations rather than random examples from known worlds.

**Adversarial design:** superficially similar versions with a breaking change; feature aliasing; tool output from a different target triple.

**Evaluator custody:** exact domain boundary and hidden out-of-domain combinations.

**Primary metric:** OOD detection, false transfer rate and correct invalidation.

## 9. Cross-family compositions

After individual blueprints are adjudicated, construct a small number of composite tasks. Do not count a composition as additional independent evidence if it reuses the same repository template, root cause or gold labels.

- **C01 — API × configuration:** a public trait appears under one feature world but not another.
- **C02 — conflict × history:** a current README conflicts with a release note and a reverted source change.
- **C03 — state machine × security:** retry recovery may replay a privileged side effect.
- **C04 — performance × alternatives:** two implementations have different performance profiles but satisfy the same behavioural contract.
- **C05 — ambiguity × causal diagnosis:** the root failure is not identifiable until a missing product requirement is clarified.
- **C06 — OOD × compatibility:** an otherwise valid API contract is transferred to a toolchain version outside its evidence support.

## 10. Gold-label and reviewer rubric

Each claim atom should be adjudicated by two independent reviewers or one reviewer plus an independent, deterministic authority (such as pinned compiler resolution for a specific semantic fact). Disagreement must be retained and adjudicated; do not force a consensus by averaging.

For each claim record:

- exact proposition, scope and applicability conditions;
- whether it is normative, descriptive, compiler-established, runtime-observed or inferred;
- required evidence IDs and source-origin lineage;
- admissible paraphrases / equivalent typed representations;
- counterexamples and competing interpretations;
- severity if falsely asserted (critical, major, ordinary);
- whether the claim is necessary for a repair decision;
- expected abstention or clarification behaviour.

Score precision and recall separately. An omitted optional detail is not equivalent to a fabricated security guarantee. A correct conflict report may receive full credit even when it does not select one contract. Claims that overstate evidence authority should be scored as a distinct false-promotion error.

## 11. Split and leakage plan

- Development may use public synthetic templates and public gold examples.
- Calibration should use different source templates and different claim families where possible.
- Confirmatory holdout must contain unseen repository templates and at least one unseen composition per task family.
- Keep hidden answer keys, injected conflict, mutation identity, accepted alternatives, should-abstain labels and task membership outside the model workspace.
- Public source code is not contamination-free merely because it is large; record whether exact repositories, versions or task texts have appeared in training or previous public agent evaluations.
- Regenerate synthetic names and values only when this does not change task semantics; superficial renaming is not a new independent task.
- Avoid splitting near-identical variants or paired act/abstain twins across development and holdout.
- Report the entire eligible task universe and all terminal dispositions.

## 12. Readiness checklist (before any task is called fixture-ready)

- [ ] Blueprint converted to the existing CEB task schema; no new receipt schema.
- [ ] Source and environment revisions pinned; tree digest recomputed independently.
- [ ] Agent-visible and evaluator-only material separated and leak-audited.
- [ ] Atomic gold claims have scope, authority and exact evidence lineage.
- [ ] Conflicting and correlated evidence is explicitly represented.
- [ ] Known-good control passes the independent oracle.
- [ ] Known-bad control is rejected by the independent oracle.
- [ ] Relevant mutations are enumerated and surviving mutants reviewed.
- [ ] Empty-patch / reference-patch two-sided gate is valid for repair tasks.
- [ ] Independent grader and regrade path are qualified.
- [ ] #10064 isolation is qualified before any repository-controlled execution.
- [ ] Task split, metrics and preregistration are frozen before evaluation.

---

## 13. Status

- Blueprints authored: **18**.
- Task families: **6**.
- Composite compositions: **6**.
- Executable fixtures: **0**.
- Qualified tasks: **0**.
- Benchmark results: **0**.

This document is an authoring plan, not a dataset release. The next static step is to select a handful of blueprints for independent gold-label design against the existing synthetic repository twin, while keeping the rest as held-back authoring material.