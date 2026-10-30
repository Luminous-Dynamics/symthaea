# BASALT evidence freshness and selective re-verification — 2026-10-30

## Purpose

A coding agent does not operate in a static repository. Source files, dependency locks, toolchains, task contracts, configuration, runtime environments, and sessions can change while a task is underway. A prior test/build/security result must not silently remain usable after one of the inputs it depended on changes.

This tranche adds a deterministic, Rust-first **evidence-state timeline auditor**. It specializes dependency-aware invalidation for BASALT's run evidence. It does not duplicate the general truth-maintenance graph in [#8998](https://github.com/Luminous-Dynamics/symthaea/pull/8998), the bitemporal source ledger in [#9109](https://github.com/Luminous-Dynamics/symthaea/pull/9109), or the clean-to-shifted analysis in [#8901](https://github.com/Luminous-Dynamics/symthaea/pull/8901).

## Executable slice

Module: `xtask/src/basalt_state.rs`

Command:

```sh
cargo run -p xtask -- basalt-state-audit --input PATH/TO/STATE-TIMELINE.json
```

Contract:
- `benchmarks/basalt/schema/state-timeline.schema.json`
- `benchmarks/basalt/examples/state-timeline.example.json`

The fixture illustrates four directly invalidated evidence records, one transitive dependency invalidation, and two fresh results produced after the source change. It also contains an unrelated README change that must not invalidate evidence that did not depend on that file.

## State and invalidation semantics

1. The initial resource ledger declares resource IDs and SHA-256 digests at generation zero.
2. Every timeline entry has one contiguous sequence number. The timeline is the order of events and evidence creation.
3. A resource-change event must name a declared resource, match its exact current `before_digest`, provide a different `after_digest`, and carry a valid event digest field. The event increments that resource's generation.
4. Evidence must pin one or more declared resources by exact digest **and generation** as they exist at that point in the timeline. A result recorded against an already stale generation is an invalid timeline.
5. Evidence may depend only on evidence recorded earlier. This makes the evidence dependency graph acyclic by construction.
6. A resource change directly invalidates evidence that pinned that resource. Invalidation propagates to all transitive evidence descendants.
7. Evidence whose resource inputs remain unchanged is preserved. Irrelevant file changes do not cause blanket invalidation.
8. Returning a resource to its earlier digest does not resurrect old evidence: generation is monotonic across every declared change. Re-verification creates a new evidence record against the current generation.
9. A result is `usable_for_success` only when it is fresh, its outcome is `pass`, and every declared evidence dependency is itself usable. A fresh failure is still a fresh failure; a stale pass is not usable.
10. The emitted re-verification plan lists stale evidence IDs in original topological/creation order. It is a deterministic candidate queue, not proof that a runner has executed the checks. A structurally valid audit may still contain stale evidence: command success means the timeline was internally consistent, not that every check is current or that the task has passed. Consumers must inspect freshness and `usable_for_success` fields.

The implementation maintains reverse resource and evidence indexes. Evidence leaves those indexes when first invalidated, so later workspace changes do not repeatedly rescan already-stale nodes. A final topological provenance pass closes the full stale-ancestor and invalidation-event history across multi-parent DAGs, independent of traversal order; a diamond-dependency regression covers converging stale parents. Bounds are explicit: 64 MiB input, 20,000 timeline entries, 100,000 initial resources, 4,096 resource pins or evidence parents per record, and 250,000 total dependency edges.

## Trust boundary

The auditor establishes consistency of the supplied timeline and derives freshness relative to its declared resource/event history. It performs a topological closure pass so a descendant reports all stale ancestors, including multiple converging paths. It does **not** establish:
- that a runner observed every real workspace change;
- that a resource digest was measured against the actual filesystem or process environment;
- that event/evidence digests came from a trusted producer;
- that a test, build, or security check was executed honestly;
- that a verifier is semantically adequate for the task;
- that a final tree or runner identity is authentic.

The `event_digest` is recomputed as SHA-256 over compact canonical JSON with the domain `basalt.resource-change.v1`, schema version, run ID, sequence, event ID, change kind, resource ID, and before/after digests. This detects payload edits but is not an issuer signature: an actor able to rewrite the timeline can also recompute the hash. The `evidence_digest` is currently format-checked but its referenced evidence bytes are not resolved here. No issuer signatures are verified in this slice. Trusted change-event delivery, signed evidence receipts, and isolated runner integration remain mandatory before confirmatory scoring; the next implementation boundary is tracked in [#9210](https://github.com/Luminous-Dynamics/symthaea/issues/9210). Confirmatory scoring therefore remains gated on authenticated receipts (#9034), isolated runner delivery, exact source/environment identity, and qualified oracles (#9115). No claim of execution or SOTA superiority follows from the example fixture.

## Research hypothesis

A dependency-aware evidence graph can preserve useful work while preventing stale evidence reuse. Compared with blanket reruns, it should reduce re-verification cost after a localized change without increasing false-fresh acceptance.

This is falsifiable. A dependency model that misses hidden inputs, incomplete events, or indirect effects can make the system less safe than simply rerunning all checks. The runner must therefore measure resource dependencies conservatively and expose unknown/untracked inputs; an unknown dependency must never be interpreted as irrelevant.

## Benchmark design

Build a separate state-continuity lane with paired clean and shifted continuations, using the same saved initial state, agent/model/scaffold, task, budget, and toolchain. Vary intervention position (early, middle, late) and change family (source, dependency lock, toolchain, task contract, configuration/environment, tool/session).

Compare four evidence policies:
1. **No invalidation:** reuse all old evidence (deliberately unsafe negative control).
2. **Global invalidation:** rerun every check after every event (safe but expensive reference policy).
3. **Content-only cache:** invalidate only when current digests differ, without generation/lineage propagation (detects the ABA reversion weakness).
4. **BASALT graph:** exact digest + monotonic generation + transitive evidence dependencies.

Include positive controls where only an irrelevant resource changes, negative controls where a required source/dependency/toolchain/contract changes, digest A→B→A reversion, stale-parent reuse, missing event, duplicated/out-of-order event, forged before-state, and fresh re-verification after a change.

Report:
- strict task completion under the current contract;
- stale-evidence false-accept rate and missed-invalidation rate;
- invalidation precision/recall against a separately constructed dependency oracle;
- unaffected-evidence retention and false invalidation rate;
- re-verification work as fraction of global-rerun work;
- cost, wall time, tool calls, and p50/p90 recovery latency;
- unresolved/untracked dependency rate and runner receipt coverage.

### Proposed gates (targets, not results or industry standards)

- 100% correct direct/transitive behavior on deterministic unit and mutation fixtures.
- Zero stale-evidence accepts across at least 300 independent negative-control cases; this is a qualification target, not proof of zero risk.
- 100% preservation on deterministic irrelevant-change controls.
- At least 30% lower re-verification work than the global-rerun policy, with no increase in stale-evidence acceptance, before claiming an efficiency advantage.
- At least 70% clean-to-shift success retention on a preregistered confirmatory cohort as a stretch target; report absolute shifted success and paired success separately.
- 100% of confirmatory timeline entries tied to independently authenticated runner receipts.

Use task/scenario clustering for confidence intervals. Do not treat repeated tests, evidence nodes, or intervention positions within a repository as independent tasks. Publish every unresolved event and untracked input; never silently drop it from the denominator.

## Industry comparison

SWE-Shift (23 October 2026) evaluates 120 scenarios across 12 change families and nine intervention positions. Its authors report lower shifted than clean success for all 17 evaluated systems; GPT-6 Astra falls from 67.35% clean to 24.79% shifted, with 20.34% paired success. These are measurements under that paper's protocol, not universal estimates or BASALT results: https://arxiv.org/abs/2610.32196

TestJack (7 October 2026) reports that 1,545 of 4,487 trials previously judged passing violated requirements (34.4%) across its five-benchmark sample, after auditing apparent passes with requirement-grounded witnesses. This is likewise study-specific evidence for strengthening evaluators, not a prevalence estimate for every benchmark: https://arxiv.org/abs/2610.10619

For static coding-agent context, keep harnesses separate:
- Official Terminal-Bench 4.0 on 30 October 2026 reports Claude Opus 5.5 via Claude Code at 64.8% ± 3.1, Sonnet 5.5 at 61.8% ± 2.9, and GPT-6 Astra via Codex at 58.2% ± 2.8: https://www.tbench.ai/?version=4.0
- Vals AI's common Mini-SWE-agent Terminal-Bench 4.0 board updated 29 October reports Opus 5.5 at 65.15%, Sonnet 5.5 at 64.14%, Kimi K3.1 Pro at 63.64%, and GPT-6 Astra at 59.60%, with per-test costs: https://vals-ai.com/benchmarks/terminal-bench-4
- Artificial Analysis Coding Agent Index v2.0 combines FrontierCode v1.1 (100 tasks), Terminal-Bench 4.0 (66 tasks), and SWE-Atlas-QnA-AA (124 tasks), with three attempts per task and equal weighting across the three lanes: https://artificialanalysis.ai/methodology/coding-agents-benchmarking

Do not compare these numbers directly with BASALT. Reproduce a public lane with identical harness, model/scaffold, budget, tool/network policy, and evaluator version; publish paired uncertainty and cost per strict verified fix. The differentiator should be measurable state continuity and evidence correctness, not a composite score that hides failure modes.

## Next implementation steps

1. Bind resource IDs to trusted Git tree/file digests, dependency lockfiles, compiler/toolchain profiles, task contract digests, test/oracle bundle digests, and environment/session receipts.
2. Add a trusted runner that emits signed, monotonic change events and evidence receipts; verify event completeness and identity independently of the agent.
3. Connect the auditor's stale set to an executable check scheduler. The scheduler must distinguish required checks, dependent checks, and checks unaffected by the change.
4. Add event omission, event-order, stale-parent, digest-reversion, and dependency-oracle mutation campaigns.
5. Compose with #8901's paired shifted-run analysis and #9115's witness-test pool. Keep the result analysis-only until runner receipt verification and the exact target workflows are qualified.
