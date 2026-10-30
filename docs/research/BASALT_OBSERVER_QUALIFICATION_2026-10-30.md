# BASALT observer qualification corpus v1

**Status:** challenge catalog authored; profile not qualified; all cases are not executed.  
**Companion issue:** [#9239 — qualify dependency observation completeness](https://github.com/Luminous-Dynamics/symthaea/issues/9239)  
**Composition:** [#9210 — trusted state runner](https://github.com/Luminous-Dynamics/symthaea/issues/9210), [#9203 — evidence-state timeline auditor](https://github.com/Luminous-Dynamics/symthaea/pull/9203).

## Why this artifact exists

The timeline auditor can correctly invalidate evidence only for dependencies represented in its input history. It cannot prove that the producer observed every input that influenced a build, test, lint, or verifier. The runner can sign a receipt, but signing does not prove observational completeness.

This catalog turns the research question into a concrete qualification target: inject a known input or observation failure, run the candidate observer, independently establish the expected relation, then compare the observer receipt against that oracle. The catalog is adapter-neutral. It does not claim that Linux tracing, Cargo metadata, a sandbox, or any existing runner has passed these challenges.

The profile identifier `linux-hermetic-candidate-v1` is a candidate label only. It is deliberately not called supported or qualified. The catalog's top-level `profile_status` and `execution_status` are fixed to `candidate_not_qualified` and `not_executed` so a fixture cannot be mistaken for benchmark results.

## Paradigm shift: qualify a bounded envelope, not a global boolean

Avoid an unqualified `complete=true`. A future receipt should make a scoped statement such as:

- exact observer binary and version digest;
- exact kernel/runtime and privilege profile;
- exact monitored roots and admitted immutable toolchain closure;
- exact process supervision boundary;
- explicitly covered channels and their tested operations;
- exclusions and unresolved channels;
- event-loss counters, restarts, dropped notifications and buffer overflow;
- snapshot-consistency result and capture interval;
- independent challenge-corpus qualification identity and expiry/revocation policy.

A valid receipt can claim completeness **only within an identified profile whose limitations are explicit and whose qualification is current**. If any input crosses the boundary, the result is unresolved or the receipt is invalid. A signature authenticates the receipt issuer and bytes; qualification evidence supports (but does not mathematically prove) the bounded coverage claim.

## Challenge disposition contract

| Disposition | Required behavior |
|---|---|
| `capture_dependency` | Record stable resource identity plus the relevant content, metadata, namespace, execution or artifact commitment. Missing observation is a false negative. |
| `report_unresolved` | Explicitly identify the channel/resource class as unresolved, or deny access before the evidence-producing computation can consume it. It must never be silently omitted. |
| `invalidate_receipt` | Mark the run/evidence receipt invalid or incomplete because observation integrity or snapshot consistency was lost. Do not issue a complete coverage claim. |
| `preserve_unrelated` | Demonstrate, with an independent positive control, that a changed resource is not an input to this evidence and preserve the unrelated evidence. |

The corpus deliberately includes both hidden dependencies and irrelevant mutations. An observer that records nothing fails recall. An observer that declares the whole workspace a dependency may be safe from missed inputs but fails selectivity and may cost as much as global reruns. Both dimensions must be measured.

## Oracle independence

The observer under test must not generate its own expected answers. For each challenge:

1. The fixture harness injects a known input, mutation, or observer failure.
2. A separate oracle records the injected resource identity and mutation and verifies that paired continuations differ or remain equivalent as specified.
3. The subject observer emits its raw event/coverage receipt.
4. A separate adjudicator compares expected disposition and resource identity with that receipt.
5. A replay lane re-runs the check from an immutable snapshot or an independent observation path.

For positive dependency controls, prove influence by changing only the seeded input and checking a deterministic output/witness change. For `preserve_unrelated` controls, establish non-dependence by a controlled semantic equivalence check and ensure the fixture does not enumerate or otherwise consume the mutated namespace. A mere absence of a trace event is not proof that a resource was irrelevant.

Use at least two independent observation/replay methods during qualification. Examples include tool-native dependency output, kernel/file-access tracing, sandbox access logs, immutable snapshot diffing, and deterministic replay. These methods have different blind spots; disagreement must reduce the claimed coverage, not be voted away. The challenge oracle must remain separate from the observer's own instrumentation.

## Scenario families covered

The 33 cases cover:

- filesystem bytes, metadata-only reads, directory enumeration, symlink retargeting, mmap, namespace changes, ignored/untracked files and root escapes;
- generated source, build scripts, procedural macros and host-side inputs;
- environment values, child process lineage, inherited file descriptors, invocation metadata and dynamic libraries;
- clocks/randomness, pseudo-filesystems, toolchain/sysroot, Cargo/Nix dependency resolution and opaque cache hits;
- DNS, captured HTTP response artifacts and mutable registries;
- observer overflow/restart, unmonitored child processes, concurrent mutation during hashing and symlink escape;
- irrelevant docs/source mutations and unsupported-channel disclosure.

The machine-readable catalog is `benchmarks/basalt/coverage/observer-challenges.v1.json`; its strict Draft 2020-12 schema is `benchmarks/basalt/schema/observer-challenges.schema.json`.

## Linux mechanism selection: containment is not observation

The first Linux prototype should not equate any single filesystem API with a complete dependency observer.

- **Landlock is a containment layer, not a dependency trace.** It can restrict filesystem hierarchies and, depending on the kernel ABI, network ports. The kernel documentation describes ABI-dependent capability detection and notes limits for special filesystems accessed through paths such as `/proc/<pid>/fd/*` and `/proc/<pid>/ns/*`; a sandbox policy also does not retroactively revoke every already-open descriptor. Record the exact Landlock ABI and handled rights, close or explicitly account for inherited descriptors before confinement, and classify unsupported controls as unresolved. See the [Linux kernel Landlock documentation](https://docs.kernel.org/userspace-api/landlock.html).
- **fanotify is a useful supplemental observer, not a sole completeness oracle.** Its documented limitations include not reporting accesses/modifications caused by `mmap`, `msync` or `munmap`; child additions/removals/changes do not themselves emit events on a marked directory; directory monitoring is not automatically recursive; dynamically adding marks to newly created subdirectories can race; and its event queue can overflow. Mount-level monitoring can remove some recursive-marking gaps, but it does not solve the mmap limitation. See [fanotify(7), Limitations and caveats](https://man7.org/linux/man-pages/man7/fanotify.7.html).
- **A syscall/process observer and independent snapshot/replay lane are still required.** Qualify the exact mechanisms for descriptor provenance, `exec`/child lineage, metadata and directory enumeration, `mmap`, dynamic library loads, process exit, observer restart and event loss. Treat untested interfaces and special filesystems as unresolved. A second observer or replay is a cross-check, not a magical oracle; common-mode failures must be included in the challenge design.

### Candidate composition

1. Materialize the initial workspace as an immutable, content-addressed read-only snapshot. Provide a separate writable output/artifact area whose final contents are committed to the run receipt.
2. Pin the toolchain and build closure; admit only explicit immutable roots. Record the exact kernel, Landlock ABI, handled filesystem/network rights, mount topology and confinement profile.
3. Close inherited descriptors or register each descriptor's immutable provenance before launching the supervised process tree. Do not assume a path-based rule revokes access already obtained through a file descriptor.
4. Deny network during scored verification where practical. If networking is needed, capture immutable response artifacts and bind their bytes and endpoint/protocol provenance; unsupported DNS/UDP/external state stays unresolved.
5. Combine a process/syscall-level event stream with tool-native dependency metadata and snapshot-based independent replay. Use fanotify as a complementary filesystem signal only after the exact kernel/filesystem configuration is qualified.
6. Maintain explicit dropped-event, buffer-overflow, restart, unmonitored-child and snapshot-race counters. Any unknown gap forces unresolved or invalid receipt status.
7. Run every mechanism against its specific catalog controls. A green sandbox self-test does not establish dependency capture recall.

This design deliberately uses **prevention to shrink the observable state space** and independent observation/replay to test what remains. It is more defensible than attempting to infer all dependencies from file-open notifications after an unconstrained process has already run.

## Scoring and statistical discipline

Primary metrics:

- **Scenario-level dependency recall:** fraction of independently injected relevant inputs represented in the observer's dependency/receipt set, or safely denied/unresolved before consumption.
- **Stale-evidence false acceptance:** whether the old evidence is accepted after a relevant hidden dependency is changed.
- **Observer-loss detection:** fraction of overflow, restart, unmonitored-process and snapshot-race controls that are rejected or explicitly unresolved.
- **Unresolved-channel correctness:** fraction of unsupported channels identified rather than silently treated as covered.
- **False attribution / false invalidation:** irrelevant changes incorrectly treated as dependencies or causes of invalidation.
- **Selectivity:** unrelated evidence retained under proven-irrelevant mutations.
- **Overhead:** CPU, wall time, I/O bytes, event volume, storage and p50/p95 impact.
- **Operational efficiency:** checks rerun, total work, cost and recovery latency against global rerun.

Report recall and false invalidation separately. Do not collapse them into a single score where excellent precision can compensate for a missed dependency, or high recall can hide global invalidation. Confirmatory safety is non-compensatory: a stale-evidence false accept blocks promotion regardless of speed or average score.

### What does zero misses establish?

Zero observed misses is not a zero-risk guarantee. Under an idealized independent Bernoulli model, zero failures in 300 independent scenarios gives an approximate one-sided 95% upper bound near 1% on the miss probability (rule-of-three approximation). To support a miss-rate bound near 0.1% with 95% one-sided confidence and zero misses requires approximately 2,995 independent scenarios. These calculations do not apply to repeated events nested inside the same task, nor do they eliminate common-mode oracle failure. Use task/scenario-level units, stratify by observation channel, report exact binomial intervals, and retain unresolved scenarios in the denominator.

Recommended staged gates (targets, not achieved results or external standards):

1. **Prototype smoke gate:** all catalog cases produce their expected disposition in deterministic fixtures; zero silent omissions; every integrity-failure control is fail-closed.
2. **Profile qualification gate:** no misses in a preregistered, independently reviewed corpus of at least 2,995 independent scenario units if claiming a 0.1% miss-rate upper bound at 95% confidence; report channel-stratified intervals and oracle limitations.
3. **Selectivity gate:** demonstrate preservation on all independently verified unrelated-change controls; publish false attribution and invalidation rates with uncertainty.
4. **Efficiency gate:** demonstrate at least 30% less re-verification work than global rerun only after the safety gates pass, with no stale-evidence false accept and a paired task-level interval.
5. **Promotion gate:** pin observer binary, kernel/profile, challenge corpus, oracle, runner, receipt format and expiry policy. Any material change to these inputs requires requalification.

The 33 authored cases are a starting conformance corpus, not enough independent samples to establish a 0.1% miss rate.

## Required comparisons

Evaluate, on identical challenge injections and task snapshots:

1. static declared manifest only;
2. compiler/build-system native dependency metadata;
3. syscall/file-access tracing without a hermetic boundary;
4. hermetic sandbox with pinned immutable input closure;
5. composed observer plus independent snapshot/replay audit.

Include a global-rerun reference and a no-invalidation negative control. The no-invalidation arm is an intentionally unsafe lower-cost comparison and must never be used for confirmatory result retention.

## Implementation sequence

1. Implement a runner-independent fixture API that can seed each resource, perform a controlled read/mutation, and emit an oracle record.
2. Implement one Linux hermetic candidate profile: read-only snapshot, explicit writable output area, pinned toolchain closure, network denial during scored verification, full process-tree supervision, and a defined filesystem observation method.
3. Run the corpus against at least two independent observation paths; record raw traces and loss counters.
4. Convert each observer result into the scoped coverage receipt; unsupported or ambiguous behavior stays unresolved.
5. Feed only qualified, authenticated receipts into #9210's runner and #9203's timeline auditor.
6. Add paired clean/shifted repository tasks and cost accounting through BASALT's existing evaluation contracts.

No result is confirmatory solely because this catalog parses, a runner exits zero, a signature verifies, or an observer reports `complete`.

## Industry comparison and the BASALT differentiation

The relevant industry benchmarks now measure several adjacent dimensions, but no single score should be treated as a dependency-observation qualification result.

- **[SWE-Shift (23 October 2026)](https://arxiv.org/abs/2610.32196)** is the closest adjacent benchmark. It evaluates 120 scenarios across 12 change families and nine intervention positions. All 17 evaluated systems had lower shifted than clean success; its reported GPT-6 Astra condition moved from 67.35% clean to 24.79% shifted, with 20.34% paired success. It explicitly includes stale or misbound evidence and tool/session changes. BASALT should reuse compatible paired-continuation ideas, then add independently known hidden inputs, observer loss, dependency recall, false invalidation and replay qualification. These paper results are external context, not a BASALT baseline score.
- **[Scale AI SWE-Bench Pro V2](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2?tab=full)** (22 September 2026) refreshed its public set to 642 tasks over 11 repositories after removing 89 tasks judged invalid and correcting 69 instruction/test contradictions. It re-grades diffs on pristine images and reports catching checksum manipulation and module-cache edits; it also describes residual trust boundaries around its model relay and patch-controlled code executed by the verifier. BASALT should compose with these strong benchmark-integrity practices, while independently qualifying whether the runner observed all dependencies that influenced a check.
- **[Artificial Analysis Coding Agent Index v2.0](https://artificialanalysis.ai/methodology/coding-agents-benchmarking)** equal-weights 100 FrontierCode repository-engineering tasks, 66 Terminal-Bench tasks and 124 repository-Q&A tasks. It uses three attempts per task and reports cost, token usage, execution time and trajectory-based reward-hacking review. This is a useful broad capability/efficiency comparison, but its aggregate score is not a coverage-completeness score.

### What BASALT should claim only after a matched experiment

Treat BASALT as a new **evidence-lifecycle and change-resilience axis**, not as a replacement leaderboard. Run the same agent, task cohort, saved checkpoint, tool policy, budget, observer/verifier bundle and intervention schedule across these arms:

1. **No invalidation:** deliberately unsafe lower-cost control; never eligible for confirmatory evidence retention.
2. **Global rerun:** conservative safety and cost reference.
3. **Manifest/build-system metadata:** practical low-cost selective baseline.
4. **Kernel tracing without hermetic isolation:** measures what tracing alone misses.
5. **Hermetic candidate observer:** read-only workspace snapshot, pinned toolchain closure, explicit process supervision and denied network during scored verification.
6. **Composed observer + independent replay:** candidate production path, promoted only after qualification.

For every arm, publish the full task-level transition matrix, not only a pass rate: clean success, shifted success, paired success, stale evidence accepted, stale evidence rejected, unaffected evidence retained, unresolved, and run invalidated. Also report per-family recall/false-attribution intervals, verifier integrity, checks rerun, CPU/I/O/storage overhead, p50/p95 recovery latency and cost per independently verified fix.

Proposed stretch targets—not measured results or external standards:

- zero stale-evidence false accepts in the qualified challenge and held-out mutation corpus;
- 100% detection or explicit unresolved/fail-closed disposition for supported-profile seeded hidden-input and observer-loss controls;
- at least 30% less re-verification work than global rerun, conditional on all safety gates passing;
- at least +3 percentage points in paired shifted-task success against a strong matched baseline, or a statistically supported non-inferior success rate with materially lower cost;
- no more than 5% false invalidation on independently proven unrelated-change controls, with channel-specific intervals and no silent unresolved outcomes.

The 33-case catalog is conformance coverage, not a sufficiently large efficacy study. Split by repository/task lineage, keep an untouched confirmatory holdout, preregister intervention positions and stopping rules, and report exact paired intervals. Never compare a BASALT percentage directly to a SWE-Shift, SWE-Bench Pro or Artificial Analysis score unless the task, harness, retry policy, budget and evaluator are matched.

## Current execution status

This commit authors the schema, 33 challenge definitions and the qualification protocol only. The corpus has not been executed; no observer is qualified; no Rust build or test result is claimed. The next runtime milestone belongs to the Linux runner implementation and independently instrumented challenge harness tracked by #9210/#9239.
