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

## Current execution status

This commit authors the schema, 33 challenge definitions and the qualification protocol only. The corpus has not been executed; no observer is qualified; no Rust build or test result is claimed. The next runtime milestone belongs to the Linux runner implementation and independently instrumented challenge harness tracked by #9210/#9239.
