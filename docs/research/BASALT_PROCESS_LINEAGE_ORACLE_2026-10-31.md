# BASALT process-lineage oracle audit — 2026-10-31

## Purpose

This slice makes process-lifecycle evidence independently adjudicable before an observer is allowed to claim that it saw a closed task process tree. It is deliberately narrower than the full dependency-observation profile: process lineage is one channel, not proof of filesystem, environment, toolchain, cache or network coverage.

The audit consumes two byte-bounded JSON inputs:
1. an oracle process graph and event ledger produced by a separately controlled supervisor;
2. the observer's raw process/event receipt, treated as untrusted.

It reconciles host boot identity, PID namespace, run/task/attempt, root process, process-instance identities, parent edges, lifecycle event occurrences, capture windows, per-source sequence numbers and health/loss counters. Output authenticity and qualification remain hard false.

## Correctness refinement

Linux permits multiple successful `execve` operations during one process lifetime. An event key of `(kind, process)` is therefore not unique for `exec`. The auditor now compares **event multiplicities**, permits multiple exec occurrences, and still requires exactly one birth event and (when a closed tree is claimed) exactly one exit event per process instance.

Two regression controls accompany this change:
- a positive case with multiple exec events for one process instance;
- a negative case in which the oracle contains a repeated exec occurrence that the observer receipt omits.

The event sequence is also converted with checked/saturating integer conversions before it is compared with the receipt's 64-bit gap counter.

## Identity and lifecycle contract

The v1 identity tuple is scoped by:
- host boot identity digest;
- PID namespace inode;
- run, task and attempt identity;
- PID plus `/proc/<pid>/stat` start-time ticks.

PID alone is not a process identity. PID reuse with a different start-time value is represented as a distinct process instance. Ambiguous duplicate identity tuples, orphan parents, cycles, incomplete event streams, sequence gaps, reported loss, restarts, unmonitored processes and capture-window mismatches fail closed.

The independent supervisor should retain pidfds for processes while they are alive and disclose whether `CLONE_PIDFD` or `pidfd_open` was used. A pidfd pins a reference to a task and can signal task termination through polling; it does not discover arbitrary descendants or observe exec events by itself. Descendant discovery still needs a separately qualified lifecycle source or supervisor mechanism. The fallback `/proc` start-time identity path must be stratified and disclosed, not silently treated as equivalent to pidfd supervision.

## Explicit v1 boundary: processes are not threads

Linux tasks and process/thread groups are not interchangeable. A `clone` operation can create a thread using `CLONE_THREAD`, not a new process. The current v1 receipt does not carry the complete `tid`/`tgid`/clone-flags contract needed to adjudicate thread-group transitions and exec de-threading. Therefore:
- v1 does **not** claim full task/thread-tree coverage;
- an adapter must not normalize a `CLONE_THREAD` task as a distinct process child;
- thread creation or group transitions outside the declared process-only profile must be recorded as unresolved/excluded and prevent a broader process-tree completeness claim;
- a future task-lineage version must model thread-group identity and clone flags explicitly, including the effects of exec in a multithreaded process.

This is a qualification boundary, not a reason to weaken the auditor's process-only controls.

## Paradigm shift: three independent planes

BASALT should join three different forms of evidence without letting one attest for itself.

1. **Constrain:** use a pinned execution profile to constrain task scope, inherited descriptors, namespaces, network and allowed external inputs. The policy identity is part of the run evidence.
2. **Observe:** collect process lifecycle using a sensor with explicit hook/source identity, per-source sequence counters and monotonic loss/overflow/restart counters. Missing or unsupported sources remain unresolved.
3. **Adjudicate:** compare observer output against a separate supervisor oracle and independently injected process-tree controls. The observer may not author the oracle or mark its own result qualified.

Disagreement is not settled by majority vote. Any unexplained discrepancy blocks the completeness claim.

## Industry comparison

The benchmark defines matched comparison arms for PID-only identity (unsafe negative control), procfs start-time identity, pidfd-supervised oracle, Falco kernel event stream, Tetragon process lifecycle events, unreconciled receipts, and BASALT's oracle reconciliation.

Relevant existing systems already provide substantial process telemetry:
- Tetragon documents default `process_exec` and `process_exit` events, including process start metadata and a unique `exec_id` for correlating recorded activity.
- Falco exposes kernel-side event and drop counters; its current metrics documentation includes event-drop and auxiliary-map drop telemetry. A fair comparison must preserve these counters rather than compare only serialized events.

BASALT should not claim to replace either telemetry engine. Its candidate contribution is a separate adjudication layer that checks exact process-instance identity, parent graph, event multiplicity, sequence integrity, event loss and process-tree closure against an independent oracle.

The broader dependency-closure benchmark remains separate. It compares the same build fixtures against Bazel strict action sandboxing, Nix sandboxed derivations, an actual SLSA Build L4 trusted platform, sccache and independent reproducible builds. SLSA Build L4's defining requirement is complete `resolvedDependencies` enforced by the trusted build platform; a BASALT receipt or an attestation file alone does not confer that level.

## Proposed gates — targets, not results

The process-lineage protocol keeps safety and efficiency separate:
- exact process-instance recall: 100%;
- parent-edge precision: 100%;
- PID-reuse separation: 100%;
- lifecycle-event occurrence recall: 100%;
- injected sequence-gap/duplicate detection: 100%;
- false acceptance under event loss, closed-tree gaps, namespace joins or ambiguous identity: zero;
- p95 audit latency: at most 200 ms for the defined 10,000-process / 100,000-event upper-bound receipt;
- p95 report size: at most 1 MiB.

The statistical unit is an independent task-attempt lineage, clustered by repository/task lineage and host boot/PID namespace. Events, syscalls, retries and repeated starts within one attempt are not independent samples. Development, qualification and untouched confirmatory lineages must be separate. These thresholds are proposed research gates, not external standards or achieved results.

## Current trust boundary and next work

This module is an **offline auditor**, not an event collector or independent supervisor. The schemas define the input/output shapes; they do not establish who authored the oracle, whether kernel event delivery was complete, or whether the receipt signer is trusted. A matching report means only that the supplied receipt reconciles with the supplied oracle claims.

Next implementation:
1. build a separate supervisor fixture runner with controlled fork/clone/vfork/exec/exit scenarios and retained pidfds where available;
2. capture an independent lifecycle stream and explicitly account for `CLONE_THREAD`;
3. compare the supervisor, kernel tracepoint/BPF stream and Falco/Tetragon arms on identical task snapshots, event rates and mutation schedules;
4. inject event loss, sequence gaps, PID reuse, namespace boundaries, observer restarts, child escape and capture-window races;
5. hold out whole task-lineage templates and kernel/namespace strata;
6. qualify the exact observer/profile only after source-level and runtime challenge results, independent oracle integrity and authenticity are established.

This PR does not claim Rust compilation, test execution, live process observation, comparative benchmark results or observer qualification.

## References

- Linux `pidfd_open(2)`: https://www.man7.org/linux/man-pages/man2/pidfd_open.2.html
- Linux `ptrace(2)` lifecycle events: https://man7.org/linux/man-pages/man2/ptrace.2.html
- Tetragon process lifecycle: https://tetragon.io/docs/use-cases/process-lifecycle/
- Tetragon process execution and `exec_id`: https://tetragon.io/docs/use-cases/process-lifecycle/process-execution/
- Falco metrics and drop counters: https://falco.org/docs/concepts/metrics/
- Falco event-drop diagnostics: https://falco.org/docs/troubleshooting/dropping/
- SLSA Build track basics: https://slsa.dev/spec/build-track-basics
- SLSA build provenance and `resolvedDependencies`: https://slsa.dev/spec/build-provenance


## Semantic event identity extension (2026-10-31)

A process event is not fully described by its event kind and process instance alone. Repeated `exec` occurrences may refer to different executable images, and `clone` events can represent different task semantics depending on the exact flags.

The oracle and observer event contracts now require:
- `exec_ordinal`: a 1-based, contiguous per-process execution occurrence number; zero for non-exec events;
- `exec_image_digest`: the SHA-256 digest of the executable image resolved against the pinned immutable snapshot; empty for non-exec events;
- `clone_flags`: the exact observed clone flag word for clone events; zero for other event kinds.

The auditor's semantic event key now includes those fields. This means an observer cannot satisfy the event-set check merely by reporting the right number of exec/clone events while substituting a different executable image or clone behavior. Exec ordinals must be unique and contiguous within each process instance. Exact duplicate semantic keys are rejected.

`CLONE_THREAD` (0x00010000) is explicitly rejected by the process-only v1 profile. Thread/task lineage requires a future contract with `tid`, `tgid`, clone flags and exec de-threading semantics. Unsupported thread behavior must remain unresolved, not be relabelled as a process child.

Executable image digests are a strong requirement. A sensor that cannot bind the executed image to exact bytes in the pinned snapshot must report this channel unsupported/unresolved. A pathname-only match is insufficient. For scripts, interpreter resolution and the actual executable image must be defined by the adapter; for memfd or other non-snapshot executables, the adapter must provide an independently bound image digest or remain unresolved.

The benchmark adds three non-compensatory safety metrics: executable-image identity recall, exec-occurrence order integrity, and clone-flag semantic integrity. New negative controls inject a wrong image digest, reordered/duplicate exec ordinals, altered clone flags and attempted `CLONE_THREAD` normalization.

These additions strengthen semantic reconciliation but do not authenticate the oracle or observer. The digest still binds bytes; it does not prove who authored the observation or that the underlying event sensor was complete.


### Semantic-field trust boundary

The auditor itself does not open executable files, recompute `exec_image_digest`, inspect the kernel's `bprm->file`, or prove that a supplied `clone_flags` word came from the actual syscall. It validates field shape, per-process ordinal invariants and exact semantic equality between the supplied oracle and receipt.

The independent supervisor/oracle work must bind executable digests to the actual executed object through a stable file handle or an immutable snapshot, record interpreter resolution for scripts, and explicitly handle memfd/anonymous executables. If the image bytes cannot be bound, the source must be unresolved; a path-only hash made after the fact is not a substitute. The clone-flags field likewise must come from a source that captures the actual syscall flags, not a workload-provided assertion.

Therefore a passing semantic event comparison is only a reconciliation result over supplied claims. It does not authenticate either producer, prove the executable digest's origin, or qualify a kernel observer. The #9519 supervisor fixture is responsible for establishing those independent bindings.
