# MACE Oracle Non-Interference Laboratory

**Status:** deterministic reference harness and synthetic tests; not production oracle certification.

This package tests whether a benchmark evaluator changes the environment whose state it is grading. It composes with the MACE evaluator-qualification contract and the existing independent-judge/oracle work. It does not create a second oracle manifest, evidence store, signature format or receipt authority.

## What is measured

The harness keeps two results separate:

- The benchmark grade: `pass`, `fail` or `indeterminate`.
- Evaluator non-interference: `pass`, `fail` or `indeterminate`.

A grader can return a task result while failing non-interference. Conversely, a grader can be non-interfering but return an indeterminate task grade. Neither result substitutes for the other.

## Invocation protocol

For a caller-pinned state scope, the harness:

1. Requires a complete observer coverage declaration for the exact expected scope. Any declared uninstrumented path or mismatched scope blocks the grade invocation.
2. Captures the pre-grade state digest and revision plus the independent observer's event cursor.
3. Invokes the evaluator with a read-only `StateReader` interface.
4. Captures the post-grade state digest/revision, event cursor and coverage declaration.
5. Requires a complete, contiguous event sequence attributed to the exact invocation.
6. Rejects any persistent state/revision change or any observed mutation event. Missing events or incomplete coverage remain indeterminate.

A caught grader error or panic produces an indeterminate grade. It does not become a task pass. If that failed invocation also mutated state, non-interference fails independently.

## Higher-order evaluator checks

- **Repeatability:** restore one exact pristine state before each repetition, reuse the same evaluator instance, and compare verdict and output digest. This can expose stateful caches, randomness, time dependence or order-sensitive internal state.
- **Order invariance:** evaluate two tasks as A→B and B→A, each from the same pristine environment and a fresh evaluator instance. The per-task outputs must match across permutations.

The reset operation must restore the exact state and start a fresh event-monitor window. A production adapter should normally use fresh disposable environment clones/snapshots and a separately scoped observer, not a mutable in-process map.

## Critical limitations

A final-state snapshot alone cannot detect a transient mutation that is later restored. The harness therefore also requires the mutation-event stream to cover all relevant paths. If that stream is incomplete, has gaps, changes scope or cannot attribute events to the invocation, the result is indeterminate.

However, a malicious or defective observer that falsely claims complete coverage can still hide events. A complete declaration is a pinned trust assumption to be independently qualified; this library cannot prove it. The simulator's own event log is not independent production evidence.

This package does not:
- sandbox or isolate an evaluator process;
- inspect a real database transaction log, filesystem, process tree, network or remote service;
- prove that every external mutation path is instrumented;
- prove oracle labels are correct or independent from task inputs;
- authenticate observer producers or qualify a signing key;
- replace #11772's human/LLM judge calibration or #10826's ground-truth provenance/common-mode analysis.

The next production step is a non-destructive adapter whose state snapshot comes from the system under test and whose event recorder is outside the grader's trust domain. The grader should run on a disposable clone; snapshots and independent transaction/event evidence should be retained before cleanup.

## Local checks

The package uses only the Go standard library. The local reference tests cover read-only evaluation, persistent mutation, transient write/restore, missing observer events, incomplete/uninstrumented coverage, scope and attribution mismatch, grader errors and panics, repeatability, order invariance and state-revision changes.

Local results are recorded in the parent PR's `RESULTS.md`. All fixtures are synthetic.


## Linux filesystem observer adapter (prototype)

On Linux, `fsobserve_linux.go` provides `ObserveDirectory`, a real local-kernel observation adapter for an existing, symlink-free regular directory tree.

It:
- Installs recursive inotify directory watches before the evaluator callback.
- Pins the root device/inode and resolves nested paths against `/proc/self/mountinfo` mount IDs, rejecting different or unknown mount boundaries.
- Refuses to run when the initial tree contains symlinks, special files, unwatchable paths or unsupported mount boundaries.
- Decodes the Linux 16-byte inotify wire header explicitly (rather than casting to Go's larger `syscall.InotifyEvent` struct).
- Fails on queue overflow, watch invalidation, unknown descriptors, new subdirectories, root replacement and event-drain errors.
- Captures a deterministic before/after tree digest as a second signal. Mutation events still reject transient writes whose final bytes are restored.

The local test set exercises read-only evaluation, persistent writes, transient write/restore, transient create/delete, nested paths, symlink rejection, dynamic-directory coverage loss, callback panic, root replacement, mount-ID path selection and actual `/proc/self/mountinfo` parsing.

**Trust boundary:** Linux inotify is a host-kernel event source, not an isolated or remote witness. The callback must wait for every child process/writer to finish. The adapter excludes symlinks, special files and newly created subdirectories; it does not independently measure remote service state, ACL/xattr contents or privileged activity outside the watched tree. Mount identity depends on the current mount namespace and `/proc/self/mountinfo`. A dishonest or compromised kernel/privileged host can still hide activity. Therefore a local `pass` is limited to this declared filesystem scope and is not proof of production observer completeness.

The in-memory oracle harness and this Linux filesystem adapter are separate experiments. The latter does not yet integrate the evaluator with a disposable service, real database transaction log or durable broker.
