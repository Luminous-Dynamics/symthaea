# BASALT event-delivery accounting and source-epoch closure — 2026-10-31

## Research thesis

A zero-drop counter is not a proof of a loss-free event stream unless the counter path has itself been challenged. BASALT should make delivery accounting a first-class, separately adjudicated artifact.

The Linux BPF ring-buffer documentation records two important constraints:
- reservation can fail in NMI context because the reservation spinlock cannot be acquired, even when the ring buffer is not full;
- ring-buffer query helper values are momentary snapshots and may be stale by the time they are consumed.

Therefore a failed `bpf_ringbuf_reserve()` must be recorded as a reservation failure with unknown cause unless the API actually identifies the cause. Ring occupancy snapshots are diagnostics, not completeness evidence.

Falco's metrics are disabled by default. Its modern eBPF metrics include auxiliary-map reentrancy and pool-full counters. `scap.n_drops` already includes those drops, and the tail-call reentrancy counter is a subset of reentrancy; summing these fields naively double-counts loss. Every industry arm must pin version/configuration and preserve the documented aggregate/subset relationships.

## Paradigm shift: every attempted event must be accounted

A sensor should allocate a monotonic sequence **before** any operation that can drop the event. Every attempted sequence must appear exactly once as:
1. published;
2. classified lost;
3. explicitly discarded because the event is required by the profile; or
4. explicitly discarded because an independently reviewed policy excludes it.

The ledger is scoped by `source_id`, `epoch_id` and `partition_id` (for example, a per-CPU producer). It records a terminal attempted-sequence high-water mark and a close fence after producers are quiesced and the userspace consumer has drained. Policy-excluded discards remain explicit, but this auditor does not evaluate the exclusion policy; any such discard blocks loss-free promotion until a separate pinned policy evaluator approves it.

The terminal high-water mark is critical. A dropped final event produces no later record, so an ordinary interior sequence-gap detector cannot see it. Only a final attempted-sequence fence can expose that tail loss.

## Loss classes

The contract distinguishes:
- ring-buffer reservation failure with unknown cause (do not assert "full" unless independently supported);
- perf-buffer lost sample;
- consumer queue overrun/backpressure;
- exporter drop;
- decoder/serialization rejection;
- intentional producer discard;
- source restart/detach/reattach;
- unmonitored producer;
- absent or unacknowledged terminal close.

A required discard blocks loss-free status. Excluded policy suppressions remain visible as a separate count; they are never silently erased from the ledger.

## New implementation slice

This PR adds:
- `xtask/src/basalt_event_delivery.rs`: bounded Rust auditor for the ledger;
- strict source-epoch ledger and audit-report JSON schemas;
- a machine-readable seven-arm, nine-metric benchmark;
- `basalt-event-delivery-audit --ledger PATH` CLI;
- thirteen unit tests authored for clean closure, tail loss, interior gaps, overlap, counter mismatch, required/excluded discards, unclosed consumers, restart, unknown epochs and source duplication.

The auditor compares each source epoch's attempted range against event singletons and classified loss/discard intervals. It checks that intervals are non-overlapping and cover the entire attempted range, including the tail. It recomputes published/lost/discarded counts and compares them against the supplied counters. It refuses loss-free status for any unreviewed policy exclusion, even when the sequence arithmetic is complete.

## Industry comparison

- Linux BPF ringbuf: challenge reservation failures, producer discards, consumer drain and close-fence behavior.
- Linux perf buffer: use native lost-sample callbacks as a separate baseline.
- Falco modern eBPF: pin version, enable metrics explicitly, retain `scap.n_drops` and its component counters without double counting.
- Tetragon: pin lifecycle-event and exporter configuration; retain available metrics/health data.
- ptrace/wait: separate comparison source, with per-thread tracing and stop/continue overhead reported.
- intentionally lossy transport: positive control that proves the audit can detect missing sequences.

The objective is not to claim the transport is infallible. It is to make every detected and undetected-loss boundary explicit and challengeable.

## Proposed gates — not results

Safety:
- 100% sequence accounting completeness;
- 100% recall on seeded transport-loss controls;
- zero false loss-free acceptances;
- 100% reported/derived counter consistency;
- 100% tail-loss detection;
- 100% required-discard disclosure;
- 100% source-epoch close integrity.

Efficiency, only after all safety gates pass:
- p95 event delivery latency <=10 ms for the pinned workload;
- p95 workload slowdown <=5% against the no-observer paired run.

These are proposed research targets, not external standards or achieved measurements. The independent statistical unit is a host-boot/source-configuration/task-lineage run; individual events and sequence numbers inside a run are not independent samples.

## Trust boundary

The auditor validates the supplied sequence ledger, event records, classified ranges, close-fence digest format and derived counter relationships. It does not evaluate the content of an exclusion policy, attach a BPF program, observe kernel events, prove the counter was incremented by the kernel, authenticate the ledger producer, or qualify an observer profile.

The source supervisor must generate the terminal close only after producers stop and the consumer drain is acknowledged. The positive-control harness must prove each relevant drop path is detected before any loss-free claim is accepted.

## References

- Linux BPF ring buffer: https://docs.kernel.org/bpf/ringbuf.html
- Falco metrics and modern eBPF auxiliary-map counters: https://falco.org/docs/concepts/metrics/
- Falco dropped-event diagnostics: https://falco.org/docs/troubleshooting/dropping/
- Tetragon process lifecycle: https://tetragon.io/docs/use-cases/process-lifecycle/
- BASALT process-lineage auditor: https://github.com/Luminous-Dynamics/symthaea/pull/9518
- BASALT semantic process-event auditor: https://github.com/Luminous-Dynamics/symthaea/pull/9540
