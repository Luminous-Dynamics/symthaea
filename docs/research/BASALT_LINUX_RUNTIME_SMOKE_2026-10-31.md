# BASALT Linux runtime syscall smoke tests — 2026-10-31

## Why this slice exists

PR #9449 inventories exposed Linux interfaces. That inventory is deliberately conservative: configuration, mount and file presence are not the same as an API being usable by this process.

This follow-on adds a separate command:

    cargo run -p xtask --bin xtask -- basalt-linux-runtime-smoke \
      --out reports/basalt/linux-runtime-smoke.v1.json

The default command performs two narrow operations:

1. Queries the Landlock ABI using the documented NULL/zero-size version query.
2. Calls fanotify_init with a notification-class, nonblocking, close-on-exec, file-handle-reporting group and immediately closes the returned descriptor.

A higher-privilege initialization check is available only when explicitly requested:

    cargo run -p xtask --bin xtask -- basalt-linux-runtime-smoke \
      --probe-permission-class \
      --out reports/basalt/linux-runtime-smoke-permission-class.v1.json

That option tests FAN_CLASS_CONTENT group initialization, which may require CAP_SYS_ADMIN. It does not generate a permission event, mark a path, or test the response protocol.

## Precise capability claims

| Probe | A pass establishes | A pass does not establish |
|---|---|---|
| Landlock ABI query | The current process can query a positive Landlock ABI version | A ruleset can be created/enforced with the rights BASALT needs; access mediation is correct |
| fanotify notification group | The kernel accepted fanotify_init flags and returned a group descriptor that was closed | Marks work, events are delivered, event queues are loss-free, mmap or remote filesystem accesses are covered |
| fanotify permission-class group (opt-in) | The kernel accepted the higher-privilege FAN_CLASS_CONTENT initialization flags and the descriptor was closed | A permission event is delivered, a response is accepted, timeout behavior is safe, or all relevant accesses are mediated |
| BPF LSM load/attach | Not attempted | Hook attachment, verifier acceptance, CO-RE relocation and loss behavior |
| process tracepoint attach | Not attempted | Event enablement, process identity, ordering or loss |
| cgroup workload scope | Not attempted | Child inheritance, process-tree closure or migration race resistance |

The Landlock query is the documented version-discovery interface. It supplies a version value when called with a null attribute, zero size and the version flag; it does not create a ruleset. Error handling preserves ENOSYS, EOPNOTSUPP, and permission denial as distinct result classes.

The fanotify tests use only group initialization. They do not call fanotify_mark, read an event, issue a permission response, or create any workload. A successful fanotify_init is evidence of group creation only. The returned descriptor is explicitly closed and the result records whether the close was confirmed.

## Safety boundary

The command:
- loads or attaches no BPF program;
- enables or attaches no tracepoint;
- adds no fanotify marks and reads no events;
- creates or enforces no Landlock ruleset;
- creates or migrates no cgroup or namespace;
- changes no host policy or sysctl;
- spawns no workload process;
- never claims event capture, observer attachment or qualification.

The output is create-new (no overwrite) and binds the observation to kernel release, architecture, a digest of the boot ID and the mount-namespace inode when readable. Each probe records its status, errno where applicable, resource-creation flag and cleanup state. A successful fanotify initialization with failed/unconfirmed close is a failed probe, not a pass.

The report schema hard-requires all observer, workload, event-capture and qualification claims to remain false. The benchmark protocol treats these as safety gates, not compensating score components.

## Paradigm shift: capability is a ladder, not a boolean

BASALT should make each layer earn its own claim:

1. **Configured** — kernel config or build options suggest a feature exists.
2. **Exposed** — relevant interface, mount or ABI entry point is visible.
3. **Operational API** — the exact syscall or query succeeds in the calling process' privilege and namespace context.
4. **Resource lifecycle verified** — the resource is explicitly closed/unwound and cleanup is confirmed.
5. **Observer attached** — exact pinned program/hook/mark set is active with declared scope.
6. **Capture-integral** — controlled task lineage is closed, event loss is accounted for, and all required resources are bound to immutable snapshots.
7. **Oracle-qualified** — held-out counterfactual mutations and independent oracles establish required recall and no stale-evidence acceptance.

These are monotone evidence levels only when the same kernel, namespace, privilege context and observer profile identity remain bound. A lower level must never be silently promoted because a different adapter passed. A Landlock ABI query cannot qualify a fanotify observer; successful fanotify group initialization cannot qualify event delivery.

## Benchmark plan and industry comparison

The machine-readable protocol is benchmarks/basalt/linux-runtime-smoke-benchmark.v1.json, with strict schema benchmarks/basalt/schema/linux-runtime-smoke.v1.schema.json.

Compare:
- kernel-config-only and the read-only BASALT inventory as conservative discovery baselines;
- direct Landlock/fanotify syscall outcomes as the independent reference;
- BASALT runtime smoke output as the audited candidate;
- Falco/Tetragon/BPF-LSM/tracepoint capture only in a later, separate observer-qualification lane.

Do not compare smoke checks to Falco or Tetragon on event throughput: they are runtime telemetry and enforcement systems, while this command does not collect events. The meaningful BASALT advantage must be demonstrated later through independent dependency oracles, exact task/resource lineage, complete event-loss accounting and resistance to stale-evidence reuse—not by claiming a narrower syscall probe is a better telemetry engine.

Proposed targets (not measured results or industry standards):
- 100% precision for non-unknown operational status against controlled syscall outcomes;
- 100% correct distinction among unsupported, disabled and permission-denied outcomes;
- 100% explicit cleanup confirmation for successfully created group descriptors;
- zero unintended host-side effects;
- zero reports claiming observer attachment, event capture, workload execution or qualification;
- 100% schema and cross-field validation;
- p95 default smoke latency no more than 150 ms and p95 report size no more than 16 KiB.

Use the host/kernel-boot/mount-namespace/privilege-context as the independent unit. Repeated probes on one boot are clustered observations, not independent samples. Include seccomp and container privilege profiles in the test strata. Inject failures into a testable syscall adapter in unit tests before relying on live hosts for denied/disabled cases.

## Next high-value work

1. Add a deterministic syscall adapter seam and failure-injection tests for every errno/cleanup path.
2. Run on a matrix of unprivileged host, root host, unprivileged container, seccomp-restricted container and Landlock-disabled kernel.
3. Add a separately pinned and allowlisted BPF attach/unload probe only after an independent teardown verifier exists.
4. Add a controlled fanotify permission-event fixture under a disposable mount or test filesystem, with bounded timeouts and exact response semantics. Do not test this against arbitrary host paths.
5. Build the full observer capture and counterfactual oracle lane. Only that lane may support a coverage envelope.

## Upstream references

- Linux kernel Landlock user API: https://docs.kernel.org/userspace-api/landlock.html
- fanotify_init(2): https://man7.org/linux/man-pages/man2/fanotify_init.2.html
- Linux eBPF syscall API and BPF_PROG_QUERY: https://docs.kernel.org/userspace-api/ebpf/syscall.html
- Falco kernel event source: https://falco.org/docs/concepts/event-sources/kernel/
- Tetragon events: https://tetragon.io/docs/concepts/events/
