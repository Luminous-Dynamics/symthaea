# BASALT read-only Linux capability inventory — 2026-10-31

## Purpose

BASALT's next gap is no longer only the shape of the coverage receipt. It is choosing a real observer profile based on the host's *actual available interfaces* without mistaking a configured kernel feature for an attached, working, loss-free observer.

The new `basalt-linux-capability-probe` command produces a read-only host inventory:

```sh
cargo run -p xtask --bin xtask -- basalt-linux-capability-probe \
  --out reports/basalt/linux-capability-inventory.v1.json
```

It records kernel release and architecture, active Linux Security Modules, a digest of `/sys/kernel/btf/vmlinux`, a digest of an accessible kernel config, relevant mount types, process scheduler tracepoint IDs, bpftool executable presence and binary digest, and cgroup membership. Kernel configuration is read only from well-known config files; absence or unreadability is `unknown`, not `disabled`.

The report distinguishes `candidate_available`, `not_available`, `unknown`, and `inaccessible`. It stores a digest and bounded summary of `/proc/self/mountinfo`, not unrelated host mount paths. The parser is capped at 16,384 lines; a larger inventory becomes `not_probed`, and mount-dependent capability statuses remain `unknown` rather than interpreting the truncated prefix as absence. It writes the report using create-new semantics and does not overwrite an earlier report.

## Non-goals and safety boundary

The probe does **not**:
- execute bpftool or any other discovered host tool (it records only the bpftool executable digest);
- load or attach BPF programs;
- attach or enable tracepoints;
- create cgroups, namespaces, mounts, Landlock rulesets, or fanotify groups;
- modify security policy or sysctls;
- execute a target workload;
- collect dependency events;
- claim that any observer has been qualified.

`candidate_available` is only an interface/configuration hint. For example, active `bpf` in `/sys/kernel/security/lsm` plus BTF visibility indicates a plausible BPF-LSM path, not that the desired hook set can be attached, that a BPF program passes the verifier, or that an event ring buffer is loss-free. The report hard-codes `observer_attached=false`, `workload_executed=false`, and `qualification_eligible=false`.

## Capability inventory

| Capability | Inventory basis | What still needs a separate test |
|---|---|---|
| BPF LSM | Active LSM list, BTF presence/digest, kernel config flags, bpftool executable identity/digest | Attach the exact pinned LSM hook set, validate CO-RE and verifier behavior, test permission/privilege requirements, record link identities and event loss |
| Process tracepoints | tracefs/debugfs mount and scheduler fork/exec/exit event IDs | Enable/consume under a supervised workload; validate event ordering, task identity, filtering and loss |
| fanotify notifications | `CONFIG_FANOTIFY` in readable kernel config | Create a notification group and marks, verify permissions, mount coverage, queue overflow and teardown |
| fanotify permission events | `CONFIG_FANOTIFY_ACCESS_PERMISSIONS` separately from base fanotify config | Exercise a permission event and verify exact response/timeout semantics |
| Landlock | `CONFIG_SECURITY_LANDLOCK` | Query the Landlock ABI, create a ruleset, verify handled access rights and special-object behavior |
| cgroup v2 process scope | cgroup2 mount and the probe process's `/proc/self/cgroup` membership | Create a dedicated workload cgroup, prove child inheritance, close the process tree and verify recursive populated state |

## Why the observer strategy must be composed

Linux fanotify is useful but cannot be treated as a complete dependency oracle. Its documented limitations include mmap-based access, remote activity on network filesystems, non-recursive directory marks with a race when new subdirectories are marked, and event queue overflow. This is why BASALT must not label a fanotify-only profile complete for mmap or unsupported mount/network behavior.

BPF LSM can instrument selected security hooks and can be used for system-wide MAC and audit policies. The exact attached hook set is part of the profile identity. BPF ring buffers support multi-producer event streams and cross-CPU reservation ordering, but record reservation fails when there is no space. BASALT must bind the deployed buffer configuration and require explicit loss counters; the presence of a ring-buffer map is not proof of zero lost events.

Landlock is valuable as an unprivileged access-control boundary, but it is not an activity recorder and the kernel documents limitations for some special filesystem objects reached through `/proc`. Cgroup v2 is useful for process-tree containment and lifecycle supervision, but PID lists can race with migration and PID reuse. The probe only detects its current namespace's cgroup v2 context; it does not certify a future workload supervisor.

Primary Linux references:
- BPF LSM programs: https://docs.kernel.org/bpf/prog_lsm.html
- BPF ring buffer: https://kernel.org/doc/html/latest/bpf/ringbuf.html
- Active LSM ordering and discovery: https://cdn.kernel.org/doc/html/latest/admin-guide/LSM/index.html
- fanotify limitations: https://man7.org/linux/man-pages/man7/fanotify.7.html
- Landlock limitations: https://docs.kernel.org/userspace-api/landlock.html
- cgroup v2 process and populated state: https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html

## Benchmark protocol for capability discovery

The capability inventory itself should be evaluated separately from observer qualification.

1. **Truthfulness:** compare every status against a controlled syscall/attach smoke test on the same kernel. A configured-but-unusable feature must never be reported as runtime-ready.
2. **Unknown handling:** inaccessible or missing config must remain unknown, and must not be inferred disabled or enabled.
3. **Profile prediction:** measure precision and recall for selecting a candidate adapter profile; false-positive capability selection is more costly than conservative unknown.
4. **Host drift:** compare inventory digests across kernel updates, boot configuration changes, LSM order changes, BTF changes, mount namespace changes and toolchain updates.
5. **Reproducibility:** two runs on the same boot and namespace should have matching stable identity fields, with time/probe IDs excluded from the comparison.
6. **Non-interference:** verify that the probe does not attach programs, change mounts/policies/sysctls, create cgroups or run a target workload.
7. **Cost:** report wall time, CPU time, bytes read and report size. Keep the inventory lightweight enough to run as a preflight on every candidate host.

Proposed targets (not results or external standards):
- 100% precision for any `candidate_available` finding when compared to an actual later runtime smoke test;
- 100% unknown classification for capabilities whose evidence is inaccessible, truncated or insufficient;
- zero side effects on host policy, mount, cgroup and BPF attachment state;
- p95 preflight below 250 ms on a warm host;
- deterministic stable-field report digest for repeated inventories on the same boot and namespace.

## Industry observer comparison

BASALT should not pretend to replace established kernel telemetry engines. It should use them as candidate observation adapters and benchmark the evidence they actually provide.

| Engine | Documented strength | BASALT comparison target |
|---|---|---|
| Falco Modern eBPF | CO-RE/BTF-based syscall event capture, configurable ring-buffer transport, and kernel-event rules | Compare event completeness, buffer-loss reporting, and task-to-resource identity on the same workload. Falco's event-source model currently does not correlate events across separate sources. |
| Tetragon | Process exec/exit visibility by default; custom policies can add tracepoint and kprobe observations and enforcement | Compare process lineage, file/network event coverage, task-scoped capture boundaries, and evidence needed to bind a build result to consumed inputs. |
| fanotify | Filesystem notification and permission-event interfaces | Keep as an independent filesystem lane, with explicit mmap, remote filesystem, recursion-race and queue-overflow exclusions. |
| Linux BPF LSM / tracepoints | Hooks selected kernel security operations and process lifecycle points | Pin exact hook set, kernel/BTF identity, attachment links, privilege requirements, and every loss counter. |
| BASALT composed observer (target) | Independent oracle, per-profile coverage envelope, root-class manifest, and evidence-retention adjudication | Demonstrate that independently observed inputs match the challenge oracle and that a changed dependency cannot leave stale evidence reusable. |

Falco documents its modern eBPF probe requirements (BTF and ring-buffer support), configurable buffer arrangements and kernel-event source architecture. Its event-source documentation says that event sources are processed separately and events from different sources are not correlated. Tetragon's documentation describes default process exec/exit events and custom tracepoint/kprobe policies. These are strong production baselines for telemetry, but their primary objectives are runtime security visibility and enforcement, not proving the dependency closure of an individual build/test execution.

Primary references:
- Falco kernel event source: https://falco.org/docs/concepts/event-sources/kernel/
- Falco event source correlation model: https://falco.org/docs/concepts/event-sources/
- Falco supported events: https://falco.org/docs/reference/rules/supported-events/
- Tetragon events: https://tetragon.io/docs/concepts/events/
- Tetragon use cases and tracing policies: https://tetragon.io/docs/use-cases/
- Tetragon observability metrics: https://tetragon.io/docs/reference/metrics/

**How BASALT could do better:** do not compete on alert throughput alone. Run Falco, Tetragon, fanotify, and a BASALT-specific BPF/tracepoint adapter against the same controlled task lineages and mutation oracle. Score each lane on seeded dependency recall, event-loss disclosure, stable resource identity, unsupported-channel honesty, and stale-evidence false acceptance. Then compose lanes without majority voting: one credible conflict or a missing required source downgrades coverage. A 100% event count is not enough if the events cannot be bound to the exact task, root snapshot and resource versions.

## Capability-inventory benchmark protocol

The machine-readable protocol is `benchmarks/basalt/linux-capability-inventory-benchmark.v1.json`, with strict schema `benchmarks/basalt/schema/linux-capability-inventory-benchmark.v1.schema.json`. It defines five comparison arms (config-only, sysfs/procfs/mount inventory, bpftool feature probe, runtime smoke tests, and BASALT preflight), eight metrics and five mandatory safety gates.

Targets include:
- at least 99% capability-status accuracy against independently established host support;
- zero false promotions when supporting evidence is unknown or inaccessible;
- 100% report/schema conformance and host-identity binding;
- zero host-side effects from the preflight;
- at least 99.9% reproducibility for stable report fields on the same boot and mount namespace;
- p95 preflight latency no more than 250 ms and p95 report size no more than 64 KiB.

These are proposed targets, not measured results or industry standards. The Linux inventory module currently has 17 authored Rust tests, including a host preflight test and a truncation downgrade test; none has been compiled or executed here. Hosts, kernel boots and mount namespaces are the independent units; repeated reads from one host do not inflate sample size. Candidate availability remains weaker than successful runtime attach or capture.

## Next step after this inventory

The next implementation should add **explicit, opt-in runtime smoke tests** as a separate command, not silently extend this read-only command. Each smoke test must have its own privilege declaration, resource limits, teardown verification, syscall/attach evidence and failure cleanup. The smoke-test report must still be distinct from a capture receipt: a feature can pass a simple attach/permission test and still fail to cover every relevant event under workload pressure.

Only then should BASALT select a Linux observation profile, start a supervised workload, collect real event records, reconcile independent lanes and issue a coverage envelope. Interface discovery is a precondition, not qualification.
