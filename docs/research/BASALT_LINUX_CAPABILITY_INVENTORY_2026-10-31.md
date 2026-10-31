# BASALT read-only Linux capability inventory — 2026-10-31

## Purpose

BASALT's next gap is no longer only the shape of the coverage receipt. It is choosing a real observer profile based on the host's *actual available interfaces* without mistaking a configured kernel feature for an attached, working, loss-free observer.

The new `basalt-linux-capability-probe` command produces a read-only host inventory:

```sh
cargo run -p xtask --bin xtask -- basalt-linux-capability-probe \
  --out reports/basalt/linux-capability-inventory.v1.json
```

It records kernel release and architecture, active Linux Security Modules, a digest of `/sys/kernel/btf/vmlinux`, a digest of an accessible kernel config, relevant mount types, process scheduler tracepoint IDs, bpftool binary/version identity, and cgroup membership. Kernel configuration is read only from well-known config files; absence or unreadability is `unknown`, not `disabled`.

The report distinguishes `candidate_available`, `not_available`, `unknown`, and `inaccessible`. It stores a digest and bounded summary of `/proc/self/mountinfo`, not unrelated host mount paths. It writes the report using create-new semantics and does not overwrite an earlier report.

## Non-goals and safety boundary

The probe does **not**:
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
| BPF LSM | Active LSM list, BTF presence/digest, kernel config flags, bpftool binary/version | Attach the exact pinned LSM hook set, validate CO-RE and verifier behavior, test permission/privilege requirements, record link identities and event loss |
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

1. **Truthfulness:** compare every status against a controlled syscall/attach smoke test on the same kernel. A configured-but- unusable feature must never be reported as runtime-ready.
2. **Unknown handling:** inaccessible or missing config must remain unknown, and must not be inferred disabled or enabled.
3. **Profile prediction:** measure precision and recall for selecting a candidate adapter profile; false-positive capability selection is more costly than conservative unknown.
4. **Host drift:** compare inventory digests across kernel updates, boot configuration changes, LSM order changes, BTF changes, mount namespace changes and toolchain updates.
5. **Reproducibility:** two runs on the same boot and namespace should have matching stable identity fields, with time/probe IDs excluded from the comparison.
6. **Non-interference:** verify that the probe does not attach programs, change mounts/policies/sysctls, create cgroups or run a target workload.
7. **Cost:** report wall time, CPU time, bytes read and report size. Keep the inventory lightweight enough to run as a preflight on every candidate host.

Proposed targets (not results or external standards):
- 100% precision for any `candidate_available` finding when compared to an actual later runtime smoke test;
- 100% unknown classification for capabilities whose evidence is inaccessible or insufficient;
- zero side effects on host policy, mount, cgroup and BPF attachment state;
- p95 preflight below 250 ms on a warm host, excluding slow external tool startup;
- deterministic stable-field report digest for repeated inventories on the same boot and namespace.

## Next step after this inventory

The next implementation should add **explicit, opt-in runtime smoke tests** as a separate command, not silently extend this read-only command. Each smoke test must have its own privilege declaration, resource limits, teardown verification, syscall/attach evidence and failure cleanup. The smoke-test report must still be distinct from a capture receipt: a feature can pass a simple attach/permission test and still fail to cover every relevant event under workload pressure.

Only then should BASALT select a Linux observation profile, start a supervised workload, collect real event records, reconcile independent lanes and issue a coverage envelope. Interface discovery is a precondition, not qualification.
