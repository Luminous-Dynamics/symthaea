# Fail-closed Linux candidate sandbox supervisor v0.1

Tracking: [#9865](https://github.com/Luminous-Dynamics/symthaea/issues/9865)  
Parent: [bounded candidate protocol PR #9863](https://github.com/Luminous-Dynamics/symthaea/pull/9863)

## Decision

A candidate policy trait and a bounded wire codec are not security boundaries. Confirmatory benchmark candidates must execute in a separate process whose filesystem, namespaces, capabilities, syscalls and resource limits are externally enforced.

The first target is Linux/NixOS using Bubblewrap for namespaces and filesystem construction, systemd/cgroup v2 for resource enforcement, and an architecture-pinned seccomp BPF profile.

## Critical fail-closed rule

Do not use Bubblewrap `--unshare-all` as the qualification profile. Current Bubblewrap documentation describes it as expanding to try variants for user and cgroup namespaces; a try variant can skip that namespace when unavailable. Instead, explicitly require every namespace the runner depends on:

`--unshare-user --unshare-pid --unshare-net --unshare-ipc --unshare-uts --unshare-cgroup`

A missing namespace, permission, feature or controller must fail the run before the candidate executable is launched. The launcher must not remove an unsupported flag and retry.

Bubblewrap also provides separate flags for read-only bind mounts, minimal proc/dev, clearing environment, new session, dropping capabilities, seccomp filters and killing children when the wrapper/parent dies. Use only the explicitly pinned options verified on the actual runner. [Upstream Bubblewrap options](https://github.com/containers/bubblewrap/blob/main/bwrap.xml).

## Filesystem plan

The candidate should see only:
- its exact immutable executable;
- each exact runtime closure object individually bound read-only;
- minimal `/proc` and `/dev`;
- ephemeral tmpfs `/tmp` and an empty home/working directory;
- standard input/output pipes for the candidate protocol.

Do not expose `/nix/store` wholesale: it can contain unrelated artifacts and potentially evaluator material. Do not expose host `/home`, data paths, caches, credentials, SSH agents, Docker sockets or broad `/run`. Use `--clearenv` and set a minimal explicit environment.

The Nix store closure is derived from the pinned executable/runtime and independently hashed. Runtime path enumeration itself must be included in the launch receipt. The evaluator's dataset, seed schedule, hidden labels, oracle digest and private result store remain outside the candidate mount/descriptor/environment/IPC namespaces.

## Namespace and syscall plan

Required namespaces: user, PID, network, IPC, UTS and cgroup. Do not share the host network. Start a new session, drop all capabilities, and make the candidate die with the supervisor.

Load a versioned seccomp BPF filter through Bubblewrap's documented FD interface. Build and test profiles per supported architecture and runtime. Minimum deny surfaces include ptrace, mount, setns, unshare, bpf, perf_event_open, open_by_handle_at, userfaultfd and socket/network syscalls. A generic filter can break thread/runtime behavior or leave escape surfaces; the final profile requires kernel and application adversarial testing. It is not sufficient to merely document a deny-list.

## Cgroup/resource plan

Use a transient **scope** and cgroup v2 for external enforcement. A scope runs the command as a child of `systemd-run` with the caller's execution environment, while the system manager controls its cgroup. Use `systemd-run --user --scope`; do not add `--pipe`, which is documented for the transient-service execution path and is not needed for scope mode. The trusted launcher must pre-open the seccomp descriptor and explicitly preserve it across its own fork/exec so Bubblewrap receives fd 3. This handoff remains an integration-test obligation, not an assumption. Scope-level controls are:
- `MemoryMax` and `MemorySwapMax=0`;
- `CPUQuota` (plus external wall-clock timeout);
- `TasksMax`;
- read back the effective systemd properties and actual cgroup control files before starting/accepting the candidate result.

Official systemd resource-control documentation maps these properties to cgroup v2 controllers and describes CPU quotas. Scope units are externally created processes and expose resource controls; service execution properties such as `ProtectSystem`, `ProtectHome` and `PrivateNetwork` are not included in this scope argv. Bubblewrap provides the namespace and filesystem isolation. Availability, delegation and controller activation are host-dependent; a requested property string is not evidence that the effective resource limit was enforced. Verify actual values and fail closed. The launcher must prove by integration test that fd 3 (the seccomp program) survives the trusted launcher's fork/exec and systemd scope launch and reaches Bubblewrap unchanged. [systemd-run](https://www.freedesktop.org/software/systemd/man/latest/systemd-run.html) and [systemd.resource-control](https://freedesktop.org/software/systemd/man/latest/systemd.resource-control.html).

The supervisor should cap stdout/stderr and frame I/O independently, kill on wall-clock deadline, malformed frame, unexpected extra bytes, resource termination or supervisor error, and record the exact reason.

## Receipt separation

1. Launch receipt: candidate digest, Nix closure digest, Bubblewrap executable/version, exact argv, seccomp profile digest/architecture, systemd properties, effective cgroup path/control values, kernel identity and supervisor digest.
2. Candidate output receipt: protocol request/action digests, bounded stderr digest, exit code, termination reason and elapsed wall time.
3. Evaluator receipt: separately bound oracle digest, task/split identity, evaluator digest and recomputed per-task/per-subject metrics.
4. System receipt: physical target, firmware/compiler/runtime, raw latency/power traces, instrument calibration and measurement uncertainty.

A frame SHA-256 is corruption detection, not sender authentication. A receipt hash is integrity binding, not a signature or hardware attestation. A process sandbox cannot establish that the Linux kernel, firmware or power instrument is honest.

## Validation ladder

Before any confirmatory run:
- verify exact binary, runtime closure and supervisor digests;
- prove all mandatory namespaces are created;
- prove network access is unavailable;
- inspect mount table from inside and prove evaluator/private paths are absent;
- prove capabilities are empty and seccomp profile active;
- verify resource limits by reading the actual candidate cgroup;
- run process-spawn, fork-bomb, memory-exhaustion, CPU-spin, file-read, descriptor, protocol-malformation, timeout and supervisor-kill negative controls;
- independently replay all receipts.

## Cgroup v2 read-back verifier tranche

The next implementation slice adds `research/temporal-bench/src/cgroup_v2_verify.rs`, exported as `cgroup_v2_verify`. It consumes the exact systemd `ControlGroup` property from the trusted launcher, checks the trusted host's mountinfo for one unified cgroup2 mount, canonicalizes the scope path beneath that mount, reads the parent controller availability/delegation and the candidate scope's effective `memory.max`, `memory.swap.max`, `cpu.max`, `pids.max`, `memory.oom.group`, `cgroup.procs` and `cgroup.events`, and rejects any mismatch. The candidate PID must belong to the expected scope and the cgroup must report `populated 1`. A digest commits the exact observed limits; it is not a signature or hardware attestation.

Twelve unit tests are authored for exact read-back, wrong scope, permissive/mismatched resource settings, disabled controllers, missing candidate PID, empty scope, unsafe paths, cgroup2 mount parsing and malformed event files. They are not yet compiled or executed. The verifier does not query systemd over D-Bus, create the scope, install seccomp, preserve fd 3, enforce wall time, or run a candidate. Those remain supervisor/integration work and no security qualification is claimed.

The scope planner no longer includes `--pipe`. FD 3 preservation is an explicit responsibility of the trusted launcher and must be verified end-to-end before a candidate run can qualify.

## Status

The first pure planning tranche is implemented in `research/temporal-bench/src/sandbox_profile.rs` and exported from the crate. It validates pinned Nix-store path shapes, nonzero candidate/Bubblewrap/seccomp/closure digests, closure uniqueness and inclusion of the candidate, bounded resource settings and seccomp fd 3. It emits deterministic Bubblewrap and systemd scope argv vectors and hashes the exact plan. Twelve tests are authored for explicit namespaces, no try-flags, exact read-only closure binds, scope/pipe semantics, seccomp fd pinning, resource bounds and digest behavior; they have not been compiled or run. No supervisor executable, seccomp filter, cgroup verifier or adversarial integration test is implemented by this document. The design is not yet a qualified security boundary.