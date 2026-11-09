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

## Adversarial qualification matrix

The versioned red-team case families and their positive/negative controls are specified in [CEB_SANDBOX_ISOLATION_ADVERSARIAL_MATRIX_V0_1.md](CEB_SANDBOX_ISOLATION_ADVERSARIAL_MATRIX_V0_1.md). This matrix is owned by #9865/#10064, not a new benchmark lane or runner. It includes entry-point bypass, namespace downgrade, mount/path races, COW semantics, inherited descriptors, network and syscall restrictions, resource exhaustion, lifecycle/process-tree termination, cross-run leakage, control-plane attacks, virtual-device bounds, policy consistency, receipt spoofing and oracle/grader separation.



## Two-phase admission: verify the live boundary before candidate release

The supported launch path is a **gated candidate start**, not “launch candidate, then inspect it”:

1. The trusted supervisor opens and hashes the architecture-pinned seccomp BPF artifact. It prepares a private one-shot gate channel and starts the pinned `systemd-run --user --scope` command with the Bubblewrap argv. The seccomp descriptor is assigned fd 3; the gate channel is fd 4. In the child launch path, the supervisor must explicitly clear `FD_CLOEXEC` only for those descriptors before the trusted exec chain.
2. Current upstream systemd source shows the scope path calls `StartTransientUnit()` with a PID reference to `systemd-run` itself, waits for the scope job, and then replaces that same PID using `execvpe_safe()`. `execvpe_safe()` calls `execve()` directly. Consequently, a deliberately non-`CLOEXEC` fd can survive this exec without using `--pipe`; the exact fd lifetime is still an integration obligation, not something proven by source inspection alone.
3. Bubblewrap consumes fd 3 as its seccomp program and starts only a **trusted, pinned gate helper** as the first process inside the namespaces. The helper does not parse or execute candidate content. The Bubblewrap profile must preserve only the explicitly planned descriptor range (currently fd 3–4); no other inherited descriptor is allowed.
4. The gate helper sends a fixed readiness token over fd 4 and blocks. At this point the untrusted candidate has not been started. The host supervisor resolves the exact gate process from the scope's `cgroup.procs`, its parent relationship to the Bubblewrap PID, `/proc/<pid>/status` (`NSpid`, `NoNewPrivs`, `Seccomp`, `CapEff`, `CapBnd`) and host-visible namespace/mount records. It verifies the exact systemd `.scope` manager properties and kernel cgroup controls with `cgroup_v2_verify`.
5. Only after every required observation agrees does the supervisor send the one-shot release token. The gate closes fd 3 and fd 4, clears any other non-protocol descriptors, and `execve()`s the exact pinned candidate binary with the bounded protocol streams. No candidate-controlled byte is interpreted before release.
6. Any missing, duplicate, unexpected or inconsistent observation kills the entire scope and records a **not-qualified** result. There is no host-mode or reduced-isolation fallback.

The trusted launcher must identify the host PID of the gate process unambiguously (for example, the unique child of the pinned Bubblewrap PID in the scope whose `NSpid` ends at the expected inner PID and whose `/proc/<pid>/exe` matches the gate digest). The gate must not fork before release. The receipt binds the gate and candidate digests, fd plan, seccomp artifact digest, Bubblewrap argv/version, systemd unit identity, namespace/mount observations, cgroup snapshot digest and release disposition.

**Important evidence ceiling:** `/proc/<pid>/status` reporting `Seccomp: 2` proves that a filter is active, not which BPF program is active. The exact filter is bound by the supervisor's artifact digest and Bubblewrap invocation; the permitted/denied syscall behavior still requires adversarial integration tests. Namespace inode differences and mountinfo establish observed namespace/mount state but do not prove the kernel is uncompromised.

### Required descriptor and gate controls

- Positive: fd 3 contains the exact pinned seccomp artifact when Bubblewrap starts; fd 4 delivers exactly one readiness/release exchange; the gate closes both before candidate `execve`.
- Negative: setting `FD_CLOEXEC` on either required descriptor prevents qualification; an unexpected inherited fd, missing fd, duplicate release, malformed token or stale token fails closed.
- Positive: the exact gate PID is present in the target scope before release, has the expected namespace/mount/seccomp status, and the legitimate candidate starts only after release.
- Negative: a candidate sentinel that would write a host-only file or read an evaluator-only sentinel must not run when any pre-release check fails.
- Failure: supervisor crash, parent death, timeout, scope activation failure, manager/kernel mismatch or incomplete gate handshake terminates the scope; a child cannot outlive the supervisor.

The current PR contains the planner and read-back verifier but **does not implement this gate helper, the descriptor handoff, procfs namespace inspection, the systemd scope lifecycle, process-tree termination or the release protocol**. All items above are design requirements pending source implementation and live Linux integration tests.

## Cgroup v2 read-back verifier tranche

The next implementation slice adds `research/temporal-bench/src/cgroup_v2_verify.rs`, exported as `cgroup_v2_verify`. It consumes the exact systemd `ControlGroup` property from the trusted launcher, checks the trusted host's mountinfo for one unified cgroup2 mount, canonicalizes the scope path beneath that mount, reads the parent controller availability/delegation and the candidate scope's effective `memory.max`, `memory.swap.max`, `cpu.max`, `pids.max`, `memory.oom.group`, `cgroup.procs` and `cgroup.events`, and rejects any mismatch. The candidate PID must belong to the expected scope and the cgroup must report `populated 1`. A digest commits the exact observed limits; it is not a signature or hardware attestation. The module also parses an explicitly selected `systemctl show` KEY=VALUE response (without `--value`) and verifies `Id`, `ActiveState`, `SubState`, `ControlGroup`, `MemoryMax`, `MemorySwapMax`, `CPUQuotaPerSecUSec`, `TasksMax`, `RuntimeMaxUSec` and `OOMPolicy` against the requested scope policy. The response must be obtained by a trusted host-side query for the exact transient unit; parsing itself does not authenticate the manager or query D-Bus. Kernel cgroup-file read-back remains required and must match the same `ControlGroup`. `CPUQuotaPerSecUSec` and `RuntimeMaxUSec` are normalized systemd timespan strings (for example `1s`, `190ms`, or `1min 30s`), not necessarily bare integers; the parser converts integer-unit components to microseconds with checked arithmetic and rejects `infinity`, unknown units, fractional/inexact values and overflow rather than silently rounding.

Twenty-three unit tests are authored for systemd unit/property matching, exact integer timespan conversion (including compound values), rejection of infinite/malformed/fractional/overflowing timespans, duplicate/missing/oversized property-response rejection, unit-to-ControlGroup identity, exact cgroup read-back, wrong scope, permissive/mismatched resource settings, disabled controllers, missing candidate PID, empty scope, exact `.scope` identity, bounded control-file reads, unsafe paths, cgroup2 mount parsing, malformed event fields and invalid octal escapes. They are not yet compiled or executed. The verifier does not query systemd over D-Bus, create the scope, install seccomp, preserve fd 3, enforce wall time, or run a candidate. Those remain supervisor/integration work and no security qualification is claimed.

The scope planner no longer includes `--pipe`. FD 3 preservation is an explicit responsibility of the trusted launcher and must be verified end-to-end before a candidate run can qualify.

## Status

The first pure planning tranche is implemented in `research/temporal-bench/src/sandbox_profile.rs` and exported from the crate. It validates pinned Nix-store path shapes, nonzero candidate/Bubblewrap/seccomp/closure digests, closure uniqueness and inclusion of the candidate, bounded resource settings and seccomp fd 3. It emits deterministic Bubblewrap and systemd scope argv vectors and hashes the exact plan. Twelve planner tests are authored for explicit namespaces, no try-flags, exact read-only closure binds, scope-mode argv, seccomp fd pinning, resource bounds and digest behavior; they have not been compiled or run. A source-only cgroup-v2 verifier is now exported by the temporal-bench crate, but its twenty-three tests are also uncompiled/unexecuted. No supervisor executable, seccomp filter installation, verified fd-3 handoff, wall-time/output supervisor or adversarial integration test is implemented. The design is not yet a qualified security boundary.