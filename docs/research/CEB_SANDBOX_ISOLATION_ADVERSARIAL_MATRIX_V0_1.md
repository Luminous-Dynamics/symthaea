# CEB / Temporal Candidate Sandbox — Adversarial Qualification Matrix V0.1

**Status:** Design-only acceptance matrix. All cases below are unexecuted until the #9865 supervisor and an independently qualified Linux runner exist.  
**Owner:** [#9865 — fail-closed Linux candidate sandbox supervisor](https://github.com/Luminous-Dynamics/symthaea/issues/9865), composed with [#10064 — execution isolation boundary](https://github.com/Luminous-Dynamics/symthaea/issues/10064).  
**Implementation context:** [#11202 — cgroup v2 and systemd scope read-back](https://github.com/Luminous-Dynamics/symthaea/pull/11202), stacked on [#9866 — pure sandbox plan](https://github.com/Luminous-Dynamics/symthaea/pull/9866).

## 1. Paradigm shift: qualify the execution path, not the sandbox label

A product setting named “sandboxed” is not evidence that the candidate actually executes inside the advertised boundary. A secure guest/container can still be defeated by an unwrapped entry point, host-side file broker, mount option, cache, volume synchronizer, stale descriptor, or control-plane API.

**Qualification proposition:** for every supported candidate entry point, the exact candidate process that consumes untrusted task content must be the process whose mount, namespace, credential, syscall, network, descriptor and resource state was independently observed and tested.

This is a *boundary-congruence* requirement:

1. **Declared policy:** the versioned profile says what should be isolated.
2. **Launch binding:** the exact executable, closure, supervisor, Bubblewrap argv, seccomp artifact and systemd unit are pinned.
3. **Manager view:** the trusted systemd manager reports the expected active `.scope`, `ControlGroup` and limits.
4. **Kernel view:** host cgroup files and process identity show the effective values for the same scope/PID.
5. **Adversarial outcome:** candidate-originated probes cannot cross the declared boundary.
6. **Receipt:** all observations bind to the exact task, source, image, policy and run identity.

A mismatch at any layer is **unqualified**, even when the candidate task itself passes.

## 2. Threat model

Assume the candidate is buggy or intentionally hostile. It can:
- run arbitrary user-space code within its candidate process;
- spawn threads and child processes, race filesystem paths and send malformed protocol frames;
- inspect its visible environment, filesystem, process namespace, mounts and inherited descriptors;
- exploit any accessible helper, filesystem broker, cache, volume API, socket, shared memory or orchestration endpoint;
- attempt CPU, memory, process, disk, I/O and wall-time exhaustion.

Out of scope for this v0.1 matrix: kernel compromise, malicious firmware, hardware side channels, and a privileged host administrator. These limits must remain explicit.

The evaluator, oracle, hidden labels, private datasets, signer keys, grading binaries, raw receipts and host credentials are trusted-side assets. They must never be mounted, inherited, proxied or otherwise exposed to the candidate.

## 3. Adversarial matrix

The rows are *attack families*, not independent statistical trials. Each requires at least one negative control (the candidate's attempted boundary crossing must be denied) and one positive control (the legitimate intended workload still works). Add multiple variants per family and stratify by supported kernel, architecture, runtime and entry point.

| ID | Attack family | Adversarial probe / failure pattern | Required observation and pass condition |
|---|---|---|---|
| SBX-01 | Entry-point bypass | Invoke every CLI, API, agent-tool, retry, patch, test and recovery path; attempt a route that launches the host command directly while the UI/API claims sandboxing. | Every untrusted entry point maps to the same qualified supervisor path. Any host-mode or unwrapped fallback is a critical failure. |
| SBX-02 | Namespace downgrade | Remove one required namespace, deny user namespaces, or substitute a `-try` option; attempt launch with only partial isolation. | Missing user/PID/network/IPC/UTS/cgroup namespace support fails before candidate execution. No downgrade/retry path exists. |
| SBX-03 | Mount and root exposure | Probe `/`, `/home`, `/run`, `/proc`, `/sys`, full `/nix/store`, data roots and evaluator directories. | Candidate sees only its enumerated read-only runtime closure, minimal proc/dev, isolated scratch and required protocol streams. |
| SBX-04 | Path and link races | Attempt `..`, encoded separators, Unicode variants, symlink replacement between check/use, hardlink pivots, archive extraction traversal and path aliasing. | Path resolution is confined at operation time, not only at an earlier string check. No write, read, delete or rename escapes the declared workspace. |
| SBX-05 | Mount/COW semantic mismatch | Write through declared `rwcopy`, overlay, snapshot or temporary-copy mount modes; inspect whether host source bytes change. | Any advertised copy-on-write mode is demonstrably copy-on-write. Host originals remain byte-identical. |
| SBX-06 | Descriptor and credential inheritance | Enumerate `/proc/self/fd`; inspect environment and accessible sockets; attempt access to SSH agent, D-Bus, Docker, cloud metadata, tokens and parent descriptors. | Only explicitly allowed protocol descriptors are inherited. No ambient credentials, evaluator descriptors, manager sockets or host service handles are reachable. |
| SBX-07 | Network boundary | Probe IPv4/IPv6, DNS, loopback, host gateway, metadata endpoints, Unix sockets, proxies, inherited sockets and alternate network namespaces. | Network is denied by default. No path can bypass the namespace or policy through a proxy/helper. Protocol pipes remain usable. |
| SBX-08 | Capabilities and syscall policy | Attempt ptrace, mount, setns, unshare, bpf, perf events, open-by-handle, userfaultfd, raw sockets and privilege transitions. | Capability set is empty; no-new-privileges and the architecture-pinned seccomp program are present; forbidden operations fail. |
| SBX-09 | Resource exhaustion | CPU spin, memory allocation, swap pressure, PID/thread flood, scratch fill, stdout/stderr flood and endless frame output. | Effective cgroup values match policy; wall-clock and I/O bounds trigger deterministic termination; host and adjacent tasks remain healthy. |
| SBX-10 | Process-tree escape and lifecycle | Fork/daemonize, double-fork, create new sessions, leave grandchildren, ignore TERM, outlive parent, kill supervisor, or exit during receipt capture. | Timeout, cancellation and supervisor death terminate the entire candidate cgroup. The launcher reaps children and records the termination reason. |
| SBX-11 | Cross-run and cross-tenant leakage | Reuse workspace, cache, snapshot, temp path, cgroup, mount, file descriptor or service across task IDs; read stale outputs from a previous run. | Each run has a unique scope and isolated writable state. No previous candidate or private evaluator artifacts are reachable. |
| SBX-12 | Control-plane and helper compromise | Reach host file brokers, cache daemons, volume sync APIs, snapshot handlers, privileged helpers, local admin ports, systemd/D-Bus and orchestration endpoints. | Candidate cannot call or mutate host-control APIs. Helpers validate identity, path, operation and lifetime at the point of use. |
| SBX-13 | Virtual-device and arithmetic boundary | Exercise guest-controlled offsets, lengths, mapping sizes, integer overflow, device command fields and shared-memory ranges in any VMM/device path. | Checked arithmetic and range validation prevent mappings or writes outside guest-owned buffers. If v0.1 does not use a VMM, record this family as not applicable with an explicit reason. |
| SBX-14 | Policy consistency | Exercise the same file/network/process operation through each supported platform backend, shell, tool API, patcher, test runner, retry and background-worker path. | Policy semantics are identical across all execution routes for the declared platform profile. Unknown backend/version is a hard failure, not permissive fallback. |
| SBX-15 | Receipt and evidence spoofing | Tamper with candidate digest, closure digest, Bubblewrap path, seccomp profile, systemd properties, cgroup path, PID, termination reason, stdout/stderr digest, replay or reorder receipts. | Receipts bind exact task, executable, policy, manager unit, kernel-observed cgroup, process identity, output and evaluator. A hash is never represented as a signature or runner attestation. |
| SBX-16 | Oracle and grader isolation | Search mounted files, environment, descriptors, process arguments, logs, caches, IPC and host APIs for oracle seeds, hidden labels, grader paths and private receipts. | Candidate cannot access evaluator bytes or alter grader outputs. The grader independently recomputes results from retained evidence. |

### Positive controls

At minimum, prove:
- the pinned candidate executable starts through the intended entry point;
- every exact runtime-closure dependency can be read while the closure remains read-only;
- the intended stdin/stdout protocol exchange succeeds within the frame limits;
- a representative legitimate workload completes within the declared resource budget;
- timeout, cancellation and normal exit are distinguishable;
- the host-side evaluator can independently read the separate oracle and grade the retained candidate output;
- a second run gets a fresh scope and writable state.

## 4. External research and industry comparison

**SandboxGym (Nebula Security, fixed source snapshots; page last updated August 15, 2026)** reports 32 high-impact isolation failures across nine open-source sandbox/VM projects, including two full escapes and at least one isolation failure in every scored product. Its methodology retains findings judged exploitable from inside the sandbox and capable of collapsing the intended boundary. It distinguishes full escapes, policy bypasses and cross-tenant/data exposure; the dominant failure class was policy bypass, including entry-point omissions, inconsistent policy application, path/symlink races, mount/COW mistakes, inherited handles and control-plane errors.

CEB interpretation is deliberately limited: this is a self-published code-scanning audit on fixed snapshots, not a controlled live-agent benchmark or an independent replication. Finding counts are not comparable system-risk scores. Its value here is threat-model coverage and concrete failure mechanisms, not a leaderboard number.

**SRE-Bench (October 2026)** evaluates 1,572 deterministic tasks over 262 binary instances from 19 private programs, with isolated-container tasks and a separate grading server. It demonstrates a useful benchmark architecture: candidate-facing artifacts and tools are separated from source/reference/grader materials. Its task success metrics measure reverse-engineering capability, not sandbox escape resistance.

Sources:
- SandboxGym methodology and results: https://sandboxgym.io/
- SRE-Bench methodology: https://daplab.cs.columbia.edu/benchmarks/sre-bench/
- Linux cgroup v2 interfaces: https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html
- systemd scope properties: https://www.freedesktop.org/software/systemd/man/latest/systemd.scope.html
- systemd resource controls: https://www.freedesktop.org/software/systemd/man/latest/systemd.resource-control.html

## 5. Qualification metrics and hard gates

Do not average away a full escape, host write, credential disclosure or oracle leak. Report separately:
- **Entrypoint coverage:** fraction of supported untrusted entry points that are proven to invoke the qualified supervisor.
- **Attack-family coverage:** number of attack families with both positive and negative controls.
- **Critical escape count:** confirmed host code execution, host memory corruption, host write, credential theft or cross-tenant/oracle disclosure.
- **Policy-bypass count:** any operation allowed by the effective kernel/host state but forbidden by the declared profile.
- **Manager/kernel congruence:** agreement of exact systemd unit properties and same-scope kernel cgroup/process observations.
- **Positive-control completion:** legitimate workloads that remain solvable under the declared profile.
- **Resource enforcement:** actual CPU/memory/swap/PID/time/stdio ceilings, plus termination and cleanup correctness.
- **Evidence completeness:** exact executable/closure/policy/seccomp/systemd/cgroup/process/termination digests and independently replayable receipts.

Hard gate: one confirmed critical escape or oracle disclosure disqualifies that exact image, supervisor, policy and runner configuration. Fixing it requires exploit replay, alternate-path replay and fresh-candidate evaluation; editing the original exploit string alone is insufficient.

A zero observed escape count can receive a statistical upper bound only over a declared, exchangeable attack population. Attack cases sharing the same vulnerability mechanism, host image or policy are clustered; repeated variants are not independent (n). Always publish counts by attack family and configuration and retain the raw cases.

## 6. Composition and no-runner rule

This matrix is an acceptance adapter for #9865 and #10064. It does not create another sandbox implementation, candidate protocol, evidence ledger, DSSE envelope, task schema or primary CEB metric.

Before a qualified runner exists, permissible work is limited to threat-model review, fixture authoring, deterministic parser/planner tests, static validators and evidence/receipt contracts. Do not execute candidate-controlled code, build scripts, tests, PoCs or graders on the host-opt-in CodeExecutor.

**Current boundary:** the cgroup-v2 verifier can parse a supplied manager response and read supplied host cgroup paths, but the supervisor does not yet query systemd, launch Bubblewrap, preserve fd 3, establish namespaces/seccomp, enforce wall-time/output limits or execute these attack cases. None of the controls in this matrix is yet qualified.
