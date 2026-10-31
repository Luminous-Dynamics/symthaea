# NixOS/systemd Evaluator Isolation Runbook (v0.1)

**Tracking:** EVAL-ORACLE-002 / #9621 and EVAL-ORACLE-001 / #9437  
**Status:** module and intake primitive authored; no Nix evaluation, host deployment, adversarial execution or Rust tests run  
**Scope:** opt-in research prototype only

## 1. Authority model

The prototype separates three OS identities:

1. **Task generator:** receives a task-generation credential and writes both public task data and oracle artifacts. It is trusted but does not receive the evaluator's signing key.
2. **Candidate:** receives only public task data and a candidate-controlled exchange directory. It receives no seed, oracle, evaluator credential, signing key, or inherited socket activation descriptor.
3. **Evaluator:** reads public task, oracle and the run-bound candidate artifact; copies the candidate artifact into a private directory; evaluates only that copy; signs a receipt using an evaluator-only credential.

This is stronger than application-level hidden fields. It is not yet a complete security boundary, because root and the kernel remain trusted and host-specific isolation needs adversarial tests.

## 2. NixOS module

File: research/evaluator-isolation/nixos-module.nix

The module is imported explicitly and defaults to disabled. It defines separate users/groups, tmpfiles ownership/modes and ordered oneshot systemd units:

taskgen -> candidate -> evaluator

The CPU-only sandbox profile uses:
- ProtectSystem=strict, ProtectHome=yes;
- PrivateTmp=yes, PrivateDevices=yes, PrivateNetwork=yes;
- NoNewPrivileges=yes, empty capability bounding/ambient sets;
- ProtectKernelTunables, ProtectKernelModules, ProtectControlGroups, ProtectClock;
- ProtectProc=invisible, ProcSubset=pid;
- RestrictNamespaces, RestrictSUIDSGID, LockPersonality, MemoryDenyWriteExecute;
- native syscall architecture and @system-service filter;
- explicit per-service ReadOnlyPaths, ReadWritePaths and InaccessiblePaths.

The candidate sees only public files, owns its work directory and writes to a narrow exchange path. The evaluator sees oracle files and can traverse/read a known prediction filename in the exchange directory, but cannot list that directory or access candidate work. Its signing credential is delivered with LoadCredentialEncrypted=; taskgen receives a different credential. Both are hidden from the candidate.

The module intentionally does not enable itself in the repository's default NixOS setup. It expects operator-supplied absolute executable paths and encrypted credential files. LoadCredentialEncrypted= requires systemd 254+; systemd's credential feature exposes immutable per-unit credentials through a service-specific credential directory. See https://systemd.io/CREDENTIALS/.

### Directory permissions

| Directory | Owner/group | Mode | Purpose |
|---|---|---:|---|
| public | generator/public | 2750 | read-only to candidate and evaluator |
| oracle | generator/oracle | 2750 | inaccessible to candidate |
| exchange | candidate/exchange | 2710 | candidate write; evaluator opens known filename |
| candidate-work | candidate/candidate | 0700 | private candidate scratch |
| evaluator-private | evaluator/evaluator | 0700 | imported immutable-to-candidate copy |
| receipts | evaluator/evaluator | 0700 | evaluator receipt storage |

The setgid directories ensure newly created files inherit their intended group. Candidate exchange artifact mode is expected to be 0640 under umask 0027. The evaluator opens a known run-bound filename using a directory FD; it does not enumerate the untrusted directory.

The current module has fixed paths and only supports a serial research run. It must not be used for concurrent or multi-tenant jobs until run-scoped directories, cleanup, atomic run manifests and replay-resistant run IDs are implemented.

## 3. Safe candidate artifact intake

Package: research/evaluator-isolation

The new Rust library uses Linux openat2 to constrain paths relative to already-open directory descriptors:
- RESOLVE_BENEATH;
- RESOLVE_NO_SYMLINKS;
- RESOLVE_NO_MAGICLINKS;
- O_NOFOLLOW, O_NONBLOCK, O_CLOEXEC.

It rejects path traversal, symlinks, magic links, hard links (link count other than one), non-regular files, oversize input and an existing private destination. It copies bytes from the same opened candidate FD into a newly created 0600 evaluator-private file and hashes exactly the bytes copied. It checks source identity/size/timestamps before and after intake and removes a partial private copy on failure.

If the kernel lacks openat2 (ENOSYS), the code refuses the operation; it does not fall back to openat, canonicalize, prefix string checks or an insecure path traversal.

This does not by itself create an atomic snapshot of a maliciously-written source file. The evaluator binds and parses the private copied byte sequence only, checks the expected run/task/model/schema digests, and treats any malformed or mismatched artifact as invalid. The digest must be included in the signed receipt.

## 4. Required systemd and host checks

On a disposable target NixOS host, import the module, set executable/credential paths and rebuild. Before starting any candidate code:

```sh
systemd --version
systemd-analyze verify symthaea-temporal-taskgen.service symthaea-temporal-candidate.service symthaea-temporal-evaluator.service
systemd-analyze security symthaea-temporal-taskgen.service
systemd-analyze security symthaea-temporal-candidate.service
systemd-analyze security symthaea-temporal-evaluator.service
systemctl cat symthaea-temporal-taskgen.service
systemctl cat symthaea-temporal-candidate.service
systemctl cat symthaea-temporal-evaluator.service
```

Then execute the host attack suite in the README. systemd-analyze security is diagnostic and not a proof. PrivateNetwork=yes is not a blanket AF_UNIX restriction; inspect and deny relevant host sockets explicitly. systemd may be unable to apply some namespace restrictions on a host lacking the required kernel features; never assume that directives alone establish isolation.

## 5. Evidence to preserve

For each qualification attempt, capture:
- NixOS generation, kernel and systemd versions;
- Nix flake lock, module source digest and unit files;
- UID/GID/group membership and directory mode/ownership manifest;
- candidate mount namespace, mountinfo, visible paths, open FDs and AF_UNIX socket inventory;
- attack input digest and exact command / exit code;
- evaluator copy digest, size, run/task/model binding and disposition;
- failed/invalid/timeout cases, not just successful cases.

## 6. Remaining work

- Wire the taskgen/candidate/evaluator binaries to run-specific artifacts and enforce unique run IDs.
- Add signed receipt production/verification using an audited signature library and evaluator-only key; no hand-rolled crypto.
- Add persistent replay/nonce database and key rotation/revocation model.
- Add process/UID attack harness and test descriptor inheritance, /proc, AF_UNIX, symlink, hardlink, TOCTOU, oversized file, stale output and replay attacks.
- Run Nix evaluation and systemd unit verification on actual target NixOS.
- Run adversarial isolation tests before any confirmatory temporal benchmark.

## 7. Security claim ceiling

The module and intake source are authored, not security-qualified. Even a future passing test only supports the declared NixOS/kernel/systemd/filesystem and threat model. It does not establish resistance to kernel compromise, root, privileged operator, hypervisor or hardware side-channel attacks.

