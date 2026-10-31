# Symthaea evaluator isolation prototype

This directory contains a Linux-only candidate-artifact intake primitive and an optional NixOS module. Nothing is imported into the default NixOS or Symthaea configuration.

## Contents

- src/lib.rs: safe intake of one candidate-generated artifact using Linux openat2.
- nixos-module.nix: separate task-generator, candidate and evaluator service UIDs, directory permissions, systemd mount restrictions and per-service encrypted credentials.
- Cargo.toml: standalone research package.

## NixOS integration

Import this module explicitly in a test NixOS host. Supply three absolute executable paths for the task generator, candidate runner and evaluator. The module does not invent binaries or modify the user's default configuration.

The defaults expect two encrypted credential files:
- /etc/credstore.encrypted/symthaea-temporal-generator-seed
- /etc/credstore.encrypted/symthaea-temporal-evaluator-signing-key

Provision them with systemd-creds encrypt on the target machine, using a TPM2/local system credential key as supported by that host. Do not put plaintext keys or task seeds in Nix store paths, unit files, command arguments, or candidate environment variables. The evaluator signing key is delivered only to the evaluator unit using LoadCredentialEncrypted=; the task-generation seed is delivered only to taskgen. systemd credentials are immutable per activation and are exposed to the receiving service via its private credential directory. This feature requires systemd 254 or newer.

Example module configuration:

```nix
{
  imports = [ /path/to/symthaea/research/evaluator-isolation/nixos-module.nix ];

  services.symthaea-temporal-isolation = {
    enable = true;
    taskgenExecutable = "/nix/store/.../bin/symthaea-taskgen";
    candidateExecutable = "/nix/store/.../bin/symthaea-candidate";
    evaluatorExecutable = "/nix/store/.../bin/symthaea-evaluator";
    generatorSeedCredential = "/etc/credstore.encrypted/symthaea-temporal-generator-seed";
    evaluatorSigningKeyCredential = "/etc/credstore.encrypted/symthaea-temporal-evaluator-signing-key";
  };
}
```

After rebuilding the disposable test host, inspect generated units and their effective settings before running any candidate:

```sh
systemd-analyze verify symthaea-temporal-taskgen.service symthaea-temporal-candidate.service symthaea-temporal-evaluator.service
systemd-analyze security symthaea-temporal-taskgen.service
systemd-analyze security symthaea-temporal-candidate.service
systemd-analyze security symthaea-temporal-evaluator.service
systemctl cat symthaea-temporal-taskgen.service
systemctl cat symthaea-temporal-candidate.service
systemctl cat symthaea-temporal-evaluator.service
```

The evaluator unit requires the candidate unit; candidate requires taskgen. Start only the evaluator unit to exercise the ordered chain after verifying all executable paths and credentials. Do not run untrusted code on a production host as part of initial qualification.

## Directory / permission model

| Path | Owner / group | Mode | Generator | Candidate | Evaluator |
|---|---|---:|---|---|---|
| public | generator / public | 2750 | read/write | read-only | read-only |
| oracle | generator / oracle | 2750 | read/write | inaccessible | read-only |
| exchange | candidate / exchange | 2710 | inaccessible | read/write | traverse + read known artifact |
| candidate-work | candidate / candidate | 0700 | inaccessible | read/write | inaccessible |
| evaluator-private | evaluator / evaluator | 0700 | inaccessible | inaccessible | read/write |
| receipts | evaluator / evaluator | 0700 | inaccessible | inaccessible | read/write |

The candidate is not a member of the oracle group and does not receive the evaluator signing credential or generator seed. The module also places each service in a private IPC namespace, restricts address families to AF_UNIX, and hides common host AF_UNIX endpoints including the Nix daemon socket. The exchange directory is intentionally group execute-only to the evaluator: the evaluator opens a run-bound filename supplied by trusted metadata instead of listing candidate-controlled filenames. Candidate artifacts are expected to be created with mode 0640 under the setgid exchange directory (the unit uses umask 0027).

The NixOS module currently uses fixed directories and is intended for one serial research run at a time. It is not yet a multi-tenant or concurrent-run scheduler. The calling executables must bind exact run IDs, filenames and task/model/config digests. A later revision should add run-scoped subdirectories with trusted creation and cleanup.

## openat2 intake

The library:
- rejects filenames containing path traversal or multiple path components;
- opens configured directories without following symlink or magic-link path components;
- opens the untrusted prediction with openat2, RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS, O_NOFOLLOW, O_NONBLOCK, and O_CLOEXEC;
- fails closed on ENOSYS; it has no openat, canonicalize, or string-prefix fallback;
- rejects non-regular files, link count other than one, and artifacts exceeding a caller-supplied limit;
- creates a new mode-0600 evaluator-private file with O_EXCL;
- copies bytes from the same opened source FD, hashes the exact bytes written to the private copy, checks source metadata before/after, and syncs the private copy.

After import, only parse and score the private copy. Do not reopen the candidate-controlled exchange path. The copied bytes are what the digest binds; an attacker may still race writes to the exchange file during intake, so the evaluator must validate the complete artifact schema and compare its run/task/model binding to protected evaluator metadata. If source metadata changes or the byte count differs, intake rejects the artifact.

## Receipts and replay consumption

The evaluator-only receipt module is `src/receipt.rs`. It encodes a versioned canonical binary payload and signs it using Ed25519 from `ed25519-dalek` 3.0.0. The signing key is read only from the evaluator's systemd credential directory; the key ID, task/oracle commitments, prediction/model/config/evaluator profile hashes, run/nonce/counter, three separate TED dimensions and measured resource costs are bound into the signature. Verification uses `verify_strict` against a trusted public key allow-list.

The local `ReplayLedger` consumes nonce, run-ID and per-key monotonic-counter markers after signature verification. The private evaluator ledger directory is enforced as evaluator-owned, directory-only and inaccessible to group/other users. Marker and counter writes are synced. A crash can consume an identifier without recording a successful result, but cannot permit replay of that identifier. Restoring an old ledger backup can roll back replay state; this prototype is not a distributed monotonic database. Do not accept receipts after unverified ledger restore.

The receipt and replay code are source-authored; their Rust tests have not been executed and the code is not wired into a working evaluator binary yet.

## Mandatory host attacks

Run these with the candidate UID, then preserve stdout/stderr, exit status and the evaluator's raw audit record:

1. Read, list, stat and traverse oracle, evaluator-private, receipt and credential paths.
2. Access evaluator/taskgen processes through /proc/<pid>/fd, /proc/<pid>/root, environment and process metadata.
3. Open system bus, systemd private socket, Nix daemon and any host AF_UNIX sockets not explicitly needed.
4. Submit symlink, magic-link, hard-link, FIFO, directory, device, absolute, .., nested, oversized, truncated, sparse, duplicate and malformed artifacts.
5. Race repeated rename/write/truncate attempts against evaluator intake; verify only the private copied bytes are parsed and their digest is receipt-bound.
6. Attempt to replace oracle/task/prediction after commitment and replay a receipt against a different run/task/model/config digest.
7. Verify unsupported namespaces or missing openat2 produce an explicit failed qualification, never a permissive fallback.

Systemd's private network namespace does not block every AF_UNIX connection. The module hides common system sockets, including the D-Bus/systemd sockets, the Nix daemon socket, Docker/containerd, udev, NetworkManager, Avahi and Podman endpoints. The host's actual socket inventory must still be inspected. Filesystem namespace controls can also be unavailable or degrade on some kernels or service-manager configurations. A successful systemd-analyze security score is diagnostic only, not evidence of confidentiality. A source-equivalent three-unit fixture passed `systemd-analyze verify` with systemd 257.9 in the authoring container. The Nix module itself has not been evaluated because Nix is unavailable; `systemd-analyze security` could not be run because this container is not booted with systemd as PID 1. Validate the generated units with real access attempts under the exact target NixOS generation, kernel and systemd release.

## Claim ceiling

This is an isolation deployment prototype and source-level artifact intake library. It is not yet a working end-to-end taskgen/candidate/evaluator pipeline or verified sandbox. A canonical Ed25519 receipt and local replay-ledger primitive are now source-authored, but are not wired into a running evaluator and are not security-qualified. It does not protect against kernel/host-root compromise, hardware side channels, privileged operators or a malicious hypervisor. The Ed25519 receipt module and replay ledger are source-authored; no signature or replay tests have run. No security claim is made until the attack suite passes on the target host.
