# Trusted CPU Runner Host Lifecycle Contract

## Purpose

This document freezes the host-lifecycle boundary for `symthaea-trusted-cpu-v1`.

The word **ephemeral** has two different meanings that must not be conflated:

```text
ephemeral runner registration/process
        !=
ephemeral machine / VM / host
```

Symthaea's current NixOS recovery design qualifies the first property and strongly cleans runner-owned state between jobs. It does **not** destroy and recreate the physical/virtual host after each job.

That distinction is part of the security and evidence contract.

## Current v1 profile — persistent hardened recovery appliance

The current `ci/nixos-ephemeral-runner-v1` design is intentionally narrow.

At the repository-pinned nixpkgs revision, the runner service is configured so that:

- GitHub runner registration uses `--ephemeral`, so one registration accepts at most one job before deregistration;
- `Restart=on-success` causes a successful one-job runner to restart and register again;
- ephemeral service startup wipes the runner state directory before the next registration;
- the runtime-backed work directory is cleared on every service start;
- the runner uses a dynamic user and strict systemd filesystem/device/process restrictions;
- the original access-token path and private comparison token are inaccessible to job code;
- runner self-update is disabled;
- default GitHub labels are disabled and only `symthaea-trusted-cpu-v1` is advertised;
- the DynamicUser runner receives only a fixed supplementary group for the local Stage-F authorization socket;
- the Stage-F authorization ledger is root-owned persistent host state outside the runner state/work directories.

These properties are mechanically frozen by `nix/tests/eval-github-actions-runner.nix`.

However, the underlying machine remains the same machine unless the operator separately destroys or reverts it.

The following state can therefore persist across registrations/jobs outside the cleaned runner directories:

- the Nix store and build cache;
- the root-owned Stage-F authorization-consumption ledger;
- operating-system state not owned by the runner service;
- host/network identity and routing environment;
- externally retained system/runner logs;
- kernel/firmware/device state;
- any operator-added filesystem or service state outside the runner sandbox.

Therefore:

```text
clean runner state
        !=
fresh host state
```

## One-time Stage-F authorization ledger

The persistent appliance carries a dedicated root-owned ledger at:

```text
/var/lib/symthaea-stage-f-authorizations
```

It is **not** the upstream GitHub runner `StateDirectory` and must never be folded into the directory that ephemeral runner startup clears.

Job code cannot write this directory directly. A root-owned, socket-activated local service owns the minimal protocol:

```text
BOOT_ID
CONSUME <64-hex authorization nonce> <64-hex authorization-capsule SHA-256>
```

The socket is reachable only by the runner's fixed supplementary group. Successful `CONSUME` atomically creates a root-owned nonce marker and records the authorization SHA-256. A second attempt to consume the same nonce fails closed.

The intended theorem is:

```text
valid Stage-F authorization.v2
+ qualified host boot identity
+ unused 256-bit nonce
+ atomic root-owned CONSUME
= exactly one eligible execution attempt
```

The ledger provides **authority replay prevention**, not scientific evidence. Its contents do not qualify SE-001, classify a failure, or grant repair authority.

A trusted job that can reach the socket can at worst consume a known authorization early, causing denial of service. It cannot delete a root-owned marker, mint an authorization capsule, make an old boot current, or turn a consumed authorization into additional authority. The recovery host remains an isolated trust appliance; do not add unrelated local users/services to the socket group.

### Boot binding

Stage-D smoke v2 records the Linux kernel boot ID after successfully querying the root authorization consumer. `recovery-eligibility.v5` carries that exact boot ID, and every Stage-F authorization.v2 must bind it.

Immediately before consuming a nonce, the Stage-F workflow asks the root consumer for the current boot ID and requires an exact match.

Therefore:

```text
host reboot
=> prior smoke.v2 stale
=> prior recovery-eligibility.v5 unusable
=> prior Stage-F authorization.v2 unusable
```

A reboot requires a fresh Stage-D smoke, fresh Stage-E join, and a newly reviewed Stage-F authorization. This prevents loss/replacement of host-local state across a reboot from silently reopening prior execution authority.

Root/operator mutation remains inside the host trust root. If the authorization ledger is deleted, rolled back, restored from an older snapshot, corrupted, or otherwise administratively altered, invalidate the current runner-substrate evidence and repeat Stage D/E before issuing any new Stage-F authorization.

Do not garbage-collect nonce markers during a qualified boot. If ledger maintenance is necessary, perform it only across an explicit substrate reset and fresh smoke/eligibility generation.

## Allowed scope for the persistent-appliance profile

A persistent hardened host is acceptable only for the **current reviewed recovery class**:

```text
trusted main-owned workflow
+ workflow_dispatch only
+ permissions: {}
+ exact hard-coded/reviewed target identity
+ exact diff/path or theorem boundary
+ no repository/environment secrets
+ correctness/reproducibility only
```

The exact consumer set remains allowlisted by `nix/tests/eval-trusted-runner-routing.nix`.

Do not route to this persistent-host profile:

- automatic `pull_request`, `push`, or `schedule` workflows;
- fork code;
- arbitrary branch-selected workflow definitions;
- user-supplied shell commands;
- generic build/test jobs whose source identity is not separately authorized;
- workloads requiring repository/environment secrets;
- GPU jobs or jobs requiring privileged device access;
- a future generic Tier-Q candidate executor unless it receives a separately qualified disposable-host design.

The repository is public. The absence of a secret does not make arbitrary public-repository code safe to execute on a persistent self-hosted machine.

## Preferred future profile — disposable machine per candidate job

Any future executor that runs broader candidate product code should raise the lifecycle boundary from **runner-ephemeral** to **host-ephemeral**.

Preferred sequence:

```text
trusted scheduler / main-owned executor
        ↓
create clean VM/image from qualified immutable generation
        ↓
register one-job runner
        ↓
execute one exact candidate qualification
        ↓
persist only declared evidence/log outputs
        ↓
deregister runner
        ↓
destroy VM or revert to trusted immutable snapshot
```

The next job must not inherit writable state from the previous candidate.

Acceptable implementations may include a newly provisioned VM, a one-shot cloud instance, a disposable NixOS VM, or another architecture that demonstrates equivalent destruction/reversion semantics. Merely deleting `_work`, `RUNNER_TEMP`, or GitHub runner credential files is not equivalent.

A future host-ephemeral design needs a replacement one-time authorization theorem. It must not assume the persistent-host ledger survives machine destruction.

## Nix-store boundary

The Nix store is intentionally useful as a deterministic dependency/cache substrate, but on a persistent appliance it is cross-job host state.

Rules:

1. no credential or secret may enter a derivation, flake input, store path, build log, or fixed-output artifact;
2. no scientific claim may treat a warm store as performance-equivalent to another runtime;
3. cache warmth must not alter correctness semantics;
4. store growth must be bounded operationally with disk monitoring and a reviewed garbage-collection policy;
5. a candidate must not gain authority merely because an artifact already exists in the store;
6. any future shared binary cache requires its own provenance/trust policy.

A Nix-store cache hit is a build optimization, not new evidence.

## External logs are required operational evidence

GitHub's current self-hosted-runner guidance recommends forwarding and preserving ephemeral runner application logs externally before production deployment. Symthaea should treat that as part of the operational acceptance boundary, not an optional debugging convenience.

Before Stage D/E recovery use, establish external retention for at least:

- runner `Runner_*` logs;
- runner `Worker_*` job logs where available;
- relevant systemd journal records for `github-runner-symthaea-validation.service`;
- `symthaea-stage-f-authorization@*.service` authorization-consumer records;
- host boot/generation identity;
- runner registration/restart timestamps;
- disk/GC/resource exhaustion events relevant to correctness availability.

The external destination must not expose a write path back into the runner and must have a declared retention period.

Do not place GitHub access tokens or other secrets in diagnostic output. Stage-F authorization nonces are authority identifiers, not credentials, but should still be retained only as part of the declared authorization/evidence record.

## Post-job persistent-host checks

For the v1 persistent-appliance profile, each completed recovery job should be followed by an operator-visible or automatically retained record proving at least:

```text
old runner registration completed/deregistered
service restarted successfully
new registration has the same unique capability label
runner work directory began clean
runner state was rebuilt from the external credential
no unexpected service/sandbox policy drift
host boot identity still equals the qualified smoke boot
consumed Stage-F nonce marker remains present and root-owned
host disk remained within declared bounds
```

Failure of these checks does not retroactively convert a completed test into a scientific FAIL, but it can invalidate the **runner-substrate qualification** for subsequent jobs until repaired.

## Evidence semantics

Keep these distinctions explicit:

```text
runner process ephemeral != host ephemeral
runner cleanup PASS != candidate PASS
candidate correctness PASS != integration PASS
candidate correctness PASS != performance equivalence
host reuse != evidence reuse
cache reuse != independent evidence
new runner registration != new scientific observation
Stage-F authorization != scientific authority
Stage-F consumption != candidate PASS
consumed authorization != reusable authorization
host reboot != same recovery-eligibility epoch
```

A recovered correctness result must continue to bind the exact source, harness, toolchain, runner/hardware context, host boot, authorization, authorization consumption, and command/gate identity.

## Activation gate for current v1

The persistent-appliance recovery profile is eligible for activation only after all of the following are true:

1. the exact recovery branch generation receives fresh external bootstrap authorization;
2. Stage-A bootstrap validation succeeds on that exact generation;
3. the eval-only runner and routing tests pass, including one-time authorization ledger isolation;
4. the external access-token file satisfies the existing root ownership/mode/no-newline contract;
5. the host contains no unrelated credentials, production mounts, privileged control sockets, or sensitive LAN reachability;
6. external runner/system/authorization-consumer log retention is configured and tested;
7. the runner appears with exactly `symthaea-trusted-cpu-v1` and no default labels;
8. the main-only smoke.v2 succeeds and binds the current host boot;
9. recovery-eligibility.v5 joins the exact promotion and smoke identities;
10. each Stage-F execution receives its own exact one-use authorization.v2 and unique 256-bit nonce;
11. only the exact reviewed recovery workflows are dispatched;
12. the operator records that this is a **persistent hardened host** profile, not a disposable-VM profile.

## Future Tier-Q gate

The Tier-Q qualification-capsule executor proposed in #217 must **not** inherit permission to use this persistent-appliance profile merely because both are called trusted CPU qualification.

Before Tier Q can execute general candidate product code on self-hosted capacity, separately qualify:

- disposable VM/host provisioning or equivalent immutable snapshot/revert semantics;
- externally retained runner logs;
- candidate-source admission and exact-SHA binding;
- a main-owned, allowlisted qualification profile rather than candidate-supplied arbitrary shell;
- no secret exposure;
- network egress policy appropriate to unmerged candidate code;
- cleanup/destruction proof after each job;
- denial of fork/untrusted external sources;
- one-time authorization semantics appropriate to an ephemeral host.

Until then, Tier Q should prefer GitHub-hosted runners, and `symthaea-trusted-cpu-v1` remains a narrow recovery capability.

## Upstream guidance boundary

This contract follows current GitHub guidance that ephemeral self-hosted runners receive one job, should be provided a clean environment between jobs, and should have diagnostic logs forwarded externally. GitHub also warns that self-hosted runners for public repositories can be exposed to dangerous fork/PR code.

Repository policy is intentionally stricter than relying on the `--ephemeral` flag alone.
