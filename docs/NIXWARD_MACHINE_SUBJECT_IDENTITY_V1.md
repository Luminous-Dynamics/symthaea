# Nixward Machine Subject Identity v1

**Status:** architecture contract only. No machine-identity implementation or authorization widening is claimed by this document.

**Tracks:** #4973, #4892, #4929, #4948.

## Purpose

Nixward's governed action intent needs an exact `subject_identity`, but the current `SystemStateSnapshot` exposes mutable/operational state such as NixOS generation and services rather than a canonical machine identity.

Do not fill that gap by silently treating hostname, NixOS generation, `/etc/machine-id`, boot ID, hardware serial, or any one key as universally equivalent to "the machine".

The identity boundary must make lifecycle semantics explicit.

## Core theorem

```text
hostname
!= operating-system instance identity
!= boot identity
!= hardware/device identity
!= cryptographic controller identity
!= organizational asset identity
```

and:

```text
same copied identity bytes
!= proof of same physical or logical machine
```

Images, VMs, containers, backups, and clones can copy state unless their provisioning lifecycle regenerates or rebinds it correctly.

Likewise:

```text
new machine-instance identity
!= necessarily a new physical device
```

A reinstall may intentionally create a new OS-instance identity on the same hardware.

## Identity dimensions

Keep these dimensions distinct even when a deployment profile binds several together.

### `InstallationIdentity`

Identity of one installed OS instance / Nixward installation lifecycle.

Expected to change for a fresh installation unless an explicit continuity/recovery profile says otherwise.

### `MachineInstanceIdentity`

Stable identity for the running OS instance across ordinary reboots.

A systemd-derived application-specific machine identity may be one input on supported Linux profiles, but must not be exposed raw across trust boundaries by default.

### `BootIdentity`

Identity of one kernel boot. Useful for currentness/replay ceilings, never a replacement for stable machine identity.

### `DeviceIdentity`

Optional identity anchored to hardware, TPM/device keys, or another qualified physical-device mechanism.

Not every VM/container/server profile has or should require this.

### `ControllerIdentity`

Cryptographic identity of the actor/agent/controller participating in authorization, for example a local authority key or Xenia-bound operator/device identity.

Possession of this identity does not prove physical-device identity unless a separate binding establishes that relation.

### `AssetIdentity`

Organizational/logical identity such as inventory asset, node role, cluster member, or Mycelix/Xenia resource reference.

Human-readable hostname may be metadata or an alias here; it is not canonical proof by itself.

## Subject identity profile

Define a versioned profile equivalent to:

```text
NixMachineSubjectProfileV1 {
  profile_identity,
  required_identity_dimensions[],
  optional_identity_dimensions[],
  continuity_rules,
  clone_rules,
  reinstall_rules,
  recovery_rules,
  disclosure_policy,
}
```

Then derive an exact action subject reference from the evidence required by that profile.

Do not hard-code one universal identity recipe for desktop, server, VM, container, recovery image, and fleet deployment.

## Local Linux profile

A first local Linux/NixOS profile may combine:

- a Nixward installation identity stored under protected state;
- an application-specific derivation of the host's machine identity where available;
- current NixOS system generation / exact system profile target as state, not identity;
- optional boot-specific identity for strict replay/currentness windows;
- optional local controller/device public-key identity.

Raw `/etc/machine-id` should not be exported as a public ecosystem identifier.

Systemd recommends application-specific derivation when machine identity is used in untrusted environments. Keep that privacy separation.

## Clone semantics

A clone-capable profile must specify what happens when a disk/image is duplicated.

At minimum distinguish:

```text
intentional continuity clone/recovery
  -> retains declared logical/installation lineage only under an explicit receipt/profile

new independent instance from template
  -> receives a distinct installation/instance identity
```

A copied identifier with no provenance receipt must not automatically establish continuity.

## Reinstall semantics

Fresh reinstall on the same hardware should normally produce a new `InstallationIdentity`.

If an operator wants continuity to an existing asset/device identity, record an explicit binding/migration receipt rather than reusing bytes and claiming nothing changed.

## Recovery semantics

Spore recovery may deliberately restore configuration/state while preserving or rotating particular identity dimensions.

Recovery must record which dimensions were:

- preserved;
- regenerated;
- rebound;
- unavailable;
- intentionally revoked.

`restored filesystem` is not enough to imply `same authorization subject`.

## NixOS generation/currentness

A NixOS generation is **state**, not machine identity.

For governed mutation a policy may require both:

```text
subject_identity == S
current_generation == G
```

Changing `G` should invalidate an exact-generation permit while leaving `S` unchanged.

Where generation is unavailable, preserve `None`/unknown rather than inventing generation zero.

## Config currentness

For config transitions additionally bind:

```text
subject S
+ config target T
+ exact pre-image digest A
+ candidate digest C
+ generation/state predicate G?
```

#4934 provides the first filesystem pre-image enforcement layer.

## Privacy

Machine identity is potentially correlating metadata.

Rules:

- do not publish raw machine identifiers through Mycelix/network-federation receipts by default;
- support audience/profile-specific pseudonymous or application-specific subject references;
- keep local stable identity separable from public/federated aliases;
- a disclosure alias cannot recreate a stronger hidden hardware/device identity claim.

## Xenia composition

Xenia may bind an authorization permit to a subject/resource identity, but Nixward must validate that the permit's subject profile matches the local subject it is about to mutate.

```text
valid Xenia signature
!= permit applies to this local machine
```

The binding must cover exact subject/profile identity and state predicates.

## Spore composition

Spore bootstrap/recovery should be able to operate before a long-lived identity exists.

Support a staged lifecycle:

```text
Unprovisioned
  -> BootstrapIdentity
  -> ProvisionedInstallationIdentity
  -> optional Device/Controller binding
```

Do not force installer/recovery media to reuse the installed system's authority identity merely to inspect or restore it.

## Network Twin composition

Network Twin node/resource identity may map to a Nixward machine subject only through an explicit mapping/binding receipt.

Matching hostname, IP address, MAC address, topology label, or friendly node name is insufficient identity equivalence.

## First qualification corpus

Freeze deterministic fixtures for:

1. same subject + generation G -> exact subject identity stable;
2. same subject + generation G+1 -> subject stable while state predicate changes;
3. hostname rename -> subject identity unchanged under profiles that do not treat hostname as identity;
4. fresh install on same device -> new installation identity;
5. cloned template without reprovisioning -> duplicate/ambiguous subject detected under profile;
6. explicit recovery continuity receipt -> permitted binding where profile allows it;
7. raw machine-id cannot be serialized into a public sanitized receipt by default;
8. boot identity changes after reboot while stable machine-instance identity remains;
9. Xenia permit for subject A cannot authorize subject B even with identical hostname;
10. Network Twin node label match alone cannot establish Nixward subject mapping;
11. missing identity input remains unknown/unsupported rather than synthesized;
12. reinstall migration receipt preserves old and new identities rather than rewriting history.

## Exit gate

- identity dimensions and lifecycle rules frozen;
- generation explicitly remains state/currentness, not identity;
- hostname and raw machine-id are not universal canonical identity;
- clone/reinstall/recovery semantics explicit;
- privacy/disclosure profile defined;
- Xenia, Spore, and Network Twin mappings remain explicit boundaries;
- deterministic fixtures qualify before privileged daemon integration claims machine-bound authorization.
