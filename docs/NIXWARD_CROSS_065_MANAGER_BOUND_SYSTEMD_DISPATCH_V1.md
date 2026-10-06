# Nixward CROSS-065 — Manager-Bound systemd Lifecycle Dispatch V1

## Purpose

CROSS-065 closes a second-order systemd manager-incarnation race between:

1. pre-arming the `JobRemoved` watcher;
2. dispatching the lifecycle mutation.

Using the well-known D-Bus name `org.freedesktop.systemd1` for both steps leaves a rollover window: systemd could restart after the watcher captures one manager incarnation, and the later well-known-name method call could resolve to the replacement incarnation.

That would break the provenance relation between the observation channel and the effect dispatch.

## Exact-destination contract

The typed lifecycle mutation transport now exposes:

```
dispatch_lifecycle_for_manager_owner(operation, manager_owner)
```

The caller supplies the unique D-Bus owner captured by the read-only observer.

The transport first checks that the supplied unique name is a syntactically valid D-Bus unique name and that it currently owns `org.freedesktop.systemd1`.

The actual mutation method call is then sent to the **unique owner name**, not the well-known name.

The sequence is therefore:

```
systemd well-known owner lookup
        ↓
captured unique manager owner
        ↓
pre-armed JobRemoved watcher
        ↓
current-owner verification
        ↓
D-Bus call addressed to captured unique owner
        ↓
exact Job object path
```

## Rollover behavior

If systemd restarts after the owner check but before the lifecycle method call, the destination remains the old unique name.

Because that old connection identity is no longer the owner of systemd, the method call fails instead of being redirected to the replacement manager incarnation.

This closes the dangerous "observe old manager, mutate new manager" retargeting case without pretending the two D-Bus operations form an atomic transaction.

D-Bus unique names identify a single connection lifetime and are not reused for another connection on the same bus, which makes the unique destination suitable for preserving this incarnation distinction. citeturn117926search0turn117926search1

## Transport boundary

This module remains transport-only.

It:

- accepts only `NixServiceOperationV1`;
- maps Start/Stop/Restart/Reload to the native systemd Manager methods;
- uses the exact D-Bus Job object path returned by systemd;
- does not accept shell text or `systemctl` command material;
- does not accept authorization records;
- does not decide whether an effect is permitted.

The caller remains responsible for authorization, currentness, pre-state checks, watcher arming, Job capture, and post-state evidence.

## Job-path qualification

Returned Job paths must:

- be in the exact `/org/freedesktop/systemd1/job/` namespace;
- contain exactly one terminal path component;
- encode a non-zero `u32` Job ID.

Stronger identity correlation remains the observer's responsibility, where the Job ID is checked against the Job object path and native Job properties.

## Claim ceiling

CROSS-065 does **not** claim:

- atomic CAS semantics across D-Bus owner lookup and mutation dispatch;
- immunity from all concurrent service-state changes;
- cryptographic attestation of systemd;
- that a failed old-incarnation call can itself prove the absence of another effect.

Its specific guarantee is narrower and testable: a governed lifecycle transport call cannot silently follow the mutable systemd well-known name into a different manager incarnation after the caller has bound the call to a captured unique owner.

## Relationship to the preceding tranches

- CROSS-061 pre-arms `JobRemoved` before dispatch.
- CROSS-063 carries the systemd manager incarnation into Job evidence and receipts.
- CROSS-064 provides the native typed lifecycle mutation transport.
- CROSS-060 remains responsible for binding service-definition identity into authorization.
- CROSS-062 verifies stability using repeated observation sequences.

## Qualification

The tranche is not PASS merely because the branch is mergeable or the code is syntactically plausible.

Exact-head GitHub Actions must pass, including the checked-in Nixward boundary scan, focused formatting, and typed mutation transport tests.
