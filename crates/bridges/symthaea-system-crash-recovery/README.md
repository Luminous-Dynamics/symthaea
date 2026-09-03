# Symthaea System Crash Recovery

Fail-closed recovery coordinator for the durable Agency Kernel systemd vertical
slice.

## Goal

A process crash may reduce availability. It must not:

- recreate capability authority;
- erase or release an uncertain external effect;
- turn missing attempt evidence into permission to retry;
- trust a locally advanced anti-rollback frontier merely because it is local;
- reuse the pre-crash Xenia authorization as a new execution grant.

## Protocol

```text
independently trusted CheckpointHead
              |
              v
load + re-hash SQLite Agency Kernel frontier
              |
        exact equality?
          /       \
        no         yes
        |           |
    CONTAIN       verify checkpoint against exact grant
                    |
                    v
             scan local attempt evidence read-only
                    |
          validate grant/plan/world/reservation bindings
                    |
                    v
         every stranded Reserved -> OutcomeUnknown
                    |
                    v
              CAS one successor
                    |
                    v
               re-read frontier
                    |
                    v
          QuiescentNoAuthority + final head
                    |
                    v
       publish final head to independent trust domain
                    |
                    v
       obtain fresh Xenia authority for new effects
```

The coordinator has no `ServiceBackend`, no restart method, and no capability
verification entry point. Its successful output is `QuiescentNoAuthority`.

## Monotone recovery

The only runtime-accounting mutation this crate performs is:

```text
Reserved -> OutcomeUnknown
```

Both states remain fully charged, so recovery cannot increase remaining use or
risk capacity.

Local `ProvenNotDispatched` evidence is not sufficient to release a reservation
after restart. The current attempt-evidence journal is locally hash chained, but
its live head is not yet independently anchored across process loss.

## External-head publication gap

If recovery advances the SQLite checkpoint frontier, the returned
`external_anchor_update_required` flag is true. The new `final_head` must be
durably acknowledged by the external trust domain before fresh authority is
accepted.

If the process crashes after the local CAS but before that publication, the old
trusted head no longer equals the local frontier. The next recovery attempt
fails closed with `TrustedHeadMismatch`.

This intentionally trades availability for anti-rollback safety. A later
protocol can make local-CAS + external-head publication a witnessed two-phase
ceremony, but v0.1 does not pretend those stores are atomic.

## Attempt cross-checks

Before any recovery CAS, every incomplete attempt must:

- bind the exact capability grant digest;
- bind the exact plan digest required by the grant;
- bind the exact pre-effect world digest required by the grant;
- match one reservation already present in the authenticated checkpoint;
- not claim checkpoint sequences from the future;
- not duplicate another incomplete attempt for the same reservation;
- represent a crash cut compatible with the current broker/evidence ordering.

These checks are performed before the durable runtime state is changed.

## Non-claims

This crate does not yet provide:

- a concrete external checkpoint-head trust provider;
- Xenia/TPM/remote-witness head publication;
- signed or remotely witnessed attempt-evidence heads;
- trusted authority time;
- read-only systemd effect reconciliation after reboot;
- automatic compensation or retries;
- measured workload attestation;
- hardware power-loss guarantees for SQLite;
- fresh post-recovery Xenia authorization.

Those remain independent trust boundaries and later composition work.
