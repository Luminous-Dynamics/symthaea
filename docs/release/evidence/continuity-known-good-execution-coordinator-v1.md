# Continuity known-good execution coordinator v1 — qualification subject

Status: **implementation subject only; not qualified evidence**.

Exact branch:

`architecture/continuity-known-good-execution-coordinator-v1`

The exact implementation subject is the branch head containing:

- `execution_coordinator.rs` type-level durability gate;
- exact qualified-anchor predecessor accessor;
- public crate export surface;
- architecture theorem document.

No local or CI execution result is claimed by this record. A passing exact-head CI run must be attached separately before this subject is promoted from draft/implementation status.

The critical theorem under qualification is:

```text
A-bound trusted eligibility
  -> pending attempt exposes intent only
  -> exact intent durably reconstructed as spent
  -> fresh qualified journal anchor directly extends exact predecessor
  -> ready physical execution token
```

Failure of any exact journal, eligibility, anchor-root, predecessor, sequence, time, subject, trusted-epoch, or prepared-attempt binding must fail closed.
