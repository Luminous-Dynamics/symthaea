# Symthaea IoT Actuation Guard — Admission Reservation

This crate is the crash-durable **pre-semantic** sequence journal for the privileged IoT guard.

A successful reservation means only:

- Xenia transport was already verified by an upstream fixed verifier;
- the envelope passed static local device-policy checks;
- its command sequence and exact envelope/receipt/trust commitments were atomically written and read back from durable storage.

It does **not** mean the device is currently healthy, observations are safe, the command is semantically accepted, a controller interlock is valid, or physical I/O may occur.

The journal uses a pinned Linux directory, owner-only files, safe cross-process `File::lock`, `O_NOFOLLOW`, temp-file `sync_all`, atomic rename, directory `sync_all`, and exact read-back.

Its returned `AdmissionReservationHead` is suitable for later independent anchoring. This filesystem implementation intentionally does not claim hardware anti-rollback against a privileged attacker restoring older disk state.
