# Local checkpoint witness journal v1

This package implements a single-writer, append-only checkpoint issuer for a pinned database ID and Ed25519 key ID. It is a local reference component for testing witness behavior; it is **not** a remote witness service or proof of administrative independence.

## Protocol

`Open` loads a caller-supplied Ed25519 private key into process memory, takes an exclusive Linux `flock` on the witness journal, and validates every signed record in sequence. The key is not read from or written to disk by this package. On non-Linux platforms, opening fails closed because the reference writer-lock primitive is Linux-only.

`Issue(ObservedHead)` rejects wrong database identity and malformed heads, journal rollback, same-sequence equivocation, and sequence-only movement without a new digest. An identical head retry returns the exact same signed checkpoint to recover from a lost acknowledgement. New journal progress increments the checkpoint sequence and predecessor digest. A backward issue clock is rejected before append. The record is appended and fsynced before acknowledgement; ambiguous write/sync/directory-sync outcomes poison the running instance until restart and reconciliation.

Replay rejects unterminated, malformed, tampered, wrong-identity, non-contiguous, broken-predecessor, non-advancing and clock-regressed histories. The private key is injected by the caller and retained in process memory only.

## Tests and trust limits

Tests cover genesis/successor signing, verifier interoperability, exact retry, equivocation, rollback, invalid observations, restart recovery, tampered signature, torn tail, exclusive writer lock, 32 concurrent identical requests, and clock rollback.

This is a **single-host local witness**: it shares the host kernel, storage and administrator. The caller supplies the key; key custody/HSM, policy, rotation and revocation are external. A copied private key can sign a fork outside this journal, and peer split-view detection still requires checkpoint exchange or a common monitor. The package provides neither gossip nor a transparency log. It does not prove that the observer saw every real-world effect or qualify physical power-loss behavior.
