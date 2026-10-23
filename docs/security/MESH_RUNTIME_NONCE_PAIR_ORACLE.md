# Runtime AEAD nonce-pair oracle

**Status:** test-only diagnostic harness and legacy witnesses; not remediation.

Related: #8376 (source audit), #8471 (bounded nonce model), #8508 (fragment AEAD/MTU containment), #8533 (clone model), #8534 (runtime implementation target), and #8544 (v3 design proposal).

## What the oracle checks

The oracle observes outputs from the real `encrypt_packet_typed` and `encrypt_fragment` helpers and tracks the invariant-relevant pair:

`(effective key bytes, emitted nonce bytes)`

It does not use a sender ID, epoch, sequence, or claimed session label as a substitute for the effective key. Keys remain only as in-memory test keys; the helper never logs, serializes, or prints them. For tests, retaining the exact key array as an in-memory map key avoids introducing a truncated identifier or a separate diagnostic hash as the safety oracle.

The nonce prefix is read from the actual encryption helper output. Thus the test observes what is emitted, rather than independently reimplementing the nonce constructor and testing that duplicate implementation against itself.

## Newly added executable witness: packet/fragment cross-API overlap

The packet helper creates the 12-byte nonce:

`source_id[0..6] | payload_type | epoch | sequence_le[4]`

The fragment helper creates:

`source_id[0..8] | thought_id_le[2] | fragment_index | 0`

These layouts can be byte-for-byte identical when the two APIs use the same effective key. The deterministic test in `src/swarm/mesh/tests.rs` uses:

- source bytes: `11 22 33 44 55 66 00 88`
- packet type: `0x00` (`WisdomVector`)
- packet epoch: `0x88`
- packet sequence: `0x00030201`, little-endian `01 02 03 00`
- fragment thought ID: `0x0201`, little-endian `01 02`
- fragment index: `3`

Both helpers therefore emit nonce bytes:

`11 22 33 44 55 66 00 88 01 02 03 00`

With a shared effective key, this is a repeated AEAD pair even though the plaintexts and calling APIs differ. The test also checks the ciphertext XOR relation caused by reuse. This is a **conditional code-path witness**, not a claim that every deployment shares a key across packet and fragment encryption.

The important design lesson is cross-domain: nonce uniqueness must be checked across every invocation that uses the same effective key, not within each helper in isolation. Separate, reviewed keys per purpose can prevent this particular cross-API collision; the implementation and key schedule must actually enforce that separation.

## Positive control

`runtime_nonce_pair_oracle_detects_collision_and_accepts_rekey_control` first records one packet invocation, then requires the oracle to detect a repeated `(key, nonce)` pair caused by truncated sender identity. It then sends the same nonce bytes under a different test key and requires no same-key collision. This guards against the bad metric of treating nonce bytes alone as the invariant, without regard to the effective key.

The test keys are retained only within process memory. Failure output identifies scenario labels, not key bytes.

## Evidence semantics

The two new tests and the older `legacy_*` tests are **legacy vulnerability witnesses**: they currently pass only if the documented legacy behavior is reproducible. They are not acceptance tests for a secure replacement. A future v3 implementation must add strict-path tests with the opposite expectation: the same hostile traces must be rejected or use a provably distinct key namespace, while fresh-session positive controls continue to work.

A witness being present in source is not execution evidence. Record Rust toolchain, exact commit, feature set, test command, exit code, output, and source digest. Do not promote this work to a security pass until the focused tests actually compile and run on the exact head and implementation-level strict tests exist.

## Benchmark contract

For each supported configuration, track at minimum:

- duplicate effective-key/nonce pairs observed;
- unique pairs and total actual AEAD invocations;
- legacy witness kill rate (registered defects reproduced for the intended reason);
- strict positive-control retention;
- false acceptance of a duplicate pair (target: zero);
- malformed/truncated encryption-output handling;
- packet and fragment API coverage;
- test runtime and memory overhead.

Release target: zero repeated `(effective key, nonce)` pairs over the declared strict-path fault corpus. The legacy detector itself must be tested using known collision witnesses; random sampling alone is not sufficient. If key freshness, rollback resistance, or sender exclusivity cannot be established, the runtime must fail closed rather than turn absence of collisions in a finite run into a guarantee.

This oracle is a test instrument, not a production nonce allocator, cryptographic proof, or replacement for independent review.
