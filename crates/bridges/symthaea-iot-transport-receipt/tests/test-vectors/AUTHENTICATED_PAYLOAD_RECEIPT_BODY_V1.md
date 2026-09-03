# Xenia authenticated payload receipt body v1 neutral wire vector

This is the Symthaea relying-party copy of the neutral Xenia receipt-body wire oracle. The literal `.hex` file must remain byte-identical to the vector carried by the Xenia qualification branch.

The expected bytes were independently constructed from bincode 1.3's documented helper-function format, not emitted by either repository.

Typed values:

- `schema`: `xenia-authenticated-payload-receipt-v1`
- `attestor_id`: `xenia-host-a`
- `key_id`: `transport-attestor-1`
- `signature_algorithm`: `ed25519-rfc8032+ml-dsa-65-fips204`
- `session_evidence_digest`: bytes `0x01..=0x20`
- `peer_role`: `Viewer` (variant discriminant 1)
- `peer_identity_fingerprint`: bytes `0x21..=0x40`
- `transcript_hash`: bytes `0x41..=0x60`
- `session_context_hash`: bytes `0x61..=0x80`
- `telemetry_enabled`: `true`
- `input_control_enabled`: `false`
- `payload_type`: `0x70`
- `payload_len`: `0x00001234`
- `payload_digest`: bytes `0x81..=0xA0`
- `sealed_envelope_digest`: bytes `0xA1..=0xC0`
- `opened_at_unix_ms`: `0x0102030405060708`
- `expires_at_unix_ms`: `0x01020304050617E9`

Frozen neutral result:

- canonical body length: **354 bytes**
- SHA-256 canonical body: `3b740e18f66fc89b2deeadfdba406bf91d9d59d2dd837d0230abd4b171a05c8d`
- SHA-256 of `b"xenia-authenticated-payload-receipt-v1\0" || canonical_body`: `cc0ddd150502e1864305643a204ce36f2ebbcfcb06c71db9017e691c4f642e86`

The test requires Symthaea's mirrored type to serialize byte-for-byte to the same literal vector and to round-trip back to the same typed values. This proves wire-shape compatibility only; trust, signatures, freshness, device semantics and physical authority remain independent gates.
