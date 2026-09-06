# Effect Reconciliation Challenge Wire Format v1

This document defines the normative cross-language byte representation of
`EffectReconciliationChallengeV1`.

The challenge is **not authority to actuate** and is **not evidence that a physical effect occurred**.
It only identifies one exact rollback-protected unresolved attempt for fresh device-class-specific
outcome evidence.

## Constants

Wire magic, exact bytes including the trailing NUL:

`SYMTHAEA-IOT-RECON-CHALLENGE-V1\0`

Challenge digest domain, exact bytes including the trailing NUL:

`symthaea-iot-effect-reconciliation-challenge-v1\0`

Schema version: `1`.

Maximum challenge lifetime: `5000 ms`.

All multi-byte integers are unsigned and encoded **big-endian**.

All strings are UTF-8 byte strings encoded as:

```text
u32 byte_length, big-endian
exact UTF-8 bytes
```

The byte length is the UTF-8 encoded byte count. No Unicode normalization, case folding, trimming,
or locale transformation is performed by the wire encoder. Identity canonicalization must already
have happened before the challenge is constructed.

## Canonical field order

```text
wire_magic: fixed bytes
schema_version: u16
nonce: [u8; 32]
journal_generation: u64
journal_digest: [u8; 32]
correlation_digest: [u8; 32]
command_digest: [u8; 32]
envelope_digest: [u8; 32]
composition_digest: [u8; 32]
device: length-prefixed UTF-8
operation: length-prefixed UTF-8
executor: length-prefixed UTF-8
sequence: u64
adapter_id: length-prefixed UTF-8
source_state_tag: u8
source_state_payload: conditional
attempt_common_fenced_at_unix_ms: u64
attempt_wall_valid_until_unix_ms: u64
issued_at_unix_ms: u64
expires_at_unix_ms: u64
```

Source-state tags are normative:

```text
0 = Prepared
    payload: none

1 = AdapterAcknowledged
    payload: adapter_evidence_digest [u8; 32]

2 = AdapterIndeterminate
    payload: none
```

Any other source-state tag is invalid for v1.

`AbandonedBeforePort` has no wire tag because that state must never produce a reconciliation
challenge.

## Digest

Let `wire` be the exact canonical byte sequence above.

The challenge digest is:

```text
BLAKE3(
    "symthaea-iot-effect-reconciliation-challenge-v1\0"
    || u64_be(len(wire))
    || wire
)
```

A verifier must never hash a separately decoded/re-serialized structure and assume equivalence.
It must verify the exact locally issued challenge or reproduce this canonical encoding exactly.

## Freshness

A challenge is fresh only when:

```text
now_unix_ms >= issued_at_unix_ms
and
now_unix_ms < expires_at_unix_ms
```

The lifetime must be non-zero and no greater than 5000 ms.

Challenge issuance may occur after the original actuation deadline, because reconciliation is not
actuation. However:

```text
issued_at_unix_ms >= attempt_common_fenced_at_unix_ms
```

must hold so a regressed wall clock cannot make reconciliation appear to precede the original
linearized attempt.

## Golden vector v1

The following deterministic example is for cross-language encoder qualification only. Production
challenges obtain their nonce and issue time from guard-owned runtime sources.

```text
schema_version                    = 1
nonce                             = a1 repeated 32 bytes
journal_generation                = 7
journal_digest                    = 11 repeated 32 bytes
correlation_digest                = 22 repeated 32 bytes
command_digest                    = 33 repeated 32 bytes
envelope_digest                   = 44 repeated 32 bytes
composition_digest                = 55 repeated 32 bytes
device                            = "iot:valve:72"
operation                         = "open"
executor                          = "gateway:a"
sequence                          = 41
adapter_id                        = "hal:valve-72"
source_state                      = AdapterAcknowledged
adapter_evidence_digest           = 66 repeated 32 bytes
attempt_common_fenced_at_unix_ms  = 1000000
attempt_wall_valid_until_unix_ms  = 1001000
issued_at_unix_ms                 = 2000000
expires_at_unix_ms                = 2005000
```

Canonical wire length: **360 bytes**.

Canonical wire hex:

```text
53594d54484145412d494f542d5245434f4e2d4348414c4c454e47452d5631000001a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a10000000000000007111111111111111111111111111111111111111111111111111111111111111122222222222222222222222222222222222222222222222222222222222222223333333333333333333333333333333333333333333333333333333333333333444444444444444444444444444444444444444444444444444444444444444455555555555555555555555555555555555555555555555555555555555555550000000c696f743a76616c76653a3732000000046f70656e00000009676174657761793a6100000000000000290000000c68616c3a76616c76652d373201666666666666666666666666666666666666666666666666666666666666666600000000000f424000000000000f462800000000001e848000000000001e9808
```

An implementation that does not reproduce these 360 bytes exactly is not wire-compatible with v1.

## Verification boundary

A later device-class verifier should bind signed outcome evidence to the exact challenge bytes or
challenge digest and then re-establish **current** device trust/policy. A later journal-closing layer
must additionally prove that the current rollback-protected journal head still equals the challenge
journal head before writing any terminal reconciliation state.

Neither this wire format nor a syntactically valid challenge grants physical-effect authority.
