# Current Xenia transport revalidation v0.2

This crate performs the one-shot transition from `ExactXeniaTransportEvidence` to `RevalidatedXeniaTransport` under independently anchored **current** Xenia transport trust.

`CurrentXeniaTransportRevalidator` owns a verified `TransportTrustRegistry` plus the exact independently retained `TransportTrustHead`. Construction fails unless those heads match. Revalidation consumes both the guard and the exact-evidence capsule, obtains wall-clock time internally, and calls the existing fixed `verify_xenia_physical_effect_receipt` implementation over the retained canonical bytes.

There is no caller-selectable cryptographic provider and no caller-selectable production timestamp. The original transport generation must equal the independently anchored current generation; any successor generation forces fresh Xenia submission even if the same attestor key remains active.

## Natural-expiry correction

Head equality alone is not enough for a later physical boundary. The exact selected transport key and the current transport-trust snapshot can expire naturally without changing the trust head.

After fixed current verification succeeds, v0.2 locates the exact `(attestor_id, key_id)` record selected by the signed receipt and retains its domain-separated key-record digest. It also retains three exclusive deadlines:

- signed receipt expiry;
- selected transport-attestor key `not_after`;
- current transport-trust snapshot expiry.

`valid_until_unix_ms` is the minimum of those three deadlines. This is evidence, not an authorizing lease. A later final/JIT boundary must obtain current time and current trust again, require the same transport generation and exact key-record identity/digest, and fail closed if any current revocable state changed or any deadline elapsed.

The output is opaque, non-clone and non-serializable. It drops the portable raw receipt/payload bytes. It creates no final permit, JIT capability, HAL handle or claim that a physical effect occurred.
