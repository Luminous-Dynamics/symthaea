# Current Xenia transport revalidation

This crate performs the one-shot transition from `ExactXeniaTransportEvidence` to `RevalidatedXeniaTransport` under independently anchored **current** Xenia transport trust.

`CurrentXeniaTransportRevalidator` owns a verified `TransportTrustRegistry` plus the exact independently retained `TransportTrustHead`. Construction fails unless those heads match. Revalidation then consumes both the guard and the exact-evidence capsule, obtains wall-clock time internally, and calls the existing fixed `verify_xenia_physical_effect_receipt` implementation over the retained canonical bytes.

There is no caller-selectable crypto provider and no caller-selectable production timestamp.

For physical-effect lineages, trust generation is strict: the original `TransportTrustHead` retained in exact evidence must equal the independently anchored current head. Any successor generation therefore requires fresh Xenia submission even when the same transport key remains active.

The output is opaque, non-clone and non-serializable. It drops the portable raw receipt/payload bytes and retains only the newly verified transport proof, the exact-evidence commitment, revalidation time and signed receipt expiry.

`RevalidatedXeniaTransport` is **not authority**. Its `valid_until_unix_ms` is evidence rather than a lease. Corrected final composition must independently obtain current time again and combine this proof with current semantic/controller/interlock state before any JIT/HAL boundary is considered.
