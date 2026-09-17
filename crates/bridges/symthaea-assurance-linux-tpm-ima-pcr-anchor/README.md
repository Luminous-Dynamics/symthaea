# symthaea-assurance-linux-tpm-ima-pcr-anchor

Narrow ASSURE-RUNTIME-002 composition boundary for Linux TPM2 + IMA evidence.

This crate does **not** infer runtime integrity from a PCR number or a serializable report. It composes two capability-bearing results:

- a raw TPM2 quote already verified by the pinned `tpm2_checkquote` adapter and independent TPMS_ATTEST parser;
- a canonical Linux IMA SHA-256 replay already verified against an expected final PCR.

Before minting `VerifiedLinuxTpmImaPcrAnchor`, it also re-runs the existing semantic attestation-possession theorem from the original platform qualification, challenge, AK binding, quote artifacts, and verification receipt. The re-qualified possession identities must match the opaque raw-quote capability.

The central theorem is intentionally small:

```text
exact checkquote policy commitment
+ exact possession policy commitment
+ exact IMA replay policy commitment
+ opaque VerifiedTpm2Quote
+ re-qualified semantic AK possession
+ opaque VerifiedImaReplay
+ selected quote-authenticated IMA PCR == independently replayed IMA PCR
-> VerifiedLinuxTpmImaPcrAnchor
```

The capability is non-serializable. Reports are evidence only.

## Non-claims

This crate does not establish:

- EK/manufacturer-chain trust;
- firmware or kernel correctness outside the measured-policy assumptions;
- trusted wall-clock time;
- absence of compromise between measurements;
- complete verifier executable/dependency/configuration coverage by itself;
- runtime checkpoint continuity;
- verifier independence;
- readiness, deployment permission, or physical authority.

A follow-on runtime-measurement binding must give semantic roles to the IMA policy's required measurements before this becomes the full provider boundary described by #2875.
