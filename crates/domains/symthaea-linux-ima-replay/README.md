# symthaea-linux-ima-replay

Fail-closed replay of canonical Linux IMA runtime measurements for the first real-provider tranche of ASSURE-RUNTIME-002 (#2875).

This crate deliberately targets one narrow, reviewable profile:

- canonical little-endian `binary_runtime_measurements_sha256` framing;
- SHA-256 template digests and PCR extends;
- reviewed `ima-ng` and `ima-sig` templates only;
- SHA-256 `d-ng` event digests;
- bounded record/template/field resources;
- independent recomputation of each template digest from exact binary template data;
- exact ordered PCR replay against an externally supplied expected PCR value; and
- optional required content measurements, with event-name digests usable only in conjunction with content digests.

Malformed/truncated framing, unsupported templates, template-hash substitution, malformed `d-ng`/`n-ng` fields, and resource-limit violations are invalid. Missing required content or a replayed-PCR mismatch is blocked. Only exact agreement mints the non-serializable `VerifiedImaReplay` capability.

This crate does **not** authenticate the expected PCR. A later TPM2 quote adapter must supply that value from a verified fresh quote. It also does not claim trusted time, TPM/firmware/kernel correctness, process identity, readiness, or physical authority.
