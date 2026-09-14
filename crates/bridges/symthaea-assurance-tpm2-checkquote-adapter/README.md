# symthaea-assurance-tpm2-checkquote-adapter

Raw TPM2 quote verification producer for the existing `symthaea-assurance-tpm2-attestation-possession` semantic gate.

This crate deliberately separates two verification paths:

1. a reviewed, exact-digest-pinned `tpm2_checkquote` executable performs the cryptographic signature check; and
2. Symthaea independently parses the signed `TPMS_ATTEST` bytes and verifies the TPM-generated magic, quote attestation type, exact challenge nonce, exact AK qualified signer, exact SHA-256 PCR selection, and exact quoted PCR composite.

The first profile is intentionally narrow: one SHA-256 PCR bank, PCR indices 0..23, SHA-256 quote hashing, normalized PCR-value input, and an exact 32-byte challenge nonce.

The adapter emits the existing serializable `Tpm2QuoteArtifacts` and `QuoteVerificationReceipt` types only after both verification paths agree. It also mints a non-serializable `VerifiedTpm2Quote` capability that retains the exact authenticated structured PCR values for later TPM+IMA composition.

This crate does **not** establish AK manufacturer/EK-chain trust, trusted time, TPM/firmware correctness, verifier dependency-closure integrity, IMA replay, runtime continuity, readiness, or physical authority. Those remain separate gates.
