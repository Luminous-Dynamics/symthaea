# symthaea-assurance-trust-store-attestation

Typed receipts for independent verification of trust-store checkpoint and recovery attestation evidence.

An `attestation_ref` is only a pointer to evidence. It does not prove that the evidence was checked. This crate records the result of an external verifier binding that evidence to the exact checkpoint/recovery state.

It verifies provenance and exact binding only; it does not implement TPM/HSM/secure-element cryptography or hardware I/O.

The verification actor may not be the trust store or hardware instance it is verifying. Receipts are evidence only and never grant physical authority.
