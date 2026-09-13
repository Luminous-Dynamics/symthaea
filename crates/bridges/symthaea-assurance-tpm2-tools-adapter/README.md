# symthaea-assurance-tpm2-tools-adapter

Read-only Linux TPM 2.0 adapter for the monotonic assurance trust-store contract.

This crate deliberately does **not** provision, increment, reset, undefine, or otherwise mutate TPM state. It executes only `tpm2_nvreadpublic` and `tpm2_nvread` through a shell-free executor, validates that the configured NV index is an 8-byte TPM counter, and binds the observed counter value into Symthaea trust-store evidence.

The adapter requires explicit absolute tool paths plus reviewed BLAKE3 digests, an explicit device TCTI (`device:/dev/tpmrm0` or `device:/dev/tpm0`), an exact NV handle, and an externally reviewed counter epoch. A successful hardware read is still **not** an attestation verification receipt; independent attestation verification remains a separate assurance layer.

Production Linux execution is available through `SystemTpm2ToolsExecutor`; tests use a fake executor and therefore do not claim real TPM hardware qualification.

The crate never grants physical authority and contains no targeting, interception, firing, jamming, spoofing, weapon-control, or engagement-optimization logic.
