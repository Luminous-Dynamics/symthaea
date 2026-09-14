# symthaea-assurance-tpm2-tools-adapter

Read-only Linux TPM 2.0 adapter for the monotonic assurance trust-store contract.

This crate deliberately does **not** provision, increment, reset, undefine, or otherwise mutate TPM state. It executes only `tpm2_nvreadpublic` and `tpm2_nvread` through a shell-free executor, validates that the configured NV index is an 8-byte TPM counter, and binds the observed counter value into Symthaea trust-store evidence.

The reviewed policy binds an explicit device TCTI (`device:/dev/tpmrm0` or `device:/dev/tpm0`), exact NV handle, exact TPM NV Name, externally reviewed counter epoch and rollback floor, absolute `tpm2-tools` paths, and BLAKE3 digests of the exact executables. The Linux executor removes `TPM2TOOLS_TCTI`, `LD_PRELOAD`, and `LD_LIBRARY_PATH` before execution so environment overrides cannot silently replace the reviewed transport or dynamic-loader configuration.

The NV Name is treated as part of the public-identity evidence for the index. The adapter also requires the public attributes to identify the index as a counter and the public data size to be exactly eight bytes before it accepts a counter read.

A successful read is **not** a TPM attestation-verification receipt and does not establish physical TPM identity, firmware correctness, boot integrity, or TPM implementation correctness. Those remain separate evidence obligations. Likewise, hashing the executable bytes does not yet prove the complete dynamic-link or Nix closure; a later qualification tranche should bind the exact runtime closure and platform identity.

The v1 adapter accepts no authentication secret and creates no TPM authorization/policy sessions. Deployments that need secret-bearing authorization or richer TPM policies require a separately reviewed adapter rather than placing credentials on a process command line.

Production Linux execution is available through `SystemTpm2ToolsExecutor`; ordinary tests use a fake executor and therefore qualify parser/command/binding semantics only. They do **not** claim real TPM hardware qualification.

The crate never grants physical authority and contains no targeting, interception, firing, jamming, spoofing, weapon-control, or engagement-optimization logic.
