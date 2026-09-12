# symthaea-trust-root-crucible

Adversarial qualification for the monotonic trust-root and recovery stack.

The crucible combines positive and negative controls for checkpoint monotonicity, backup/restore rollback resistance, exact recovery continuity, one-shot recovery acceptance, current recovered-segment selection, and independent attestation verification.

The report passes only when every reviewed scenario produces the fail-closed or recovery outcome expected by policy.

It is descriptive test evidence only. It does not verify real TPM/HSM cryptography, does not discharge the wider safety case, and never grants physical authority.
