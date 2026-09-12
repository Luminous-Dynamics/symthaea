# symthaea-trust-root-readiness-crucible

End-to-end qualification of the trust-root-backed readiness theorem.

Unlike the component trust-root crucible, this fixture begins from a genuinely `Ready` anchored/signer-governed safety-evidence stack and then composes the current trust-store segment and an independently verified checkpoint attestation.

Negative controls deliberately break the anchor/checkpoint binding, attestation receipt, trusted-time ordering, trust-store identity, and current-segment monotonicity. A reviewed recovered-segment positive control demonstrates that accepted recovery history can preserve readiness without weakening the trust-root gate.

The report is test evidence only. It does not perform real TPM/HSM/secure-element cryptography, cannot discharge the wider safety case, and never grants physical authority.
