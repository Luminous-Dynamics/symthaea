# symthaea-assurance-trust-store-recovery

Evidence model for completing a reviewed replacement of a failed assurance trust store.

The parent `symthaea-assurance-trust-store` crate determines whether recovery is *eligible*. This crate proves that the replacement actually preserved continuity:

- same logical trust-store identity,
- exact previous checkpoint and backup digests,
- exact reviewed recovery authorization,
- authorized replacement trust-store reference and fresh counter epoch,
- independently evidenced replacement-store attestation,
- restored anchor digest and policy revision no weaker than the pre-loss state,
- first replacement checkpoint linked to the exact old checkpoint digest,
- no retroactive or expired recovery completion.

It does not perform hardware I/O and does not claim a TPM/HSM/secure-element implementation. A concrete adapter should supply the attestation and counter evidence consumed here.

This crate never grants physical authority.
