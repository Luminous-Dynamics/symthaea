# symthaea-assurance-tpm-ima-pcr-binding

Pure capability theorem connecting verified TPM2 quote evidence to independently replayed Linux IMA state.

The bridge accepts only the non-serializable `VerifiedTpm2Quote` and `VerifiedImaReplay` capabilities produced by the existing raw-quote and IMA-replay gates. A reviewed policy pins the exact upstream quote-policy digest, exact IMA-policy digest, and the required PCR index.

Qualification requires the IMA replay to target that exact PCR, the verified TPM quote to contain that PCR, and the TPM-authenticated PCR value to exactly equal the independently replayed IMA PCR. Only then is non-serializable `VerifiedTpmImaPcrBinding` minted.

The binding explicitly carries the quote challenge, AK, quote artifact, verification receipt, PCR selection, IMA measurement-list digest, matched required measurements, upstream qualification digests, and quote timing evidence for later provider composition.

This crate is deliberately pure: it does not consume a one-shot challenge, mutate an acceptance ledger, establish platform/EK trust, establish trusted time, prove executable dependency-closure completeness, provide runtime continuity, establish readiness, or grant physical authority.
