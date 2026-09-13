# TPM / Reference Cross-Stage Verifier Diversity

This bridge closes the common-cause gap between two verification stages that are intentionally represented separately:

1. campaign recomputation of the complete TPM/reference chain; and
2. obligation-specific review that emits a canonical `SafetyEvidenceReceipt` for DA-038 through DA-042.

The base TPM/reference evidence bridge already requires different verifier identities. This crate strengthens that rule by reusing the canonical `VerifierFaultDomainProfile` from `symthaea-evidence-verifier-diversity` and applying a reviewed cross-stage policy to organization, review-process, toolchain, and evidence-source fault domains.

A policy may require any or all of those domains to differ. Missing or malformed profiles fail closed. A shared domain required to be distinct blocks qualification. The obligation review must also remain within the reviewed time bound after campaign recomputation.

A successful qualification emits a canonical safety receipt whose evidence digest is the digest of the cross-stage qualification itself. That digest content-binds the original TPM/reference candidate, the reviewed diversity policy, both reviewed verifier profiles, and the obligation-specific verification record.

This layer does not replace the later active-receipt verifier-diversity/deep-readiness gate. It closes the otherwise-hidden relationship between the campaign verifier and obligation verifier before the receipt enters the generic readiness pipeline.

Assurance only. It grants no physical authority.