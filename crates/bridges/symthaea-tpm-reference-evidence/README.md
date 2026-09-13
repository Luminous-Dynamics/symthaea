# TPM / Reference Candidate Evidence

This bridge converts a verified end-to-end `TpmReferenceChainReport` into typed **candidate evidence** for DA-038 through DA-042.

A structurally plausible report is not enough. Candidate creation is governed by a reviewed `TpmReferenceEvidencePolicy` that pins:

- the exact top-level TPM/reference campaign-policy digest;
- the exact lower-policy-bundle digest;
- the authorized campaign-recomputation verifier;
- the exact campaign-verification tool digest; and
- the maximum campaign-to-verification delay.

A separate `TpmReferenceCampaignVerification` must then bind the exact report digest, both policy digests, verifier/tool identity, verification time, and explicit confirmation that the full chain was recomputed and both policy identities were checked.

This campaign verification is still not deployment readiness and does not discharge any formal obligation. Each emitted candidate requires a second, obligation-specific review before a canonical `SafetyEvidenceReceipt` exists. The bridge enforces that this second review uses a **different verifier identity** from the campaign recomputation verifier and occurs no earlier than the campaign verification. Broader common-cause independence across verifier organization, process, tooling, or evidence source remains the responsibility of the verifier-diversity/deep-readiness layer rather than being inferred from two different verifier strings.

Each candidate content-binds the durable campaign reference, campaign digest, evidence-admission-policy digest, campaign-verification digest, campaign-verifier identity/time, qualification time, rationale, and the obligation-specific sub-artifacts. The bridge therefore does not treat one broad campaign report as implicit proof of every TPM/reference facet.

Bindings are atomic:

- DA-038: platform qualification + before/after TPM counter observations;
- DA-039: fresh AK-possession record;
- DA-040: measured-boot replay record;
- DA-041: reference evaluation + exact current manifest + exact current manifest-signature receipt;
- DA-042: reference lineage + exact current manifest + exact current manifest-signature receipt.

All evidence remains assurance-only and has zero physical authority.