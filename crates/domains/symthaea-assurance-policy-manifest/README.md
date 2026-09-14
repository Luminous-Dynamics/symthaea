# symthaea-assurance-policy-manifest

Versioned provenance for the policies that define what counts as acceptable safety evidence.

The manifest binds an exact safety contract and deployment/configuration/model/calibration state to exact reviewed policy digests for:

- evidence lifecycle
- deployment scope
- evidence quarantine
- trusted time
- requalification
- verifier diversity
- atomic evidence coverage
- evidence dependency / anti-circularity

A manifest has a deterministic BLAKE3 content digest and explicit predecessor lineage. Revision changes therefore produce a new manifest identity instead of silently changing the rules underneath existing readiness evidence.

Cryptographic signature verification is intentionally external. `ManifestSignatureVerificationReceipt` records the exact manifest digest, signer/key/signature references, verification process, and verification time. The crate does not pretend that the presence of a signature string proves cryptographic validity.

`PolicyScopedSafetyReceipt` binds an already deployment-scoped safety receipt to one exact verified manifest digest. Changing a policy digest, safety contract, deployment configuration, model manifest, calibration manifest, or manifest revision therefore makes the old receipt inapplicable to the new manifest.

This crate never discharges a safety obligation and never grants physical authority.
