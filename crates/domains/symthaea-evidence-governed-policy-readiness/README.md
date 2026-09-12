# symthaea-evidence-governed-policy-readiness

Top-level assurance composition that requires both:

1. policy-scoped deep readiness at the exact signed lineage tip, and
2. independent authorization of every manifest signer/key under a provisioned signing-authority governance digest.

The signing-authority governance object is not accepted merely because it is structurally valid. Its deterministic digest must equal the externally provisioned expected digest supplied by the deployment trust root.

This means an attacker cannot preserve readiness by swapping in a weaker authority policy that authorizes an otherwise-untrusted replacement key.

The gate recomputes both lower assessments and can only retain or reduce readiness. Invalid authority provenance or governance-digest substitution yields `Invalid`; ordinary lower assurance degradation remains `Blocked` or `Invalid` according to the lower reports.

The report never grants physical authority.
