# symthaea-evidence-lineage-readiness

Final policy-provenance gate over Symthaea's policy-scoped deep evidence readiness.

A current assurance manifest is eligible for readiness only when:

- the complete signed policy lineage is valid,
- the current manifest is present in that lineage,
- the current manifest is exactly the lineage tip rather than an older signed revision,
- the signature-verification receipt used for readiness is exactly the receipt recorded at the lineage tip,
- every policy-scoped evidence receipt used by the current assessment names that same signature-verification receipt,
- and the underlying policy-manifest deep-readiness assessment remains ready.

A valid but older manifest is treated as a rollback/stale-policy condition and blocks readiness. A manifest absent from the lineage, an invalid lineage, signature-record splicing, duplicate evidence identities, or malformed inputs fail closed as invalid.

This crate recomputes the lower readiness assessment; it does not accept a caller-authored `Ready` report as proof.

The report is descriptive safety evidence only and never grants physical authority.
