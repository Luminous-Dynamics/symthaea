# symthaea-evidence-independence-completeness

Relation-scoped completeness overlay for the bounded verifier-independence graph from ASSURE-015A.

The parent graph can establish ancestry and common causes, but its first tranche exposes one coarse `lineage_complete` bit per node. This crate leaves that subject unchanged and adds a stronger reviewed theorem:

```text
no shared ancestor
+ every policy-required relation closure is complete
-> Separated

known shared ancestor
-> Correlated

missing/incomplete required relation closure
-> Indeterminate
```

Completeness is asserted per `(node, relation)` pair. Policies choose the exact relation classes that matter for each independence axis, so uncertainty about an irrelevant relation does not poison a narrower theorem.

This is still evidence assurance only. It does not authenticate completeness claims, establish issuer trust, provide revocation/supersession/contradiction lifecycle, grant readiness, or confer physical authority.
