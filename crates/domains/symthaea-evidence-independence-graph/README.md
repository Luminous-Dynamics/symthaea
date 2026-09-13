# symthaea-evidence-independence-graph

Bounded, reviewable common-cause lineage for verifier and evidence independence.

The existing `VerifierFaultDomainProfile` records direct organization, review-process, toolchain, and evidence-source domains. This crate adds a stronger additive layer: those domain IDs become entry points into an explicit reviewed DAG of control, operation, dependency, derivation, and governance ancestry.

The central rule is:

```text
different leaf IDs != independence
```

A policy-qualified pair is `Separated` only when every required lineage is structurally valid, sufficiently complete for the reviewed policy, and has no shared ancestor. A known common cause is `Correlated`. Missing ancestry or a policy depth bound produces `Indeterminate`; malformed graphs, cycles, profile/graph mismatches, and invalid policies are `Invalid`.

This tranche deliberately does not authenticate graph assertions, implement profile lifecycle, or grant readiness/physical authority. Those remain separate follow-on layers. The graph is bounded and content-addressed so future provenance/lifecycle machinery can bind the exact reviewed subject.
