# Typed multiplex connectome claims V0.1

This directory specifies a relation-type boundary across anatomical synapses, reconstruction correspondence, statistical dependence, functional association and causal-effect claims.

- Protocol: `CONNECTOME_TYPED_CLAIM_QUERY_PROTOCOL_V0.1.md`
- Schema: `connectome-typed-claim-bundle-v0.1.schema.json`
- Known-answer fixture: `fixtures/connectome-typed-claim-bundle-known-answer-v0.1.json`
- Research issue: [#10349](https://github.com/Luminous-Dynamics/symthaea/issues/10349)

The JSON Schema validates shape and relation-specific evidence profiles. The semantic verifier requires endpoint-contiguous ordered paths and standalone causal-effect claims; mixed anatomical/statistical evidence cannot promote causality. The schema itself does not authenticate evidence, validate statistical-family completeness or establish causal identification. All `CLAIMED_*` statuses are producer assertions.
