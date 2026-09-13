# symthaea-evidence-independence-provenance

Scoped cryptographic provenance for the verifier-independence assurance stack.

This crate authenticates the exact verifier-profile revision and the exact graph/completeness revisions it claims to describe. It deliberately separates:

```text
signature valid
!= issuer trusted
!= issuer authorized for this claim scope
!= fault-domain claim substantively true
!= verifier independent
```

A successful verification therefore produces historical/authenticity evidence only. Current lifecycle, revocation/supersession/contradiction semantics, trusted-time currentness, and independence assessment remain separate layers.

The trust policy is the explicit reviewed root for this protocol; the implementation does not recurse indefinitely by requiring another verifier profile for the issuer policy itself.
