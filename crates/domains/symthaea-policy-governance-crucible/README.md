# symthaea-policy-governance-crucible

Adversarial qualification suite for the complete assurance-policy governance chain.

The crucible exercises the real anchored + signer-governed readiness stack and requires reviewed outcomes for both positive and negative controls.

Covered attacks/failures include:

- truncated signed-lineage rollback,
- deleted intermediate policy revision,
- policy-threshold change without requalification,
- unreviewed signer/key replacement,
- reviewed signer/key rotation,
- signer-governance policy substitution,
- old external-anchor substitution,
- uncheckpointed forward policy revision,
- policy-manifest substitution outside the signed lineage,
- and late external checkpointing.

The report passes only when every scenario produces its reviewed `Ready`, `Blocked`, or `Invalid` outcome. It is descriptive test evidence only and never grants physical authority.
