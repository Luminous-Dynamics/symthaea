# TPM / Reference End-to-End Crucible

This crate composes the real TPM/reference assurance APIs into one fail-closed chain.

It recomputes:

1. TPM/platform/runtime qualification from reviewed counter observations;
2. fresh nonce-bound AK possession;
3. ordered measured-boot replay to the fresh quote;
4. measured-state evaluation against signed/versioned reference integrity; and
5. the authorized current reference-manifest lineage.

The campaign contract is content-addressed at two levels:

- `TpmReferenceChainPolicy::policy_digest()` identifies the exact top-level campaign semantics, including expected adapter/store/runtime subject and campaign freshness windows;
- `lower_policy_bundle_digest()` identifies the exact adapter, platform, possession, replay, reference-integrity, and reference-lineage policies, including verifier/tool identities, Final Events requirements, unknown-measurement policy, current RIM tip, and signer/key revision ranges.

A lower policy cannot retain the same textual `policy_id` while silently weakening these semantics. Any content change changes the bundle digest and fails the externally pinned campaign policy before lower qualification begins.

The composition also checks cross-layer bindings that the individual crates cannot establish alone, including:

- exact runtime subject continuity;
- exact source event-log/final-events continuity into normalized measurements;
- exact current manifest-tip continuity;
- exact current signature-receipt continuity, preventing valid-signature splicing;
- current freshness of platform, possession, replay, and reference evidence.

A passing report content-addresses both policy layers and all accepted evidence facets. It is assurance evidence only. It does not prove general TPM correctness, runtime integrity after measured boot, absence of compromise, or physical authority.
