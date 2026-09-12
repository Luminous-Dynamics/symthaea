# symthaea-assurance-policy-lineage

Validates a contiguous, signed lineage of assurance-policy manifests.

A revision number and predecessor digest are not sufficient on their own: the predecessor must actually be present as the immediately previous revision, every revision must preserve the same manifest/deployment identity, and every manifest must have a matching external signature-verification receipt.

The assessor rejects gaps, duplicate revisions, duplicate manifest digests, invalid manifests/signatures, manifest-id or deployment-id changes inside one lineage, and predecessor-hash mismatches.

Configuration, model, calibration, safety-contract, and policy digests may change across reviewed revisions; those changes are already part of each manifest's exact content digest and require the manifest's reviewed change record.

This crate never establishes deployment readiness and never grants physical authority.
