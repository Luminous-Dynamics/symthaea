# REA typed claim evaluator v0

This is the first executable claim-evaluation slice for issue #10629. It deliberately supports only one positive rule:

- provider_reported_observation: a selected, semantically verified REA Evidence record reports an exact normalized_result for the same subject digest, operation and represented environment scope.

The evaluator re-runs verify_evidence_value for every supplied premise. It does not trust a caller-constructed VerifiedEvidence struct or an unverified Evidence ID. Claim JSON import rejects duplicate object keys recursively, including inside asserted_value.

## Dispositions

- supported_within_scope: at least one applicable Evidence record has exact normalized-result equality, and no applicable record contradicts it.
- refuted_within_scope: applicable records exist and all differ from the asserted value.
- conflicted: applicable same-scope records both match and differ.
- underdetermined: referenced evidence is missing or the requested scope contains a dimension not established by REA Evidence v1.
- inapplicable: all evaluated evidence is outside the stated subject, operation, predicate or represented scope.
- unverified: no predicate-specific rule is registered, or unverified assumptions are present.

V0 treats exact JSON equality as an observation comparison, not a semantic equivalence relation. It does not interpret arbitrary normalized-result schemas.

## Explicit non-claims

A supported provider-reported observation does not prove the underlying artifact proposition. This evaluator does not establish member-content integrity, whole-archive identity/completeness, producer authenticity, cryptographic signature validity, signer identity/trust, package authorization, platform acceptance, provider independence, causal correctness, benignness, runtime safety or MASVS control pass.

Those claim predicates are represented in the closed vocabulary but return unverified until a dedicated verifier and evidence contract is implemented. Unknown provider lineage is not treated as independent. Platform-version-specific claims remain underdetermined because REA Evidence v1 does not encode platform version in its environment contract.

## Deterministic derivation

The evaluator emits a derivation object with:

- exact claim ID, predicate and scope;
- the rule/version selected;
- each Evidence ID and its provider, confidence, authority and assessment;
- missing IDs and unmet obligations;
- a falsification condition and claim ceiling;
- SHA-256 over RFC 8785 canonical bytes of the derivation body, excluding the digest field itself.

The derivation digest is an integrity commitment, not an issuer signature or trust anchor.

## Tests

The module tests cover exact provider-report support, disagreement/refutation, same-scope conflict preservation, subject and operation mismatch, platform-version scope underdetermination, unimplemented predicate fail-closed behavior, missing evidence, assumptions, forged VerifiedEvidence metadata, recursive duplicate-key rejection and deterministic derivation commitments.

The isolated REA Evidence compatibility harness also compiles the root adapter and executes a root API smoke test. These tests are authored in this branch; do not represent them as passing until the exact-head workflow completes.
