# REA typed claim evaluator v0

This is the first executable claim-evaluation slice for issue #10629. It deliberately supports only one positive rule:

- provider_reported_observation: a selected, semantically verified REA Evidence record reports an JCS-canonical normalized_result for the same subject digest, operation and represented environment scope.

The evaluator re-runs verify_evidence_value for every supplied premise. It does not trust a caller-constructed VerifiedEvidence struct or an unverified Evidence ID. Claim JSON import rejects duplicate object keys recursively, including inside asserted_value.

## Dispositions

- supported_within_scope: at least one applicable Evidence record has exact normalized-result equality, and no applicable record contradicts it.
- refuted_within_scope: applicable records exist and all differ from the asserted value.
- conflicted: applicable same-scope records both match and differ.
- underdetermined: referenced evidence is missing or the requested scope contains a dimension not established by REA Evidence v1.
- inapplicable: all evaluated evidence is outside the stated subject, operation, predicate or represented scope.
- unverified: no predicate-specific rule is registered, or unverified assumptions are present.

V0 compares normalized_result and asserted_value by RFC 8785 JCS canonical bytes. This gives JSON-number equivalence under the binary64/JCS rules (for example, 42 and 42.0 canonicalize identically) while preserving array order and string code points. This remains structural JSON equality, not domain-semantic equivalence; it does not interpret arbitrary normalized-result schemas.

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

The derivation commits to the complete canonical claim request, not merely to the final disposition. It carries the request SHA-256 and reproduces the asserted value, source predicate type, ordered Evidence ID list and assumptions in the result body. Two different assertions that both fail to match the observed normalized result therefore cannot collapse to the same derivation commitment.

Raw claim import also requires every schema-required field to be present, including nullable source_predicate_type and the evidence_ids / assumptions arrays. Serde defaults do not make omitted request fields silently equivalent to explicit null or empty arrays.

## Versioned schemas

- Request: `docs/research/rea-claim-evaluation-v0.schema.json` (`luminous.rea.claim.v0`).
- Result: `docs/research/rea-claim-evaluation-result-v0.schema.json` (`luminous.rea.claim-evaluation.v0`).

The result schema requires the complete asserted value, source predicate, ordered requested Evidence IDs, assumptions, per-premise assessments, missing IDs, unmet obligations, claim ceiling, canonical request digest and derivation digest. Consumers should validate the result against the result schema before treating it as a portable derivation record.

### Confidence and authority are preserved, not promoted

The v0 positive rule is intentionally record-level: it qualifies that the semantically verified REA Evidence record contains the exact normalized value under the requested scope. It does **not** require the record's `confidence` to be `observed` or its `authority` to be a particular class, because doing so would silently turn one generic record-content predicate into an authority policy.

Instead, every EvidenceAssessment carries the record's confidence and authority verbatim, and the claim ceiling says the result does not prove the underlying proposition. A record labelled `confidence: inferred` and `authority: analyst-inference` can therefore support only the narrow statement that the record reports that value; the assessment must retain those labels. It cannot be promoted to an observed artifact fact, authenticated producer statement, independent corroboration or trusted conclusion. Any stronger rule must declare its confidence/authority preconditions in its own predicate-specific verifier and policy contract.
