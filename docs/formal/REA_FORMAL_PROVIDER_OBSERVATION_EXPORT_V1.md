# REA formal provider-observation export v1

## Purpose and scope

\`src/rea_compat/formal.rs\` projects a validated REA typed claim evaluation into a versioned formal-closure envelope. It is intentionally limited to the v0 meta-claim \`provider_reported_observation\`:

> The named provider reported this normalized result for this exact subject, operation, represented environment and request.

It does not turn that report into a claim that the artifact is intact, authentic, authorized, complete, benign, safe or accepted by a platform.

## Wire schema

- Request: \`luminous.rea.claim.v0\`
- Result: \`luminous.rea.claim-evaluation.v0\`
- Formal envelope: \`luminous.rea.formal-provider-observation.v1\`
- Mapping rule: \`rea-provider-observation-projection/v1\`
- Formal evidence class: \`provider_observation\`
- Formal envelope schema: \`docs/formal/rea-formal-provider-observation-v1.schema.json\`

The formal binding carries the entire request identity: request SHA-256, derivation SHA-256, request/result schemas, evaluator version/rule ID and numeric rule generation, claim ID, subject SHA-256, operation, optional platform/version, environment, asserted value, source predicate type, ordered Evidence IDs, assumptions and falsification condition.

The numeric generation field is an explicit wire value. For evaluator v0 it is \`0\`. It is not inferred from a textual version label. The evaluator version and rule ID are retained alongside it.

## Semantic mapping

| REA disposition | Formal semantic polarity | Meaning |
|---|---|---|
| \`supported_within_scope\` | \`positive_support\` | Supports only the provider-reported-observation meta-claim |
| \`refuted_within_scope\` | \`semantic_counterevidence\` | Counters only that exact provider-reported-observation meta-claim |
| \`conflicted\` | \`no_semantic_conclusion\` | Supporting and contradicting observations coexist |
| \`underdetermined\` | \`no_semantic_conclusion\` | Missing or unrepresented evidence/scope prevents conclusion |
| \`inapplicable\` | \`no_semantic_conclusion\` | The supplied records do not apply to the requested scope |
| \`unverified\` | \`no_semantic_conclusion\` | No authorized v0 rule or assumption verifier supports the claim |

The exporter rejects every predicate other than \`provider_reported_observation\`. It recomputes the request digest from the reconstructed request, validates the evaluator/rule version, and checks that every requested Evidence ID occurs exactly once as an assessment or a missing marker. It never majority-votes away conflict or turns confidence/authority descriptors into an ordinal trust score.

## Validation versus independent replay

The exporter validates internal consistency of the typed REA result and recomputes the request digest. It does **not** authenticate the producer or independently reproduce the result's derivation. Its derivation SHA-256 is a byte commitment, not a signature or trust anchor.

Consumers requiring independently reproduced semantics must separately require a matching ReplayMatch from the independent replay verifier (#10740). A formal export by itself is not a replay receipt.

## Formal binding relationship

\`formal/lean/assurance/REAProviderObservation.lean\` consumes the same conceptual fields in \`ClaimBinding\`. The Lean adapter retains generic closure generation identity as well as the exact textual evaluator and rule identifiers, so a new rule generation cannot silently reuse a prior result.

The string-valued subject, operation and environment fields remain exact source values in the export. The existing generic closure kernel's abstract subject/claim identity is a separate identity layer; this envelope does not claim that a SHA-256 digest is a producer-authenticated identity proof.

## Claim ceiling

The formal envelope is evidence that a provider-observation meta-claim was mapped consistently under a named v0 contract. It does not prove that the provider told the truth, that its parser is correct, that the archive/member bytes were completely observed, that the source is authentic, that the signer is trusted, that sources are independent, or that the underlying artifact is safe.
