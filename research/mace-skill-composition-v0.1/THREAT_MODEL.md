# MACE-CoS Guard — threat model

## Protected properties

- Confidential/restricted input is not downgraded through skill output declarations.
- Scope, purpose, audience and target remain bound across artifact handoffs.
- Individually admitted skills cannot silently create an unsafe selected composition.
- Skills cannot create capabilities or mint effect authority.
- External action authorization remains tied to an exact principal, action, target and argument digest.
- Expired, revoked, stale or mismatched inputs do not become admissible by composition.

## In-scope adversarial cases

1. Confidential extraction followed by a summary step declaring a public output.
2. Pairwise conflicts across selected skills, including one-sided declarations.
3. Attempted capability grant by a skill.
4. External action without a separate receipt.
5. Authorization replay by reusing a nonce.
6. Target rebinding and mismatched action arguments.
7. Cross-scope, cross-purpose or cross-audience source use.
8. Revoked/expired skills and stale sources.
9. Unknown labels/sink kinds and forward references.
10. Exposure mode/audience mismatch and token-budget overrun.
11. Duplicate IDs, invalid digests and malformed authorization fields.
12. Plan mutation and report-digest sensitivity.

## Explicitly out of scope for v0.1

- Natural-language prompt injection or semantic conflict extraction.
- Implicit information flow through model hidden state, timing, external side effects or covert channels.
- Rich product lattices for compartments, jurisdiction, retention, consent and licensing.
- Declassification, aggregation privacy or differential privacy.
- Cryptographic signatures, issuer authentication and source custody.
- Dynamic revocation between validation and dispatch.
- Sandboxing, action execution, model quality and production operations.
- Compromised or dishonest trusted verifier.

## Trust assumptions

The structured skill registry, policy context, source manifest and verifier-status assertion are supplied as inputs. SHA-256 establishes content identity only. The independent broker must authenticate source custody and signature authority, and must repeat the authorization check immediately before the side effect.
