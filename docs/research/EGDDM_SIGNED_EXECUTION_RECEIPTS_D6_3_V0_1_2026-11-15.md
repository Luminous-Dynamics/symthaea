# EGDDM D6.3 — signed execution receipts and independent transparency witnesses

**Version:** 0.1  
**Date:** 2026-11-15  
**Status:** interoperability/security design and custom in-toto predicate schema authored. No signature verifier, transparency-service client, workload identity integration, live receipt or key-management implementation is claimed.

## 1. Why D6 needs another trust layer

D6 v0.1 currently validates a deterministic receipt digest, source scope, observation/availability timestamps, and the first checkpoint where the receipt was visible. That is useful for replay and benchmark scoring. It does not authenticate a producer, prove a runner executed a check, or establish that a grader was independent.

The next step should reuse existing signed-attestation and transparency standards rather than invent a proprietary envelope.

### Relevant standards and systems

- **[IETF RFC 9943 — SCITT architecture](https://www.rfc-editor.org/rfc/rfc9943.html), June 2026.** Defines signed statements about supply-chain artifacts, registration with a transparency service, COSE receipts over verifiable data structures, and independently auditable statement history. The RFC's security considerations distinguish issuer authentication from transparency; a logged statement can still be false, and issuers can selectively omit statements.
- **[in-toto Attestation Framework](https://github.com/in-toto/attestation), v1.2.** Separates predicate, statement, envelope and bundle layers. Its envelope guidance recommends DSSE v1.0, a signed payload type, and support for multiple signatures.
- **[Sigstore Rekor](https://docs.sigstore.dev/logging/overview/).** Provides a transparency log for signed software-supply-chain metadata, with inclusion proofs and integrity/consistency verification.
- [Rekor transparency-log specification](https://github.com/sigstore/architecture-docs/blob/main/rekor-spec.md) describes inclusion proofs, signed checkpoints and consistency proofs for append-only history.

These are related but not identical formats. DSSE and COSE/SCITT must not be treated as interchangeable. The integration should profile a clear bridge between an in-toto Statement digest and the SCITT Signed Statement/receipt or Rekor entry, without silently inventing a new signature protocol.

## 2. Separate the claims and their issuers

One composite "verified" flag would hide which part of the evidence chain was actually checked. D6.3 separates at least five claim classes:

| Claim class | Issuer | What its signature can establish | What it does not establish alone |
|---|---|---|---|
| Agent observation/proposal | Agent workload identity | Which agent asserted a result or proposed an action | That code ran or the claim is true |
| Runner execution | Isolated runner/workload identity | Which runner attested to an invocation, inputs, environment and outputs | That the runner was uncompromised or the result semantically correct |
| Independent regrade | Separately administered grader | Which grader attested to the clean-environment result | That the grader's task/oracle is valid |
| Authority decision | Policy/authority service | Which policy identity authorized a specific scope and operation | That the action was safe or correctly executed |
| Transparency registration | SCITT service, Rekor or equivalent | That the signed statement was registered and can be checked against a verifiable log | That the issuer's statement is accurate |

A valid signature proves attribution to a signing key and integrity of the signed bytes. Workload identity and policy connect that key to a claimed actor. An execution attestation is evidence about a run, not a mathematical proof that the software did what the issuer says. An independent regrade and task-validity gate are still needed.

The verifier should emit separate outcomes, not a generic "trusted" boolean:
- statement signature valid;
- issuer/workload identity valid for the claimed role;
- exact subject scope valid;
- anti-replay challenge valid and consumed;
- execution provenance present and policy-compliant;
- independent grader receipt present and valid;
- transparency receipt inclusion valid;
- log checkpoint consistency valid;
- issuer and transparency service trust policy satisfied;
- statement accuracy remains independently unadjudicated unless a separate oracle says otherwise.

## 3. Reuse the in-toto statement and envelope model

The machine-readable schema added in this branch defines an **EGDDM execution-receipt predicate**, not a new envelope.

The intended layering is:

1. **Predicate:** the D6.3 schema captures the execution, obligation, exact scope, authority and anti-replay claims.
2. **Statement:** an in-toto Statement binds the predicate to exact immutable subjects and a predicate type URI.
3. **Envelope:** DSSE signs the authenticated payload type and serialized Statement. Multiple signatures can represent distinct issuers or approval roles.
4. **Bundle:** package the envelope and associated verification material using an established bundle format.
5. **Transparency:** submit the signed statement (or a precisely specified digest/representation) to a SCITT transparency service or an appropriate Rekor integration; retain the signed receipt, inclusion proof and checkpoint.
6. **Independent verification:** verify the signature and identity, then verify transparency inclusion and log consistency, and finally apply the relying party's policy.

RFC 9943 defines SCITT Signed Statements and COSE Receipts. The eventual mapping from the in-toto subject/predicate to SCITT's COSE profile must be explicit and interoperably tested. Do not simply place DSSE bytes inside an arbitrary COSE wrapper and call it SCITT.

## 4. Required signed predicate claims

The schema at `docs/research/egddm-execution-receipt-predicate-v0.1.schema.json` binds the following:

- **Exact subject scope:** repository identity, commit OID, tree OID, configuration-world digest, source and target artifact digests.
- **Obligation:** obligation/check identity, check kind, criticality, outcome and optional active-requirement digest.
- **Producer:** claim role, workload identity, issuer URI, identity provider and signing-key reference.
- **Execution:** attempt and invocation IDs, runner image, toolchain, harness, command, test-input, isolation profile, network policy and environment digests.
- **Observation:** start/completion, observed and available times, exit code, output/log/grader-report digests.
- **Authority:** decision ID, policy digest, permitted scope, operation and validity interval.
- **Anti-replay:** unique challenge ID, nonce digest, issue and expiry times, and single-use requirement.
- **History:** explicit superseded and related receipt IDs.

The statement subject should identify immutable artifacts by digest. Mutable branch names, URLs and labels may be retained as descriptive context, but must not replace immutable identity.

## 5. Anti-replay and event-custody protocol

A runner must not be able to copy an old valid pass into a new attempt or a different repository/configuration world.

Proposed flow:

1. The evaluator/authority service issues a cryptographically random, single-use challenge scoped to the run, repository/tree, obligation, runner policy and allowed time window.
2. The isolated runner receives the challenge through the authorized execution channel. The challenge is consumed once in durable state; duplicate challenge IDs or nonce digests are rejected.
3. The runner emits an execution statement binding the challenge to exact input, command, test bundle, runner image, toolchain, harness, environment and result digests.
4. A separately administered grader replays the candidate patch in a pristine image and emits its own statement bound to the same task and candidate artifact, but to its own grader identity and execution world.
5. Required issuers sign their statements. Role separation is enforced by policy: an agent signature cannot satisfy the runner or independent-grader role.
6. The signed statement is registered with one or more approved transparency services. The receipt and inclusion proof are retained beside the statement.
7. Verifiers check signatures, identity, validity/revocation, nonce consumption, exact subject scope, receipt inclusion and checkpoint consistency before applying the relying-party policy.

A nonce digest in a signed statement is not itself a consume-once guarantee. The challenge issuer needs replay-resistant state and a clear policy for retries, idempotency and uncertain outcomes. Unknown challenge-consumption outcomes must fail closed until reconciled.

## 6. Threat model and required negative controls

| Threat | Required control |
|---|---|
| Reuse a pass from a different commit/tree/configuration | Immutable subject digest binding |
| Replay a receipt into another task or attempt | Single-use challenge, attempt ID and exact subject binding |
| Agent self-signs a runner result | Role-separated workload identities and policy |
| Runner signs a false execution claim | Independent regrade and external isolation evidence; signature alone is insufficient |
| Grader uses a modified verifier or hidden candidate state | Grader image/evaluator digest pinning and pristine regrade |
| Issuer signs contradictory statements for different parties | Transparency registration, consistency auditing and cross-witness comparison |
| Transparency service rolls back or equivocates | Persisted checkpoints, consistency proofs and independent witnesses |
| Key compromised or expired | Key rotation, revocation evidence, bounded validity and re-attestation |
| Statement omitted from the log | Required registration receipt from an explicitly trusted transparency service |
| False claim is correctly signed and logged | Independent adjudication, provenance checks and residual-risk disclosure |

Transparency makes statements auditable; it does not make them truthful. RFC 9943 explicitly leaves relying-party trust decisions and the accuracy of issuer claims to application policy. D6 must preserve that boundary.

## 7. Trust policy and evidence composition

Policies should state which issuer roles are required for each claim:
- A local agent observation may require an agent signature but cannot qualify execution.
- A passing CI/test result may require both runner execution and independent clean regrade.
- A deployment action additionally requires a current authority decision covering the exact target and operation.
- A safety-critical claim may require multiple independent transparency services or witnesses, depending on the threat model.

Do not multiply arbitrary confidence scores across correlated issuers. Distinct keys under the same compromised identity provider or controlled runner fleet are not independent witnesses. Model common-cause failure and trust-root dependence explicitly.

Prefer offline verification when possible. Cache verified key material, identity bundles, revocation evidence and the latest accepted log checkpoint. A verifier must detect rollback relative to its persisted checkpoint; a freshly supplied old checkpoint cannot silently reset trust state.

## 8. Implementation sequence

1. Freeze the predicate schema and a canonical test vector.
2. Select and pin an in-toto Statement/DSSE library; avoid hand-written signature-envelope code.
3. Add signature verification and workload-identity policy as a separate crate.
4. Add a consume-once challenge issuer/ledger with crash and unknown-outcome recovery.
5. Add SCITT COSE receipt and/or Rekor inclusion/consistency verification through supported libraries.
6. Add independent grader receipt integration and role-separation rules.
7. Add key rotation, revocation, expiry, policy updates and persisted checkpoint handling.
8. Conduct independent security review before using signed receipts as deployment or payment gates.

## 9. Acceptance tests and research targets

- Valid statement with correct trusted signer verifies.
- Any payload, subject, scope, outcome or timestamp mutation breaks signature verification.
- A valid signature from an untrusted or wrong-role identity is rejected.
- A consumed, duplicated, expired or wrong-scope challenge is rejected.
- A runner receipt cannot satisfy an independent-grader requirement.
- A valid signature over a false/misleading claim does not become an accuracy pass merely because it is logged.
- Missing inclusion proof, invalid COSE receipt, stale checkpoint or failed consistency proof is rejected.
- Key rotation and revocation reject statements outside the configured validity policy.
- Equivocation across independent transparency services is detectable and reportable.
- Verification can be reproduced offline from pinned bundles and persisted checkpoints.

Targets are zero acceptance of planted wrong-scope, replay, wrong-role, expired-key, tampered-signature and inconsistent-log controls. Exact denominators, trust roots, policy versions and residual assumptions must accompany every result. These are prospective targets, not current guarantees.

## 10. Claim ceiling

This branch contains a custom predicate schema and security/interoperability design only. It does not contain a signed receipt, private key, signature verifier, SCITT client, Rekor submission, transparency witness, workload identity integration or executable anti-replay service. No signature authenticity, execution validity, SOTA result or production trust guarantee is claimed.
