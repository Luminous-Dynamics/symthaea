# OAR interoperability profile for MON-009/010/011 v1

**Status:** interoperability mapping against OAR draft -01 (25 Nov 2026) and the public OAR repository / OAR 0.3 vector corpus. This is not a claim of OAR -01 conformance.

## 1. Positioning

Open Action Record (OAR) already defines action-level records for decision, approval, execution/refusal and confirmation. MON MUST reuse OAR-compatible lifecycle semantics and MUST NOT define a competing action receipt merely to duplicate those records.

The MON-specific contribution is a qualified join across:
- the source and transformation lineage that supports a decision (MON-010);
- the exact decision, approval, authority and action intent;
- the destination commit and independent readback (MON-009);
- the causal join / trust-domain and completeness conditions across the records (MON-011).

OAR is an action evidence format. It is not by itself a factual source-provenance graph, proof that source figures are true, a database transaction protocol, or proof of physical-world effect.

## 2. OAR fields and MON evidence mapping

| OAR evidence | MON interpretation | Required join |
|---|---|---|
| Decision record / `decision_ref` | Authorized decision node | Exact decision payload digest is the parent of approval, intent and execution |
| `authorization_ref` and authority set | Delegated credentials and policy authorizations considered | Bind authority-set digest and individual credential digests to this one decision/action |
| Commit-time `authority_check` | Authorization recheck at the mutation boundary | Bind exact authority set, decision, intent, policy/scope and check time; verify status as-of sink commit |
| Approval binding / nonce / decision digest | Human approval for the precise action | Match action instance, display commitment, decision digest, approver, decision and approval outcome |
| `instance_commitment` | Privacy-preserving action/request commitment | Bind exact request semantics without exposing protected data |
| `state_before_commitment`, `state_after_commitment`, `target_ref_commitment` | Sink-side affected resource/state | Match the exact sink resource and before/after revision |
| Confirmation statement | Read-only system-of-record observation | Match execution reference, sink digest, resulting revision and state commitment |
| `committed`, `refused`, `unresolved` | Outcome semantics | Only confirmed committed state maps to COMMITTED; refused is not executed; unresolved never means success |

The MON source graph is linked to the decision through exact source, claim, transformation and replay digests. These links should be carried by OAR-compatible external references or profile extensions agreed with OAR/SCITT, rather than silently adding undocumented core fields.

## 3. Outcome handling

MON adopts OAR's three outcome classes:
- `committed`: exact state change and read-only confirmation are present and qualify;
- `refused`: the action was held/denied/not executed because authorization or policy failed;
- `unresolved`: an outcome is unknown, unconfirmed, missing or contradictory.

An API success, accepted dispatch or producer confirmation MUST NOT be promoted to `committed` without the defined system-of-record confirmation. An unresolved outcome remains unresolved even if the agent claims success.

MON may use additional internal diagnostic statuses (`qualified`, `indeterminate`, `contradicted`, `stale`, `unauthorized`) but maps them to OAR outcomes only at the action boundary. No diagnostic status changes OAR's semantics.

## 4. Existing OAR reference vectors to reuse

The public `openactionrecord/oar` repository provides a manifest with 18 main and 6 compatibility cases. Its vector README says all supported verifiers must reach the manifest's expected result; published test keys are test-only. Cases relevant to MON include:

| OAR vector / case | Attack or assertion | MON regression mapping |
|---|---|---|
| `authority-revoked.json` | Authority valid at decision but revoked before commit; action must be refused | MON-009 commit-time authority / revocation race |
| `authority-not-rechecked.json` | Execution lacks required commit-time recheck | Missing authority-check record → unresolved / refused by profile policy |
| `authority-stale.json` | Authority check is not fresh/current | Commit-time freshness fails closed |
| `authority-evidence-misplaced.json` | Recheck evidence names authority outside the decision's authority set | Authority-set digest and membership mismatch |
| `approval-flipped.json` | Approval outcome conflicts with the decision binding | Reject approval / exact binding mismatch |
| `tampered-record.json` | Signed record changed | Authentication / chain verification failure |
| `removed-record.json` | Lifecycle record removed | Completeness / predecessor / checkpoint failure |
| `not-confirmed.json` | API reports success while readback does not confirm change; another action is denied | Outcome remains unresolved / denied, not committed |
| `erased-values.json` | Disclosure values removed while commitments still verify | Preserve verification without requiring disclosure of protected values |
| `compat/product-evidence-single-action-tampered.json` | Product evidence record modified | Cross-implementation adapter tamper control |

MON should invoke the maintained OAR verifiers against the actual official corpus. The Go CEJI preflight is a normalized semantic checker; it is not a replacement verifier and does not reproduce OAR's Ed25519/JCS/sealing/timestamp checks.

## 5. Executable outcome mapping

The small standard-library Go adapter in `oar-outcome/` implements the OAR outcome projection as a pure mapping: `confirmed` → `committed`; `denied`, `rejected`, `expired`, `refused` → `refused`; all other and unknown outcomes → `unresolved`. It has three named tests covering positive confirmation, the full refusal vocabulary, and non-terminal/unconfirmed/unknown cases. Local Go 1.23.2 preflight: race ×5 passed, vet passed, 100% statement coverage. This is not a signed OAR bundle parser or a cryptographic test-vector run.

## 6. Draft version and implementation boundary

OAR -01 is an active Internet-Draft published 25 November 2026. Its implementation-status section states that the public OAR format/repository implements every construction from -00, while new -01 additions are planned and not yet implemented in the referenced implementation. The public repository's current specification reports OAR 0.3 and contains the 24-case main/compatibility vector corpus.

Therefore:
- pin OAR draft version and record wire version separately;
- test backward-compatible OAR 0.3/-00 fixtures first;
- do not label the current reference implementation as fully OAR -01 conformant;
- create a separate draft -01 profile adapter only once the new fields and vectors are implemented and stable;
- test approval display-binding and authority recheck edge cases against OAR vectors before adding MON composition vectors.

## 7. Independence policy

OAR permits confirmation by the issuer or another party with read access. MON deployment policy may require the observer to be independently administered from the sink, action producer and authorization service. This is a stricter deployment qualification. It MUST NOT be described as a universal OAR requirement.

Similarly, a signature over an issuer's statement does not prove that the source figure is true. The source, transformation and replay layers provide provenance/reproducibility evidence, not truth certification.

## References

- OAR -01 Internet-Draft (25 Nov 2026): https://www.ietf.org/ietf-ftp/internet-drafts/draft-malhotra-scitt-oar-01.html
- OAR public format and implementation: https://github.com/openactionrecord/oar
- OAR main/compatibility vectors: https://github.com/openactionrecord/oar/tree/main/vectors

**Not claimed:** no OAR -01 wire adapter, OAR cryptographic verifier, or full 24-vector execution is included in this MON preflight.
