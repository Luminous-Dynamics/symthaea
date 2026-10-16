# SYM-FIN-SYS-016 — Trust appraisal layers for socioeconomic run evidence

**Status:** research design, not a qualified production control.  
**Claim ceiling:** provenance of synthetic/deterministic validator execution only.  
**Normative boundary:** a valid report, digest or signature is not proof that a modeled economic event happened or that the modeled mechanism is fair/effective.

## Why this design separates claims from authority

[RFC 9334 (RATS Architecture)](https://www.rfc-editor.org/rfc/rfc9334.html) distinguishes an Attester that produces Evidence, a Verifier that appraises Evidence under a policy and emits Attestation Results, and a Relying Party that evaluates those results against its own policy. The design here borrows that separation of duties; it is not an implementation of device-level remote attestation.

A single `verified=true` bit is inadequate for socioeconomic evidence. At minimum, the consumer must distinguish these independent questions:

1. **Byte integrity:** are these the exact bytes named by the declared digest?
2. **Issuer/authentication:** is the source identity authenticated and authorized for this evidence type?
3. **Execution provenance:** which repository revision, workflow, runner context and inputs produced this artifact?
4. **Semantic/domain validity:** does the evidence establish the actual claim in the relevant domain, under the relevant unit, time, scope and dispute rules?
5. **Authority:** what action, if any, may this consumer take based on the appraised result?

Success at one layer must never imply success at the next layer.

## Mapping the current experiment

| Role / layer | Current experiment | What it can establish | What it cannot establish |
|---|---|---|---|
| Evidence producer | Test job and report writer | Produces a run report, output log and exact input hashes | That an economically meaningful real-world event occurred |
| Byte-integrity checker | `validate_socioeconomic_run_bundle.py` | Supplied files match manifest digests/lengths and pointer bindings | Who created the files or whether their statements are true |
| Execution report | `qualification-report.json` | Records source SHA, event SHA, workflow ref/SHA, run metadata, input digests and test counts | Independent authentication on its own; the report is generated inside the workflow |
| Attestation envelope | GitHub Actions artifact attestation, manual trusted-main path only | Cryptographically ties the signed subject to GitHub's asserted workflow/repository/event provenance | That the test logic was sufficient, the simulation is truthful, or the economic mechanism is valid |
| Independent verifier | `gh attestation verify` plus local schema/digest checks | Evaluates signature, signer identity/policy and report structure when run with expected identity constraints | Domain truth beyond its appraisal policy |
| Relying party | Reviewer or downstream integration policy | Decides whether this level of evidence is sufficient for the intended purpose | Must not inherit authority automatically from a green workflow |

## Trust policy for consumers

The consumer should receive an evidence result vector, not one unqualified boolean:

- `byte_integrity`: `pass | fail | unknown`
- `source_authentication`: `pass | fail | unknown`
- `workflow_provenance`: `pass | fail | unknown`
- `test_execution`: `pass | fail | unknown`
- `domain_claim_validity`: `pass | fail | unknown`
- `permitted_authority`: a separately configured capability set, empty by default for research artifacts

The current workflow only has enough information to target the first four dimensions for its validator-test claim. It must leave `domain_claim_validity=unknown` and `permitted_authority=empty`. Do not translate an attested test report into a right to reserve resources, settle assets, change reputation, or import data into a production ledger.

## Threat model and non-goals

Consider at least: malformed/double-key JSON; path traversal/symlink aliases; tampered payload bytes; a report bound to the wrong source SHA; synthetic merge SHA mistaken for PR head SHA; test inventory shrinkage; spoofed or mismatched test-log counts; a report/predicate digest mismatch; use of a PR-defined workflow to sign its own claims; compromised dependencies/runner; stale or revoked signing identity; and an authentic but semantically false domain statement.

Existing work directly addresses structural parsing, payload integrity, report/predicate agreement, test-count drift and trust-boundary flags. A signed workflow identity reduces ambiguity about execution provenance. It does not itself close compromised-runner, malicious-but-reviewed test logic, domain-truth or authority-policy risks. Those remain separate gates.

The manual attestation path is restricted to a workflow dispatch on `main`, and it qualifies only the event's own default-branch SHA. PR and push runs upload reports/logs but do not sign them. This design intentionally avoids choosing an arbitrary candidate SHA in a trusted signing run and signing a report generated by candidate-controlled test/report code.

## Proposed qualification gates

| Gate | Acceptance criterion |
|---|---|
| Exact source binding | Report source SHA equals the checkout SHA; workflow SHA/ref are separately recorded and consistent with the run context |
| Input identity | 100% of required validator/test/schema/dependency/workflow inputs have recomputed exact-byte digests and lengths |
| Test execution | Per-suite command exit is successful; observed test summaries equal the pinned suite inventory and source-discovered function counts |
| Predicate binding | Signed subject digest equals report bytes; predicate's source/run/workflow/count/authority fields match the report |
| Signer policy | Independent verification accepts only the expected repository, workflow identity, ref and event policy |
| Authority non-escalation | All authority flags remain false; schema-valid report cannot authorize any external action |
| Negative controls | Every required tamper/identity/count/path/authority counterexample is rejected, with control IDs preserved |
| Domain validity | Explicitly `unknown` for this workflow; require separate domain-specific evidence and independent appraisal before elevating it |

## Research frontier

The promising direction is not "a hash for every claim" but **typed evidence with explicit appraisal policies and narrow authority**. Socioeconomic adapters should preserve the claim's issuer, time, unit, scope, provenance level, uncertainty and appeal path. A downstream mechanism should ask "what does this evidence entitle me to conclude or do?" rather than "is it verified?"

This is compatible with a future run-bundle format that can carry several evidence classes and independently appraised results. It does not remove the need for field-specific validators, human governance, adversarial testing, or external evidence for real-world claims.
