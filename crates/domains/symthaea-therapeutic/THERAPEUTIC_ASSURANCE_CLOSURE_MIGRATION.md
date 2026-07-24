# Therapeutic Assurance Closure Migration — Series IX

Series IX closes the gap between isolated safety evidence and a reviewable release
candidate. It adds no therapeutic capability, diagnosis, medication logic, or
production autonomy. It binds denial semantics, provenance, reproducibility,
independent review, privacy-preserving evidence export, and authenticated release
evidence to one exact source tree.

## Required migration order

1. Register bounded denial definitions in `DenialCatalog`; never emit free-form denial reasons into portable evidence.
2. Build `TherapeuticProvenanceGraph` from source, configuration, policy, model, verification, and release nodes.
3. Reject duplicate critical node kinds, cycles, dangling endpoints, and disconnected release lineage.
4. Produce two authenticated `ReproducibilityRun` records from distinct runner identities.
5. Compare runs through `compare_reproducibility_runs`; output and test-vector digests must match.
6. Collect distinct safety, privacy, security, and release approvals through `evaluate_independent_review`.
7. Ensure the release reviewer is organizationally independent from the author commitment.
8. Export only typed evidence digests through `EvidenceExportManifest`, bound to an intended recipient and expiry.
9. Evaluate prior release gates through `evaluate_pre_assurance_closure`.
10. Compose all evidence with `TherapeuticAssuranceClosureCoordinator`.
11. Record the closure digest as the `assurance-closure` release gate.
12. Qualify a monotonic canary candidate through `ReleaseCandidateRegistry`.
13. Require a separate human release decision; closure and candidate qualification never authorize production.

## Closure invariants

- Denial codes are stable, bounded, and catalogued; raw prompts and narratives are not denial metadata.
- Critical provenance kinds occur exactly once and are reachable from the source-tree node.
- Provenance binds the exact configuration, policy, model registry, verification receipt, and release evidence authenticator.
- Evidence exports contain typed identifiers, schema versions, and digests only.
- Evidence exports are authenticated, recipient-bound, purpose-bound, and expiring.
- Reproducibility requires distinct authenticated runners using one manifest and source tree.
- Independent review requires four distinct reviewer commitments and all required roles.
- Review evidence is source-bound, proposal-bound, expiring, and authenticated.
- Decision receipts carry catalogued codes and cryptographic commitments, not free-form therapeutic content.
- Candidate generations are monotonic and rollback-resistant.
- Assurance closure authorizes only release-candidate qualification.
- Release-candidate qualification authorizes only a bounded canary.
- Neither gate establishes clinical validity or permits general availability.

## Workspace verification

Run `scripts/verify-assurance-closure.sh` from the complete Symthaea workspace.
Its successful result supplies the `assurance-closure` release gate. The gate must
remain `NotRun` when compilation, tests, reproducibility runs, independent review,
or evidence verification were not actually executed. Static archive validation is
integrity evidence only and must not be represented as executable verification.
