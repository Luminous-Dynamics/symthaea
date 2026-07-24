# Therapeutic Governance Migration — Series II

This series extends the fail-closed safety work with enforceable consent,
privacy, jurisdiction, evidence, orchestration, retention, and deployment gates.
It does not convert experimental therapeutic models into clinical devices.

## New mandatory boundaries

Production callers should use `TherapeuticSafetyKernel::process` rather than
calling crisis detection, ethics evaluation, or `ScopeGuard` independently.
The kernel fixes the release order:

1. purpose-scoped consent validation;
2. crisis triage and context clarification;
3. reviewed jurisdiction resource validation;
4. evidence authorization for proposed interventions;
5. ethical authorization with contraindication and supervision facts;
6. fail-closed response scope guarding;
7. redacted, hash-chained release receipts.

A draft response is not renderable until this sequence completes.

## Consent migration

Replace blanket consent booleans with `ConsentReceipt`. Grant only the purposes
needed for the operation. `SupportiveConversation` does not imply
`InterventionExecution`, raw narrative storage, human disclosure, quality
evaluation, or research.

Derive `SubjectRef` with a deployment-specific secret salt stored outside
therapeutic snapshots. Never reuse the salt across deployments.

## Jurisdiction migration

Provide `JurisdictionPolicy` through deployment configuration. Resource records
and reporting rules require review deadlines. Expired or incomplete policy is a
hard deployment-readiness failure. Unknown reporting cases default to qualified
human review, never automatic reporting.

The built-in generic safety-plan template remains jurisdiction-neutral and must
not be represented as localized emergency guidance.

## Evidence migration

Register every model, threshold, proxy, and intervention-selection claim in an
`EvidenceRegistry`. A declared use still requires the minimum evidence maturity
for that use, current review metadata, explicit uncertainty, source identifiers,
population scope, and a calibration version.

Research simulation authorization does not authorize clinical assessment,
medication selection, crisis detection, or autonomous treatment.

## Privacy and deletion migration

Use the `redacted_*` summary methods for telemetry and ordinary export. Default
state snapshots can contain highly sensitive raw content and should remain in an
encrypted, access-controlled store.

Apply `TherapeuticRetentionPolicy` on every persistence boundary and deletion
request. Deleting narrative, formulation, or shadow content also removes derived
HDC encodings, pressure histories, and dream-queue references.

Audit receipts contain keyed fingerprints, decisions, policy versions, and
reason codes. They deliberately exclude raw therapeutic text and contacts.

## Deployment readiness

Construct `TherapeuticDeploymentManifest` and require `evaluate(...).ready`
before serving traffic. A supportive production deployment rejects:

- experimental computational psychiatry;
- experimental consciousness protocols;
- named diagnostic hypotheses;
- legacy clinical-scale analogues;
- raw sensitive export capability;
- autonomous intervention capability.

Research sandboxes must not have production therapeutic-data access.

Persist the configuration fingerprint with release evidence so policy changes
can be correlated with audit receipts and evaluation results.

## Verification

Run `scripts/verify-governance.sh` from the Symthaea workspace root. The script
checks the default supportive surface first, then each quarantined feature and
the combined research feature set.

The safety kernel still requires external evaluation of crisis sensitivity,
context classification, resource correctness, fairness, accessibility, and
human escalation operations before any real deployment.
