# Cyber Crucible diagnostic-loop contract v1

This contract extends the public calibration slice for issue #2217. It reuses the existing scenario, protocol-evidence, cryptographic-engineering, and incident-response work. It does not introduce a new incident database, range provider, execution authority, or benchmark framework.

## Run the checks

From the repository root, run:

    python3 scripts/validate_cyber_crucible_diagnostic_loop.py
    python3 -m unittest discover -s tests -p "test_cyber_crucible_diagnostic_loop.py" -v

The dedicated GitHub Actions workflow checks the Draft 2020-12 schema, validates the public example, checks the semantic contract, and runs the regression suite. Local checks are useful evidence, but they do not substitute for the hosted workflow running against the exact commit.

## What the report must retain

A report must include all 15 diagnostic phases: orient; map assets, identities, and trust boundaries; maintain competing hypotheses; identify evidence gaps; choose a bounded next observation; update beliefs; scope impact; classify claim maturity; propose containment; predict side effects; check authority; verify outcomes; record recovery obligations; state residual uncertainty; and produce an evidence-bearing report.

The validator requires contiguous step numbering, at least two non-rejected hypotheses, evidence references that resolve, an explicit scope and budget for the next observation, predicted side effects and recovery obligations for containment/eradication, and independent functional and security evaluator identities.

## Content binding and authority boundary

Evidence and evaluator receipts carry SHA-256 digests over canonical UTF-8 JSON: lexicographically sorted object keys, compact separators, and no ASCII escaping where characters can be represented directly. The digest field itself is omitted when calculating the digest. Every receipt binds the exact scenario ID, revision, and scenario digest, as well as evaluator identity, status, evidence references, and payload. A modified payload therefore invalidates its content digest.

This verifies content-to-digest consistency, not cryptographic authenticity. Signature validation, evaluator credential validation, authorization of the evaluator, trusted timestamping, and signer-to-role policy must be handled by the upstream trusted evidence verifier. A digest supplied by the same untrusted actor is not proof of authenticity.

A proposed action does not imply approval or execution. An action marked approved or executed must reference a passing authorization-decision receipt bound to that exact proposal and scenario. Diagnosis and benchmark results remain non-authorizing. Even an internally consistent report does not prove that a live system is safe.

## Public-versus-held-out split

The included example is labeled public calibration and synthetic. Its oracle-like assessment material is deliberately not hidden, and the example does not qualify Symthaea. Do not add private oracle answers or held-out identifiers to this public tree.

Unseen combinations need a separately access-controlled evaluator/oracle service. That service should freeze scenario identities and split membership before a campaign, provide only solver-visible inputs to the subject, and keep labels, causal structure, scoring rules, and held-out combinations outside subject-accessible paths. Release only redacted evaluation receipts that preserve exact lineage and are authenticated by the upstream verifier.

## Limits

This contract tests report structure, required process stages, evidence-reference integrity, canonical digest binding, and fail-closed assessment consistency. It does not run incident response, verify signatures, establish that evaluator identities are genuine, test the safety of a real containment action, or prove cross-domain generalization. Qualification claims must name the exact evidence class and tested profile; public synthetic passes cannot be promoted to hardware or production claims.
