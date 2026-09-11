# Continuity Fenced Resource Evidence Auditor V1

## Status

`INDEPENDENT_STRUCTURAL_AUDITOR / NOT_VERIFIER_QUALIFICATION`

This child adds an implementation independent from the fenced-resource harness for auditing the retained evidence bundle introduced by #1659.

It grants no execution authority and does not establish that the observations are truthful.

## Input boundary

The auditor accepts exactly one evidence directory containing exactly:

- `campaign-manifest.json`;
- `campaign-observations.json`;
- `campaign-run-context.json`;
- `campaign-summary.json`;
- `SHA256SUMS`.

Unknown extra files or missing files fail closed.

## Independent reconstruction

`verify-continuity-fenced-resource-evidence.py` independently checks and re-derives:

1. the exact four-file `SHA256SUMS` manifest;
2. closed field sets for manifest, observations, context and summary;
3. exact Git subject syntax and subject agreement between context/summary;
4. the fixed nine V1 obligations and required evidence bases;
5. every observation record ID from obligation + basis + canonical raw evidence;
6. exact manifest-to-observation record/timestamp linkage;
7. campaign interval containment for every observation;
8. the V1 complete-set ID;
9. the canonical campaign-manifest SHA-256;
10. the #1578 evidence-manifest SHA-256;
11. the #1578 canonical campaign-preimage SHA-256;
12. exact summary/oracle reconstruction;
13. the fixed V1 contained-lab property set and final state.

A successful audit returns a compact JSON report containing the exact subject and reconstructed identities.

## Adversarial external harness

`test-continuity-fenced-resource-evidence-auditor.py` starts from one valid retained bundle, copies it, mutates it from outside the auditor implementation, recomputes `SHA256SUMS`, and still requires fail-closed rejection.

V1 adversarial cases are:

- unknown/shadow campaign-manifest field;
- raw observation payload mutation;
- required evidence-basis substitution;
- exact subject substitution in both retained context and summary;
- removal of one obligation;
- forged oracle/preimage summary;
- backend-generation drift with superficially repaired manifest-summary fields.

The point of recomputing `SHA256SUMS` after each mutation is to prove that ordinary file-integrity consistency cannot substitute for semantic identity validation.

## Workflow ordering

The dedicated workflow now requires, in order:

```text
contained resource scenarios
-> retained bundle generation
-> sha256sum verification
-> independent semantic auditor
-> external adversarial auditor harness
-> digest publication
-> artifact upload
```

The uploaded artifact therefore cannot be produced by this workflow unless both the structural auditor and its mutation corpus pass.

## Non-claims

An auditor PASS establishes only that the retained bytes are internally coherent under the frozen V1 structural algorithms.

It does **not** establish:

- truth of the raw observations;
- current verifier admission;
- campaign invalidation/currentness policy;
- cryptographic authentication of the resource permit;
- hardware monotonic state;
- a real NETCONF/Redfish/gNOI/storage/BMC enforcement boundary;
- production mutation permission.

Those remain downstream #1550/#1528 responsibilities.
