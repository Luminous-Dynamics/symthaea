# WCARE-48H — GitHub-host run attestation protocol v1

Status: `PREREGISTERED_HOST_ATTESTATION`
Authority: `MeasurementOnly`
Tracks: #2992
Lineage parent: repaired WCARE-47 `0321a22986e5c6bf1a6b099d30dc75ccc33d9deb`

## Purpose

Independently attest GitHub's hosted record for the exact WCARE-48 generation event. Receipt self-assertion is not host provenance.

The governing chain is:

`FINAL structurally valid != receipt run identity bound != GitHub host run attested`

## Exact host subject

- repository: `Luminous-Dynamics/symthaea`
- workflow path: `.github/workflows/wcare48-lock-generation.yml`
- workflow name: `WCARE-48 Observed Lock Generation`
- run id: `34836223949`
- run number: `1`
- run attempt: `1`
- event: `push`
- branch: `wcare-48-observed-lock-generation`
- PREPARED SHA: `52d1d9fb741250ab8bcab205113689a8cc9431bb`
- WCARE-48V verifier subject: `890ab746618f0a57853db1e38cedb7c1b500a89d`

A later run or retry requires a new preregistration.

## Evidence inputs

The verifier consumes independently retrieved:

1. GitHub workflow-run JSON for the exact run id;
2. GitHub jobs JSON for that run;
3. decoded hosted log text for the exact `generate` job;
4. the repository branch head after execution;
5. a WCARE-48V result from exact verifier subject `890ab746618f0a57853db1e38cedb7c1b500a89d`.

SHA-256 commitments to the run JSON, jobs JSON, decoded log evidence, and WCARE-48V result are emitted in the host-attestation result.

## Host theorem

`HOST_RUN_ATTESTED` requires:

- exact run identity/path/name/event/branch/PREPARED head/run number/attempt;
- hosted run completed successfully;
- exactly one `generate` job completed successfully;
- all five frozen ceremony steps completed successfully;
- hosted log contains `PASS_WCARE48_GENERATION_PRECOMMIT`;
- hosted log contains one exact lock SHA-256, candidate Git blob, and positive package census;
- hosted `git commit` output exposes a commit prefix uniquely matching the candidate FINAL commit;
- hosted log records successful push to `wcare-48-observed-lock-generation`;
- branch head equals the candidate FINAL commit;
- candidate is a direct child of frozen PREPARED;
- exact WCARE-48V result says `FINAL_CHILD_VALID`, `final_child_structurally_valid=true`, `receipt_run_identity_bound=true`, and `github_host_run_attested=false` for the same candidate;
- committed receipt run identity and lock SHA/blob/package count agree with hosted log evidence.

## Outcomes

- `HOST_RUN_ATTESTED`
- `HOST_RUN_FAILED`
- `HOST_RUN_INDETERMINATE`
- `INVALID_HOST_ATTESTATION_PROTOCOL`

Before the exact run completes, the only valid state is `HOST_RUN_INDETERMINATE: exact_run_not_completed`.

## Non-claims

This is evidence from the GitHub hosting/provider fault domain. It does not establish independently authenticated builder identity, external preregistration, independent-host reproducibility, trusted hardware, lock admission, WCARE-42 executable qualification, subject correctness, runtime authority, consciousness, phenomenal experience, suffering, moral patienthood, binding consent, veto/self-preservation authority, or solved alignment.
