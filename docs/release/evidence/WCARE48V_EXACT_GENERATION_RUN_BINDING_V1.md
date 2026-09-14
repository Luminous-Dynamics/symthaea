# WCARE-48V — preregistered receipt run-identity binding v1

Status: `PREREGISTERED_RECEIPT_RUN_BINDING`
Authority: `MeasurementOnly`
Tracks: #2982

## Purpose

Tighten the WCARE-48V FINAL-child theorem so a structurally consistent receipt from a later or replacement workflow run cannot silently claim the identity of the already-preregistered WCARE-48 generation event.

The exact receipt run identity is:

- GitHub workflow run id: `34836223949`
- workflow run number: `1`
- workflow run attempt: `1`
- PREPARED head: `52d1d9fb741250ab8bcab205113689a8cc9431bb`
- workflow: `.github/workflows/wcare48-lock-generation.yml`
- branch: `wcare-48-observed-lock-generation`

The governing distinction is:

`receipt consistent != receipt names preregistered run != GitHub-hosted run attested`

## Rule

The receipt-bound WCARE-48V decision requires both:

1. `scripts/wcare48v_verify_final.py` reaches `FINAL_CHILD_VALID`; and
2. `scripts/wcare48v_verify_bound_final.py` confirms the receipt carries exactly run id `34836223949`, run number `1`, and run attempt `1`.

The resulting field is deliberately named:

`receipt_run_identity_bound = true`

and the same verifier must keep:

`github_host_run_attested = false`.

That is intentional. A commit and receipt can self-consistently name a GitHub run without independently proving that GitHub's hosted record for that run executed successfully and produced that exact commit. Hosted-run attestation is a separate evidence theorem and must use GitHub-host evidence rather than receipt self-assertion.

A later workflow run or later attempt is not grandfathered into this receipt-binding theorem. If the exact first attempt fails before producing a FINAL child, any replacement generation attempt requires its own explicit preregistration/binding rather than widening this contract after seeing the outcome.

## Promotion boundary

Receipt run-identity binding still does not establish GitHub-host run attestation, lock admission, builder authentication, external preregistration, independent-host reproducibility, WCARE-42 executable qualification, runtime authority, consciousness, phenomenal experience, suffering, moral patienthood, binding consent, veto/self-preservation authority, or solved alignment.
