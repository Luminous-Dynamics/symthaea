# DE-001 draft revocation probe

Ephemeral CI-only probe for `CI-STACK-005`.

This file has no product, research, qualification, evidence, or scientific authority. The child PR containing it exists only to materialize ordinary `pull_request` workflow runs on a base that already contains the exact-head draft-revocation workflow, then exercise the `converted_to_draft` cancellation transition.

Expected lifecycle:

1. open the child PR ready-for-review;
2. observe automatic `pull_request` runs for the child head;
3. convert the PR to draft;
4. require the base-branch `pull_request_target` revoker to converge on the exact child head and cancel any still-live automatic PR runs;
5. close the probe after evidence capture.
