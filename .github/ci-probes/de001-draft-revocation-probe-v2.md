# DE-001 runnerless draft revocation probe v2

Ephemeral CI-only probe for `CI-STACK-005`.

This file has no product, research, qualification, evidence, optimizer, or scientific authority.

The child PR containing it exists only to demonstrate that the base branch's `CI Draft Revocation Signal` can cancel an automatic full-CI run on a pure `converted_to_draft` transition by joining CI's exact concurrency group, without requiring a hosted runner or an additional commit.

Expected lifecycle:

1. open this child PR ready-for-review;
2. observe an automatic full-CI run for the exact child head;
3. convert the PR to draft without changing the head;
4. require `CI Draft Revocation Signal` to appear for the same head and complete with its sole job skipped;
5. require the previously live automatic full-CI run to become cancelled;
6. close the probe after evidence capture.
