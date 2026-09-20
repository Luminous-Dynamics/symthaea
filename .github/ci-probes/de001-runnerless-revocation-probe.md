# DE-001 runnerless revocation probe

Ephemeral CI-only probe for `CI-STACK-005`.

This file has no product, research, qualification, evidence, or scientific authority. The child PR containing it exists only to materialize an ordinary ready-state `CI` run on a base that already contains `.github/workflows/draft-ci-concurrency-revocation.yml`, then exercise `converted_to_draft`.

Success criterion:

1. a ready-state automatic `CI` run exists for this exact head;
2. conversion to draft emits `CI Draft Revocation Signal` for the same PR ref;
3. that signal's job skips before runner allocation;
4. workflow-level concurrency cancels the older automatic `CI` run;
5. no `workflow_dispatch` evidence run is affected.
