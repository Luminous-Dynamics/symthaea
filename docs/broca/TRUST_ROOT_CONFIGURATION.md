# Broca trust-root repository configuration

The code-level trust root is the base-owned Broca Independent Trust Anchor workflow.

It becomes a merge-control mechanism only after repository policy requires its status.

## Required status

Require the exact commit status context:

`Broca / Independent Trust Anchor`

The rule should apply to the branch that receives the Broca compiler changes.

GitHub supports required checks as either checks or commit statuses. Where repository policy supports selecting the expected source, select the GitHub App that owns the status rather than allowing any source to satisfy the requirement.

## Required semantics

The independent status must be required on the protected target branch.

A successful status is only valid for the latest commit SHA. A later push invalidates the previous status and requires a new independent verdict.

The verifier already enforces an additional source snapshot lock through:

`docs/broca/independent_trust_policy_v1.json`

That policy records the exact approved PR head, base ref/base SHA, and every changed-file Git blob SHA.

## Root-of-trust separation

Do not configure the merge rule so that the PR-controlled Broca Matrix alone can satisfy the release/control-plane requirement.

The intended hierarchy is:

`protected branch policy -> base-owned independent trust anchor -> PR-controlled Broca execution evidence`

The Broca Matrix remains necessary because it performs the actual compiler execution. It is not, by itself, the final authority over whether its own evidence is admissible.

## Operational note

The integration used to maintain this repository does not currently expose branch-protection configuration. The repository owner therefore needs to apply the required status rule through GitHub repository/ruleset administration.

Once configured, a successful `Broca / Independent Trust Anchor` status becomes a real merge prerequisite instead of merely an informative status.

## Actions execution protection

The trust workflow has write authority over commit statuses. GitHub's Actions workflow execution protections can independently restrict both who may trigger a workflow and which events are permitted; these controls sit outside the workflow file and therefore remain an external governance root.

For this trust anchor, configure an execution policy targeting the exact workflow path and allow only a small trusted administrator/team or dedicated automation identity to invoke workflow_dispatch. Permit workflow_run as the automated trigger. Do not grant general contributors the ability to manually execute the trust anchor merely because they can contribute code. GitHub documents actor and event allowlists for this purpose. https://docs.github.com/en/actions/how-tos/administer/control-workflow-execution

## Status-source authority

Require Broca / Independent Trust Anchor as a required status on the protected target branch and select the specific GitHub App that is authorized to create that status. Selecting any source leaves the status spoofable by another writer with repository write access. GitHub explicitly supports app-specific sources for required status checks. https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets

## Merge-queue compatibility

If a merge queue is enabled later, re-qualify this design against the queue's merge_group execution model. GitHub requires required GitHub Actions checks to report for merge_group events when a merge queue is used; a PR-only workflow can otherwise leave the queue waiting indefinitely. The trust anchor should not be declared queue-compatible without fresh evidence for the actual ruleset configuration. https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-a-merge-queue

## Configuration status boundary

These settings cannot currently be verified from the available repository integration because branch-protection administration is not exposed. The implementation therefore treats them as explicit prerequisites, not as assumed configuration. A PASS from the verifier must never be interpreted as proof that these external governance controls are installed.
