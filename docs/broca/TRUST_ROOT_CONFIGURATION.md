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
