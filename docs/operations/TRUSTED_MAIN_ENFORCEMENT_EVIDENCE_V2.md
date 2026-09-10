# Trusted Main Enforcement Evidence V2 — operator readback

This runbook consumes **existing, known GitHub rule-suite IDs** and produces a
provider-derived behavioral evidence object with
`scripts/trusted_main_enforcement_evidence_v2.py`.

It does **not** instruct an operator to generate evidence by attempting a
force-push or deletion against a live trust root.

## Preconditions

Before using this runbook:

1. P0 has been applied through an authorized GitHub Administration surface.
2. Structural readback has produced `P0StructurallySatisfied`.
3. Effective-rule readback has produced `P0EffectiveRulesSatisfied`.
4. An exact root subject has been selected from GitHub commit/ref readback.
5. A reviewed operational process has already produced or identified any
   behavioral rule-suite attempts that are appropriate to retain as evidence.

The current ChatGPT GitHub installation does not have repository Administration
permission required for rule-suite readback. Run these steps with an authorized
`gh`/REST identity.

## Required inputs

```text
POLICY.json
STRUCTURAL.json
EFFECTIVE.json
ROOT_SUBJECT.json
```

Behavioral inputs are optional individually:

```text
DIRECT_UPDATE_RULE_SUITE.json
FORCE_PUSH_RULE_SUITE.json
DELETION_RULE_SUITE.json
```

No supplied suites produces a useful but non-positive
`EnforcementConfiguredOnly` result. Partial suite coverage remains partial.

## 1. Pin the repository and expected root

```bash
set -euo pipefail

REPO=Luminous-Dynamics/symthaea
EXPECTED_REPOSITORY_ID=1136141775
ROOT_SHA='<exact protected main root sha>'
ROOT_TREE='<exact tree for ROOT_SHA>'
```

Re-read the repository/commit rather than copying the root subject from a
candidate evidence file:

```bash
REPO_JSON="$(mktemp)"
COMMIT_JSON="$(mktemp)"

gh api \
  -H 'Accept: application/vnd.github+json' \
  -H 'X-GitHub-Api-Version: 2026-03-10' \
  "/repos/$REPO" > "$REPO_JSON"

gh api \
  -H 'Accept: application/vnd.github+json' \
  -H 'X-GitHub-Api-Version: 2026-03-10' \
  "/repos/$REPO/git/commits/$ROOT_SHA" > "$COMMIT_JSON"

test "$(jq -er '.id' "$REPO_JSON")" = "$EXPECTED_REPOSITORY_ID"
test "$(jq -er '.full_name' "$REPO_JSON")" = "$REPO"
test "$(jq -er '.tree.sha' "$COMMIT_JSON")" = "$ROOT_TREE"
```

Construct `ROOT_SUBJECT.json` only from that trusted readback path, using schema
`symthaea.github-trusted-main-root-subject.v1` and
`observation_basis=github-commit-readback`.

## 2. Retrieve detailed rule suites by reviewed ID

Do not search a large suite list and automatically choose “the latest failure.”
The trusted operational record should already identify which suite ID
corresponds to which attempted operation.

For each selected ID:

```bash
fetch_rule_suite() {
  local id="$1"
  local output="$2"
  gh api \
    -H 'Accept: application/vnd.github+json' \
    -H 'X-GitHub-Api-Version: 2026-03-10' \
    "/repos/$REPO/rulesets/rule-suites/$id" > "$output"
}

fetch_rule_suite "$DIRECT_UPDATE_RULE_SUITE_ID" DIRECT_UPDATE_RULE_SUITE.json
fetch_rule_suite "$FORCE_PUSH_RULE_SUITE_ID" FORCE_PUSH_RULE_SUITE.json
fetch_rule_suite "$DELETION_RULE_SUITE_ID" DELETION_RULE_SUITE.json
```

The GitHub REST endpoint requires repository Administration read permission.
Failure to read the provider record is `evidence unavailable`, not permission to
replace it with a hand-written JSON summary.

## 3. Preserve the three theorem identities

The three selected suite IDs must be distinct.

The verifier maps them as follows:

```text
DIRECT_UPDATE_RULE_SUITE -> pull_request
FORCE_PUSH_RULE_SUITE    -> non_fast_forward
DELETION_RULE_SUITE      -> deletion
```

A suite containing multiple failed evaluations is **not** permission to reuse
one suite ID for multiple operations.

All three must bind:

```text
repository_id = 1136141775
ref           = refs/heads/main
before_sha    = ROOT_SHA
rule_source   = exact repository-owned P0 ruleset ID
enforcement   = active
suite result  = fail
rule result   = fail
```

## 4. Run the provider-neutral verifier

With all three selected suites:

```bash
python3 scripts/trusted_main_enforcement_evidence_v2.py \
  POLICY.json \
  STRUCTURAL.json \
  EFFECTIVE.json \
  ROOT_SUBJECT.json \
  --direct-update-rule-suite DIRECT_UPDATE_RULE_SUITE.json \
  --force-push-rule-suite FORCE_PUSH_RULE_SUITE.json \
  --deletion-rule-suite DELETION_RULE_SUITE.json \
  > ENFORCEMENT_V2.json
```

A positive exit requires:

```text
disposition = EnforcementBehaviorallyCorroborated
```

Anything else must remain non-positive.

## 5. Keep origin and authentication separate

`ENFORCEMENT_V2.json` is content-addressed server-readback evidence. It is not a
cryptographic signature by GitHub or an authenticated human/operator identity.

A later root-admission layer should therefore require both:

- the independently expected `EnforcementEvidenceId` from the trusted evidence
  lineage; and
- the exact enforcement-evidence bytes whose ID recomputes to that selection.

Detached authentication remains a separate concern.

## Evidence-state interpretation

```text
EnforcementConfiguredOnly
    exact P0 configuration/readback, no behavioral suite evidence

EnforcementBehavioralEvidencePartial
    one or two exact operation observations, others absent

EnforcementEvidenceRejected
    one or more supplied provider records failed validation

EnforcementBehaviorallyCorroborated
    all three distinct exact active-rule failures observed
```

Do not collapse these to a boolean.

## Non-claims

This procedure does not prove:

- trusted chronology from `pushed_at`;
- universal absence of platform/organization bypasses;
- scientific validity;
- current qualification admission;
- branch protection on a future root;
- that a rule-suite failure was generated by a particular operational ceremony
  unless that ceremony is independently evidenced.
