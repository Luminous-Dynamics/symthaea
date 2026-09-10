# Trusted `main` P0 — effective active-rule verification

This is the second, read-only server-state check for #330 after the detailed
repository-owned P0 ruleset has been created/reconciled and passed
`trusted_main_ruleset.py verify`.

It uses GitHub's **Get rules for a branch** endpoint. GitHub documents that this
endpoint returns all **active** rules that apply to the named branch regardless
of whether the rules originate at repository or organization level; rulesets in
`evaluate` or `disabled` state are not returned.

This is stronger than inspecting the ruleset definition alone, but it remains a
server readback—not a behavioral direct-push test.

## Inputs

Reuse the canonical policy and detailed ruleset readback from
`TRUSTED_MAIN_P0_APPLY.md`:

```bash
POLICY=docs/security/trusted-main-protection-p0.v1.json
# RULESET_READBACK must be the exact detailed repository-owned trusted-main-p0
# ruleset JSON already selected and verified by the primary P0 runbook.
```

## Capture GitHub's effective active rules for `main`

```bash
set -euo pipefail

EFFECTIVE_RULES="$(mktemp)"
gh api \
  -H 'Accept: application/vnd.github+json' \
  -H 'X-GitHub-Api-Version: 2026-03-10' \
  '/repos/Luminous-Dynamics/symthaea/rules/branches/main?per_page=100' \
  > "$EFFECTIVE_RULES"
```

Do not filter the JSON before preserving it. Organization-level rules are useful
context and should remain visible in the raw evidence even though they cannot
substitute for the repository-owned P0 identity.

## Verify the repository-owned P0 projection

```bash
EFFECTIVE_VERIFICATION="$(mktemp)"
python3 scripts/trusted_main_effective_rules.py \
  "$POLICY" \
  "$RULESET_READBACK" \
  "$EFFECTIVE_RULES" \
  > "$EFFECTIVE_VERIFICATION"

cat "$EFFECTIVE_VERIFICATION"
jq -e '.disposition == "P0EffectiveRulesSatisfied"' \
  "$EFFECTIVE_VERIFICATION" >/dev/null
```

A positive result requires the exact detailed `trusted-main-p0` ruleset ID to
contribute exactly one active rule of each required type:

```text
deletion
non_fast_forward
pull_request
```

and every matching rule must report:

```text
ruleset_source_type = Repository
ruleset_source      = Luminous-Dynamics/symthaea
```

Rules from other ruleset IDs—including inherited organization rules—are not
counted toward P0. Any additional active rule emitted by the **same P0 ruleset**
is treated as P0 policy drift and rejects the theorem.

## Evidence semantics

Preserve together:

```text
P0 policy_id
primary structural verification_id
P0 ruleset_id
raw effective-rules JSON
effective-rules verification_id
```

The two read-only theorems mean different things:

```text
P0StructurallySatisfied
  = the repository-owned ruleset definition/readback matches reviewed P0

P0EffectiveRulesSatisfied
  = GitHub reports that exact ruleset's required rules as active on main
```

Neither means:

```text
an ordinary non-bypass direct push was behaviorally observed to fail
```

That remains separate #330 evidence and should not be simulated using an
administrator/bypass identity.

## Exact verifier execution evidence

The new verifier and tests were materialized locally and their Git blob IDs were
independently recomputed to match the repository blobs exactly:

```text
scripts/trusted_main_effective_rules.py
  git blob: 1917f5f8f36a98d63703c959f2d1980ef3eac43b

tests/python/test_trusted_main_effective_rules.py
  git blob: 2183650ec26e26527d9b51479ba210b63580c01c

unittest result: 9 passed / 0 failed
```

This execution qualifies the verifier implementation only. It does not claim
that P0 is currently applied to GitHub.
