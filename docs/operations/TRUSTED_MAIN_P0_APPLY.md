# Trusted `main` P0 protection — apply and read back

This runbook applies the minimal **P0** GitHub ruleset defined by
`docs/security/trusted-main-protection-p0.v1.json` and then verifies the
server readback with `scripts/trusted_main_ruleset.py`.

P0 is intentionally small. It protects the trust root without coupling merges
to the currently saturated full CI matrix. It does **not** install P1 required
qualification checks, activate a trusted self-hosted runner, or qualify any
source/scientific result.

## Preconditions

- Work from a reviewed clean checkout of the exact P0 branch/commit.
- `gh auth status` must show an identity with repository Administration write
  permission. Do not paste tokens into the repository or shell history.
- Confirm the repository is exactly `Luminous-Dynamics/symthaea`.
- Review the rendered payload before applying it.
- Do not add bypass actors unless they have been explicitly reviewed and added
  to the canonical policy first.

## 1. Render the reviewed payload

`render` emits a provenance wrapper containing both the content-addressed policy
ID and the actual GitHub API request. **Only `.request` is submitted to GitHub.**

```bash
set -euo pipefail

POLICY=docs/security/trusted-main-protection-p0.v1.json
RENDERED="$(mktemp)"
PAYLOAD="$(mktemp)"

python3 scripts/trusted_main_ruleset.py render "$POLICY" > "$RENDERED"
POLICY_ID="$(jq -er '.policy_id' "$RENDERED")"
jq -e '.request | type == "object"' "$RENDERED" >/dev/null
jq '.request' "$RENDERED" > "$PAYLOAD"

printf 'policy_id=%s\n' "$POLICY_ID"
cat "$PAYLOAD"
```

The request body must contain exactly the P0 intent: an active branch ruleset
for `refs/heads/main`, no exclusions, no bypass actors, and the three rules
`deletion`, `non_fast_forward`, and `pull_request`. P0 intentionally has no
required-status-check or required-workflow rule.

Do not pass `$RENDERED` itself to `gh api --input`; its `policy_id` wrapper is
provenance metadata, not part of GitHub's create-ruleset request schema.

## 2. Select the repository-owned P0 rule before applying anything

GitHub's repository ruleset listing includes inherited parent rulesets by
default. P0 is specifically a **repository-owned** trust root, so first query
with `includes_parents=false` and resolve the name case-insensitively.

```bash
REPO_RULESETS="$(mktemp)"
gh api \
  -H 'Accept: application/vnd.github+json' \
  -H 'X-GitHub-Api-Version: 2026-03-10' \
  '/repos/Luminous-Dynamics/symthaea/rulesets?includes_parents=false' \
  > "$REPO_RULESETS"

MATCH_COUNT="$(jq '[.[] | select((.name | ascii_downcase) == "trusted-main-p0")] | length' "$REPO_RULESETS")"
case "$MATCH_COUNT" in
  0) printf '%s\n' 'no existing repository-owned trusted-main-p0 ruleset' ;;
  1) printf '%s\n' 'one existing repository-owned trusted-main-p0 ruleset found' ;;
  *) printf 'error: ambiguous repository-owned trusted-main-p0 rulesets: %s\n' "$MATCH_COUNT" >&2; exit 1 ;;
esac
```

A same-named organization ruleset is not a match here. It may be useful defense
in depth, but it does not satisfy this repository-owned P0 policy.

## 3. Create or reconcile the repository-owned rule

If `MATCH_COUNT=0`, create the reviewed **inner request body**:

```bash
if [[ "$MATCH_COUNT" == 0 ]]; then
  CREATED="$(mktemp)"
  gh api \
    --method POST \
    -H 'Accept: application/vnd.github+json' \
    -H 'X-GitHub-Api-Version: 2026-03-10' \
    /repos/Luminous-Dynamics/symthaea/rulesets \
    --input "$PAYLOAD" > "$CREATED"

  RULESET_ID="$(jq -er '.id' "$CREATED")"
  printf 'created repository-owned ruleset id=%s\n' "$RULESET_ID"
else
  RULESET_ID="$(jq -er '[.[] | select((.name | ascii_downcase) == "trusted-main-p0")][0].id' "$REPO_RULESETS")"
  printf 'existing repository-owned ruleset id=%s; read back and reconcile before changing it\n' "$RULESET_ID"
fi
```

If an existing rule differs from the rendered policy, do not silently create a
second overlapping rule. Review the delta and update the existing rule through
an explicitly authorized administration change.

## 4. Capture fresh server readbacks

```bash
RULESET_READBACK="$(mktemp)"
BRANCH_READBACK="$(mktemp)"
REPO_READBACK="$(mktemp)"
ALL_RULESETS="$(mktemp)"

gh api \
  -H 'Accept: application/vnd.github+json' \
  -H 'X-GitHub-Api-Version: 2026-03-10' \
  "/repos/Luminous-Dynamics/symthaea/rulesets/$RULESET_ID" \
  > "$RULESET_READBACK"

gh api \
  -H 'Accept: application/vnd.github+json' \
  -H 'X-GitHub-Api-Version: 2026-03-10' \
  /repos/Luminous-Dynamics/symthaea/branches/main \
  > "$BRANCH_READBACK"

gh api \
  -H 'Accept: application/vnd.github+json' \
  -H 'X-GitHub-Api-Version: 2026-03-10' \
  /repos/Luminous-Dynamics/symthaea \
  > "$REPO_READBACK"

# Separate audit surface: include inherited organization/parent rulesets.
gh api \
  -H 'Accept: application/vnd.github+json' \
  -H 'X-GitHub-Api-Version: 2026-03-10' \
  '/repos/Luminous-Dynamics/symthaea/rulesets?includes_parents=true' \
  > "$ALL_RULESETS"
```

## 5. Run structural verification

Repository ownership is verifier-owned now. `P0StructurallySatisfied` requires
all of the following from the detailed GitHub ruleset readback:

```text
source_type = Repository
source      = Luminous-Dynamics/symthaea
name        = trusted-main-p0
enforcement = active
```

along with the exact ref/rule/bypass/PR-policy checks.

```bash
VERIFICATION="$(mktemp)"
python3 scripts/trusted_main_ruleset.py verify \
  "$POLICY" \
  "$RULESET_READBACK" \
  "$BRANCH_READBACK" \
  "$REPO_READBACK" \
  > "$VERIFICATION"

cat "$VERIFICATION"
jq -e '
  .disposition == "P0StructurallySatisfied" and
  .policy_id == $policy_id and
  .ruleset_source_type == "Repository" and
  .ruleset_source == "Luminous-Dynamics/symthaea"
' --arg policy_id "$POLICY_ID" "$VERIFICATION" >/dev/null
```

Structural verification is necessary but not sufficient. Preserve the exact
policy ID, ruleset ID, ruleset source, repository ID, and verification ID in
the #330 evidence record. Preserve `ALL_RULESETS` separately so inherited
organization rules remain visible as defense-in-depth context rather than being
silently conflated with the repository root.

### Verifier execution evidence for this tranche

The verifier regression suite was executed locally against materialized source
bytes whose Git blob IDs were independently recomputed and matched GitHub's
repository blobs exactly:

```text
scripts/trusted_main_ruleset.py
  git blob: 0ecebc5aae80633325b23df2c7ad01bc42804cbf

tests/python/test_trusted_main_ruleset.py
  git blob: 55e599aa7e0d772eb44b5f19cbd96e5aed39aa5a

unittest result: 24 passed / 0 failed
```

This is executable evidence about the P0 verifier implementation only. It does
**not** prove that GitHub has applied the ruleset, that `main` is currently
protected, or that a non-bypass direct push has been rejected server-side.

## 6. Do **not** conflate structural readback with a destructive negative test

Do not test protection by casually attempting a real direct push or force-push
to `main` from an administrator account. If that identity has bypass authority,
the result is ambiguous; if protection is misconfigured, the test could mutate
the trust root.

#330's separate negative-enforcement requirement should be satisfied using a
reviewed non-bypass test identity or another GitHub-supported server-side test
that cannot mutate `main` on failure. Until that evidence exists, record:

```text
P0 structural readback: satisfied
negative direct-push enforcement: not evaluated
```

## 7. Re-read after any ruleset edit

Any later change to the P0 ruleset, bypass actors, target conditions, source
ownership, repository identity, or protection mechanism invalidates the
previous structural readback. Capture a new readback and verification lineage.
P1 admission checks should be introduced as a separate reviewed policy
transition rather than silently added to P0.
