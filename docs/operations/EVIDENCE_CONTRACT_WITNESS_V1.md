# Evidence Contract Witness V1 — base-owned execution contract

This document describes the trusted/base-owned phase that may eventually consume
a low-trust `Evidence Contract Qualification` candidate receipt.

It is **not** a workflow installation and does not authorize #1119's bootstrap.
Until #330 protects the predecessor trust root, this remains design/conformance
infrastructure only.

## Core rule

The trusted phase executes **no candidate code**. Candidate workflow files,
scripts, Git trees, and artifacts are hostile data. The trusted phase derives
provider metadata and recipe identities independently, then supplies those facts
to `scripts/evidence_contract_witness.py`.

```text
candidate self-report
    != trusted observation
```

Never source/eval/import a candidate-provided file in the trusted phase.

## Required trusted inputs

For one candidate run, independently obtain:

- repository numeric ID and full name;
- exact candidate head SHA and Git tree SHA;
- Actions run ID and run attempt;
- provider run conclusion and workflow ID;
- exact receipt artifact ID and canonical artifact name;
- SHA-256 of the downloaded artifact ZIP bytes;
- SHA-256 of candidate `scripts/qualify-evidence-contract.sh` bytes;
- SHA-256 of candidate `.github/workflows/evidence-contract.yml` bytes;
- complete expected `butlin_*authority*_regression.rs` target list;
- exact predecessor SHA/tree and predecessor recipe hashes, unless this is the
  explicit first bootstrap.

The receipt is not authoritative for any of those facts.

## 1. Read the provider run

```bash
set -euo pipefail

REPO=Luminous-Dynamics/symthaea
RUN_ID='<candidate run id>'
RUN_JSON="$(mktemp)"

gh api \
  -H 'Accept: application/vnd.github+json' \
  -H 'X-GitHub-Api-Version: 2026-03-10' \
  "/repos/$REPO/actions/runs/$RUN_ID" > "$RUN_JSON"

HEAD_SHA="$(jq -er '.head_sha' "$RUN_JSON")"
RUN_ATTEMPT="$(jq -er '.run_attempt' "$RUN_JSON")"
RUN_CONCLUSION="$(jq -er '.conclusion' "$RUN_JSON")"
WORKFLOW_ID="$(jq -er '.workflow_id' "$RUN_JSON")"
REPOSITORY_ID="$(jq -er '.repository.id' "$RUN_JSON")"
REPOSITORY_NAME="$(jq -er '.repository.full_name' "$RUN_JSON")"

test "$REPOSITORY_NAME" = "$REPO"
test "$REPOSITORY_ID" = 1136141775
```

Do not accept a receipt whose run ID/attempt disagrees with this provider
record.

## 2. Resolve the exact candidate tree without checking out or executing it

```bash
COMMIT_JSON="$(mktemp)"
gh api \
  -H 'Accept: application/vnd.github+json' \
  -H 'X-GitHub-Api-Version: 2026-03-10' \
  "/repos/$REPO/git/commits/$HEAD_SHA" > "$COMMIT_JSON"

HEAD_TREE="$(jq -er '.tree.sha' "$COMMIT_JSON")"
```

The Git tree SHA, not a synthetic merge ref, is the source-tree subject.

## 3. Hash candidate recipe bytes independently

Fetch the two candidate files as data via the GitHub Contents API. Do not run
either file.

```bash
hash_candidate_file() {
  local path="$1"
  gh api \
    -H 'Accept: application/vnd.github+json' \
    -H 'X-GitHub-Api-Version: 2026-03-10' \
    "/repos/$REPO/contents/$path?ref=$HEAD_SHA" \
    --jq '.content' \
    | tr -d '\n' \
    | base64 --decode \
    | sha256sum \
    | awk '{print $1}'
}

CANDIDATE_QUALIFIER_SHA256="$(hash_candidate_file scripts/qualify-evidence-contract.sh)"
CANDIDATE_WORKFLOW_SHA256="$(hash_candidate_file .github/workflows/evidence-contract.yml)"
```

If the API response is not a regular file or does not contain the expected
content encoding, fail closed rather than falling back to receipt-provided
hashes.

## 4. Derive the authority-test target set from a complete Git tree

Use the candidate tree as data. GitHub's recursive tree response can be
truncated for large trees; truncation is **not** permission to infer that no
additional authority tests exist.

```bash
TREE_JSON="$(mktemp)"
gh api \
  -H 'Accept: application/vnd.github+json' \
  -H 'X-GitHub-Api-Version: 2026-03-10' \
  "/repos/$REPO/git/trees/$HEAD_TREE?recursive=1" > "$TREE_JSON"

jq -e '.truncated == false' "$TREE_JSON" >/dev/null

EXPECTED_TARGETS="$({
  jq -r '.tree[] | select(.type == "blob") | .path' "$TREE_JSON" \
    | sed -nE 's#^crates/domains/symthaea-psych-bench/tests/(butlin_[A-Za-z0-9_.-]*authority[A-Za-z0-9_.-]*_regression)\.rs$#\1#p' \
    | LC_ALL=C sort -u
} | paste -sd, -)"

if [[ -z "$EXPECTED_TARGETS" ]]; then
  EXPECTED_TARGETS=none
fi
```

A future complete-diff/router primitive may replace this discovery mechanism,
but it must preserve the same fail-closed completeness theorem.

## 5. Resolve exactly one canonical receipt artifact

```bash
ARTIFACTS_JSON="$(mktemp)"
gh api \
  -H 'Accept: application/vnd.github+json' \
  -H 'X-GitHub-Api-Version: 2026-03-10' \
  "/repos/$REPO/actions/runs/$RUN_ID/artifacts" > "$ARTIFACTS_JSON"

ARTIFACT_NAME="evidence-contract-qualification-$RUN_ID-$RUN_ATTEMPT"
MATCH_COUNT="$(jq --arg name "$ARTIFACT_NAME" '[.artifacts[] | select(.name == $name and .expired == false)] | length' "$ARTIFACTS_JSON")"
test "$MATCH_COUNT" = 1

ARTIFACT_ID="$(jq -er --arg name "$ARTIFACT_NAME" '.artifacts[] | select(.name == $name and .expired == false) | .id' "$ARTIFACTS_JSON")"
PROVIDER_ARCHIVE_DIGEST="$(jq -er --arg name "$ARTIFACT_NAME" '.artifacts[] | select(.name == $name and .expired == false) | .digest' "$ARTIFACTS_JSON")"
```

The provider digest and downloaded ZIP digest are different from the receipt
content digest. Keep all three identities separate.

## 6. Download and inspect the ZIP without extracting candidate paths

```bash
ARTIFACT_ZIP="$(mktemp)"
gh api \
  -H 'Accept: application/vnd.github+json' \
  -H 'X-GitHub-Api-Version: 2026-03-10' \
  "/repos/$REPO/actions/artifacts/$ARTIFACT_ID/zip" > "$ARTIFACT_ZIP"

ARCHIVE_SHA256="$(sha256sum "$ARTIFACT_ZIP" | awk '{print $1}')"
test "$PROVIDER_ARCHIVE_DIGEST" = "sha256:$ARCHIVE_SHA256"

RECEIPT="$(mktemp)"
python3 - "$ARTIFACT_ZIP" "$RECEIPT" <<'PY'
import pathlib
import sys
import zipfile

archive = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
expected = "evidence-contract-qualification-v1.tsv"

with zipfile.ZipFile(archive) as zf:
    infos = zf.infolist()
    if len(infos) != 1 or infos[0].filename != expected or infos[0].is_dir():
        raise SystemExit("artifact must contain exactly the canonical receipt file")
    # Read member bytes directly; do not extract archive-controlled paths.
    data = zf.read(infos[0])
out.write_bytes(data)
PY
```

Do not use `unzip` into a trusted workspace for candidate-controlled archives.

## 7. Compare against the predecessor recipe root

For ordinary post-bootstrap witnessing, independently hash the trusted
predecessor's workflow and qualifier bytes using the same data-only mechanism.
Bind those hashes together with the exact predecessor commit/tree.

A moving `main` is not automatically the predecessor for an older candidate.
The predecessor root must be selected from the attempt's historical lineage.
Current admissibility of that root is a separate #931 decision.

For #1119 itself, use only `--bootstrap-no-predecessor`. Its successful candidate
run must remain `BootstrapNoPredecessor`, never self-authorized.

## 8. Invoke the strict witness verifier

Conceptually:

```bash
python3 scripts/evidence_contract_witness.py "$RECEIPT" \
  --expected-subject-sha "$HEAD_SHA" \
  --expected-subject-tree "$HEAD_TREE" \
  --expected-run-id "$RUN_ID" \
  --expected-run-attempt "$RUN_ATTEMPT" \
  --expected-run-conclusion "$RUN_CONCLUSION" \
  --expected-workflow-id "$WORKFLOW_ID" \
  --expected-artifact-id "$ARTIFACT_ID" \
  --expected-artifact-name "$ARTIFACT_NAME" \
  --expected-artifact-archive-sha256 "$ARCHIVE_SHA256" \
  --expected-repository "$REPOSITORY_NAME" \
  --expected-repository-id "$REPOSITORY_ID" \
  --expected-candidate-qualifier-sha256 "$CANDIDATE_QUALIFIER_SHA256" \
  --expected-candidate-workflow-sha256 "$CANDIDATE_WORKFLOW_SHA256" \
  --expected-authority-integration-targets "$EXPECTED_TARGETS" \
  --trusted-predecessor-sha "$TRUSTED_PREDECESSOR_SHA" \
  --trusted-predecessor-tree "$TRUSTED_PREDECESSOR_TREE" \
  --trusted-qualifier-sha256 "$TRUSTED_QUALIFIER_SHA256" \
  --trusted-workflow-sha256 "$TRUSTED_WORKFLOW_SHA256"
```

A non-authorizing but structurally valid disposition must not be turned into a
successful admission by shell glue. Preserve the verifier's exit semantics.

## Non-claims

A valid witness object still does not establish scientific evidence, full-repo
CI success, environment reproducibility, detached signer identity, or current
admission. Those remain separate layers (#905/#955/#931).
