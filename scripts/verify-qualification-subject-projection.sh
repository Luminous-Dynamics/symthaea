#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Prove that a qualification commit is an exact source projection of its
# promoted candidate after removing only explicitly named verifier-harness
# paths. This proves repository-source equivalence only; it grants no runtime,
# operational, policy, availability, or scientific authority.

set -euo pipefail

if [[ "$#" -lt 3 ]]; then
    echo "usage: $0 <candidate-sha> <qualifier-sha> <qualifier-only-path>..." >&2
    exit 2
fi

candidate_sha="$1"
qualifier_sha="$2"
shift 2
excluded_paths=("$@")
receipt_path="${QUALIFICATION_SUBJECT_RECEIPT:-${TMPDIR:-/tmp}/qualification-subject-projection-v1.tsv}"
status="FAIL"
stage="arguments"
candidate_tree="unavailable"
qualifier_tree="unavailable"
candidate_projection_tree="unavailable"
qualifier_projection_tree="unavailable"
projection_manifest_sha256="unavailable"
verifier_sha="$(git rev-parse HEAD 2>/dev/null || printf 'unavailable')"
verifier_tree="$(git rev-parse 'HEAD^{tree}' 2>/dev/null || printf 'unavailable')"

sha256_file() {
    local path="$1"
    if [[ -f "$path" ]]; then
        sha256sum "$path" | awk '{print $1}'
    else
        printf 'unavailable'
    fi
}

write_receipt() {
    local exit_code="$1"
    local final_status="$status"
    local terminal_stage="$stage"
    local tmp_path="${receipt_path}.tmp.$$"

    set +e
    [[ "$exit_code" -eq 0 ]] || final_status="FAIL"
    [[ "$final_status" != "PASS" ]] || terminal_stage="none"
    mkdir -p "$(dirname "$receipt_path")" || return 1
    {
        printf 'schema\tqualification-subject-projection-v1\n'
        printf 'status\t%s\n' "$final_status"
        printf 'exit_code\t%s\n' "$exit_code"
        printf 'terminal_stage\t%s\n' "$terminal_stage"
        printf 'scope\trepository-source-projection-only\n'
        printf 'runtime_authority\tnone\n'
        printf 'operational_authority\tnone\n'
        printf 'policy_authority\tnone\n'
        printf 'availability_authority\tnone\n'
        printf 'scientific_authority\tnone\n'
        printf 'candidate_sha\t%s\n' "$candidate_sha"
        printf 'candidate_tree\t%s\n' "$candidate_tree"
        printf 'qualifier_sha\t%s\n' "$qualifier_sha"
        printf 'qualifier_tree\t%s\n' "$qualifier_tree"
        printf 'candidate_projection_tree\t%s\n' "$candidate_projection_tree"
        printf 'qualifier_projection_tree\t%s\n' "$qualifier_projection_tree"
        printf 'projection_manifest_sha256\t%s\n' "$projection_manifest_sha256"
        printf 'git_object_format\t%s\n' "$(git rev-parse --show-object-format 2>/dev/null || printf 'unavailable')"
        printf 'excluded_path_count\t%s\n' "${#excluded_paths[@]}"
        local path
        for path in "${excluded_paths[@]}"; do
            printf 'excluded_qualifier_only_path\t%s\n' "$path"
        done
        printf 'verifier_sha\t%s\n' "$verifier_sha"
        printf 'verifier_tree\t%s\n' "$verifier_tree"
        printf 'verifier_script_sha256\t%s\n' "$(sha256_file scripts/verify-qualification-subject-projection.sh)"
        printf 'verifier_workflow_sha256\t%s\n' "$(sha256_file .github/workflows/qualification-subject-projection.yml)"
        printf 'github_run_id\t%s\n' "${GITHUB_RUN_ID:-not-applicable}"
        printf 'github_run_attempt\t%s\n' "${GITHUB_RUN_ATTEMPT:-not-applicable}"
    } > "$tmp_path" || { rm -f "$tmp_path"; return 1; }
    mv "$tmp_path" "$receipt_path" || { rm -f "$tmp_path"; return 1; }
    echo "qualification-subject-projection receipt=$receipt_path status=$final_status stage=$terminal_stage"
}

finish() {
    local exit_code=$?
    trap - EXIT
    if [[ "$exit_code" -eq 0 && "$status" != "PASS" ]]; then
        echo "error: verifier exited without terminal PASS (stage=$stage)" >&2
        exit_code=1
    fi
    write_receipt "$exit_code" || {
        echo "error: projection receipt could not be persisted" >&2
        [[ "$exit_code" -ne 0 ]] || exit_code=1
    }
    exit "$exit_code"
}
trap finish EXIT

stage="validate_exact_shas"
sha_re='^[0-9a-f]{40}$'
[[ "$candidate_sha" =~ $sha_re ]] || { echo 'error: candidate must be an exact lowercase 40-hex commit SHA' >&2; exit 1; }
[[ "$qualifier_sha" =~ $sha_re ]] || { echo 'error: qualifier must be an exact lowercase 40-hex commit SHA' >&2; exit 1; }
[[ "$candidate_sha" != "$qualifier_sha" ]] || { echo 'error: candidate and qualifier SHAs must be distinct' >&2; exit 1; }

git cat-file -e "${candidate_sha}^{commit}"
git cat-file -e "${qualifier_sha}^{commit}"
[[ "$(git rev-parse "${candidate_sha}^{commit}")" == "$candidate_sha" ]]
[[ "$(git rev-parse "${qualifier_sha}^{commit}")" == "$qualifier_sha" ]]
candidate_tree="$(git rev-parse "${candidate_sha}^{tree}")"
qualifier_tree="$(git rev-parse "${qualifier_sha}^{tree}")"

stage="validate_exclusion_policy"
declare -A seen=()
for path in "${excluded_paths[@]}"; do
    [[ -n "$path" && "$path" != /* && "$path" != *$'\n'* && "$path" != *$'\t'* ]] || {
        echo "error: invalid excluded path: $path" >&2
        exit 1
    }
    [[ "$path" != ".." && "$path" != ../* && "$path" != */../* && "$path" != */.. ]] || {
        echo "error: parent traversal is forbidden in excluded paths: $path" >&2
        exit 1
    }
    [[ -z "${seen[$path]+x}" ]] || { echo "error: duplicate excluded path: $path" >&2; exit 1; }
    seen[$path]=1

    # Exclusions are deliberately qualifier-only harness files. Refuse to
    # exclude any path that already exists in the promoted candidate.
    if git cat-file -e "${candidate_sha}:$path" 2>/dev/null; then
        echo "error: excluded path exists in candidate and could mask source drift: $path" >&2
        exit 1
    fi
    [[ "$(git cat-file -t "${qualifier_sha}:$path" 2>/dev/null || true)" == "blob" ]] || {
        echo "error: excluded qualifier-only path is absent or not a blob: $path" >&2
        exit 1
    }
done

stage="first_parent_binding"
first_parent="$(git rev-parse "${qualifier_sha}^1")"
[[ "$first_parent" == "$candidate_sha" ]] || {
    echo "error: qualifier first parent is not the promoted candidate" >&2
    echo "candidate=$candidate_sha first_parent=$first_parent" >&2
    exit 1
}

stage="exact_difference_set"
mapfile -t actual_diff < <(git diff --name-only --no-renames "$candidate_sha" "$qualifier_sha" | LC_ALL=C sort)
mapfile -t expected_diff < <(printf '%s\n' "${excluded_paths[@]}" | LC_ALL=C sort)
if [[ "${#actual_diff[@]}" -ne "${#expected_diff[@]}" ]]; then
    echo 'error: qualifier/candidate difference count exceeds exact harness allowlist' >&2
    printf 'actual: %s\n' "${actual_diff[*]}" >&2
    printf 'expected: %s\n' "${expected_diff[*]}" >&2
    exit 1
fi
for i in "${!actual_diff[@]}"; do
    [[ "${actual_diff[$i]}" == "${expected_diff[$i]}" ]] || {
        echo 'error: qualifier/candidate differences are not exactly the harness allowlist' >&2
        printf 'actual: %s\n' "${actual_diff[*]}" >&2
        printf 'expected: %s\n' "${expected_diff[*]}" >&2
        exit 1
    }
done

project_tree() {
    local ref="$1"
    local index
    index="$(mktemp)"
    rm -f "$index"
    GIT_INDEX_FILE="$index" git read-tree "${ref}^{tree}"
    local path
    for path in "${excluded_paths[@]}"; do
        GIT_INDEX_FILE="$index" git update-index --force-remove -- "$path"
    done
    GIT_INDEX_FILE="$index" git write-tree
    rm -f "$index"
}

stage="projected_tree_identity"
candidate_projection_tree="$(project_tree "$candidate_sha")"
qualifier_projection_tree="$(project_tree "$qualifier_sha")"
[[ "$candidate_projection_tree" == "$qualifier_projection_tree" ]] || {
    echo 'error: projected Git trees differ after exact harness removal' >&2
    exit 1
}

stage="canonical_projection_manifest"
candidate_manifest="$(mktemp)"
qualifier_manifest="$(mktemp)"
cleanup_manifests() { rm -f "$candidate_manifest" "$qualifier_manifest"; }
trap 'cleanup_manifests; finish' EXIT
python3 - "$candidate_sha" "$qualifier_sha" "$candidate_manifest" "$qualifier_manifest" "${excluded_paths[@]}" <<'PY'
import subprocess
import sys

candidate, qualifier, candidate_out, qualifier_out, *excluded = sys.argv[1:]
excluded_set = set(excluded)

def manifest(ref: str) -> bytes:
    raw = subprocess.check_output(["git", "ls-tree", "-rz", "--full-tree", ref])
    entries = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        meta, path = record.split(b"\t", 1)
        decoded = path.decode("utf-8")
        if decoded in excluded_set:
            continue
        mode, kind, oid = meta.split(b" ", 2)
        entries.append((path, mode, kind, oid))
    entries.sort(key=lambda item: item[0])
    out = bytearray(b"schema\tqualification-subject-projection-manifest-v1\n")
    for path, mode, kind, oid in entries:
        if b"\n" in path or b"\t" in path:
            raise SystemExit("repository path contains unsupported tab/newline for V1 canonical manifest")
        out.extend(mode + b"\t" + kind + b"\t" + oid + b"\t" + path + b"\n")
    return bytes(out)

left = manifest(candidate)
right = manifest(qualifier)
open(candidate_out, "wb").write(left)
open(qualifier_out, "wb").write(right)
if left != right:
    raise SystemExit("canonical projected ls-tree manifests differ")
PY
projection_manifest_sha256="$(sha256sum "$candidate_manifest" | awk '{print $1}')"
cleanup_manifests
trap finish EXIT

stage="clean_verifier_checkout"
if ! git diff --quiet --ignore-submodules -- || ! git diff --cached --quiet --ignore-submodules --; then
    echo 'error: verifier checkout has tracked modifications' >&2
    exit 1
fi
if [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
    echo 'error: verifier checkout has untracked files' >&2
    exit 1
fi

status="PASS"
stage="complete"
