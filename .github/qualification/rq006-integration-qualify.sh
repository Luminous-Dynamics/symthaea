#!/usr/bin/env bash
set -euo pipefail

manifest="Cargo.toml"
overlay=".github/qualification/rq006-example-targets.toml.fragment"
work="${RUNNER_TEMP:-/tmp}/rq006-integration"
metadata="${RUNNER_TEMP:-/tmp}/rq006-cargo-metadata.json"
backup="$work/Cargo.toml.product"
projected="$work/Cargo.toml.projected"
projection_diff="$work/Cargo.toml.projection.diff"
mkdir -p "$work"

[[ -f "$manifest" ]]
[[ -f "$overlay" ]]

# Reject accidental scope growth in the qualification overlay itself. Comments and
# whitespace are harmless; the parsed TOML may contain only the two exact example
# tables below, with no features, dependencies, package metadata, or other targets.
python3 - "$overlay" <<'PY'
import pathlib
import sys
import tomllib

overlay_path = pathlib.Path(sys.argv[1])
data = tomllib.loads(overlay_path.read_text())
if set(data) != {"example"}:
    raise SystemExit(f"unexpected qualification-overlay keys: {sorted(data)}")
expected = [
    {
        "name": "benchmark_metacognitive_context",
        "path": "examples/benchmark_metacognitive_context.rs",
    },
    {
        "name": "build_reasoning_capability_matrix",
        "path": "examples/build_reasoning_capability_matrix.rs",
    },
]
if data["example"] != expected:
    raise SystemExit(f"qualification-overlay target set changed: {data['example']!r}")
PY

cp "$manifest" "$backup"
product_manifest_sha256="$(sha256sum "$manifest" | awk '{print $1}')"
overlay_sha256="$(sha256sum "$overlay" | awk '{print $1}')"

restore_manifest() {
  cp "$backup" "$manifest"
  restored_sha256="$(sha256sum "$manifest" | awk '{print $1}')"
  if [[ "$restored_sha256" != "$product_manifest_sha256" ]]; then
    echo "failed to restore frozen product Cargo.toml" >&2
    exit 91
  fi
}
trap restore_manifest EXIT

# Cargo resolves every globbed workspace member before package scoping. The frozen
# repository currently contains unrelated assurance manifests that cannot inherit
# missing workspace.package fields, so an unprojected `cargo -p symthaea` fails before
# touching the RQ-006 subject. Narrow only the workspace membership frontier; preserve
# the package, dependency tables, features, profiles, source paths, and lockfile.
python3 - "$manifest" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
lines = path.read_text().splitlines(keepends=True)

def replace_workspace_array(lines, key, replacement):
    workspace_start = next(i for i, line in enumerate(lines) if line.strip() == "[workspace]")
    workspace_end = next(
        i
        for i, line in enumerate(lines[workspace_start + 1 :], workspace_start + 1)
        if line.startswith("[") and line.strip() != "[workspace]"
    )
    start = None
    for i in range(workspace_start + 1, workspace_end):
        if lines[i].strip().startswith(f"{key} ="):
            start = i
            break
    if start is None:
        raise SystemExit(f"missing [workspace] {key}")

    balance = 0
    end = None
    for i in range(start, workspace_end):
        # The frozen membership arrays contain quoted paths/comments only; bracket
        # balance therefore identifies the exact TOML array extent deterministically.
        balance += lines[i].count("[") - lines[i].count("]")
        if balance == 0:
            end = i
            break
    if end is None:
        raise SystemExit(f"unterminated [workspace] {key} array")
    return lines[:start] + [replacement + "\n"] + lines[end + 1 :]

lines = replace_workspace_array(lines, "members", "members = []")
lines = replace_workspace_array(lines, "default-members", 'default-members = ["."]')
path.write_text("".join(lines))
PY

cat "$overlay" >> "$manifest"
cp "$manifest" "$projected"
diff -u "$backup" "$projected" > "$projection_diff" || true

# Prove that the temporary projection changed only the intended membership frontier
# plus an exact append of the two qualification examples. Existing frozen example
# declarations are product semantics and must remain present, ordered, and unchanged.
python3 - "$backup" "$projected" <<'PY'
import copy
import pathlib
import sys
import tomllib

before_path, after_path = map(pathlib.Path, sys.argv[1:])
before = tomllib.loads(before_path.read_text())
after = tomllib.loads(after_path.read_text())

expected_examples = [
    {"name": "benchmark_metacognitive_context", "path": "examples/benchmark_metacognitive_context.rs"},
    {"name": "build_reasoning_capability_matrix", "path": "examples/build_reasoning_capability_matrix.rs"},
]
before_examples = before.get("example", [])
after_examples = after.get("example", [])
if not isinstance(before_examples, list) or not isinstance(after_examples, list):
    raise SystemExit("root example declarations are not TOML array-of-table lists")
if after_examples != before_examples + expected_examples:
    raise SystemExit("projected manifest did not preserve frozen examples plus the exact RQ-006 append")

before_cmp = copy.deepcopy(before)
after_cmp = copy.deepcopy(after)
if "example" in before_cmp:
    after_cmp["example"] = copy.deepcopy(before_cmp["example"])
else:
    after_cmp.pop("example", None)
for doc in (before_cmp, after_cmp):
    workspace = doc.get("workspace")
    if not isinstance(workspace, dict):
        raise SystemExit("missing [workspace]")
    workspace.pop("members", None)
    workspace.pop("default-members", None)
if before_cmp != after_cmp:
    raise SystemExit("qualification projection changed manifest semantics outside workspace membership/examples")

workspace = after.get("workspace", {})
if workspace.get("members") != []:
    raise SystemExit(f"projected workspace members changed: {workspace.get('members')!r}")
if workspace.get("default-members") != ["."]:
    raise SystemExit(f"projected default-members changed: {workspace.get('default-members')!r}")
print(f"manifest_projection=PASS members=0 default_members=root existing_examples={len(before_examples)} appended_examples=2")
PY

projected_manifest_sha256="$(sha256sum "$projected" | awk '{print $1}')"
projection_diff_sha256="$(sha256sum "$projection_diff" | awk '{print $1}')"

# Cargo now sees the root package and only local path dependencies reachable from it,
# rather than eagerly parsing every globbed repository crate. This is still the exact
# frozen source/dependency graph for the symthaea package; no dependency is stubbed.
cargo metadata --locked --no-deps --format-version 1 > "$metadata"
python3 - "$metadata" "$projection_diff" "$product_manifest_sha256" "$projected_manifest_sha256" "$projection_diff_sha256" <<'PY'
import json
import pathlib
import sys

metadata_path = pathlib.Path(sys.argv[1])
projection_diff = pathlib.Path(sys.argv[2])
product_hash, projected_hash, diff_hash = sys.argv[3:]
data = json.loads(metadata_path.read_text())
roots = [package for package in data["packages"] if package["name"] == "symthaea"]
if len(roots) != 1:
    raise SystemExit(f"expected exactly one symthaea package, got {len(roots)}")
root = roots[0]
examples = {
    target["name"]: target
    for target in root["targets"]
    if "example" in target.get("kind", [])
}
required = {
    "benchmark_metacognitive_context": "examples/benchmark_metacognitive_context.rs",
    "build_reasoning_capability_matrix": "examples/build_reasoning_capability_matrix.rs",
}
for name, suffix in required.items():
    target = examples.get(name)
    if target is None:
        raise SystemExit(f"qualification overlay failed to register: {name}")
    src = pathlib.Path(target["src_path"]).as_posix()
    if not src.endswith(suffix):
        raise SystemExit(f"qualification target {name} resolved unexpected source: {src}")

data["_rq006_qualification_projection"] = {
    "schema": "symthaea.rq006.workspace-membership-projection.v1",
    "product_manifest_sha256": product_hash,
    "projected_manifest_sha256": projected_hash,
    "projection_diff_sha256": diff_hash,
    "workspace_members": [],
    "default_members": ["."],
    "registered_examples": required,
    "unified_diff": projection_diff.read_text(),
}
metadata_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
print(f"target_projection=PASS examples={len(required)} workspace_members={len(data.get('workspace_members', []))}")
PY

# Scope formatting and compilation to the exact root package. Local dependencies are
# compiled normally from their checked-in manifests/source; the source Cargo.lock is
# immutable and --locked prevents dependency reconciliation from changing it.
#
# If the frozen product is not Rust-1.96-formatted, remain RED but emit a deterministic,
# read-only remediation artifact. The artifact is restricted to Rust files already in
# the PR's reviewed reasoning scope, binds each source file to its frozen product blob,
# and restores the checkout before exiting. This does not weaken the format gate and
# cannot push or mint a new product subject.
if ! cargo fmt -p symthaea -- --check; then
  remediation="${RUNNER_TEMP:-/tmp}/rq006-rustfmt-remediation"
  rm -rf "$remediation"
  mkdir -p "$remediation/files"

  cargo fmt -p symthaea
  mapfile -t formatted_paths < <(git diff --name-only --diff-filter=M | grep -E '\.rs$' || true)
  if [[ "${#formatted_paths[@]}" -eq 0 ]]; then
    echo 'rustfmt failed but produced no Rust remediation paths' >&2
    exit 92
  fi

  python3 - "$RUNNER_TEMP/changed-paths.txt" "${formatted_paths[@]}" <<'PYFMT'
import pathlib
import sys

changed = set(pathlib.Path(sys.argv[1]).read_text().splitlines())
paths = sys.argv[2:]
exact = {
    'examples/benchmark_metacognitive_context.rs',
    'examples/build_reasoning_capability_matrix.rs',
    'src/cognitive_loop/accessors/consciousness.rs',
    'src/cognitive_loop/cycle_subsystems.rs',
    'src/cognitive_loop/primitive_tier.rs',
    'src/consciousness/epistemic_conflict/calibrator.rs',
    'src/consciousness/unified_intelligence.rs',
}

def allowed(path: str) -> bool:
    return path in exact or path.startswith('src/intelligence/')

bad_scope = [path for path in paths if not allowed(path)]
not_in_pr = [path for path in paths if path not in changed]
if bad_scope or not_in_pr:
    raise SystemExit(
        f'rustfmt remediation escaped reviewed RQ scope: bad_scope={bad_scope!r} '
        f'not_in_pr={not_in_pr!r}'
    )
PYFMT

  : > "$remediation/formatted-paths.txt"
  printf 'path\tproduct_blob_sha1\tformatted_sha256\tbytes\n' > "$remediation/formatted-census.tsv"
  for path in "${formatted_paths[@]}"; do
    printf '%s\n' "$path" >> "$remediation/formatted-paths.txt"
    mkdir -p "$remediation/files/$(dirname "$path")"
    cp "$path" "$remediation/files/$path"
    printf '%s\t%s\t%s\t%s\n' \
      "$path" \
      "$(git rev-parse "$RQ006_PRODUCT_HEAD:$path")" \
      "$(sha256sum "$path" | awk '{print $1}')" \
      "$(wc -c < "$path" | tr -d ' ')" \
      >> "$remediation/formatted-census.tsv"
  done
  git diff -- "${formatted_paths[@]}" > "$remediation/rustfmt.patch"
  {
    echo 'schema=symthaea.rq006.rustfmt-remediation.v1'
    echo "product_head=$RQ006_PRODUCT_HEAD"
    echo "qualification_head=$(git rev-parse HEAD)"
    echo 'rustfmt_toolchain=1.96.0'
    echo "formatted_path_count=${#formatted_paths[@]}"
    echo "formatted_paths_sha256=$(sha256sum "$remediation/formatted-paths.txt" | awk '{print $1}')"
    echo "formatted_census_sha256=$(sha256sum "$remediation/formatted-census.tsv" | awk '{print $1}')"
    echo "rustfmt_patch_sha256=$(sha256sum "$remediation/rustfmt.patch" | awk '{print $1}')"
    echo 'qualification_result=FAIL_PRODUCT_FORMAT'
  } > "$remediation/remediation-receipt.txt"

  # The workflow already retains the metadata JSON on every outcome. Embed the exact
  # remediation bytes there so no new artifact transport or write permission is needed.
  python3 - "$metadata" "$remediation" <<'PYEMBED'
import base64
import json
import pathlib
import sys

metadata_path = pathlib.Path(sys.argv[1])
remediation = pathlib.Path(sys.argv[2])
data = json.loads(metadata_path.read_text())
rows = []
for line in (remediation / "formatted-census.tsv").read_text().splitlines()[1:]:
    path, product_blob, formatted_sha256, byte_count = line.split("\t")
    payload = (remediation / "files" / path).read_bytes()
    rows.append({
        "path": path,
        "product_blob_sha1": product_blob,
        "formatted_sha256": formatted_sha256,
        "bytes": int(byte_count),
        "content_base64": base64.b64encode(payload).decode("ascii"),
    })
data["_rq006_rustfmt_remediation"] = {
    "schema": "symthaea.rq006.rustfmt-remediation.v1",
    "qualification_result": "FAIL_PRODUCT_FORMAT",
    "rustfmt_toolchain": "1.96.0",
    "product_head": pathlib.Path(remediation / "remediation-receipt.txt").read_text().split("product_head=", 1)[1].splitlines()[0],
    "formatted_paths": (remediation / "formatted-paths.txt").read_text().splitlines(),
    "formatted_census_tsv": (remediation / "formatted-census.tsv").read_text(),
    "rustfmt_patch": (remediation / "rustfmt.patch").read_text(),
    "remediation_receipt": (remediation / "remediation-receipt.txt").read_text(),
    "files": rows,
}
metadata_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PYEMBED

  git restore --source=HEAD --worktree -- "${formatted_paths[@]}"
  unexpected="$(git diff --name-only | grep -v '^Cargo.toml$' || true)"
  if [[ -n "$unexpected" ]]; then
    echo "failed to restore formatter remediation paths: $unexpected" >&2
    exit 93
  fi
  echo 'qualification_contract_revision=rq006-v11-format-remediation-artifact'
  echo "rustfmt_remediation=GENERATED files=${#formatted_paths[@]} result=FAIL_PRODUCT_FORMAT"
  exit 88
fi

cargo check --locked -p symthaea --lib \
  --example benchmark_metacognitive_context \
  --example build_reasoning_capability_matrix
cargo test --locked -p symthaea --lib
cargo clippy --locked -p symthaea --lib \
  --example benchmark_metacognitive_context \
  --example build_reasoning_capability_matrix \
  -- -D warnings

printf 'qualification_contract_revision=%s\n' 'rq006-v11-format-remediation-artifact'
printf 'qualification_overlay_sha256=%s\n' "$overlay_sha256"
printf 'product_manifest_sha256=%s\n' "$product_manifest_sha256"
printf 'projected_manifest_sha256=%s\n' "$projected_manifest_sha256"
printf 'projection_diff_sha256=%s\n' "$projection_diff_sha256"
printf 'retained_metadata_sha256=%s\n' "$(sha256sum "$metadata" | awk '{print $1}')"
