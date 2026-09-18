#!/usr/bin/env bash
set -euo pipefail

: "${PRODUCT_HEAD:?}"
: "${PRODUCT_PARENT:?}"
: "${PACKAGE:?}"
: "${MANIFEST_PATH:?}"
: "${PRODUCT_PATH:?}"
: "${EXPECTED_BLOB:?}"
: "${QUALIFIER_PATH:?}"
: "${WORKFLOW_PATH:?}"

work="${RUNNER_TEMP:-/tmp}/assure-rust196-single-package"
rm -rf "$work"
mkdir -p "$work"

test "$(git rev-parse HEAD^)" = "$PRODUCT_HEAD"
test "$(git rev-parse "$PRODUCT_HEAD^")" = "$PRODUCT_PARENT"
test "$(git rev-list --count "$PRODUCT_PARENT..$PRODUCT_HEAD")" = "1"
test "$(git diff --name-status "$PRODUCT_PARENT" "$PRODUCT_HEAD")" = $'M\t'"$PRODUCT_PATH"

mapfile -t qualifier_delta < <(git diff --name-status "$PRODUCT_HEAD" HEAD)
test "${#qualifier_delta[@]}" = "2"
expected_a=$'A\t'"$QUALIFIER_PATH"
expected_b=$'A\t'"$WORKFLOW_PATH"
printf '%s\n' "${qualifier_delta[@]}" | sort > "$work/qualifier-delta.actual"
printf '%s\n' "$expected_a" "$expected_b" | sort > "$work/qualifier-delta.expected"
cmp "$work/qualifier-delta.expected" "$work/qualifier-delta.actual"

test "$(git rev-parse "$PRODUCT_HEAD:$PRODUCT_PATH")" = "$EXPECTED_BLOB"
test "$(git hash-object "$PRODUCT_PATH")" = "$EXPECTED_BLOB"
product_sha256="$(git show "$PRODUCT_HEAD:$PRODUCT_PATH" | sha256sum | awk '{print $1}')"

pre_head="$(git rev-parse HEAD)"
pre_tree="$(git rev-parse 'HEAD^{tree}')"
pre_lock="$(sha256sum Cargo.lock | awk '{print $1}')"

rustc --version | tee "$work/rustc-version.txt"
cargo --version | tee "$work/cargo-version.txt"
cargo fmt --manifest-path "$MANIFEST_PATH" -- --check

# Network is permitted only for cache acquisition. No authority is inferred here.
cargo fetch --locked --manifest-path "$MANIFEST_PATH"
export CARGO_NET_OFFLINE=true

cargo metadata --locked --offline --format-version 1 \
  --manifest-path "$MANIFEST_PATH" > "$work/metadata.json"

python3 - "$work/metadata.json" "$PACKAGE" "$MANIFEST_PATH" "$work/local-closure.tsv" <<'PY'
import json, pathlib, subprocess, sys
metadata_path, package_name, manifest_path, out_path = sys.argv[1:]
repo = pathlib.Path.cwd().resolve()
meta = json.loads(pathlib.Path(metadata_path).read_text())
packages = {p["id"]: p for p in meta["packages"]}
nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
target_manifest = (repo / manifest_path).resolve()
matches = [
    p for p in meta["packages"]
    if p["name"] == package_name and pathlib.Path(p["manifest_path"]).resolve() == target_manifest
]
if len(matches) != 1:
    raise SystemExit(f"expected one subject package, found {len(matches)}")
root = matches[0]["id"]
seen, stack = set(), [root]
while stack:
    pid = stack.pop()
    if pid in seen:
        continue
    seen.add(pid)
    node = nodes.get(pid)
    if node is None:
        raise SystemExit(f"missing resolve node: {pid}")
    stack.extend(dep["pkg"] for dep in node.get("deps", []))
rows = []
for pid in sorted(seen):
    p = packages[pid]
    if p.get("source") is not None:
        continue
    manifest = pathlib.Path(p["manifest_path"]).resolve()
    try:
        rel_manifest = manifest.relative_to(repo)
    except ValueError:
        raise SystemExit(f"local manifest escapes repository: {manifest}")
    package_dir = rel_manifest.parent
    tree = subprocess.check_output(
        ["git", "rev-parse", f"HEAD:{package_dir.as_posix()}"], text=True
    ).strip()
    rows.append((p["name"], str(p["version"]), rel_manifest.as_posix(), tree))
pathlib.Path(out_path).write_text(
    "package\tversion\tmanifest\tgit_tree_sha1\n"
    + "".join("\t".join(row) + "\n" for row in rows)
)
print(f"resolved_packages={len(seen)}")
print(f"local_packages={len(rows)}")
PY

cargo check --locked --offline --manifest-path "$MANIFEST_PATH" -p "$PACKAGE" \
  2>&1 | tee "$work/check.log"
cargo test --locked --offline --manifest-path "$MANIFEST_PATH" -p "$PACKAGE" -- --list \
  > "$work/test-list.txt"
cargo test --locked --offline --manifest-path "$MANIFEST_PATH" -p "$PACKAGE" \
  2>&1 | tee "$work/test.log"
cargo clippy --locked --offline --manifest-path "$MANIFEST_PATH" -p "$PACKAGE" \
  --all-targets -- -D warnings 2>&1 | tee "$work/clippy.log"

test "$(git rev-parse HEAD)" = "$pre_head"
test "$(git rev-parse 'HEAD^{tree}')" = "$pre_tree"
test "$(sha256sum Cargo.lock | awk '{print $1}')" = "$pre_lock"
test -z "$(git status --porcelain --untracked-files=no)"

cat > "$work/contract-outcome.txt" <<EOF
contract_result=PASS
qualification_result=PENDING_POSTFLIGHT
authority=ContractPassedPendingPostflight
subject_head=$pre_head
subject_tree=$pre_tree
product_head=$PRODUCT_HEAD
product_parent=$PRODUCT_PARENT
package=$PACKAGE
manifest_path=$MANIFEST_PATH
product_path=$PRODUCT_PATH
product_blob=$EXPECTED_BLOB
product_sha256=$product_sha256
cargo_lock_sha256=$pre_lock
EOF
