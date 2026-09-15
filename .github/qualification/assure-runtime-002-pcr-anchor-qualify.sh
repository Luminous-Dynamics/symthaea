#!/usr/bin/env bash
set -euo pipefail

package="symthaea-assurance-linux-tpm-ima-pcr-anchor"
subject_manifest="crates/bridges/symthaea-assurance-linux-tpm-ima-pcr-anchor/Cargo.toml"
work="${RUNNER_TEMP:-/tmp}/assure-runtime-002-pcr-anchor"
capsule="$work/capsule"
closure_tsv="$work/capsule-closure.tsv"
source_census="$work/capsule-source-census.tsv"
projection_tsv="$work/capsule-manifest-projection.tsv"
lock_audit="$work/capsule-lock-audit.json"
source_lock="$work/Cargo.lock.source"
capsule_lock="$work/Cargo.lock.capsule"
lock_diff="$work/Cargo.lock.capsule.diff"
capsule_root_manifest="$work/Cargo.toml.capsule"

rm -rf "$work"
mkdir -p "$work" "$capsule"

[[ -f "$subject_manifest" ]]
[[ -f Cargo.toml ]]
[[ -f Cargo.lock ]]
cp Cargo.lock "$source_lock"
source_lock_sha256="$(sha256sum Cargo.lock | awk '{print $1}')"

python3 - "$subject_manifest" "$capsule" "$closure_tsv" "$source_census" "$projection_tsv" <<'PY'
from __future__ import annotations
import hashlib, json, os, pathlib, subprocess, sys, tomllib

subject_manifest = pathlib.Path(sys.argv[1]).resolve()
repo = pathlib.Path.cwd().resolve()
capsule = pathlib.Path(sys.argv[2]).resolve()
closure_tsv = pathlib.Path(sys.argv[3])
source_census = pathlib.Path(sys.argv[4])
projection_tsv = pathlib.Path(sys.argv[5])
root = tomllib.loads((repo / "Cargo.toml").read_text())
workspace = root.get("workspace", {})
workspace_deps = workspace.get("dependencies", {})
workspace_package = workspace.get("package", {})
product_head = os.environ["PRODUCT_HEAD"]
subject_rel = subject_manifest.relative_to(repo)
subject_dir = subject_rel.parent.as_posix()

def dependency_tables(doc, include_dev):
    tables = []
    for name in ("dependencies", "build-dependencies"):
        value = doc.get(name)
        if isinstance(value, dict):
            tables.append((name, value))
    if include_dev and isinstance(doc.get("dev-dependencies"), dict):
        tables.append(("dev-dependencies", doc["dev-dependencies"]))
    targets = doc.get("target", {})
    if isinstance(targets, dict):
        for target_name, target_doc in targets.items():
            if not isinstance(target_doc, dict):
                continue
            for name in ("dependencies", "build-dependencies"):
                value = target_doc.get(name)
                if isinstance(value, dict):
                    tables.append((f"target.{target_name}.{name}", value))
            if include_dev and isinstance(target_doc.get("dev-dependencies"), dict):
                tables.append((f"target.{target_name}.dev-dependencies", target_doc["dev-dependencies"]))
    return tables

def local_path_for(dep_name, spec, manifest_dir):
    if isinstance(spec, dict) and spec.get("workspace") is True:
        if dep_name not in workspace_deps:
            raise SystemExit(f"workspace dependency missing from root: {dep_name}")
        root_spec = workspace_deps[dep_name]
        if isinstance(root_spec, dict) and "path" in root_spec:
            return (repo / root_spec["path"]).resolve()
        return None
    if isinstance(spec, dict) and "path" in spec:
        return (manifest_dir / spec["path"]).resolve()
    return None

def assert_within_repo(path):
    try:
        path.relative_to(repo)
    except ValueError as exc:
        raise SystemExit(f"local dependency escapes repository: {path}") from exc

queue = [subject_manifest]
seen = {}
required_workspace_deps = set()
while queue:
    manifest = queue.pop(0).resolve()
    if manifest in seen:
        continue
    assert_within_repo(manifest)
    if not manifest.is_file():
        raise SystemExit(f"missing local dependency manifest: {manifest}")
    doc = tomllib.loads(manifest.read_text())
    seen[manifest] = doc
    package_doc = doc.get("package", {})
    for field, value in package_doc.items():
        if isinstance(value, dict) and value.get("workspace") is True and field not in workspace_package:
            raise SystemExit(f"focused closure requires missing workspace.package.{field}: {manifest.relative_to(repo)}")
    for _, table in dependency_tables(doc, include_dev=(manifest == subject_manifest)):
        for dep_name, spec in table.items():
            if isinstance(spec, dict) and spec.get("workspace") is True:
                required_workspace_deps.add(dep_name)
            local = local_path_for(dep_name, spec, manifest.parent)
            if local is not None:
                assert_within_repo(local)
                queue.append(local / "Cargo.toml")

changed = True
while changed:
    changed = False
    for dep_name in sorted(required_workspace_deps):
        spec = workspace_deps[dep_name]
        if isinstance(spec, dict) and "path" in spec:
            dep_manifest = (repo / spec["path"] / "Cargo.toml").resolve()
            if dep_manifest not in seen:
                queue.append(dep_manifest)
                changed = True
    while queue:
        manifest = queue.pop(0).resolve()
        if manifest in seen:
            continue
        if not manifest.is_file():
            raise SystemExit(f"missing workspace path dependency manifest: {manifest}")
        doc = tomllib.loads(manifest.read_text())
        seen[manifest] = doc
        for field, value in doc.get("package", {}).items():
            if isinstance(value, dict) and value.get("workspace") is True and field not in workspace_package:
                raise SystemExit(f"focused closure requires missing workspace.package.{field}: {manifest.relative_to(repo)}")
        for _, table in dependency_tables(doc, include_dev=False):
            for dep_name, spec in table.items():
                if isinstance(spec, dict) and spec.get("workspace") is True:
                    if dep_name not in workspace_deps:
                        raise SystemExit(f"workspace dependency missing from root: {dep_name}")
                    required_workspace_deps.add(dep_name)
                local = local_path_for(dep_name, spec, manifest.parent)
                if local is not None:
                    assert_within_repo(local)
                    queue.append(local / "Cargo.toml")

def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()

def tracked_files(package_dir):
    rel = package_dir.relative_to(repo).as_posix()
    raw = subprocess.check_output(["git", "ls-files", "-z", "--", f"{rel}/"])
    return [pathlib.Path(item.decode()) for item in raw.split(b"\0") if item]

def strip_non_subject_dev_sections(text):
    out, skipping, removed = [], False, 0
    for line in text.splitlines(keepends=True):
        header = line.strip()
        if header.startswith("[") and header.endswith("]"):
            normalized = header.lower().replace('"', "").replace("'", "")
            is_dev = normalized == "[dev-dependencies]" or normalized.endswith(".dev-dependencies]")
            if is_dev:
                skipping, removed = True, removed + 1
                continue
            skipping = False
        if not skipping:
            out.append(line)
    return "".join(out), removed

closure_rows, source_rows, projection_rows = [], [], []
for manifest, doc in sorted(seen.items(), key=lambda item: item[0].as_posix()):
    package_dir = manifest.parent
    rel_dir = package_dir.relative_to(repo)
    package_name = doc.get("package", {}).get("name")
    if not isinstance(package_name, str):
        raise SystemExit(f"package name missing: {manifest}")
    tree_sha = git("rev-parse", f"{product_head}:{rel_dir.as_posix()}")
    closure_rows.append((package_name, rel_dir.as_posix(), tree_sha))
    files = tracked_files(package_dir)
    if not files:
        raise SystemExit(f"no tracked files for package: {rel_dir}")
    for rel_file in files:
        src, dst = repo / rel_file, capsule / rel_file
        dst.parent.mkdir(parents=True, exist_ok=True)
        data = src.read_bytes()
        dst.write_bytes(data)
        source_rows.append((rel_file.as_posix(), git("rev-parse", f"{product_head}:{rel_file.as_posix()}"), hashlib.sha256(data).hexdigest(), len(data)))
    copied_manifest = capsule / manifest.relative_to(repo)
    original = copied_manifest.read_text()
    if manifest == subject_manifest:
        projected, removed, kind = original, 0, "subject-exact"
    else:
        projected, removed = strip_non_subject_dev_sections(original)
        kind = "dependency-strip-dev" if removed else "dependency-exact"
        copied_manifest.write_text(projected)
    projection_rows.append((manifest.relative_to(repo).as_posix(), kind, str(removed), hashlib.sha256(original.encode()).hexdigest(), hashlib.sha256(projected.encode()).hexdigest()))

closure_tsv.write_text("package\tpath\tgit_tree_sha1\n" + "".join("\t".join(r) + "\n" for r in closure_rows))
source_census.write_text("path\tgit_blob_sha1\tsha256\tbytes\n" + "".join(f"{p}\t{b}\t{s}\t{n}\n" for p,b,s,n in source_rows))
projection_tsv.write_text("manifest\tprojection\tremoved_dev_sections\tsource_sha256\tprojected_sha256\n" + "".join("\t".join(r) + "\n" for r in projection_rows))
metadata = {"subject_dir": subject_dir, "members": [r[1] for r in closure_rows], "workspace_package": workspace_package, "workspace_dependencies": {name: workspace_deps[name] for name in sorted(required_workspace_deps)}}
(capsule / ".qualification-workspace.json").write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
PY

python3 - "$capsule/.qualification-workspace.json" "$capsule/Cargo.toml" <<'PY'
import json, pathlib, sys
metadata = json.loads(pathlib.Path(sys.argv[1]).read_text())
out = pathlib.Path(sys.argv[2])
def q(v): return json.dumps(v, ensure_ascii=False)
def tv(v):
    if isinstance(v, str): return q(v)
    if isinstance(v, bool): return "true" if v else "false"
    if isinstance(v, int): return str(v)
    if isinstance(v, list): return "[" + ", ".join(tv(x) for x in v) + "]"
    if isinstance(v, dict): return "{ " + ", ".join(f"{k} = {tv(x)}" for k,x in v.items()) + " }"
    raise TypeError(v)
lines = ["[workspace]", 'resolver = "2"', "members = ["]
lines += [f"  {q(m)}," for m in metadata["members"]]
lines += ["]", f"default-members = [{q(metadata['subject_dir'])}]", "", "[workspace.package]"]
lines += [f"{k} = {tv(v)}" for k,v in metadata["workspace_package"].items()]
lines += ["", "[workspace.dependencies]"]
lines += [f"{k} = {tv(v)}" for k,v in metadata["workspace_dependencies"].items()]
lines.append("")
out.write_text("\n".join(lines))
PY

cp "$capsule/Cargo.toml" "$capsule_root_manifest"

python3 - "$source_census" "$projection_tsv" "$capsule" <<'PY'
import hashlib, os, pathlib, subprocess, sys
census, projections, capsule = map(pathlib.Path, sys.argv[1:])
product_head = os.environ["PRODUCT_HEAD"]
projected = {}
for line in projections.read_text().splitlines()[1:]:
    manifest, kind, _, source_sha, projected_sha = line.split("\t")
    projected[manifest] = (kind, source_sha, projected_sha)

for line in census.read_text().splitlines()[1:]:
    path, expected_blob, expected_sha, expected_size = line.split("\t")
    source_data = pathlib.Path(path).read_bytes()
    if hashlib.sha256(source_data).hexdigest() != expected_sha or len(source_data) != int(expected_size):
        raise SystemExit(f"checked-out source drifted while building capsule: {path}")
    blob = subprocess.check_output(["git", "rev-parse", f"{product_head}:{path}"], text=True).strip()
    if blob != expected_blob:
        raise SystemExit(f"Git blob identity changed while building capsule: {path}")

    capsule_data = (capsule / path).read_bytes()
    if path in projected and projected[path][2] != projected[path][1]:
        if hashlib.sha256(capsule_data).hexdigest() != projected[path][2]:
            raise SystemExit(f"projected manifest digest mismatch: {path}")
    else:
        if capsule_data != source_data:
            raise SystemExit(f"copied source mismatch: {path}")
print(f"capsule_source_census=PASS files={len(census.read_text().splitlines()) - 1}")
PY

subject_capsule_manifest="$capsule/$subject_manifest"
cmp "$subject_manifest" "$subject_capsule_manifest"
cargo fmt --manifest-path "$capsule/Cargo.toml" -p "$package" -- --check

cp "$source_lock" "$capsule/Cargo.lock"
cargo check --manifest-path "$capsule/Cargo.toml" -p "$package"
cp "$capsule/Cargo.lock" "$capsule_lock"
diff -u "$source_lock" "$capsule_lock" > "$lock_diff" || true

python3 - "$source_lock" "$capsule_lock" "$lock_audit" "$closure_tsv" <<'PY'
import json, pathlib, sys, tomllib
source_path, capsule_path, audit_path, closure_path = map(pathlib.Path, sys.argv[1:])
source, capsule = tomllib.loads(source_path.read_text()), tomllib.loads(capsule_path.read_text())
def key(p): return (p["name"], p["version"], p.get("source", ""))
source_packages = {key(p): p for p in source.get("package", [])}
closure_names = {line.split("\t")[0] for line in closure_path.read_text().splitlines()[1:]}
capsule_local = {p["name"] for p in capsule.get("package", []) if "source" not in p}
if capsule_local != closure_names:
    raise SystemExit(f"capsule local package closure mismatch: expected={sorted(closure_names)!r} actual={sorted(capsule_local)!r}")
external = [p for p in capsule.get("package", []) if "source" in p]
missing, changed = [], []
for p in external:
    sp = source_packages.get(key(p))
    if sp is None: missing.append(key(p))
    elif sp != p: changed.append(key(p))
if missing: raise SystemExit(f"capsule introduced external packages absent from source lock: {missing!r}")
if changed: raise SystemExit(f"capsule external package entries drifted from source lock: {changed!r}")
summary = {"source_lock_version": source.get("version"), "capsule_lock_version": capsule.get("version"), "focused_local_packages": sorted(capsule_local), "external_packages_bound_to_source_lock": len(external), "source_package_count": len(source_packages), "capsule_package_count": len(capsule.get("package", []))}
audit_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(audit_path.read_text(), end="")
PY

capsule_lock_sha256="$(sha256sum "$capsule/Cargo.lock" | awk '{print $1}')"
printf 'source_lock_sha256=%s\n' "$source_lock_sha256"
printf 'capsule_lock_sha256=%s\n' "$capsule_lock_sha256"
printf 'closure_sha256=%s\n' "$(sha256sum "$closure_tsv" | awk '{print $1}')"
printf 'source_census_sha256=%s\n' "$(sha256sum "$source_census" | awk '{print $1}')"
printf 'manifest_projection_sha256=%s\n' "$(sha256sum "$projection_tsv" | awk '{print $1}')"
printf 'capsule_manifest_sha256=%s\n' "$(sha256sum "$capsule/Cargo.toml" | awk '{print $1}')"

cargo check --locked --manifest-path "$capsule/Cargo.toml" -p "$package"
cargo test --locked --manifest-path "$capsule/Cargo.toml" -p "$package"
cargo clippy --locked --manifest-path "$capsule/Cargo.toml" -p "$package" --all-targets -- -D warnings

test "$(sha256sum "$capsule/Cargo.lock" | awk '{print $1}')" = "$capsule_lock_sha256"
test "$(sha256sum Cargo.lock | awk '{print $1}')" = "$source_lock_sha256"
printf 'qualification_result=PASS\n'
