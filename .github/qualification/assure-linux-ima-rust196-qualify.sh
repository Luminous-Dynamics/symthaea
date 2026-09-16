#!/usr/bin/env bash
set -euo pipefail

package="symthaea-linux-ima-replay"
crate="crates/domains/symthaea-linux-ima-replay"
work="${RUNNER_TEMP:-/tmp}/assure-linux-ima-rust196-remediation"
capsule="$work/capsule"
census="$work/source-census.tsv"
remediation="$work/rustfmt-remediation"

rm -rf "$work"
mkdir -p "$capsule/$crate" "$remediation/formatted"

printf 'path\tgit_blob_sha1\tsha256\tbytes\n' > "$census"
while IFS= read -r -d '' path; do
  mkdir -p "$capsule/$(dirname "$path")"
  cp "$path" "$capsule/$path"
  printf '%s\t%s\t%s\t%s\n' \
    "$path" \
    "$(git rev-parse "$PRODUCT_HEAD:$path")" \
    "$(sha256sum "$path" | awk '{print $1}')" \
    "$(wc -c < "$path" | tr -d ' ')" \
    >> "$census"
done < <(git ls-files -z -- "$crate/")

python3 - "$census" "$capsule" <<'PY'
import hashlib
import os
import pathlib
import subprocess
import sys

census = pathlib.Path(sys.argv[1])
capsule = pathlib.Path(sys.argv[2])
product = os.environ["PRODUCT_HEAD"]
rows = census.read_text().splitlines()[1:]
if not rows:
    raise SystemExit("empty source census")
for row in rows:
    path, blob, digest, size = row.split("\t")
    source = pathlib.Path(path).read_bytes()
    copied = (capsule / path).read_bytes()
    if source != copied:
        raise SystemExit(f"capsule copy mismatch: {path}")
    if hashlib.sha256(source).hexdigest() != digest or len(source) != int(size):
        raise SystemExit(f"source census mismatch: {path}")
    actual_blob = subprocess.check_output(
        ["git", "rev-parse", f"{product}:{path}"], text=True
    ).strip()
    if actual_blob != blob:
        raise SystemExit(f"product blob mismatch: {path}")
print(f"ima_remediation_source_census=PASS files={len(rows)}")
PY

cat > "$capsule/Cargo.toml" <<'TOML'
[workspace]
resolver = "2"
members = ["crates/domains/symthaea-linux-ima-replay"]
default-members = ["crates/domains/symthaea-linux-ima-replay"]

[workspace.dependencies]
blake3 = "1.5"
serde = { version = "1.0", features = ["derive"] }
TOML

cmp "$crate/Cargo.toml" "$capsule/$crate/Cargo.toml"

fmt_check="$work/rustfmt-check.log"
set +e
cargo fmt --manifest-path "$capsule/Cargo.toml" -p "$package" -- --check >"$fmt_check" 2>&1
fmt_status=$?
set -e
cat "$fmt_check"
if [[ "$fmt_status" -eq 0 ]]; then
  echo 'remediation_error=FORMAT_ALREADY_CLEAN'
  exit 2
fi

cargo fmt --manifest-path "$capsule/Cargo.toml" -p "$package"
cargo fmt --manifest-path "$capsule/Cargo.toml" -p "$package" -- --check

python3 - "$census" "$capsule" "$remediation" "$PRODUCT_HEAD" <<'PY'
import hashlib
import pathlib
import shutil
import sys

census = pathlib.Path(sys.argv[1])
capsule = pathlib.Path(sys.argv[2])
remediation = pathlib.Path(sys.argv[3])
product = sys.argv[4]
rows = census.read_text().splitlines()[1:]
changed = []
for row in rows:
    path, blob, original_sha256, _size = row.split("\t")
    formatted = (capsule / path).read_bytes()
    formatted_sha256 = hashlib.sha256(formatted).hexdigest()
    if formatted_sha256 != original_sha256:
        changed.append((path, blob, original_sha256, formatted_sha256, len(formatted)))

expected = "crates/domains/symthaea-linux-ima-replay/src/lib.rs"
if [item[0] for item in changed] != [expected]:
    raise SystemExit(f"unexpected rustfmt remediation paths: {[item[0] for item in changed]!r}")

out = remediation / "formatted" / expected
out.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(capsule / expected, out)
formatted_sha256 = hashlib.sha256(out.read_bytes()).hexdigest()
if formatted_sha256 != changed[0][3]:
    raise SystemExit("formatted copy digest mismatch")

(remediation / "remediation.tsv").write_text(
    "path\toriginal_git_blob_sha1\toriginal_sha256\tformatted_sha256\tformatted_bytes\n"
    + "\t".join(map(str, changed[0]))
    + "\n"
)
(remediation / "receipt.txt").write_text(
    "classification=FAIL_PRODUCT_FORMAT\n"
    f"product_head={product}\n"
    "rustfmt=1.96.0\n"
    "changed_paths=1\n"
    f"formatted_corpus_sha256={formatted_sha256}\n"
)
print(f"rustfmt_remediation=GENERATED_RED paths=1 corpus_sha256={formatted_sha256}")
PY

cp "$fmt_check" "$remediation/rustfmt-check.log"
cat "$remediation/receipt.txt"
exit 1
