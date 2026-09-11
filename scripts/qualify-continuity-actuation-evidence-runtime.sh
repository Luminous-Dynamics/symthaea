#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

runtime_subject="${RUNTIME_SUBJECT_SHA:?RUNTIME_SUBJECT_SHA is required}"
qualifier_sha="${QUALIFIER_SHA:?QUALIFIER_SHA is required}"
receipt="${CONTINUITY_RUNTIME_RECEIPT:?CONTINUITY_RUNTIME_RECEIPT is required}"
manifest="crates/core/symthaea-continuity/Cargo.toml"

actual_head="$(git rev-parse HEAD)"
if [[ "$actual_head" != "$qualifier_sha" ]]; then
  echo "qualifier HEAD mismatch: expected=$qualifier_sha actual=$actual_head" >&2
  exit 1
fi

if ! git cat-file -e "${runtime_subject}^{commit}"; then
  echo "runtime subject is not available as a commit: $runtime_subject" >&2
  exit 1
fi
if ! git merge-base --is-ancestor "$runtime_subject" "$qualifier_sha"; then
  echo "runtime subject is not an ancestor of qualifier head" >&2
  exit 1
fi

# The qualifier may add CI/docs only. Runtime crate/environment inputs must remain
# byte-for-byte identical to the frozen subject under qualification.
git diff --exit-code "$runtime_subject" "$qualifier_sha" -- \
  crates/core/symthaea-continuity \
  Cargo.toml \
  Cargo.lock \
  rust-toolchain.toml

runtime_tree="$(git show -s --format=%T "$runtime_subject")"
qualifier_tree="$(git show -s --format=%T "$qualifier_sha")"
lock_sha256="$(sha256sum Cargo.lock | awk '{print $1}')"
manifest_sha256="$(sha256sum "$manifest" | awk '{print $1}')"
toolchain_sha256="$(sha256sum rust-toolchain.toml | awk '{print $1}')"
rustc_version="$(rustc --version)"
cargo_version="$(cargo --version)"

cargo fmt --manifest-path "$manifest" -- --check
cargo check --locked --manifest-path "$manifest" --all-targets
cargo test --locked --manifest-path "$manifest" --lib
cargo test --locked --manifest-path "$manifest" --doc
cargo clippy --locked --manifest-path "$manifest" --all-targets -- -D warnings

mkdir -p "$(dirname "$receipt")"
cat > "$receipt" <<EOF
schema\tsymthaea-continuity-actuation-evidence-runtime-qualification-v1
runtime_subject_sha\t$runtime_subject
runtime_subject_tree\t$runtime_tree
qualifier_sha\t$qualifier_sha
qualifier_tree\t$qualifier_tree
cargo_lock_sha256\t$lock_sha256
crate_manifest_sha256\t$manifest_sha256
rust_toolchain_sha256\t$toolchain_sha256
rustc_version\t$rustc_version
cargo_version\t$cargo_version
runtime_inputs_unchanged\tPASS
cargo_fmt\tPASS
cargo_check_all_targets\tPASS
cargo_test_lib\tPASS
cargo_test_doc\tPASS
cargo_clippy_all_targets_deny_warnings\tPASS
status\tPASS
EOF

cat "$receipt"
