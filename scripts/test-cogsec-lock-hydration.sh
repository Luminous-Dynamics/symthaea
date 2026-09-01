#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
TARGET="$ROOT/scripts/cogsec-hydrate-lock-and-qualify.sh"

[[ -f "$TARGET" ]] || {
  printf 'ERROR: missing %s\n' "$TARGET" >&2
  exit 1
}

bash -n "$TARGET"

TMP="$(mktemp -d "${TMPDIR:-/tmp}/cogsec-lock-protocol-test.XXXXXX")"
cleanup() {
  rm -rf "$TMP"
}
trap cleanup EXIT INT TERM

write_common_stubs() {
  local repo="$1"
  mkdir -p "$repo/scripts" "$repo/bin"

  cat > "$repo/scripts/cogsec-focused-qualification.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
[[ -z "$(git diff -- Cargo.lock)" ]]
printf 'FOCUSED_QUALIFICATION_STUB_PASS\n'
EOF

  cat > "$repo/bin/rustc" <<'EOF'
#!/usr/bin/env bash
printf 'rustc 1.96.0 (protocol-test-stub)\n'
EOF

  cat > "$repo/bin/cargo" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--version" ]]; then
  printf 'cargo 1.96.0 (protocol-test-stub)\n'
  exit 0
fi
if [[ "${1:-}" == "metadata" ]]; then
  locked=0
  for arg in "$@"; do
    [[ "$arg" == "--locked" ]] && locked=1
  done
  if ! grep -Fqx 'name = "symthaea-cogsec"' Cargo.lock; then
    [[ $locked -eq 0 ]] || exit 101
    cat >> Cargo.lock <<'LOCK'

[[package]]
name = "symthaea-cogsec"
version = "0.1.0"

[[package]]
name = "symthaea-cogsec-evidence"
version = "0.1.0"

[[package]]
name = "symthaea-cogsec-qualification"
version = "0.1.0"

[[package]]
name = "symthaea-cogsec-shadow-runtime"
version = "0.1.0"
LOCK
  fi
  printf '{}\n'
  exit 0
fi
exit 2
EOF

  chmod +x \
    "$repo/scripts/cogsec-focused-qualification.sh" \
    "$repo/bin/rustc" \
    "$repo/bin/cargo"
}

init_repo() {
  local repo="$1"
  mkdir -p "$repo"
  git -C "$repo" init -q
  git -C "$repo" config user.email cogsec-protocol-test@example.invalid
  git -C "$repo" config user.name cogsec-protocol-test
  write_common_stubs "$repo"
  cat > "$repo/Cargo.lock" <<'EOF'
# protocol test baseline
version = 4

[[package]]
name = "existing"
version = "1.0.0"
EOF
  git -C "$repo" add .
  git -C "$repo" commit -qm baseline
}

run_target() {
  local repo="$1"
  shift
  (
    cd "$repo"
    PATH="$repo/bin:$PATH" bash "$TARGET" "$@"
  )
}

printf '[1/5] syntax\n'
printf 'PASS: bash -n\n'

printf '\n[2/5] two-pass hydration / qualification\n'
HAPPY="$TMP/happy"
init_repo "$HAPPY"
run_target "$HAPPY" > "$TMP/happy-pass1.out"
! git -C "$HAPPY" diff --quiet -- Cargo.lock
grep -q 'HYDRATED:' "$TMP/happy-pass1.out"
! grep -q 'FOCUSED_QUALIFICATION_STUB_PASS' "$TMP/happy-pass1.out"
git -C "$HAPPY" add Cargo.lock
git -C "$HAPPY" commit -qm 'hydrate lock'
run_target "$HAPPY" > "$TMP/happy-pass2.out"
grep -q 'FOCUSED_QUALIFICATION_STUB_PASS' "$TMP/happy-pass2.out"
grep -q 'PASS: committed CogSec lock state passed focused qualification.' "$TMP/happy-pass2.out"
git -C "$HAPPY" diff --quiet
printf 'PASS: uncommitted hydration cannot become qualification\n'

printf '\n[3/5] non-additive lock churn rollback\n'
CHURN="$TMP/churn"
init_repo "$CHURN"
cat > "$CHURN/bin/cargo" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--version" ]]; then
  printf 'cargo 1.96.0 (protocol-test-stub)\n'
  exit 0
fi
if [[ "${1:-}" == "metadata" ]]; then
  locked=0
  for arg in "$@"; do
    [[ "$arg" == "--locked" ]] && locked=1
  done
  if ! grep -Fqx 'name = "symthaea-cogsec"' Cargo.lock; then
    [[ $locked -eq 0 ]] || exit 101
    sed -i '/name = "existing"/d' Cargo.lock
    for package in \
      symthaea-cogsec \
      symthaea-cogsec-evidence \
      symthaea-cogsec-qualification \
      symthaea-cogsec-shadow-runtime
    do
      printf '\n[[package]]\nname = "%s"\nversion = "0.1.0"\n' "$package" >> Cargo.lock
    done
  fi
  printf '{}\n'
  exit 0
fi
exit 2
EOF
chmod +x "$CHURN/bin/cargo"
git -C "$CHURN" add bin/cargo
git -C "$CHURN" commit -qm 'install churn stub'
CHURN_LOCK_BEFORE="$(sha256sum "$CHURN/Cargo.lock" | awk '{print $1}')"
set +e
run_target "$CHURN" > "$TMP/churn.out" 2> "$TMP/churn.err"
CHURN_RC=$?
set -e
[[ $CHURN_RC -ne 0 ]]
[[ "$(sha256sum "$CHURN/Cargo.lock" | awk '{print $1}')" == "$CHURN_LOCK_BEFORE" ]]
grep -q 'additive-only' "$TMP/churn.err"
printf 'PASS: unrelated lock rewrite rejected and restored\n'

printf '\n[4/5] dirty/untracked worktree rejection\n'
DIRTY="$TMP/dirty"
init_repo "$DIRTY"
printf 'contaminant\n' > "$DIRTY/untracked-contaminant"
DIRTY_LOCK_BEFORE="$(sha256sum "$DIRTY/Cargo.lock" | awk '{print $1}')"
set +e
run_target "$DIRTY" > "$TMP/dirty.out" 2> "$TMP/dirty.err"
DIRTY_RC=$?
set -e
[[ $DIRTY_RC -ne 0 ]]
[[ "$(sha256sum "$DIRTY/Cargo.lock" | awk '{print $1}')" == "$DIRTY_LOCK_BEFORE" ]]
grep -q 'working tree must be completely clean' "$TMP/dirty.err"
printf 'PASS: contaminated worktree rejected before mutation\n'

printf '\n[5/5] pinned-toolchain rejection\n'
TOOLCHAIN="$TMP/toolchain"
init_repo "$TOOLCHAIN"
cat > "$TOOLCHAIN/bin/rustc" <<'EOF'
#!/usr/bin/env bash
printf 'rustc 1.95.0 (protocol-test-stub)\n'
EOF
chmod +x "$TOOLCHAIN/bin/rustc"
git -C "$TOOLCHAIN" add bin/rustc
git -C "$TOOLCHAIN" commit -qm 'install wrong toolchain stub'
TOOLCHAIN_LOCK_BEFORE="$(sha256sum "$TOOLCHAIN/Cargo.lock" | awk '{print $1}')"
set +e
run_target "$TOOLCHAIN" > "$TMP/toolchain.out" 2> "$TMP/toolchain.err"
TOOLCHAIN_RC=$?
set -e
[[ $TOOLCHAIN_RC -ne 0 ]]
[[ "$(sha256sum "$TOOLCHAIN/Cargo.lock" | awk '{print $1}')" == "$TOOLCHAIN_LOCK_BEFORE" ]]
grep -q 'rustc 1.96.0 required' "$TMP/toolchain.err"
printf 'PASS: wrong pinned toolchain rejected before mutation\n'

printf '\nPASS: CogSec lock hydration protocol self-test\n'
printf 'NOTE: Cargo/rustc are stubs; this is control-flow evidence, not workspace qualification.\n'
