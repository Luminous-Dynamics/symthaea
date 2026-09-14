#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: run_hls_validity_evidence_v2.sh \
  --subject-sha <40-hex> \
  --base-evidence-out <path-outside-repo> \
  --supplement-out <path-outside-repo> \
  [--protocol smoke|research-v0]

Fresh-builds the evidence-v2 executable in an isolated Cargo target directory,
then executes only that binary against the exact clean subject checkout.
EOF
}

subject_sha=""
protocol="smoke"
base_output=""
supplement_output=""

while (($#)); do
  case "$1" in
    --subject-sha)
      subject_sha="${2:-}"
      shift 2
      ;;
    --protocol)
      protocol="${2:-}"
      shift 2
      ;;
    --base-evidence-out)
      base_output="${2:-}"
      shift 2
      ;;
    --supplement-out)
      supplement_output="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unexpected argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "$subject_sha" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "--subject-sha must be exactly 40 hexadecimal characters" >&2
  exit 2
fi
subject_sha="${subject_sha,,}"

case "$protocol" in
  smoke|research-v0) ;;
  *)
    echo "--protocol must be smoke or research-v0" >&2
    exit 2
    ;;
esac

if [[ -z "$base_output" || -z "$supplement_output" ]]; then
  echo "--base-evidence-out and --supplement-out are required" >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
repo_root="$(realpath "$repo_root")"
cd "$repo_root"

head_sha="$(git rev-parse HEAD)"
if [[ "$head_sha" != "$subject_sha" ]]; then
  echo "subject mismatch: required $subject_sha, checkout is $head_sha" >&2
  exit 1
fi

if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "evidence qualification requires a clean checkout" >&2
  git status --short >&2
  exit 1
fi

normalize_external_output() {
  local requested="$1"
  local parent
  parent="$(dirname "$requested")"
  mkdir -p "$parent"
  local absolute_parent
  absolute_parent="$(realpath "$parent")"
  local absolute="$absolute_parent/$(basename "$requested")"
  case "$absolute" in
    "$repo_root"|"$repo_root"/*)
      echo "evidence outputs must be outside the repository: $absolute" >&2
      return 1
      ;;
  esac
  printf '%s\n' "$absolute"
}

base_output="$(normalize_external_output "$base_output")"
supplement_output="$(normalize_external_output "$supplement_output")"
if [[ "$base_output" == "$supplement_output" ]]; then
  echo "base and supplement evidence outputs must be different files" >&2
  exit 2
fi

tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/symthaea-hls-validity-evidence-v2.XXXXXX")"
cleanup() {
  rm -rf "$tmp_root"
}
trap cleanup EXIT INT TERM

export CARGO_TARGET_DIR="$tmp_root/target"

echo "subject_sha=$subject_sha" >&2
echo "subject_tree=$(git rev-parse 'HEAD^{tree}')" >&2
echo "cargo=$(cargo -V)" >&2
echo "rustc=$(rustc -Vv | tr '\n' ';')" >&2

echo "fresh-building validity_capacity_evidence_v2 with --locked" >&2
cargo build --locked -p symthaea-hdc-ltc --example validity_capacity_evidence_v2

binary="$CARGO_TARGET_DIR/debug/examples/validity_capacity_evidence_v2"
if [[ ! -x "$binary" ]]; then
  echo "fresh evidence-v2 binary not found at $binary" >&2
  exit 1
fi

"$binary" \
  --subject-sha "$subject_sha" \
  --protocol "$protocol" \
  --base-evidence-out "$base_output" \
  > "$supplement_output"

post_head="$(git rev-parse HEAD)"
if [[ "$post_head" != "$subject_sha" ]]; then
  echo "subject changed during evidence qualification" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "checkout changed during evidence qualification" >&2
  git status --short >&2
  exit 1
fi

base_sha="$(sha256sum "$base_output" | awk '{print $1}')"
supplement_sha="$(sha256sum "$supplement_output" | awk '{print $1}')"
printf 'BASE_EVIDENCE_SHA256=%s\n' "$base_sha"
printf 'SUPPLEMENT_EVIDENCE_SHA256=%s\n' "$supplement_sha"
printf 'SUBJECT_SHA=%s\n' "$subject_sha"
