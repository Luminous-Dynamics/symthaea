#!/usr/bin/env bash
set -euo pipefail

package="symthaea-therapeutic"
repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

scripts/verify-verification-depth.sh

modules=(
  denial_catalog provenance_graph evidence_export reproducibility
  decision_receipt review_independence release_candidate assurance_closure
)
for module in "${modules[@]}"; do
  test -f "crates/domains/symthaea-therapeutic/src/${module}.rs" \
    || test -f "src/${module}.rs"
done

for module in "${modules[@]}"; do
  cargo test -p "$package" --no-default-features "${module}::tests"
done

if git grep -nE '^(<<<<<<<|=======|>>>>>>>)' -- . \
    ':!THERAPEUTIC_ASSURANCE_CLOSURE_MIGRATION.md'; then
  echo "merge conflict marker detected" >&2
  exit 1
fi

if ! git grep -q '"assurance-closure"' -- src/release_evidence.rs \
  crates/domains/symthaea-therapeutic/src/release_evidence.rs 2>/dev/null; then
  echo "assurance-closure production gate is missing" >&2
  exit 1
fi

for module in "${modules[@]}"; do
  path="src/${module}.rs"
  if [[ ! -f "$path" ]]; then
    path="crates/domains/symthaea-therapeutic/src/${module}.rs"
  fi
  if awk '/#\[cfg\(test\)\]/{exit} {print}' "$path" \
      | grep -nE '\b(unwrap|expect)\s*\(|panic!\s*\(|todo!\s*\(|unimplemented!\s*\('; then
    echo "non-test panic surface detected in ${path}" >&2
    exit 1
  fi
done

if git grep -nEi '(raw_therapeutic_text|contact_information|crisis_narrative|prompt_text|narrative_text)' \
    -- src/evidence_export.rs src/decision_receipt.rs src/reproducibility.rs 2>/dev/null; then
  echo "content-bearing assurance field detected" >&2
  exit 1
fi

printf 'source_tree=%s\n' "$(git write-tree)"
printf 'head_commit=%s\n' "$(git rev-parse HEAD)"
printf 'crate=%s\n' "$package"
echo 'therapeutic Series IX assurance-closure lane passed'
