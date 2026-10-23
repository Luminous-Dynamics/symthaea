#!/usr/bin/env bash
# Read-only repository inventory for the math-transfer campaign.
# It does not fetch, checkout, reset, clean, stash, modify, or test any repository.
set -uo pipefail

ROOT="$HOME/src/luminous-dynamics-pr-review"
if [ "$#" -ge 1 ]; then ROOT="$1"; fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$PWD/math-campaign-preflight-$STAMP"
if [ "$#" -ge 2 ]; then OUT="$2"; fi
mkdir -p "$OUT" || { echo "Cannot create output directory: $OUT" >&2; exit 2; }

redact() {
  sed -E 's#(https?://)[^/@[:space:]]+:[^/@[:space:]]+@#\1[REDACTED]@#g'
}

{
  echo "# Math campaign local preflight"
  echo
  echo "UTC: $STAMP"
  echo "Expected clone root: $ROOT"
  echo "Output directory: $OUT"
  echo
  echo "## Host tool availability"
  for tool in git rg rustc cargo nix lean lake elan gh sha256sum; do
    if command -v "$tool" >/dev/null 2>&1; then
      printf -- '- %s: %s\n' "$tool" "$(command -v "$tool")"
    else
      printf -- '- %s: unavailable on PATH\n' "$tool"
    fi
  done
  if command -v rustc >/dev/null 2>&1; then
    echo
    echo '### rustc -Vv'
    rustc -Vv 2>&1 || true
  fi
  if command -v cargo >/dev/null 2>&1; then
    echo
    echo '### cargo --version'
    cargo --version 2>&1 || true
  fi
  if command -v nix >/dev/null 2>&1; then
    echo
    echo '### nix --version'
    nix --version 2>&1 || true
  fi
} > "$OUT/summary.md"

for repo in symthaea symtropy mycelix; do
  path="$ROOT/$repo"
  report="$OUT/$repo.md"
  {
    echo "# $repo local inventory"
    echo
    printf 'Path: %s\n' "$path"
    echo
    if [[ ! -d "$path/.git" && ! -f "$path/.git" ]]; then
      echo "STATUS: repository clone not found at expected path. No repository changes made."
      continue
    fi
    echo '## Identity'
    printf 'Top-level: '; git -C "$path" rev-parse --show-toplevel 2>&1 || true
    printf 'HEAD: '; git -C "$path" rev-parse HEAD 2>&1 || true
    printf 'Branch: '; git -C "$path" branch --show-current 2>&1 || true
    printf 'Latest commit: '; git -C "$path" log -1 --format='%H%n%ci%n%s' 2>&1 || true
    echo
    echo '## Status (inspect before editing)'
    git -C "$path" status --short --branch 2>&1 || true
    echo
    echo '## Diff summary (staged and unstaged; no patch contents)'
    git -C "$path" diff --stat 2>&1 || true
    git -C "$path" diff --cached --stat 2>&1 || true
    echo
    echo '## Tracked math, research and test file candidates (first 250 paths)'
    if command -v rg >/dev/null 2>&1; then
      git -C "$path" ls-files 2>/dev/null | rg -i '(math|fep|entropy|information|probab|bayes|gaussian|lyapunov|stability|conserv|thermod|physics|quorum|byzant|trust|govern|allocation|routing|metric|formal|proof|research).*(\.rs|\.lean|\.md|\.json|\.yaml|\.yml|\.toml)$|(^|/)(tests?|docs|research)/' | head -n 250 || true
    else
      git -C "$path" ls-files 2>/dev/null | grep -Ei '(math|fep|entropy|information|probab|bayes|gaussian|lyapunov|stability|conserv|thermod|physics|quorum|byzant|trust|govern|allocation|routing|metric|formal|proof|research).*(\.rs|\.lean|\.md|\.json|\.yaml|\.yml|\.toml)$|(^|/)(tests?|docs|research)/' | head -n 250 || true
    fi
  } > "$report"
done

cat >> "$OUT/summary.md" <<EOF2

## Generated reports

- summary.md
- symthaea.md
- symtropy.md
- mycelix.md

This script performs no Git mutations and no network access. Review each repository status before sharing results with an agent. Reports include local paths, branch names and repository status; redact anything sensitive before sharing outside the trusted environment.
EOF2

printf 'Preflight complete: %s\n' "$OUT"
printf 'Review: %s\n' "$OUT/summary.md"
