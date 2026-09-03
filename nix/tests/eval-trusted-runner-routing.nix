# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#
# Eval/build-time regression test for the trusted CPU scheduling capability.
# This never contacts GitHub and never requires runner credentials.

{ pkgs }:

let
  workflowsDir = ../../.github/workflows;
in
pkgs.runCommand "eval-trusted-runner-routing" { } ''
  set -euo pipefail

  workflows='${workflowsDir}'
  label='symthaea-trusted-cpu-v1'

  # Capability ownership is intentionally tiny. Any new consumer must receive a
  # separate threat-model review and update this exact allowlist deliberately.
  expected="$(cat <<'EOF'
self-hosted-ai-assurance-budget-recovery.yml
self-hosted-ai-assurance-foundation-recovery.yml
self-hosted-runner-smoke.yml
self-hosted-sym-arch-002a-core-recovery.yml
EOF
)"

  actual="$(grep -RlF -- "$label" "$workflows" \
    | sed 's#.*/##' \
    | LC_ALL=C sort)"

  if [ "$actual" != "$expected" ]; then
    echo 'trusted CPU routing capability consumer set changed' >&2
    echo 'expected:' >&2
    printf '%s\n' "$expected" >&2
    echo 'actual:' >&2
    printf '%s\n' "$actual" >&2
    exit 1
  fi

  for name in $expected; do
    file="$workflows/$name"
    test -f "$file"

    # Every trusted-CPU workflow must remain operator-dispatched, tokenless, and
    # main-only. It may not gain automatic PR/push/schedule triggers.
    grep -F -- 'workflow_dispatch:' "$file" >/dev/null
    grep -F -- 'permissions: {}' "$file" >/dev/null
    grep -F -- "github.ref == 'refs/heads/main'" "$file" >/dev/null
    grep -F -- 'runs-on: [symthaea-trusted-cpu-v1]' "$file" >/dev/null

    if grep -Eq '^[[:space:]]*(pull_request|push|schedule):' "$file"; then
      echo "$name gained an automatic trigger" >&2
      exit 1
    fi
  done

  touch "$out"
''
