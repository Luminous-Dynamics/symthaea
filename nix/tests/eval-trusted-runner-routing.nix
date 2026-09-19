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
arc3-protocol-trusted-cpu-qualify.yml
self-hosted-ai-assurance-foundation-recovery.yml
self-hosted-rca-canonical-lineage-recovery.yml
self-hosted-runner-smoke.yml
self-hosted-se001q-evidence-recovery.yml
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

  arc3="$workflows/arc3-protocol-trusted-cpu-qualify.yml"

  # ARC3 recovery is intentionally narrower than the generic trusted-CPU
  # capability: one main-owned recipe, one exact reviewed product subject, no
  # caller-controlled subject/ref, no token-bearing actions, and no JS Actions.
  grep -F -- 'ARC3_SUBJECT_SHA: 6ea96737aff361920c181891a71f80a9f481ddef' "$arc3" >/dev/null
  grep -F -- 'ARC3_SUBJECT_BRANCH: feat/arc3-protocol-contract' "$arc3" >/dev/null

  if grep -Eq '^[[:space:]]+inputs:' "$arc3"; then
    echo 'ARC3 trusted CPU qualifier gained workflow inputs' >&2
    exit 1
  fi
  if grep -Eq '^[[:space:]]+-[[:space:]]+uses:' "$arc3"; then
    echo 'ARC3 trusted CPU qualifier gained a third-party/local Action' >&2
    exit 1
  fi
  if grep -Eq '^[[:space:]]*(GH_TOKEN|GITHUB_TOKEN):' "$arc3"; then
    echo 'ARC3 trusted CPU qualifier gained an explicit GitHub token' >&2
    exit 1
  fi

  # Freeze the substantive correctness profile. Removing one of these commands
  # must fail policy evaluation before the runner is considered trusted for
  # ARC3 qualification.
  grep -F -- 'python3 scripts/arc3_protocol_oracle.py' "$arc3" >/dev/null
  grep -F -- 'cargo fmt -p symthaea-arc3-protocol -p symthaea-psych-bench -- --check' "$arc3" >/dev/null
  grep -F -- 'cargo check --locked -p symthaea-arc3-protocol' "$arc3" >/dev/null
  grep -F -- 'cargo test --locked -p symthaea-arc3-protocol' "$arc3" >/dev/null
  grep -F -- 'cargo clippy --locked -p symthaea-arc3-protocol --all-targets -- -D warnings' "$arc3" >/dev/null
  grep -F -- 'cargo check --locked -p symthaea-psych-bench' "$arc3" >/dev/null
  grep -F -- 'cargo test --locked -p symthaea-psych-bench --lib' "$arc3" >/dev/null

  # Exact-subject branch binding and both recipe/subject immutability checks are
  # qualification semantics, not optional diagnostics.
  grep -F -- 'git ls-remote https://github.com/Luminous-Dynamics/symthaea.git' "$arc3" >/dev/null
  grep -F -- 'test "$advertised" = "$ARC3_SUBJECT_SHA"' "$arc3" >/dev/null
  grep -F -- 'git -C "$subject_dir" diff --exit-code' "$arc3" >/dev/null
  grep -F -- 'test "$(git rev-parse HEAD)" = "$GITHUB_SHA"' "$arc3" >/dev/null
  grep -F -- 'result=CANDIDATE_PASS' "$arc3" >/dev/null
  grep -F -- 'scope=correctness-only' "$arc3" >/dev/null

  se001q="$workflows/self-hosted-se001q-evidence-recovery.yml"

  # SE-001Q is an independent-provider observation lane, not a generic command
  # runner. Its four inputs are exact evidence/authority bytes and hashes only.
  grep -F -- 'recovery_eligibility_manifest_base64:' "$se001q" >/dev/null
  grep -F -- 'recovery_eligibility_manifest_sha256:' "$se001q" >/dev/null
  grep -F -- 'stage_f_authorization_base64:' "$se001q" >/dev/null
  grep -F -- 'stage_f_authorization_sha256:' "$se001q" >/dev/null
  input_count="$(grep -Ec '^[[:space:]]{6}(recovery_eligibility_manifest_base64|recovery_eligibility_manifest_sha256|stage_f_authorization_base64|stage_f_authorization_sha256):' "$se001q")"
  test "$input_count" = '4'

  if grep -Eq '^[[:space:]]+-[[:space:]]+uses:' "$se001q"; then
    echo 'SE-001Q trusted recovery gained a third-party/local Action' >&2
    exit 1
  fi
  if grep -Eq '^[[:space:]]*(GH_TOKEN|GITHUB_TOKEN):' "$se001q"; then
    echo 'SE-001Q trusted recovery gained an explicit GitHub token' >&2
    exit 1
  fi

  # Freeze exact subject/reference identity and the provider-authority boundary.
  grep -F -- 'TARGET_COMMIT: 47de7f2a306cffb66b5505786220590aa5f42e90' "$se001q" >/dev/null
  grep -F -- 'TARGET_TREE: 7abdf2ede579b2b729b057ca64470d11eb551031' "$se001q" >/dev/null
  grep -F -- 'HOSTED_VERIFIER_COMMIT: 3dd33625162ec7137ef06c8079794a8d2aecc95d' "$se001q" >/dev/null
  grep -F -- "test \"\$ELIGIBILITY_SCHEMA\" = 'symthaea.trusted-runner.recovery-eligibility.v5'" "$se001q" >/dev/null
  grep -F -- "test \"\$AUTH_SCHEMA\" = 'symthaea.trusted-runner.stage-f-authorization.v2'" "$se001q" >/dev/null
  grep -F -- "test \"\$AUTH_MAX_USES\" = '1'" "$se001q" >/dev/null
  grep -F -- "test \"\$AUTH_CONSUMPTION_MODE\" = 'root-owned-host-ledger-v1'" "$se001q" >/dev/null
  grep -F -- "test \"\$AUTH_SCOPE\" = 'se001q-independent-provider-reobservation-only'" "$se001q" >/dev/null
  grep -F -- "test \"\$AUTH_QUALIFICATION_CLAIM\" = 'NONE'" "$se001q" >/dev/null
  grep -F -- "test \"\$AUTH_REPAIR_CLAIM\" = 'NONE'" "$se001q" >/dev/null

  # One-use authority and boot binding are mandatory before the Rust replay.
  grep -F -- 'socket=/run/symthaea-stage-f-authorization.sock' "$se001q" >/dev/null
  grep -F -- 'request("BOOT_ID\\n")' "$se001q" >/dev/null
  grep -F -- 'request(f"CONSUME {nonce} {authorization_sha256}\\n")' "$se001q" >/dev/null
  grep -F -- 'STAGE_F_AUTHORIZATION_CONSUMED=PASS' "$se001q" >/dev/null
  grep -F -- 'STAGE_F_AUTHORIZATION_CONSUMED:-' "$se001q" >/dev/null
  grep -F -- 'NOT_DEMONSTRATED' "$se001q" >/dev/null
  grep -F -- 'symthaea.trusted-runner.stage-f-consumption.v1' "$se001q" >/dev/null
  grep -F -- 'symthaea.se001q.trusted-cpu-execution-binding.v2' "$se001q" >/dev/null
  grep -F -- 'symthaea.se001q.trusted-cpu-partial-rejection.v4' "$se001q" >/dev/null
  grep -F -- 'symthaea.se001q.trusted-cpu-partial-manifest.v4' "$se001q" >/dev/null

  # The trusted helper remains the only implementation of the frozen Rust gate
  # capture; unmerged EV2.4 Python is authenticated as reference data, not run.
  grep -F -- '--command python3 "$HARNESS_DIR/nix/ci/se001q-trusted-replay.py"' "$se001q" >/dev/null
  grep -F -- 'This run cannot replace the hosted EV2.4 run, qualify SE-001, or grant repair authority.' "$se001q" >/dev/null

  touch "$out"
''
