#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Local, runner-independent smoke checks for the Language–Code Capability Fabric.
# A successful run qualifies only the synthetic fixture/auditor contracts; it
# does not measure Symthaea model quality or validate live coding-agent behavior.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for required in cargo rustc sha256sum; do
    if ! command -v "$required" >/dev/null 2>&1; then
        echo "[language-code-capability] missing required executable: $required" >&2
        exit 127
    fi
done

git_head="$(git rev-parse HEAD 2>/dev/null || true)"
if [[ ! "$git_head" =~ ^[0-9a-f]{40}$ ]]; then
    echo "[language-code-capability] refusing to run without an exact Git HEAD" >&2
    exit 2
fi

OUT_DIR="${LANGUAGE_CODE_CAPABILITY_OUT_DIR:-target/language-code-capability/smoke}"
mkdir -p "$OUT_DIR"
MANIFEST="$OUT_DIR/run-manifest.txt"
: > "$MANIFEST"

record() {
    printf '%s=%s\n' "$1" "$2" >> "$MANIFEST"
}

record "schema_version" "language-code-capability-local-run-v1"
record "git_head" "$git_head"
record "cargo_version" "$(cargo --version)"
record "rustc_version" "$(rustc --version)"
record "host" "$(rustc -vV | sed -n 's/^host: //p')"
record "smoke_corpus_sha256" "$(
    sha256sum fixtures/language-code-capability/smoke-v1.json | cut -d ' ' -f 1
)"
record "smoke_schema_sha256" "$(
    sha256sum fixtures/language-code-capability/smoke-v1.schema.json | cut -d ' ' -f 1
)"
record "trajectory_corpus_sha256" "$(
    sha256sum fixtures/language-code-capability/trajectory-smoke-v1.json | cut -d ' ' -f 1
)"
record "trajectory_schema_sha256" "$(
    sha256sum fixtures/language-code-capability/trajectory-smoke-v1.schema.json | cut -d ' ' -f 1
)"
record "claim_ceiling" "synthetic auditor checks only; no model benchmark evaluated"

run_logged() {
    local name="$1"
    shift
    local log="$OUT_DIR/$name.log"

    printf '[language-code-capability] %s\n' "$name"
    record "${name}_command" "$*"

    if "$@" >"$log" 2>&1; then
        record "${name}_status" "passed"
        echo "[language-code-capability] PASS: $name"
    else
        local exit_code=$?
        record "${name}_status" "failed"
        record "${name}_exit_code" "$exit_code"
        echo "[language-code-capability] FAIL: $name (exit $exit_code)" >&2
        cat "$log" >&2
        exit "$exit_code"
    fi
}

run_logged "smoke_auditor_tests" cargo test --locked --example language_code_capability_smoke
run_logged "trajectory_auditor_tests" \
    cargo test --locked --example language_code_capability_trajectory_audit
run_logged "receipt_verifier_tests" \
    cargo test --locked --example language_code_capability_receipt_verify \
    --features receipt-verification
run_logged "smoke_auditor_run" cargo run --locked --example language_code_capability_smoke
run_logged "trajectory_auditor_run" \
    cargo run --locked --example language_code_capability_trajectory_audit
run_logged "receipt_negative_controls" \
    cargo run --locked --example language_code_capability_receipt_verify \
    --features receipt-verification

record "overall_status" "passed"
echo "[language-code-capability] all local synthetic-auditor gates passed"
echo "[language-code-capability] manifest: $MANIFEST"
