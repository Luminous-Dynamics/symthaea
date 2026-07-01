#!/usr/bin/env bash
# Compliance Dashboard — Automated compliance status check
#
# Runs all compliance-relevant test suites and outputs a summary.
# Designed for auditors and CI to verify current compliance state.
#
# Usage:
#   bash symthaea/scripts/compliance-dashboard.sh [--json]

set -euo pipefail

cd "$(dirname "$0")/.."

JSON_MODE=false
if [ "${1:-}" = "--json" ]; then
    JSON_MODE=true
fi

# Colors (disabled in JSON mode)
if $JSON_MODE; then
    GREEN="" YELLOW="" RED="" RESET="" BOLD=""
else
    GREEN="\033[32m" YELLOW="\033[33m" RED="\033[31m" RESET="\033[0m" BOLD="\033[1m"
fi

# Track results
declare -A SUITE_STATUS
declare -A SUITE_COUNT
TOTAL_PASS=0
TOTAL_FAIL=0
SUITES_PASS=0
SUITES_FAIL=0

run_suite() {
    local name="$1"
    local cmd="$2"
    local feature_note="${3:-}"

    if ! $JSON_MODE; then
        printf "  %-45s " "$name"
    fi

    local output
    local exit_code=0
    output=$(eval "$cmd" 2>&1) || exit_code=$?

    # Parse test counts from cargo test output
    local passed=0 failed=0
    if echo "$output" | grep -qE "^test result:"; then
        passed=$(echo "$output" | grep -oP '\d+ passed' | grep -oP '\d+' || echo "0")
        failed=$(echo "$output" | grep -oP '\d+ failed' | grep -oP '\d+' || echo "0")
    fi

    SUITE_COUNT["${name}_pass"]=$passed
    SUITE_COUNT["${name}_fail"]=$failed
    TOTAL_PASS=$((TOTAL_PASS + passed))
    TOTAL_FAIL=$((TOTAL_FAIL + failed))

    if [ $exit_code -eq 0 ]; then
        SUITE_STATUS["$name"]="PASS"
        SUITES_PASS=$((SUITES_PASS + 1))
        if ! $JSON_MODE; then
            printf "${GREEN}PASS${RESET}  (%s tests)\n" "$passed"
        fi
    else
        SUITE_STATUS["$name"]="FAIL"
        SUITES_FAIL=$((SUITES_FAIL + 1))
        if ! $JSON_MODE; then
            printf "${RED}FAIL${RESET}  (%s passed, %s failed)\n" "$passed" "$failed"
        fi
    fi
}

if ! $JSON_MODE; then
    echo ""
    echo -e "${BOLD}═══════════════════════════════════════════════════════${RESET}"
    echo -e "${BOLD}  Symthaea Compliance Dashboard$(date +'  %Y-%m-%d %H:%M')${RESET}"
    echo -e "${BOLD}═══════════════════════════════════════════════════════${RESET}"
    echo ""
    echo -e "${BOLD}Safety & Governance${RESET}"
fi

run_suite "Safety Agent (unit tests)" \
    "cargo test --lib --features safety-agents -- safety::agent 2>&1"

run_suite "Safety Agent (escalation soak)" \
    "cargo test --test safety_agent_escalation_soak --features safety-agents 2>&1"

run_suite "Safety Audit Report" \
    "cargo test --lib --features safety-agents -- safety::audit 2>&1"

if ! $JSON_MODE; then
    echo ""
    echo -e "${BOLD}Ethics & Moral Reasoning${RESET}"
fi

run_suite "Adversarial Moral Algebra" \
    "cargo test --test adversarial_moral_algebra 2>&1"

run_suite "Moral Algebra (unit tests)" \
    "cargo test --lib -- hdc::moral_algebra 2>&1"

run_suite "Ethics Engine" \
    "cargo test --lib -- cognitive_loop::ethics_engine 2>&1"

if ! $JSON_MODE; then
    echo ""
    echo -e "${BOLD}Consciousness & Substrate${RESET}"
fi

run_suite "Substrate Independence" \
    "cargo test -p symthaea-core --lib substrate_independence 2>&1"

run_suite "Substrate Validation" \
    "cargo test -p symthaea-core --lib substrate_validation 2>&1"

run_suite "Consciousness Engine" \
    "cargo test --lib -- cognitive_loop::consciousness_engine 2>&1"

if ! $JSON_MODE; then
    echo ""
    echo -e "${BOLD}Robustness & Stability${RESET}"
fi

run_suite "Proptest Feedback Stability" \
    "cargo test --test proptest_feedback_stability 2>&1"

run_suite "Proptest Threshold Sensitivity" \
    "cargo test --test proptest_threshold_sensitivity 2>&1"

run_suite "Calibration E2E" \
    "cargo test --test calibration_e2e 2>&1"

run_suite "Substrate Simulation" \
    "cargo test --test substrate_simulation 2>&1"

if ! $JSON_MODE; then
    echo ""
    echo -e "${BOLD}Governance Enforcement${RESET}"
fi

# Check governance hook is installed (skip in CI — hooks are local dev tools)
HOOK_STATUS="PASS"
if [ -z "$CI" ]; then
    GIT_DIR=$(git rev-parse --git-dir 2>/dev/null || echo ".git")
    if [ -f "$GIT_DIR/hooks/commit-msg" ] && grep -q "check-class-a-changes" "$GIT_DIR/hooks/commit-msg" 2>/dev/null; then
        HOOK_STATUS="PASS"
    else
        HOOK_STATUS="FAIL"
    fi
fi
SUITE_STATUS["Commit-msg governance hook"]=$HOOK_STATUS
if ! $JSON_MODE; then
    printf "  %-45s " "Commit-msg governance hook"
    if [ "$HOOK_STATUS" = "PASS" ]; then
        printf "${GREEN}PASS${RESET}  (installed)\n"
        SUITES_PASS=$((SUITES_PASS + 1))
    else
        printf "${RED}FAIL${RESET}  (not installed)\n"
        SUITES_FAIL=$((SUITES_FAIL + 1))
    fi
fi

# Check compliance docs exist
DOCS_MISSING=0
DOCS_CHECKED=0
for doc in \
    "docs/compliance/AI_RISK_REGISTER.md" \
    "docs/compliance/COMPLIANCE_MATRIX.md" \
    "docs/compliance/EU_AI_ACT_CLASSIFICATION.md" \
    "docs/compliance/GOVERNANCE_CHARTER.md" \
    "docs/compliance/DATA_GOVERNANCE.md" \
    "docs/compliance/INCIDENT_RUNBOOK.md" \
    "docs/compliance/TECHNICAL_DOSSIER.md" \
    "docs/compliance/CONFORMITY_ASSESSMENT.md" \
    "docs/compliance/SDLC.md" \
    "docs/compliance/RISK_TREATMENT_PLAN.md" \
    "docs/compliance/EXPLAINABILITY_FRAMEWORK.md" \
    "docs/compliance/QMS.md" \
    "docs/compliance/POST_MARKET_MONITORING.md" \
    "docs/compliance/TRANSPARENCY_OBLIGATIONS.md" \
    "docs/compliance/HUMAN_OVERSIGHT.md" \
    "docs/compliance/ANNEX_IV_TECHNICAL_DOCUMENTATION.md" \
    "docs/compliance/VALUE_VERIFICATION.md" \
    "docs/compliance/DEVELOPMENT_PROCEDURES.md" \
    "docs/compliance/DATA_QUALITY_FRAMEWORK.md" \
    "docs/compliance/ACCOUNTABILITY_MATRIX.md" \
    "docs/compliance/BENEFITS_COSTS_ANALYSIS.md" \
    "docs/compliance/RESOURCE_ALLOCATION.md" \
    "docs/compliance/EXTERNAL_FEEDBACK_PROTOCOL.md" \
    "docs/compliance/VALUE_VALIDATION_PROTOCOL.md"; do
    DOCS_CHECKED=$((DOCS_CHECKED + 1))
    if [ ! -f "$doc" ]; then
        DOCS_MISSING=$((DOCS_MISSING + 1))
    fi
done

DOC_STATUS="PASS"
if [ $DOCS_MISSING -gt 0 ]; then
    DOC_STATUS="FAIL"
fi
SUITE_STATUS["Compliance documentation"]=$DOC_STATUS
if ! $JSON_MODE; then
    printf "  %-45s " "Compliance documentation"
    if [ "$DOC_STATUS" = "PASS" ]; then
        printf "${GREEN}PASS${RESET}  (%s/%s docs present)\n" "$((DOCS_CHECKED - DOCS_MISSING))" "$DOCS_CHECKED"
        SUITES_PASS=$((SUITES_PASS + 1))
    else
        printf "${YELLOW}PARTIAL${RESET}  (%s/%s docs present)\n" "$((DOCS_CHECKED - DOCS_MISSING))" "$DOCS_CHECKED"
        SUITES_FAIL=$((SUITES_FAIL + 1))
    fi
fi

# Summary
TOTAL_SUITES=$((SUITES_PASS + SUITES_FAIL))

if $JSON_MODE; then
    cat <<ENDJSON
{
  "timestamp": "$(date -Iseconds)",
  "suites_pass": $SUITES_PASS,
  "suites_fail": $SUITES_FAIL,
  "suites_total": $TOTAL_SUITES,
  "tests_pass": $TOTAL_PASS,
  "tests_fail": $TOTAL_FAIL,
  "tests_total": $((TOTAL_PASS + TOTAL_FAIL)),
  "overall": "$([ $SUITES_FAIL -eq 0 ] && echo "PASS" || echo "FAIL")"
}
ENDJSON
else
    echo ""
    echo -e "${BOLD}═══════════════════════════════════════════════════════${RESET}"
    if [ $SUITES_FAIL -eq 0 ]; then
        echo -e "  ${GREEN}${BOLD}ALL SUITES PASS${RESET}  ($SUITES_PASS/$TOTAL_SUITES suites, $TOTAL_PASS tests)"
    else
        echo -e "  ${RED}${BOLD}$SUITES_FAIL SUITE(S) FAILED${RESET}  ($SUITES_PASS/$TOTAL_SUITES passed, $TOTAL_PASS/$((TOTAL_PASS + TOTAL_FAIL)) tests)"
    fi
    echo -e "${BOLD}═══════════════════════════════════════════════════════${RESET}"
    echo ""
fi

exit $SUITES_FAIL
