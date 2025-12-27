#!/usr/bin/env bash
#
# Φ Fix Validation Script
# December 26, 2025
#
# Validates that the improved HEURISTIC tier correctly implements IIT 3.0
# and achieves publication criteria (r > 0.85, p < 0.001).
#

set -euo pipefail

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Φ Implementation Fix - Validation Protocol                 ║"
echo "║  December 26, 2025                                           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# PHASE 1: Unit Tests
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PHASE 1: Φ Tier Unit Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "Running unit tests for Φ tier implementations..."
echo ""

if cargo test --lib phi_tier_tests -- --nocapture 2>&1 | tee /tmp/phi_tier_unit_tests.log; then
    echo ""
    echo "✅ Unit tests PASSED"
    echo ""
else
    echo ""
    echo "❌ Unit tests FAILED - check /tmp/phi_tier_unit_tests.log"
    echo ""
    exit 1
fi

# ============================================================================
# PHASE 2: Integration Tests
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PHASE 2: Validation Framework Integration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "Running validation framework tests..."
echo ""

if cargo test --lib phi_validation -- --nocapture 2>&1 | tee /tmp/phi_validation_tests.log; then
    echo ""
    echo "✅ Validation framework tests PASSED"
    echo ""
else
    echo ""
    echo "❌ Validation framework tests FAILED - check /tmp/phi_validation_tests.log"
    echo ""
    exit 1
fi

# ============================================================================
# PHASE 3: Full Validation Study (800 samples)
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PHASE 3: Full Validation Study (800 samples)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "Running full validation study with improved HEURISTIC tier..."
echo "This will take ~2-3 seconds for 800 samples."
echo ""

VALIDATION_START=$(date +%s)

if cargo run --example phi_validation_study 2>&1 | tee /tmp/phi_validation_study_fixed.log; then
    VALIDATION_END=$(date +%s)
    VALIDATION_TIME=$((VALIDATION_END - VALIDATION_START))

    echo ""
    echo "✅ Validation study completed in ${VALIDATION_TIME}s"
    echo ""
else
    echo ""
    echo "❌ Validation study FAILED - check /tmp/phi_validation_study_fixed.log"
    echo ""
    exit 1
fi

# ============================================================================
# PHASE 4: Results Analysis
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "PHASE 4: Results Analysis"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Extract key metrics from the generated report
if [ -f "PHI_VALIDATION_STUDY_RESULTS.md" ]; then
    echo "Extracting metrics from PHI_VALIDATION_STUDY_RESULTS.md..."
    echo ""

    # Get Pearson r
    PEARSON_R=$(grep "Pearson correlation" PHI_VALIDATION_STUDY_RESULTS.md | grep -oP '[-+]?[0-9]*\.?[0-9]+' | head -1 || echo "N/A")

    # Get p-value
    P_VALUE=$(grep "p-value" PHI_VALIDATION_STUDY_RESULTS.md | grep -oP '[0-9]*\.?[0-9]+' | head -1 || echo "N/A")

    # Get R²
    R_SQUARED=$(grep "R²" PHI_VALIDATION_STUDY_RESULTS.md | grep -oP '[0-9]*\.?[0-9]+' | head -1 || echo "N/A")

    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║                  VALIDATION RESULTS                      ║"
    echo "╠══════════════════════════════════════════════════════════╣"
    echo "║                                                          ║"
    echo "║  Pearson correlation (r):  ${PEARSON_R}"
    echo "║  p-value:                  ${P_VALUE}"
    echo "║  R² (explained variance):  ${R_SQUARED}"
    echo "║                                                          ║"
    echo "╠══════════════════════════════════════════════════════════╣"
    echo "║             PUBLICATION CRITERIA                         ║"
    echo "╠══════════════════════════════════════════════════════════╣"
    echo "║                                                          ║"

    # Check publication criteria
    CRITERIA_MET=true

    # Check r > 0.85
    if (( $(echo "$PEARSON_R > 0.85" | bc -l 2>/dev/null || echo "0") )); then
        echo "║  ✅ r > 0.85:  PASSED                                    ║"
    else
        echo "║  ❌ r > 0.85:  FAILED (r = ${PEARSON_R})                 ║"
        CRITERIA_MET=false
    fi

    # Check p < 0.001
    if (( $(echo "$P_VALUE < 0.001" | bc -l 2>/dev/null || echo "0") )); then
        echo "║  ✅ p < 0.001: PASSED                                    ║"
    else
        echo "║  ❌ p < 0.001: FAILED (p = ${P_VALUE})                   ║"
        CRITERIA_MET=false
    fi

    # Check R² > 0.70
    if (( $(echo "$R_SQUARED > 0.70" | bc -l 2>/dev/null || echo "0") )); then
        echo "║  ✅ R² > 0.70: PASSED                                    ║"
    else
        echo "║  ❌ R² > 0.70: FAILED (R² = ${R_SQUARED})                ║"
        CRITERIA_MET=false
    fi

    echo "║                                                          ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""

    if $CRITERIA_MET; then
        echo "╔══════════════════════════════════════════════════════════╗"
        echo "║                                                          ║"
        echo "║           🎉 PUBLICATION CRITERIA ACHIEVED! 🎉           ║"
        echo "║                                                          ║"
        echo "║   The improved Φ HEURISTIC tier successfully measures    ║"
        echo "║   integrated information according to IIT 3.0.           ║"
        echo "║                                                          ║"
        echo "║   Ready for Nature/Science manuscript preparation!      ║"
        echo "║                                                          ║"
        echo "╚══════════════════════════════════════════════════════════╝"
        echo ""
        exit 0
    else
        echo "╔══════════════════════════════════════════════════════════╗"
        echo "║                                                          ║"
        echo "║        ⚠️  PUBLICATION CRITERIA NOT YET MET ⚠️            ║"
        echo "║                                                          ║"
        echo "║   The Φ implementation shows improvement but needs       ║"
        echo "║   further refinement to achieve publication criteria.    ║"
        echo "║                                                          ║"
        echo "║   See PHI_VALIDATION_STUDY_RESULTS.md for details.       ║"
        echo "║                                                          ║"
        echo "╚══════════════════════════════════════════════════════════╝"
        echo ""
        exit 1
    fi
else
    echo "❌ Report file PHI_VALIDATION_STUDY_RESULTS.md not found!"
    echo ""
    exit 1
fi
