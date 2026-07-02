#!/usr/bin/env bash
"""
Auto-Triage: Early Warning System for Artifact Issues
======================================================

Scans latest artifacts for common issues:
1. Per-bucket FPR violations (>0.12)
2. Missing expected files
3. NaN values in metrics
4. Malformed JSON

Writes ALERTS.md with actionable diagnostics.

Usage:
    ./scripts/auto_triage.sh
    watch -n 300 ./scripts/auto_triage.sh  # Every 5 minutes

Author: Luminous Dynamics
Date: November 8, 2025
"""

set -euo pipefail

# Configuration
RESULTS_DIR="results"
ALERTS_FILE="ALERTS.md"
FPR_THRESHOLD=0.12
MIN_FILES_PER_ARTIFACT=6

# Colors for terminal output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo "======================================================================"
echo "Auto-Triage: Scanning Artifacts for Issues"
echo "======================================================================"
echo ""

# Find latest artifacts
LATEST_ARTIFACTS=$(ls -td "$RESULTS_DIR"/artifacts_* 2>/dev/null | head -n 10)

if [ -z "$LATEST_ARTIFACTS" ]; then
    echo "ℹ️  No artifacts found yet in $RESULTS_DIR/"
    exit 0
fi

ARTIFACT_COUNT=$(echo "$LATEST_ARTIFACTS" | wc -l)
echo "📊 Scanning $ARTIFACT_COUNT latest artifacts..."
echo ""

# Initialize alert tracking
ALERT_COUNT=0
ALERTS_CONTENT="# Artifact Alerts\n\n**Generated**: $(date -u '+%Y-%m-%d %H:%M:%S UTC')\n\n"

# ==================================================================
# Check 1: Per-Bucket FPR Violations
# ==================================================================
echo "🔍 Check 1: Per-Bucket FPR Compliance..."

FPR_VIOLATIONS=""
for artifact_dir in $LATEST_ARTIFACTS; do
    fpr_file="$artifact_dir/per_bucket_fpr.json"

    if [ ! -f "$fpr_file" ]; then
        continue
    fi

    # Extract buckets with FPR > threshold
    violations=$(jq -r --arg thresh "$FPR_THRESHOLD" \
        '.buckets[] | select(.fpr > ($thresh | tonumber)) | "\(.profile): FPR=\(.fpr) n=\(.n)"' \
        "$fpr_file" 2>/dev/null || echo "")

    if [ -n "$violations" ]; then
        experiment=$(basename "$artifact_dir")
        FPR_VIOLATIONS+="**$experiment**:\n"
        while IFS= read -r line; do
            FPR_VIOLATIONS+="  - $line\n"
        done <<< "$violations"
        FPR_VIOLATIONS+="\n"
        ((ALERT_COUNT++))
    fi
done

if [ -n "$FPR_VIOLATIONS" ]; then
    echo -e "${RED}❌ FPR violations detected${NC}"
    ALERTS_CONTENT+="## ❌ FPR Violations (Threshold: $FPR_THRESHOLD)\n\n$FPR_VIOLATIONS\n"
else
    echo -e "${GREEN}✅ All buckets within FPR threshold${NC}"
fi

# ==================================================================
# Check 2: Missing Files
# ==================================================================
echo "🔍 Check 2: Missing Files..."

MISSING_FILES=""
EXPECTED_FILES=(
    "detection_metrics.json"
    "per_bucket_fpr.json"
    "ema_vs_raw.json"
    "coord_median_diagnostics.json"
    "bootstrap_ci.json"
    "model_metrics.json"
)

for artifact_dir in $LATEST_ARTIFACTS; do
    experiment=$(basename "$artifact_dir")
    missing_in_this=""

    for expected_file in "${EXPECTED_FILES[@]}"; do
        if [ ! -f "$artifact_dir/$expected_file" ]; then
            missing_in_this+="  - $expected_file\n"
        fi
    done

    if [ -n "$missing_in_this" ]; then
        MISSING_FILES+="**$experiment**:\n$missing_in_this\n"
        ((ALERT_COUNT++))
    fi
done

if [ -n "$MISSING_FILES" ]; then
    echo -e "${RED}❌ Missing files detected${NC}"
    ALERTS_CONTENT+="## ❌ Missing Files\n\n$MISSING_FILES\n"
else
    echo -e "${GREEN}✅ All expected files present${NC}"
fi

# ==================================================================
# Check 3: NaN Values in Metrics
# ==================================================================
echo "🔍 Check 3: NaN Values..."

NAN_VALUES=""
for artifact_dir in $LATEST_ARTIFACTS; do
    detection_file="$artifact_dir/detection_metrics.json"

    if [ ! -f "$detection_file" ]; then
        continue
    fi

    # Check for NaN in critical metrics
    has_nan=$(jq -r 'select(.auroc == "NaN" or .auroc == null or (.auroc | type) == "string") | true' \
        "$detection_file" 2>/dev/null || echo "")

    if [ "$has_nan" == "true" ]; then
        experiment=$(basename "$artifact_dir")
        NAN_VALUES+="**$experiment**: AUROC is NaN or invalid\n"
        ((ALERT_COUNT++))
    fi
done

if [ -n "$NAN_VALUES" ]; then
    echo -e "${RED}❌ NaN values detected${NC}"
    ALERTS_CONTENT+="## ❌ NaN Values in Metrics\n\n$NAN_VALUES\n"
else
    echo -e "${GREEN}✅ No NaN values detected${NC}"
fi

# ==================================================================
# Check 4: Malformed JSON
# ==================================================================
echo "🔍 Check 4: Malformed JSON..."

MALFORMED_JSON=""
for artifact_dir in $LATEST_ARTIFACTS; do
    experiment=$(basename "$artifact_dir")

    for json_file in "$artifact_dir"/*.json; do
        if [ ! -f "$json_file" ]; then
            continue
        fi

        # Test JSON parsing
        if ! jq empty "$json_file" 2>/dev/null; then
            filename=$(basename "$json_file")
            MALFORMED_JSON+="**$experiment**: $filename is malformed\n"
            ((ALERT_COUNT++))
        fi
    done
done

if [ -n "$MALFORMED_JSON" ]; then
    echo -e "${RED}❌ Malformed JSON detected${NC}"
    ALERTS_CONTENT+="## ❌ Malformed JSON Files\n\n$MALFORMED_JSON\n"
else
    echo -e "${GREEN}✅ All JSON files valid${NC}"
fi

# ==================================================================
# Write ALERTS.md
# ==================================================================
echo ""

if [ $ALERT_COUNT -eq 0 ]; then
    echo -e "${GREEN}✅ No issues detected in latest $ARTIFACT_COUNT artifacts${NC}"
    ALERTS_CONTENT+="## ✅ All Checks Passed\n\nNo issues detected in $ARTIFACT_COUNT latest artifacts.\n"
else
    echo -e "${YELLOW}⚠️  $ALERT_COUNT issue(s) detected${NC}"
    ALERTS_CONTENT="# 🚨 Artifact Alerts ($ALERT_COUNT Issues)\n\n**Generated**: $(date -u '+%Y-%m-%d %H:%M:%S UTC')\n\n$ALERTS_CONTENT"
fi

# Write to file
echo -e "$ALERTS_CONTENT" > "$ALERTS_FILE"
echo ""
echo "📝 Alert report written to: $ALERTS_FILE"

# ==================================================================
# Summary
# ==================================================================
echo ""
echo "======================================================================"
echo "Auto-Triage Complete"
echo "======================================================================"
echo "Scanned: $ARTIFACT_COUNT artifacts"
echo "Issues: $ALERT_COUNT"
echo ""

# Exit code: 0 if no issues, 1 if issues found
if [ $ALERT_COUNT -gt 0 ]; then
    exit 1
else
    exit 0
fi
