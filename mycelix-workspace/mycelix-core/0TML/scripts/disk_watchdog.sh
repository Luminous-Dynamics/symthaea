#!/usr/bin/env bash
"""
Disk Watchdog: Protect Long-Running Experiments from Space Exhaustion
======================================================================

Monitors:
1. Free disk space (alert if < 10 GB)
2. Log file growth (alert if log hasn't grown in 30 min)
3. Artifact production (alert if no new artifacts in 2 hours)

Usage:
    ./scripts/disk_watchdog.sh                # Run once
    watch -n 300 ./scripts/disk_watchdog.sh   # Every 5 minutes

Author: Luminous Dynamics
Date: November 8, 2025
"""

set -euo pipefail

# Configuration
MIN_FREE_GB=10
LOG_FILE="logs/sanity_slice_fixed.log"
LOG_STALE_MINUTES=30
ARTIFACT_DIR="results"
ARTIFACT_STALE_HOURS=2

# Colors
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

echo "======================================================================"
echo "Disk Watchdog: Monitoring Experiment Health"
echo "======================================================================"
date -u '+%Y-%m-%d %H:%M:%S UTC'
echo ""

ALERTS=0

# ==================================================================
# Check 1: Free Disk Space
# ==================================================================
echo "🔍 Checking free disk space..."

# Get free space in GB
FREE_SPACE_KB=$(df /srv/luminous-dynamics | tail -1 | awk '{print $4}')
FREE_SPACE_GB=$((FREE_SPACE_KB / 1024 / 1024))

if [ $FREE_SPACE_GB -lt $MIN_FREE_GB ]; then
    echo -e "${RED}❌ CRITICAL: Only ${FREE_SPACE_GB} GB free (threshold: ${MIN_FREE_GB} GB)${NC}"
    echo "   Action: Clean up old artifacts or increase disk space"
    ((ALERTS++))
else
    echo -e "${GREEN}✅ Disk space OK: ${FREE_SPACE_GB} GB free${NC}"
fi

# ==================================================================
# Check 2: Log File Growth
# ==================================================================
echo ""
echo "🔍 Checking log file activity..."

if [ ! -f "$LOG_FILE" ]; then
    echo -e "${YELLOW}⚠️  Log file not found: $LOG_FILE${NC}"
    echo "   This may be normal if experiments haven't started writing yet"
else
    # Get file modification time
    LOG_AGE_SECONDS=$(( $(date +%s) - $(stat -c %Y "$LOG_FILE" 2>/dev/null || echo 0) ))
    LOG_AGE_MINUTES=$((LOG_AGE_SECONDS / 60))

    if [ $LOG_AGE_MINUTES -gt $LOG_STALE_MINUTES ]; then
        echo -e "${YELLOW}⚠️  Log file hasn't been modified in ${LOG_AGE_MINUTES} minutes${NC}"
        echo "   Threshold: ${LOG_STALE_MINUTES} minutes"
        echo "   This could indicate:"
        echo "     - Python output buffering (expected, not critical)"
        echo "     - Experiment crashed (check process status)"
        echo "     - Experiment paused (check CPU usage)"
        echo ""
        echo "   Verify process is still running:"
        echo "     ps aux | grep matrix_runner"
        ((ALERTS++))
    else
        echo -e "${GREEN}✅ Log file active (last modified ${LOG_AGE_MINUTES} min ago)${NC}"
    fi

    # Check log file size
    LOG_SIZE_MB=$(du -m "$LOG_FILE" | cut -f1)
    echo "   Log size: ${LOG_SIZE_MB} MB"

    if [ $LOG_SIZE_MB -gt 1000 ]; then
        echo -e "${YELLOW}⚠️  Log file is large (${LOG_SIZE_MB} MB)${NC}"
        echo "   Consider rotating or compressing if it grows too large"
    fi
fi

# ==================================================================
# Check 3: Artifact Production
# ==================================================================
echo ""
echo "🔍 Checking artifact production..."

if [ ! -d "$ARTIFACT_DIR" ]; then
    echo -e "${YELLOW}⚠️  Artifact directory not found: $ARTIFACT_DIR${NC}"
    echo "   This is normal if no experiments have completed yet"
else
    # Count artifact directories
    ARTIFACT_COUNT=$(ls -1d "$ARTIFACT_DIR"/artifacts_* 2>/dev/null | wc -l)

    if [ $ARTIFACT_COUNT -eq 0 ]; then
        echo -e "${YELLOW}ℹ️  No artifacts produced yet${NC}"
        echo "   This is normal during initial heavy I/O phase (dataset loading)"
    else
        echo -e "${GREEN}✅ Artifacts produced: $ARTIFACT_COUNT${NC}"

        # Check for recent artifacts
        LATEST_ARTIFACT=$(ls -td "$ARTIFACT_DIR"/artifacts_* 2>/dev/null | head -n 1)

        if [ -n "$LATEST_ARTIFACT" ]; then
            ARTIFACT_AGE_SECONDS=$(( $(date +%s) - $(stat -c %Y "$LATEST_ARTIFACT" 2>/dev/null || echo 0) ))
            ARTIFACT_AGE_HOURS=$((ARTIFACT_AGE_SECONDS / 3600))
            ARTIFACT_AGE_MINUTES=$((ARTIFACT_AGE_SECONDS / 60))

            if [ $ARTIFACT_AGE_HOURS -gt $ARTIFACT_STALE_HOURS ]; then
                echo -e "${YELLOW}⚠️  No new artifacts in ${ARTIFACT_AGE_HOURS} hours${NC}"
                echo "   Last artifact: $(basename "$LATEST_ARTIFACT")"
                echo "   This could indicate the experiment has stalled"
                ((ALERTS++))
            else
                echo "   Latest artifact: $(basename "$LATEST_ARTIFACT")"
                echo "   Age: ${ARTIFACT_AGE_MINUTES} minutes"
            fi
        fi
    fi
fi

# ==================================================================
# Summary
# ==================================================================
echo ""
echo "======================================================================"

if [ $ALERTS -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed${NC}"
else
    echo -e "${YELLOW}⚠️  ${ALERTS} alert(s) detected${NC}"
fi

echo "======================================================================"
echo ""

# Exit code: 0 if OK, 1 if alerts
exit $ALERTS
