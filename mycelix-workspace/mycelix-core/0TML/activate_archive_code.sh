#!/usr/bin/env bash
#
# Archive Activation Script - Week 1 Day 1
# Systematically activate production code from archive/
#
set -euo pipefail

ARCHIVE_DIR="/srv/luminous-dynamics/Mycelix-Core/production-fl-system/archive"
TARGET_DIR="/srv/luminous-dynamics/Mycelix-Core/0TML"

echo "🚀 Archive Code Activation - Gen-4 Implementation"
echo "=================================================="
echo ""

# Create target directories
echo "📁 Creating target directories..."
mkdir -p "$TARGET_DIR/src/integration"
mkdir -p "$TARGET_DIR/src/byzantine_attacks"
mkdir -p "$TARGET_DIR/src/baselines"
mkdir -p "$TARGET_DIR/src/optimizations"
mkdir -p "$TARGET_DIR/tests/integration"

echo "✅ Directories created"
echo ""

# Phase 1: Core Dependencies (Required by other files)
echo "📦 Phase 1: Activating core dependencies..."

echo "  → pogq_system.py (PoGQ proof system)"
cp "$ARCHIVE_DIR/pogq_system.py" "$TARGET_DIR/src/pogq_system.py"

echo "  → performance_optimizations.py (GPU acceleration)"
cp "$ARCHIVE_DIR/performance_optimizations.py" "$TARGET_DIR/src/performance_optimizations.py"

echo "✅ Phase 1 complete"
echo ""

# Phase 2: Primary Integration Code
echo "🔗 Phase 2: Activating integration code..."

echo "  → holochain_pogq_integration.py"
cp "$ARCHIVE_DIR/holochain_pogq_integration.py" "$TARGET_DIR/src/integration/holochain_pogq.py"

echo "  → byzantine_fl_with_pogq.py"
cp "$ARCHIVE_DIR/byzantine_fl_with_pogq.py" "$TARGET_DIR/src/integration/byzantine_fl_pogq.py"

echo "✅ Phase 2 complete"
echo ""

# Phase 3: Attack Suite
echo "⚔️  Phase 3: Activating advanced attacks..."

echo "  → advanced_byzantine_attacks.py"
cp "$ARCHIVE_DIR/advanced_byzantine_attacks.py" "$TARGET_DIR/src/byzantine_attacks/advanced.py"

echo "✅ Phase 3 complete"
echo ""

# Phase 4: Baseline Aggregators (CRITICAL for Paper!)
echo "📊 Phase 4: Activating baseline aggregators..."

echo "  → state_of_art_aggregators.py (Multi-KRUM, Bulyan)"
cp "$ARCHIVE_DIR/state_of_art_aggregators.py" "$TARGET_DIR/src/baselines/sota_aggregators.py"

echo "✅ Phase 4 complete"
echo ""

# Phase 5: Testing Infrastructure
echo "🧪 Phase 5: Activating test suite..."

echo "  → test_pogq_integration.py"
cp "$ARCHIVE_DIR/test_pogq_integration.py" "$TARGET_DIR/tests/integration/test_pogq.py"

echo "✅ Phase 5 complete"
echo ""

# Summary
echo "📈 Activation Summary"
echo "===================="
echo ""
echo "Files Activated:"
echo "  • Core Dependencies: 2 files"
echo "  • Integration Code: 2 files"
echo "  • Attack Suite: 1 file"
echo "  • Baseline Aggregators: 1 file"
echo "  • Test Suite: 1 file"
echo ""
echo "Total: 7 production files activated"
echo ""

# Verify files exist
echo "🔍 Verification..."
REQUIRED_FILES=(
    "$TARGET_DIR/src/pogq_system.py"
    "$TARGET_DIR/src/performance_optimizations.py"
    "$TARGET_DIR/src/integration/holochain_pogq.py"
    "$TARGET_DIR/src/integration/byzantine_fl_pogq.py"
    "$TARGET_DIR/src/byzantine_attacks/advanced.py"
    "$TARGET_DIR/src/baselines/sota_aggregators.py"
    "$TARGET_DIR/tests/integration/test_pogq.py"
)

ALL_EXIST=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "  ❌ Missing: $file"
        ALL_EXIST=false
    else
        echo "  ✅ Present: $(basename "$file")"
    fi
done

echo ""
if [ "$ALL_EXIST" = true ]; then
    echo "✅ All files successfully activated!"
    echo ""
    echo "Next Steps:"
    echo "  1. Create attack registry: 0TML/src/byzantine_attacks/__init__.py"
    echo "  2. Run smoke tests: pytest 0TML/tests/integration/test_pogq.py"
    echo "  3. Begin PoGQ-v4 enhancements (Week 2)"
    exit 0
else
    echo "❌ Some files failed to activate"
    exit 1
fi
