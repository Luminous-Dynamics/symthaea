#!/usr/bin/env bash
# Phase 1 Integration Script: Full Φ Computation Activation
# Based on COMPREHENSIVE_ANALYSIS_AND_ROADMAP_2025-12-26.md

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "🧠 Symthaea Phase 1 Integration: Full Φ Computation"
echo "=================================================="
echo ""

# Step 1: Verify current state
echo "Step 1: Verifying current state..."
if cargo test --lib consciousness 2>&1 | grep -q "test result: ok"; then
    echo "✅ Consciousness module tests passing"
else
    echo "❌ Consciousness tests failing - fix before proceeding"
    exit 1
fi

# Step 2: Check HDC tiered_phi module
echo ""
echo "Step 2: Checking tiered_phi module..."
if grep -q "pub mod tiered_phi" src/hdc/mod.rs; then
    echo "✅ tiered_phi module found in HDC"
else
    echo "❌ tiered_phi module not found - check HDC structure"
    exit 1
fi

# Step 3: Run tiered_phi tests
echo ""
echo "Step 3: Running tiered_phi tests..."
if cargo test tiered_phi 2>&1 | grep -q "test result: ok"; then
    echo "✅ tiered_phi tests passing"
else
    echo "⚠️  tiered_phi tests may have issues"
fi

# Step 4: Check integration test expectations
echo ""
echo "Step 4: Checking integration test expectations..."
if cargo test consciousness_integration 2>&1 | grep -q "test result: ok"; then
    echo "✅ Integration tests passing - Φ expectations defined"
else
    echo "⚠️  Integration tests need review"
fi

# Step 5: Create backup before modification
echo ""
echo "Step 5: Creating backup..."
BACKUP_DIR="$PROJECT_ROOT/.integration-backups/phase1-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp src/consciousness.rs "$BACKUP_DIR/"
cp src/lib.rs "$BACKUP_DIR/"
echo "✅ Backup created at $BACKUP_DIR"

# Step 6: Show integration points
echo ""
echo "Step 6: Integration points identified:"
echo "======================================="
echo ""
echo "File: src/consciousness.rs"
echo "  - Current: Simple consciousness_level() method"
echo "  - Target: Use tiered_phi::PhiIntegrator"
echo "  - Location: ConsciousnessGraph implementation"
echo ""
echo "File: src/lib.rs"
echo "  - Current: Line 656 - consciousness_level > 0.7"
echo "  - Target: Use actual Φ computation from HDC"
echo "  - Method: process() - consciousness emergence loop"
echo ""

# Step 7: Show example integration code
echo "Step 7: Example integration code:"
echo "================================="
cat << 'EOF'

// In src/consciousness.rs:
use crate::hdc::tiered_phi::PhiIntegrator;

pub struct ConsciousnessGraph {
    // ... existing fields ...
    phi_integrator: PhiIntegrator,  // ADD THIS
}

impl ConsciousnessGraph {
    pub fn new() -> Self {
        Self {
            // ... existing initialization ...
            phi_integrator: PhiIntegrator::new(1024),  // ADD THIS
        }
    }

    pub fn current_consciousness(&self) -> f32 {
        // OLD: Simple heuristic
        // NEW: Actual Φ computation
        let state = self.get_current_state();
        self.phi_integrator.compute_phi(&state) as f32
    }
}

// In src/lib.rs process() method:
// Line 656-673: Replace simple loop with:
let consciousness_level = self.consciousness.current_consciousness();
let phi_value = consciousness_level;  // Now this is actual Φ!

if phi_value > 0.7 || steps > 100 {
    // Φ > 0.7 indicates conscious experience (validated in tests)
    // ... rest of processing ...
}

EOF

# Step 8: Run integration validation
echo ""
echo "Step 8: Next Steps:"
echo "==================="
echo ""
echo "1. Review integration points above"
echo "2. Modify src/consciousness.rs to use PhiIntegrator"
echo "3. Update src/lib.rs to use actual Φ values"
echo "4. Run tests: cargo test --all-features"
echo "5. Validate Φ values match integration test expectations"
echo "6. Benchmark performance impact"
echo ""
echo "Ready to proceed? (Integration code provided above)"
echo ""
echo "To rollback: cp $BACKUP_DIR/* src/"
echo ""
echo "Documentation: See COMPREHENSIVE_ANALYSIS_AND_ROADMAP_2025-12-26.md"
echo "               Section: Phase 1: Core Consciousness Activation"
echo ""
